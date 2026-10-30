import json
from itertools import groupby
from pathlib import Path

import numpy as np
from tqdm import tqdm

from routine_graph.agent.summarize_agent import SummarizeAgent
from routine_graph.database.faiss_controller import FaissController
from routine_graph.database.sqlite_controller import Database
from routine_graph.ingestion.habit_clustering import HabitCluster
from routine_graph.ingestion.routine_clustering import RoutineCluster
from routine_graph.utils.embedding import ActivityEmbedder

def ingest_persistent_events(json_path: str) -> list[tuple]:
    """Read one persistent_event_<PARTICIPANT>_<DAY>.json and return each event as a row for the events table

    Row: (participant, day, event_index, label, description, start_time, end_time), event_id is assigned by the database
    """

    #participant can contain "_" (e.g. A1_JAKE), the day is the part after the last "_"
    stem = Path(json_path).stem.removeprefix("persistent_event_")
    participant, day = stem.rsplit("_", 1)

    with open(json_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    rows = [(participant, day, idx,
             event["label"], event["description"], event["start_time"], event["end_time"])
            for idx, event in enumerate(events)]
    return rows

def build_contiguous_subsequences(event_rows: list[tuple]) -> list[tuple]:
    """Build every contiguous subsequence of length > 1 of each participant-day from events table rows

    Events of a participant-day are ordered by start_time (event_index is the order they ended)
    e.g. day events [a, b, c] -> [a, b], [a, b, c], [b, c]
    Row: (participant, day, start_position, length, event_ids (json list of int), start_time, end_time), sequence_id is assigned by the database
    """

    #sort by participant, day, start_time, end_time, event_index
    events = sorted(event_rows, key=lambda event: (event[1], event[2], event[6], event[7], event[3]))

    rows = []
    for (participant, day), day_events in groupby(events, key=lambda event: (event[1], event[2])):
        day_events = list(day_events)
        n = len(day_events)

        for start in range(n):
            for end in range(start + 2, n + 1):
                subsequence = day_events[start:end]
                rows.append((participant, day, start, end - start,
                             json.dumps([event[0] for event in subsequence]),
                             subsequence[0][6], #start_time of the first event
                             max(event[7] for event in subsequence))) #latest end_time, events can overlap
    return rows

def ingest_json(json_paths: list[str], db: Database) -> tuple[int, int]:
    """Store the events of every persistent_event json, then rebuild the sequences table from all stored events

    Return (number of events ingested, number of subsequences stored)
    """

    n_events = 0
    for json_path in tqdm(json_paths, desc="Ingest jsons"):
        event_rows = ingest_persistent_events(json_path)
        db.insert_events(event_rows)
        n_events += len(event_rows)

    #rebuild from every event in the database (not only the new ones) so each day's subsequences are complete
    sequence_rows = build_contiguous_subsequences(db.get_events())
    db.replace_sequences(sequence_rows)

    return n_events, len(sequence_rows)

def group_by_cluster(id_to_cluster: dict[int, int]) -> dict[int, list[int]]:
    """{id: cluster id} -> {cluster id: [ids]}, noise (-1) is dropped"""

    cluster_to_ids = {}
    for id_, cluster_id in id_to_cluster.items():
        if cluster_id != -1:
            cluster_to_ids.setdefault(cluster_id, []).append(id_)
    return cluster_to_ids

class IngestionPipeline:
    def __init__(self, db: Database, faiss_controller: FaissController, embedder: ActivityEmbedder, summarize_agent: SummarizeAgent,
                 habit_eps: float = 0.5, habit_min_samples: int = 5, routine_eps: float = 0.5, routine_min_samples: int = 5):
        self.db = db
        self.faiss_controller = faiss_controller
        self.embedder = embedder
        self.summarize_agent = summarize_agent
        self.habit_eps = habit_eps
        self.habit_min_samples = habit_min_samples
        self.routine_eps = routine_eps
        self.routine_min_samples = routine_min_samples

    def run(self, json_paths: list[str]) -> tuple[int, int]:
        """Ingest persistent_event jsons, embed every stored event, habit cluster events and routine cluster sequences, then summarize each cluster

        Return (number of events ingested, number of subsequences stored)
        """

        #events json -> events table -> contiguous subsequences -> sequences table
        n_events, n_sequences = ingest_json(json_paths, self.db)

        #embed every stored event and save the vectors to faiss, event_id is the faiss id
        events = self.db.get_events()
        event_ids = [event[0] for event in events]
        self.faiss_controller.add([self.embedder.embed(event[4], event[5]) for event in tqdm(events, desc="Embed events")], event_ids)

        #fetch the stored embeddings back from faiss
        vectors = self.faiss_controller.get(event_ids)
        id_to_vector = dict(zip(event_ids, vectors))

        #habit clustering on single events -> events.cluster_id
        habit_cluster = HabitCluster(list(vectors), event_ids, eps=self.habit_eps, min_samples=self.habit_min_samples, metric="cosine")
        self.db.update_event_clusters(habit_cluster.get_id_to_cluster())

        #summarize the events of each habit cluster -> habits table
        id_to_event = {event[0]: {"label": event[4], "description": event[5], "start_time": event[6], "end_time": event[7]} for event in events}
        habit_rows = []
        for habit_id, habit_event_ids in tqdm(group_by_cluster(habit_cluster.get_id_to_cluster()).items(), desc="Summarize habits"):
            summary = self.summarize_agent.summarize_events([id_to_event[event_id] for event_id in habit_event_ids])
            habit_rows.append((habit_id, summary.label, summary.description, json.dumps(habit_event_ids)))
        self.db.replace_habits(habit_rows)

        sequences = self.db.get_sequences()
        routine_rows = []
        if sequences: #no sequences e.g. only single-event days, nothing to cluster
            #each sequence row -> (length, dim) matrix of its events' embeddings
            sequence_ids = [sequence[0] for sequence in sequences]
            sequence_to_event_ids = {sequence[0]: json.loads(sequence[5]) for sequence in sequences}
            matrices = [np.stack([id_to_vector[event_id] for event_id in sequence_to_event_ids[sequence_id]]) for sequence_id in sequence_ids]

            #routine clustering on sequences -> sequences.cluster_id
            routine_cluster = RoutineCluster(matrices, sequence_ids, eps=self.routine_eps, min_samples=self.routine_min_samples)
            self.db.update_sequence_clusters(routine_cluster.get_id_to_cluster())

            #summarize the sequences of each routine cluster -> routines table
            for routine_id, routine_sequence_ids in tqdm(group_by_cluster(routine_cluster.get_id_to_cluster()).items(), desc="Summarize routines"):
                summary = self.summarize_agent.summarize_sequences([[id_to_event[event_id] for event_id in sequence_to_event_ids[sequence_id]]
                                                                    for sequence_id in routine_sequence_ids])
                routine_rows.append((routine_id, summary.label, summary.description, json.dumps(routine_sequence_ids)))
        self.db.replace_routines(routine_rows) #replaced even when empty so old routines don't stay

        return n_events, n_sequences
