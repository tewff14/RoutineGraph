"""Persistent activities to participant-specific habit and routine profiles."""

from collections import Counter, defaultdict
from datetime import date
import json
from itertools import groupby
from pathlib import Path
from uuid import uuid4

import numpy as np

from routine_graph.agent.summarize_agent import SummarizeAgent
from routine_graph.database.indexes import load_indexes
from routine_graph.database.sqlite_controller import Database
from routine_graph.ingestion.habit_clustering import HabitCluster
from routine_graph.ingestion.routine_clustering import RoutineCluster
from routine_graph.models import Activity, clock_seconds, normalized_vectors, temporal_vector
from routine_graph.utils.embedding import ActivityEmbedder


def read_persistent_events(json_path: str) -> tuple[str, str, list[tuple]]:
    with open(json_path, encoding="utf-8") as stream:
        content = json.load(stream)
    metadata = content if isinstance(content, dict) else {}
    events = metadata.get("events") if metadata else content
    if not isinstance(events, list):
        raise ValueError(f"{json_path}: expected an activity list or an object containing events")
    participant, day = metadata.get("participant"), metadata.get("day")
    if not participant or not day:
        stem = Path(json_path).stem
        if not stem.startswith("persistent_event_") or "_" not in stem.removeprefix("persistent_event_"):
            raise ValueError("Use persistent_event_<PARTICIPANT>_<DAY>.json or specify participant and day in an events object")
        participant, day = stem.removeprefix("persistent_event_").rsplit("_", 1)
    if not isinstance(participant, str) or not participant.strip() or not isinstance(day, str) or not day.strip():
        raise ValueError("Participant and day must be nonempty strings")
    rows = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"{json_path}: activity {index} must be an object")
        activity = Activity.model_validate({**({"date": metadata["date"]} if metadata.get("date") else {}), **event})
        rows.append((participant, day, index, activity.label, activity.description, activity.start_time,
                     activity.end_time, json.dumps(activity.metadata())))
    return participant, day, rows


def ingest_persistent_events(json_path: str) -> list[tuple]:
    return read_persistent_events(json_path)[2]


def build_contiguous_subsequences(event_rows: list[tuple], max_length: int | None = 5) -> list[tuple]:
    """Build windows of 2..max_length activities; None or 0 permits all lengths."""
    if max_length is not None and (max_length < 0 or max_length == 1):
        raise ValueError("Maximum sequence length must be at least 2, or 0 for all lengths")
    events = sorted(event_rows, key=lambda event: (event[1], event[2], event[6], event[7], event[3]))
    rows = []
    for (participant, day), day_events in groupby(events, key=lambda event: (event[1], event[2])):
        day_events = list(day_events)
        n = len(day_events)
        for start in range(n):
            for end in range(start + 2, min(start + (max_length or n), n) + 1):
                subsequence = day_events[start:end]
                #compare elapsed end times to retain events that cross midnight
                latest = max(subsequence, key=lambda event: clock_seconds(event[7]) +
                             (86400 if clock_seconds(event[7]) < clock_seconds(event[6]) else 0))
                rows.append((participant, day, start, end - start, json.dumps([event[0] for event in subsequence]),
                             subsequence[0][6], latest[7]))
    return rows


def ingest_json(json_paths: list[str], db: Database, max_length: int | None = 5) -> tuple[int, int]:
    if isinstance(json_paths, (str, Path)):
        raise ValueError("Pass a list of JSON paths, not a single string")
    batches = [read_persistent_events(path) for path in json_paths]
    identities = [(participant, day) for participant, day, _ in batches]
    if len(set(identities)) != len(identities):
        raise ValueError("Supply only one source JSON for each participant/day")
    with db.transaction():
        for participant, day, rows in batches:
            db.replace_day_events(participant, day, rows)
        sequences = build_contiguous_subsequences(db.get_events(), max_length)
        db.replace_sequences(sequences)
    return sum(len(rows) for _, _, rows in batches), len(sequences)


def group_by_cluster(id_to_cluster: dict[int, int]) -> dict[int, list[int]]:
    groups = {}
    for id_, cluster in id_to_cluster.items():
        if cluster != -1:
            groups.setdefault(cluster, []).append(id_)
    return groups


def recurring_clusters(mapping: dict[int, int], days: dict[int, str], min_days: int) -> list[list[int]]:
    return [ids for ids in group_by_cluster(mapping).values() if len({days[id_] for id_ in ids}) >= min_days]


def profile_statistics(occurrences: list[dict]) -> dict:
    def typical_clock(field):
        angles = np.array([clock_seconds(item[field]) for item in occurrences]) * 2 * np.pi / 86400
        angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()) % (2 * np.pi)
        seconds = round(angle * 86400 / (2 * np.pi)) % 86400
        return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    weekdays = Counter(date.fromisoformat(item["metadata"]["date"]).strftime("%A")
                       for item in occurrences if item.get("metadata", {}).get("date"))
    durations = [(clock_seconds(item["end_time"]) - clock_seconds(item["start_time"])) % 86400 for item in occurrences]
    return {"typical_start_time": typical_clock("start_time"), "typical_end_time": typical_clock("end_time"),
            "median_duration_seconds": float(np.median(durations)), "weekday_counts": dict(weekdays)}


def representative_items(items: list[dict], limit: int = 40) -> list[dict]:
    """Round-robin days so one long recording cannot consume the summary context."""
    groups = defaultdict(list)
    for item in items:
        groups[item["day"]].append(item)
    result = []
    while groups and len(result) < limit:
        for day in list(groups):
            if len(result) == limit:
                break
            result.append(groups[day].pop(0))
            if not groups[day]:
                del groups[day]
    return result


class IngestionPipeline:
    def __init__(self, db_path: str = "data/routine_graph.db", index_path: str = "data/events.faiss",
                 embedding_model: str | None = None, embedding_provider: str | None = None, embedding_dim: int | None = None,
                 summarize_model: str = "google/gemini-3.6-flash", summarize_model_provider: str = "openrouter",
                 habit_eps: float = 0.5, habit_min_samples: int = 5, routine_eps: float = 0.5, routine_min_samples: int = 5,
                 max_sequence_length: int | None = 5, min_days: int = 2, temporal_weight: float = 0.5,
                 embedding_base_url: str | None = None, embedder=None, summarize_agent=None,
                 max_routine_candidates: int = 2000):
        if min_days < 2:
            raise ValueError("Profiles require at least two distinct recording days")
        if habit_min_samples < 1 or routine_min_samples < 1 or habit_eps <= 0 or routine_eps <= 0:
            raise ValueError("DBSCAN sample counts and epsilon values must be positive")
        if not np.isfinite(temporal_weight) or temporal_weight < 0 or max_routine_candidates < 0:
            raise ValueError("Temporal weight and candidate limit must be nonnegative")
        build_contiguous_subsequences([], max_sequence_length)
        self.db = Database(db_path)
        self.index_path = index_path
        self.embedder = embedder or ActivityEmbedder(embedding_model, provider=embedding_provider,
                                                    base_url=embedding_base_url, dimensions=embedding_dim)
        self._owns_embedder = embedder is None
        self.summarize_agent = summarize_agent
        self.summarize_model = summarize_model
        self.summarize_model_provider = summarize_model_provider
        self.habit_eps, self.habit_min_samples = habit_eps, habit_min_samples
        self.routine_eps, self.routine_min_samples = routine_eps, routine_min_samples
        self.max_sequence_length, self.min_days = max_sequence_length, min_days
        self.temporal_weight, self.max_routine_candidates = temporal_weight, max_routine_candidates
        self.indexes = {}

    def _summarizer(self):
        if self.summarize_agent is None:
            self.summarize_agent = SummarizeAgent(self.summarize_model, model_provider=self.summarize_model_provider)
        return self.summarize_agent

    def close(self):
        self.db.close()
        if self._owns_embedder:
            self.embedder.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def run(self, json_paths: list[str]) -> tuple[int, int]:
        """Publish one consistent snapshot; failures keep the preceding snapshot intact."""
        with self.db.transaction():
            counts = ingest_json(json_paths, self.db, self.max_sequence_length)
            events = self.db.records("activity")
            sequences = self.db.get_sequences()
            participant_counts = Counter(sequence[1] for sequence in sequences)
            if self.max_routine_candidates and any(n > self.max_routine_candidates for n in participant_counts.values()):
                raise ValueError(f"More than {self.max_routine_candidates} routine candidates for one participant; "
                                 "reduce sequence length/data scope or explicitly raise --max-routine-candidates (0 disables the limit)")
            vectors, pending, texts = {}, [], {}
            for event in events:
                id_ = event["event_id"]
                text = ActivityEmbedder.activity_text(event["label"], event["description"])
                texts[id_] = text
                cached = self.db.cached_embedding(id_, text, self.embedder.fingerprint)
                if cached is None:
                    pending.append(id_)
                else:
                    vectors[id_] = cached
            if pending:
                embedded = self.embedder.embed_texts([texts[id_] for id_ in pending])
                if len(embedded) != len(pending):
                    raise ValueError("Embedding provider returned the wrong number of vectors")
                for id_, vector in zip(pending, normalized_vectors(embedded)):
                    vectors[id_] = vector
                    self.db.store_event_embedding(id_, texts[id_], self.embedder.fingerprint, vector)
            dimensions = {len(vector) for vector in vectors.values()}
            if len(dimensions) > 1:
                raise ValueError("Embedding dimensions changed within a run")
            dimension = next(iter(dimensions), self.db.setting("embedding_dimension"))
            id_to_event = {event["event_id"]: event for event in events}
            features = {event["event_id"]: temporal_vector(vectors[event["event_id"]], event["start_time"],
                        event["end_time"], self.temporal_weight) for event in events}
            habit_mapping, routine_mapping = {}, {}
            habit_rows, routine_rows = [], []
            participants = sorted({event["participant"] for event in events})
            for participant in participants:
                own_events = [event for event in events if event["participant"] == participant]
                ids = [event["event_id"] for event in own_events]
                clusters = HabitCluster([features[id_] for id_ in ids], ids, eps=self.habit_eps,
                                        min_samples=self.habit_min_samples, metric="cosine")
                days = {event["event_id"]: event["day"] for event in own_events}
                for members in recurring_clusters(clusters.id_to_cluster, days, self.min_days):
                    profile_id = len(habit_rows)
                    observations = [id_to_event[id_] for id_ in members]
                    summary = self._summarizer().summarize_events(representative_items(observations))
                    habit_rows.append((profile_id, summary.label, summary.description, json.dumps(members), participant,
                                       len(members), len({days[id_] for id_ in members}), json.dumps(profile_statistics(observations))))
                    habit_mapping.update({id_: profile_id for id_ in members})
                own_sequences = [sequence for sequence in sequences if sequence[1] == participant]
                sequence_ids = [sequence[0] for sequence in own_sequences]
                memberships = {sequence[0]: json.loads(sequence[5]) for sequence in own_sequences}
                matrices = [np.stack([features[id_] for id_ in memberships[sequence_id]]) for sequence_id in sequence_ids]
                clusters = RoutineCluster(matrices, sequence_ids, eps=self.routine_eps, min_samples=self.routine_min_samples)
                sequence_days = {sequence[0]: sequence[2] for sequence in own_sequences}
                sequence_lookup = {sequence[0]: sequence for sequence in own_sequences}
                for members in recurring_clusters(clusters.id_to_cluster, sequence_days, self.min_days):
                    profile_id = len(routine_rows)
                    occurrences = [{"day": sequence_lookup[id_][2], "start_time": sequence_lookup[id_][6],
                                    "end_time": sequence_lookup[id_][7],
                                    "metadata": id_to_event[memberships[id_][0]]["metadata"], "id": id_} for id_ in members]
                    selected = representative_items(occurrences, 20)
                    summary = self._summarizer().summarize_sequences([[id_to_event[id_] for id_ in memberships[item["id"]]] for item in selected])
                    routine_rows.append((profile_id, summary.label, summary.description, json.dumps(members), participant,
                                         len(members), len({sequence_days[id_] for id_ in members}), json.dumps(profile_statistics(occurrences))))
                    routine_mapping.update({id_: profile_id for id_ in members})
            self.db.update_event_clusters(habit_mapping)
            self.db.update_sequence_clusters(routine_mapping)
            self.db.replace_habits(habit_rows)
            self.db.replace_routines(routine_rows)
            profile_rows = [(channel, row[0], row[1] + ": " + row[2] + " " + row[7])
                            for channel, rows in (("habit", habit_rows), ("routine", routine_rows)) for row in rows]
            profile_vectors = self.embedder.embed_texts([row[2] for row in profile_rows]) if profile_rows else []
            if len(profile_vectors) != len(profile_rows):
                raise ValueError("Embedding provider returned the wrong number of profile vectors")
            if profile_rows:
                profile_vectors = normalized_vectors(profile_vectors, dimension)
            self.db.replace_profile_embeddings([(row[0], row[1], vector) for row, vector in zip(profile_rows, profile_vectors)])
            self.db.set_setting("embedding_configuration", self.embedder.configuration)
            self.db.set_setting("embedding_fingerprint", self.embedder.fingerprint)
            self.db.set_setting("embedding_dimension", dimension)
            self.db.set_setting("discovery", {"temporal_weight": self.temporal_weight, "min_days": self.min_days,
                "max_sequence_length": self.max_sequence_length, "habit_eps": self.habit_eps,
                "habit_min_samples": self.habit_min_samples, "routine_eps": self.routine_eps,
                "routine_min_samples": self.routine_min_samples})
            self.db.set_setting("revision", uuid4().hex)
        self.indexes = load_indexes(self.db, self.index_path, rebuild=True)
        return counts
