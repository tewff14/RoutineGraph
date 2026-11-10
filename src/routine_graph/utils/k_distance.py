import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from routine_graph.database.indexes import load_indexes
from routine_graph.database.sqlite_controller import Database
from routine_graph.ingestion.routine_clustering import RoutineCluster
from routine_graph.models import temporal_vector


def k_distance(distance_matrix: np.ndarray, k: int) -> np.ndarray:
    """Sorted distance to the k-th other neighbour (DBSCAN min_samples - 1)."""
    matrix = np.asarray(distance_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not 1 <= k < len(matrix):
        raise ValueError("Need a square distance matrix and 1 <= k < number of observations")
    return np.sort(np.sort(matrix, axis=1)[:, k])


def plot_k_distance(distances: np.ndarray, k: int, title: str, path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(distances, marker=".")
    plt.title(title)
    plt.xlabel("points sorted by k-distance")
    plt.ylabel(f"distance to k-th nearest neighbour (k={k})")
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Temporal habit/routine k-distance plots for one participant")
    parser.add_argument("--db", required=True)
    parser.add_argument("--faiss", required=True)
    parser.add_argument("--participant")
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--output", default=".")
    args = parser.parse_args()
    if not Path(args.db).is_file():
        parser.error("Run ingestion first; database must exist")
    if args.min_samples < 2:
        parser.error("min_samples must be at least 2 for a k-distance plot")
    db = Database(args.db)
    try:
        all_events = db.get_events()
        participants = {event[1] for event in all_events}
        if args.participant is None and len(participants) != 1:
            parser.error("Select --participant when there is not exactly one participant")
        participant = args.participant or next(iter(participants))
        events = [event for event in all_events if event[1] == participant]
        k = args.min_samples - 1
        if len(events) <= k:
            parser.error("Not enough activities for the selected min_samples")
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        controller = load_indexes(db, args.faiss).get("activity")
        if controller is None:
            parser.error("Re-ingest activities to populate stored vectors")
        ids = [event[0] for event in events]
        vectors = controller.get(ids)
        weight = db.setting("discovery", {}).get("temporal_weight", 0.5)
        features = np.stack([temporal_vector(vector, event[6], event[7], weight) for event, vector in zip(events, vectors)])
        distances = k_distance(np.clip(1 - features @ features.T, 0, 2), k)
        plot_k_distance(distances, k, f"Habit k-distance: {participant}", str(output / "habit_k_distance.png"))
        sequences = [sequence for sequence in db.get_sequences() if sequence[1] == participant]
        if len(sequences) > k:
            lookup = dict(zip(ids, features))
            matrices = [np.stack([lookup[id_] for id_ in json.loads(sequence[5])]) for sequence in sequences]
            distances = k_distance(RoutineCluster(matrices, [sequence[0] for sequence in sequences]).distance_matrix, k)
            plot_k_distance(distances, k, f"Routine k-distance: {participant}", str(output / "routine_k_distance.png"))
        else:
            print("Skipped routine plot: not enough subsequences for the selected min_samples")
    finally:
        db.close()


if __name__ == "__main__":
    main()
