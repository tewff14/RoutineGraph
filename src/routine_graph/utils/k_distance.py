import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg") #save to file, no window
import matplotlib.pyplot as plt
import numpy as np

from routine_graph.database.faiss_controller import FaissController
from routine_graph.database.sqlite_controller import Database
from routine_graph.ingestion.routine_clustering import RoutineCluster

def k_distance(distance_matrix: np.ndarray, k: int) -> np.ndarray:
    """Distance of every point to its k-th nearest neighbour (itself excluded), sorted ascending

    For DBSCAN use k = min_samples - 1 (sklearn's min_samples counts the point itself)
    """
    return np.sort(np.sort(distance_matrix, axis=1)[:, k]) #column 0 of each sorted row is the point itself (distance 0)

def plot_k_distance(distances: np.ndarray, k: int, title: str, path: str):
    """Save the sorted k-distance curve, a good eps is around the elbow (where the curve bends up sharply)"""

    plt.figure(figsize=(8, 5))
    plt.plot(distances, marker=".")
    plt.title(title)
    plt.xlabel("points sorted by k-distance")
    plt.ylabel(f"distance to k-th nearest neighbour (k={k})")
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="k-distance plots to choose DBSCAN eps for habit and routine clustering")
    parser.add_argument("--db", required=True, help="Specify sqlite database path (after running the ingestion pipeline)")
    parser.add_argument("--faiss", required=True, help="Specify faiss index path")
    parser.add_argument("--min_samples", type=int, default=5, help="Specify DBSCAN min_samples you plan to use")
    parser.add_argument("--output", default=".", help="Specify output directory of the plots")
    args = parser.parse_args()

    if not Path(args.faiss).exists():
        parser.error(f"{args.faiss} doesn't exist")

    db = Database(args.db)
    faiss_controller = FaissController(args.faiss, dim=0) #dim is only used to create a new index, here the file exists
    k = args.min_samples - 1

    #habits: cosine distance between event embeddings (faiss vectors are L2-normalized, so cosine distance = 1 - dot product)
    events = db.get_events()
    event_ids = [event[0] for event in events]
    vectors = faiss_controller.get(event_ids)
    habit_distances = k_distance(np.clip(1 - vectors @ vectors.T, 0, None), k)
    plot_k_distance(habit_distances, k, f"Habit k-distance ({len(events)} events, min_samples={args.min_samples})",
                    f"{args.output}/habit_k_distance.png")

    #routines: DTW distance between sequences, the same distance RoutineCluster clusters on
    sequences = db.get_sequences()
    id_to_vector = dict(zip(event_ids, vectors))
    matrices = [np.stack([id_to_vector[event_id] for event_id in json.loads(sequence[5])]) for sequence in sequences]
    routine_distances = k_distance(RoutineCluster(matrices, [sequence[0] for sequence in sequences]).distance_matrix, k)
    plot_k_distance(routine_distances, k, f"Routine k-distance ({len(sequences)} sequences, min_samples={args.min_samples})",
                    f"{args.output}/routine_k_distance.png")

    db.close()
