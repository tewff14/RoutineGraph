from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import numpy as np

def dynamic_time_warping(sequence_a: np.ndarray, sequence_b: np.ndarray) -> tuple[float, float]:
    """Calculate DTW between two sequences of activity embeddings (n, dim) and (m, dim)

    Return (distance, similarity)
    - distance: accumulated cosine distance along the optimal alignment path
    - similarity: 1 - distance / path length = mean cosine similarity of the aligned activity pairs (1 = identical)
    """

    cost = cdist(sequence_a, sequence_b, metric="cosine") #pairwise cosine distance between every activity of a and b
    n, m = cost.shape

    #accumulated[i, j] = minimum cost to align a[:i] with b[:j], path_length[i, j] = number of aligned pairs on that path
    accumulated = np.full((n + 1, m + 1), np.inf)
    accumulated[0, 0] = 0
    path_length = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            #previous cells: match (diagonal), a repeats, b repeats
            previous = [(i - 1, j - 1), (i - 1, j), (i, j - 1)]
            best = min(previous, key=lambda cell: accumulated[cell])
            accumulated[i, j] = cost[i - 1, j - 1] + accumulated[best]
            path_length[i, j] = path_length[best] + 1

    distance = float(accumulated[n, m])
    similarity = float(1 - distance / path_length[n, m])
    return distance, similarity

def contiguous_subsequences(sequence: np.ndarray) -> list[np.ndarray]:
    """Return every contiguous subsequence sequence[start:end] of length > 1 of an (n, dim) sequence

    e.g. [a, b, c] -> [a, b], [a, b, c], [b, c] (n * (n - 1) / 2 subsequences, each (length, dim))
    """

    n = len(sequence)
    return [sequence[start:end] for start in range(n) for end in range(start + 2, n + 1)]





class RoutineCluster:
    def __init__(self, sequences: list[np.ndarray], sequence_ids: list[int], eps: float = 0.5, min_samples: int = 5,
                 algorithm: str = "auto", leaf_size: int = 30, n_jobs: int | None = None):

        if len(sequences) != len(sequence_ids):
            raise ValueError(f"Got {len(sequences)} sequences but {len(sequence_ids)} sequence ids")

        self.sequences = [np.asarray(sequence) for sequence in sequences] #each (n_activities_i, embedding_dim), length can differ
        self.sequence_ids = list(sequence_ids) #unique id of each sequence (same order as sequences)

        #pairwise DTW distance matrix, use 1 - similarity (mean cosine distance along path) so it doesn't grow with sequence length
        n = len(self.sequences)
        self.distance_matrix = np.zeros((n, n))
        with tqdm(total=n * (n - 1) // 2, desc="DTW pairs") as progress: #progress counts sequence pairs
            for i in range(n):
                for j in range(i + 1, n):
                    _, similarity = dynamic_time_warping(self.sequences[i], self.sequences[j])
                    self.distance_matrix[i, j] = self.distance_matrix[j, i] = max(1 - similarity, 0.0) #clip tiny negative float error
                progress.update(n - i - 1)

        #perform DBSCAN on the precomputed DTW distances
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed",
                             algorithm=algorithm, leaf_size=leaf_size, n_jobs=n_jobs).fit(self.distance_matrix)

        self.labels = self.dbscan.labels_ #cluster id of each sequence, -1 is noise
        self.core_sample_indices = self.dbscan.core_sample_indices_ #indices of core sequences
        self.id_to_cluster = dict(zip(self.sequence_ids, self.labels.tolist())) #sequence id -> cluster id

    def get_id_to_cluster(self) -> dict[int, int]:
        """Return the mapping of sequence id -> cluster id (-1 is noise)"""
        return self.id_to_cluster
