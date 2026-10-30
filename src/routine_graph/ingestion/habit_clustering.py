from sklearn.cluster import DBSCAN
import numpy as np

class HabitCluster:
    def __init__(self, vectors: list[np.ndarray], activity_ids: list[int], eps: float = 0.5, min_samples: int = 5, metric: str = "euclidean",
                 metric_params: dict | None = None, algorithm: str = "auto", leaf_size: int = 30,
                 p: float | None = None, n_jobs: int | None = None):

        if len(vectors) != len(activity_ids):
            raise ValueError(f"Got {len(vectors)} vectors but {len(activity_ids)} activity ids")

        self.vectors = np.stack(vectors) #(n_activities, embedding_dim)
        self.activity_ids = list(activity_ids) #activity unit id of each vector (same order as vectors)

        #perform DBSCAN on the activity vectors
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params,
                             algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=n_jobs).fit(self.vectors)

        self.labels = self.dbscan.labels_ #cluster id of each vector, -1 is noise
        self.core_sample_indices = self.dbscan.core_sample_indices_ #indices of core vectors
        self.id_to_cluster = dict(zip(self.activity_ids, self.labels.tolist())) #activity unit id -> cluster id

    def get_id_to_cluster(self) -> dict[int, int]:
        """Return the mapping of sequence id -> cluster id (-1 is noise)"""
        return self.id_to_cluster