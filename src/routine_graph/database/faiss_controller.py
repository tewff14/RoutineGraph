from pathlib import Path
import os
import tempfile

import faiss
import numpy as np

from routine_graph.models import normalized_vectors


class FaissController:
    def __init__(self, index_path: str, dim: int, *, reset: bool = False):
        self.index_path = str(index_path)
        if Path(index_path).exists() and not reset:
            self.index = faiss.read_index(self.index_path)
            if dim and self.index.d != dim:
                raise ValueError(f"Stored index dimension {self.index.d} differs from requested dimension {dim}")
        else:
            if dim < 1:
                raise ValueError("A positive embedding dimension is required for a new index")
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def _save(self):
        path = Path(self.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".faiss")
        os.close(fd)
        try:
            faiss.write_index(self.index, temporary)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def add(self, vectors: list[np.ndarray], ids: list[int]):
        if len(vectors) != len(ids):
            raise ValueError(f"Got {len(vectors)} vectors but {len(ids)} ids")
        if len(set(ids)) != len(ids):
            raise ValueError("Vector IDs must be unique")
        if not ids:
            return
        matrix = normalized_vectors(vectors, self.index.d)
        ids = np.asarray(ids, dtype="int64")
        self.index.remove_ids(ids)
        self.index.add_with_ids(matrix, ids)
        self._save()

    def replace(self, vectors: list[np.ndarray], ids: list[int]):
        if len(vectors) != len(ids) or len(set(ids)) != len(ids):
            raise ValueError("Vectors need matching unique IDs")
        matrix = normalized_vectors(vectors, self.index.d) if ids else None
        self.index.reset()
        if ids:
            self.index.add_with_ids(matrix, np.asarray(ids, dtype="int64"))
        self._save()

    def get(self, ids: list[int]) -> np.ndarray:
        if not ids:
            return np.empty((0, self.index.d), dtype=np.float32)
        return self.index.reconstruct_batch(np.asarray(ids, dtype="int64"))

    def search(self, vector: np.ndarray, k: int = 5, allowed_ids: set[int] | None = None) -> list[tuple[int, float]]:
        if k < 1:
            raise ValueError("Search limit must be positive")
        if self.index.ntotal == 0 or allowed_ids == set():
            return []
        query = normalized_vectors([vector], self.index.d)
        count = self.index.ntotal if allowed_ids is not None else min(k, self.index.ntotal)
        scores, ids = self.index.search(query, count)
        matches = [(int(id_), float(score)) for id_, score in zip(ids[0], scores[0])
                   if id_ != -1 and (allowed_ids is None or id_ in allowed_ids)]
        return sorted(matches, key=lambda match: (-match[1], match[0]))[:k]
