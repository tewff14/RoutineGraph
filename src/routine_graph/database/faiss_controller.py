from pathlib import Path

import faiss
import numpy as np

class FaissController:
    def __init__(self, index_path: str, dim: int):
        """Load the faiss index from index_path, or create an empty one if the file doesn't exist yet

        Vectors are L2-normalized and stored in an inner product index, so search scores are cosine similarity
        """

        self.index_path = index_path

        if Path(index_path).exists():
            self.index = faiss.read_index(index_path)
        else:
            #IndexIDMap2 keeps our own int64 ids (and allows reconstruct(id)) on top of the flat index
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def add(self, vectors: list[np.ndarray], ids: list[int]):
        """Add vectors with their int ids (same order), same id replaces, then save the index to index_path"""

        if len(vectors) != len(ids):
            raise ValueError(f"Got {len(vectors)} vectors but {len(ids)} ids")

        vectors = np.stack(vectors).astype("float32") #faiss needs float32 (n, dim), astype also copies so input isn't changed
        faiss.normalize_L2(vectors) #in-place, inner product of normalized vectors = cosine similarity
        ids = np.asarray(ids, dtype="int64")

        self.index.remove_ids(ids) #drop old vectors with the same ids so re-adding doesn't duplicate
        self.index.add_with_ids(vectors, ids)
        faiss.write_index(self.index, self.index_path)

    def get(self, ids: list[int]) -> np.ndarray:
        """Return the stored (L2-normalized) vectors of the ids as (n, dim), same order as ids, unknown id raises RuntimeError"""
        return self.index.reconstruct_batch(np.asarray(ids, dtype="int64"))
