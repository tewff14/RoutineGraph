"""Three semantic retrieval channels and bounded evidence traversal."""

from threading import RLock

from routine_graph.database.indexes import load_indexes
from routine_graph.database.sqlite_controller import CHANNELS, Database
from routine_graph.utils.embedding import ActivityEmbedder


def compact_node(node: dict) -> dict:
    result = {key: value for key, value in node.items() if key not in ("event_ids", "sequence_ids", "metadata")}
    result["label"] = result["label"][:200]
    result["description"] = result["description"][:2000]
    metadata = node.get("metadata", {})
    result["metadata"] = {key: metadata[key] for key in ("date", "subject", "location", "typical_start_time",
        "typical_end_time", "median_duration_seconds", "weekday_counts") if key in metadata}
    for key in ("subject", "location"):
        if key in result["metadata"]:
            result["metadata"][key] = str(result["metadata"][key])[:500]
    if metadata.get("interacted_objects"):
        result["metadata"]["interacted_objects"] = [str(item)[:100] for item in metadata["interacted_objects"][:20]]
    return result


class Retriever:
    def __init__(self, db: Database, index_path: str = "data/events.faiss", embedder=None):
        self.db, self.index_path = db, index_path
        self.embedder = embedder
        self._owns_embedder = embedder is None
        self.lock = RLock()
        self.indexes = {}
        self.revision = None

    def _refresh(self):
        revision = self.db.setting("revision")
        if revision != self.revision:
            self.indexes = load_indexes(self.db, self.index_path)
            self.revision = revision

    @staticmethod
    def _validate(channel: str, limit: int, offset: int = 0):
        if channel not in CHANNELS:
            raise ValueError("Channel must be activity, habit, or routine")
        if not 1 <= limit <= 50 or offset < 0:
            raise ValueError("Limit must be between 1 and 50; offset must be nonnegative")

    def search(self, channel: str, query: str, *, participant: str | None = None,
               day: str | None = None, limit: int = 5) -> list[dict]:
        self._validate(channel, limit)
        if not query.strip():
            raise ValueError("Search query must not be blank")
        with self.lock:
            nodes = self.db.records(channel, participant, day)
            if not nodes:
                return []
            self._refresh()
            if channel not in self.indexes:
                raise ValueError("No vector snapshot is available; run ingestion first")
            if self.embedder is None:
                configuration = self.db.setting("embedding_configuration")
                if not configuration:
                    raise ValueError("No embedding configuration is available; run ingestion first")
                self.embedder = ActivityEmbedder(**configuration)
            if self.embedder.fingerprint != self.db.setting("embedding_fingerprint"):
                raise ValueError("Retrieval model differs from the stored embeddings; use the ingestion model or re-ingest")
            id_column = CHANNELS[channel][1]
            lookup = {node[id_column]: node for node in nodes}
            hits = self.indexes[channel].search(self.embedder.embed_query(query), limit, set(lookup))
            return [{"channel": channel, "id": id_, "score": score, "node": compact_node(lookup[id_])} for id_, score in hits]

    def inspect(self, channel: str, node_id: int, *, participant: str | None = None,
                limit: int = 20, offset: int = 0) -> dict:
        self._validate(channel, limit, offset)
        with self.lock:
            node = self.db.get_node(channel, node_id)
            if node is None or (participant is not None and node["participant"] != participant):
                return {"error": "Node not found for this participant"}
            edges = self.db.get_edges(channel, node_id, limit + 1, offset)
            more = len(edges) > limit
            edges = edges[:limit]
            activities = []
            for edge in edges:
                target = self.db.get_node("activity", edge["target_id"])
                if target is not None:
                    activities.append({"relation": edge["relation"], "sequence_id": edge["sequence_id"],
                                       "position": edge["position"], "node": compact_node(target)})
            return {"channel": channel, "id": node_id, "node": compact_node(node), "activities": activities,
                    "has_more": more, "next_offset": offset + limit if more else None}

    def follow(self, event_id: int, *, participant: str | None = None, direction: str = "after", limit: int = 5) -> list[dict]:
        self._validate("activity", limit)
        if direction not in ("before", "after"):
            raise ValueError("Direction must be before or after")
        with self.lock:
            node = self.db.get_node("activity", event_id)
            if node is None or (participant is not None and node["participant"] != participant):
                return []
            result = []
            while len(result) < limit:
                event_id = self.db.adjacent_activity(event_id, direction)
                if event_id is None:
                    break
                result.append(compact_node(self.db.get_node("activity", event_id)))
            return result

    def close(self):
        if self._owns_embedder and self.embedder is not None:
            self.embedder.close()
