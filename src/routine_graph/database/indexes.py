"""Recoverable FAISS caches backed by the committed SQLite snapshot."""

import json
from pathlib import Path
import os
import tempfile

from routine_graph.database.faiss_controller import FaissController
from routine_graph.database.sqlite_controller import CHANNELS, Database


def index_paths(index_path: str) -> dict[str, Path]:
    path = Path(index_path)
    return {"activity": path, "habit": path.with_name(f"{path.stem}.habits{path.suffix}"),
            "routine": path.with_name(f"{path.stem}.routines{path.suffix}")}


def load_indexes(db: Database, index_path: str, rebuild: bool = False) -> dict[str, FaissController]:
    dimension = db.setting("embedding_dimension")
    revision = db.setting("revision")
    if not dimension or not revision:
        return {}
    manifest = Path(str(index_path) + ".manifest.json")
    try:
        current = json.loads(manifest.read_text())
    except (OSError, ValueError):
        current = {}
    paths = index_paths(index_path)
    expected = {"revision": revision, "dimension": dimension}
    if not rebuild and current == expected and all(path.exists() for path in paths.values()):
        try:
            return {channel: FaissController(str(path), dimension) for channel, path in paths.items()}
        except (RuntimeError, ValueError):
            pass
    indexes = {}
    for channel in CHANNELS:
        path = paths[channel]
        #a model change may require a different dimensionality than the old cache
        controller = FaissController(str(path), dimension, reset=True)
        ids, vectors = db.vectors(channel)
        controller.replace(vectors, ids)
        indexes[channel] = controller
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=manifest.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(expected, stream)
        os.replace(temporary, manifest)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return indexes
