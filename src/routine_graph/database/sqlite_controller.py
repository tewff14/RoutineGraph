"""SQLite is the source of truth for observations, profiles, links, and vectors."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

import numpy as np


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY,
    participant TEXT NOT NULL, day TEXT NOT NULL, event_index INTEGER NOT NULL,
    label TEXT NOT NULL, description TEXT NOT NULL,
    start_time TEXT NOT NULL, end_time TEXT NOT NULL, cluster_id INTEGER,
    metadata TEXT NOT NULL DEFAULT '{}',
    UNIQUE (participant, day, event_index)
);
CREATE TABLE IF NOT EXISTS sequences (
    sequence_id INTEGER PRIMARY KEY,
    participant TEXT NOT NULL, day TEXT NOT NULL,
    start_position INTEGER NOT NULL, length INTEGER NOT NULL,
    event_ids TEXT NOT NULL, start_time TEXT NOT NULL, end_time TEXT NOT NULL,
    cluster_id INTEGER
);
CREATE TABLE IF NOT EXISTS habits (
    habit_id INTEGER PRIMARY KEY, label TEXT NOT NULL, description TEXT NOT NULL,
    event_ids TEXT NOT NULL, participant TEXT NOT NULL DEFAULT '',
    observation_count INTEGER NOT NULL DEFAULT 0, day_count INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS routines (
    routine_id INTEGER PRIMARY KEY, label TEXT NOT NULL, description TEXT NOT NULL,
    sequence_ids TEXT NOT NULL, participant TEXT NOT NULL DEFAULT '',
    observation_count INTEGER NOT NULL DEFAULT 0, day_count INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS event_embeddings (
    event_id INTEGER PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
    text TEXT NOT NULL, fingerprint TEXT NOT NULL,
    dimension INTEGER NOT NULL, vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS profile_embeddings (
    channel TEXT NOT NULL, node_id INTEGER NOT NULL,
    dimension INTEGER NOT NULL, vector BLOB NOT NULL,
    PRIMARY KEY (channel, node_id)
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_timeline ON events(participant, day, start_time, end_time, event_index);
"""

GRAPH_VIEW = """
CREATE VIEW IF NOT EXISTS graph_edges AS
WITH timeline AS (
    SELECT event_id, LEAD(event_id) OVER (
        PARTITION BY participant, day ORDER BY start_time, end_time, event_index
    ) AS next_id FROM events
)
SELECT 'activity' AS source_type, event_id AS source_id, 'activity' AS target_type,
       next_id AS target_id, 'next' AS relation, NULL AS sequence_id, NULL AS position
FROM timeline WHERE next_id IS NOT NULL
UNION ALL
SELECT 'habit', h.habit_id, 'activity', CAST(j.value AS INTEGER), 'occurrence', NULL, NULL
FROM habits h, json_each(h.event_ids) j
UNION ALL
SELECT 'routine', r.routine_id, 'activity', CAST(e.value AS INTEGER), 'step',
       s.sequence_id, CAST(e.key AS INTEGER)
FROM routines r, json_each(r.sequence_ids) j
JOIN sequences s ON s.sequence_id = j.value, json_each(s.event_ids) e;
"""

CHANNELS = {"activity": ("events", "event_id"), "habit": ("habits", "habit_id"), "routine": ("routines", "routine_id")}


class Database:
    def __init__(self, db_path: str):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        #LangChain tools may run on worker threads; retrieval serializes shared access.
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)
        #append columns so the original tuple positions remain compatible
        with self.connection:
            for table in ("events", "habits", "routines"):
                columns = {row[1] for row in self.connection.execute(f"PRAGMA table_info({table})")}
                additions = {"metadata": "TEXT NOT NULL DEFAULT '{}'"}
                if table != "events":
                    additions = {"participant": "TEXT NOT NULL DEFAULT ''", "observation_count": "INTEGER NOT NULL DEFAULT 0",
                                 "day_count": "INTEGER NOT NULL DEFAULT 0", **additions}
                for name, declaration in additions.items():
                    if name not in columns:
                        self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
        self.connection.executescript(GRAPH_VIEW)
        self._savepoint = 0

    @contextmanager
    def transaction(self):
        """Nested writes use savepoints; a pipeline failure rolls back its whole run."""
        nested = self.connection.in_transaction
        self._savepoint += 1
        name = f"write_{self._savepoint}"
        self.connection.execute(f"SAVEPOINT {name}" if nested else "BEGIN")
        try:
            yield
            self.connection.execute(f"RELEASE SAVEPOINT {name}" if nested else "COMMIT")
        except BaseException:
            if nested:
                self.connection.execute(f"ROLLBACK TO SAVEPOINT {name}")
                self.connection.execute(f"RELEASE SAVEPOINT {name}")
            else:
                self.connection.rollback()
            raise

    def insert_events(self, rows: list[tuple]):
        rows = [tuple(row) + ("{}",) if len(row) == 7 else tuple(row) for row in rows]
        with self.transaction():
            self.connection.executemany(
                "INSERT INTO events (participant,day,event_index,label,description,start_time,end_time,metadata) "
                "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT (participant,day,event_index) DO UPDATE SET "
                "label=excluded.label,description=excluded.description,start_time=excluded.start_time,"
                "end_time=excluded.end_time,metadata=excluded.metadata", rows)

    def replace_day_events(self, participant: str, day: str, rows: list[tuple]):
        with self.transaction():
            self.insert_events(rows)
            self.connection.execute("DELETE FROM events WHERE participant=? AND day=? AND event_index>=?",
                                    (participant, day, len(rows)))

    def _rows(self, table: str) -> list[tuple]:
        return self.connection.execute(f"SELECT * FROM {table}").fetchall()

    def get_events(self) -> list[tuple]:
        return self._rows("events")

    def get_sequences(self) -> list[tuple]:
        return self._rows("sequences")

    def get_habits(self) -> list[tuple]:
        return self._rows("habits")

    def get_routines(self) -> list[tuple]:
        return self._rows("routines")

    def replace_sequences(self, rows: list[tuple]):
        with self.transaction():
            self.connection.execute("DELETE FROM sequences")
            self.connection.executemany("INSERT INTO sequences "
                "(participant,day,start_position,length,event_ids,start_time,end_time) VALUES (?,?,?,?,?,?,?)", rows)

    def _update_clusters(self, table: str, id_column: str, mapping: dict[int, int]):
        with self.transaction():
            self.connection.execute(f"UPDATE {table} SET cluster_id=-1")
            self.connection.executemany(f"UPDATE {table} SET cluster_id=? WHERE {id_column}=?",
                                        [(cluster, id_) for id_, cluster in mapping.items()])

    def update_event_clusters(self, id_to_cluster: dict[int, int]):
        self._update_clusters("events", "event_id", id_to_cluster)

    def update_sequence_clusters(self, id_to_cluster: dict[int, int]):
        self._update_clusters("sequences", "sequence_id", id_to_cluster)

    def _replace_profiles(self, table: str, id_column: str, membership: str, rows: list[tuple]):
        rows = [tuple(row) + ("", 0, 0, "{}") if len(row) == 4 else tuple(row) for row in rows]
        with self.transaction():
            self.connection.execute(f"DELETE FROM {table}")
            self.connection.executemany(f"INSERT INTO {table} "
                f"({id_column},label,description,{membership},participant,observation_count,day_count,metadata) "
                "VALUES (?,?,?,?,?,?,?,?)", rows)

    def replace_habits(self, rows: list[tuple]):
        self._replace_profiles("habits", "habit_id", "event_ids", rows)

    def replace_routines(self, rows: list[tuple]):
        self._replace_profiles("routines", "routine_id", "sequence_ids", rows)

    def records(self, channel: str, participant: str | None = None, day: str | None = None) -> list[dict]:
        table, _ = CHANNELS[channel]
        sql = f"SELECT * FROM {table}"
        filters, values = [], []
        if participant is not None:
            filters.append("participant=?")
            values.append(participant)
        if day is not None:
            if channel != "activity":
                raise ValueError("Day filtering is available on the activity channel")
            filters.append("day=?")
            values.append(day)
        if filters:
            sql += " WHERE " + " AND ".join(filters)
        cursor = self.connection.execute(sql, values)
        names = [column[0] for column in cursor.description]
        return [self._record(dict(zip(names, row))) for row in cursor.fetchall()]

    @staticmethod
    def _record(record: dict) -> dict:
        record["metadata"] = json.loads(record.get("metadata", "{}"))
        for key in ("event_ids", "sequence_ids"):
            if key in record:
                record[key] = json.loads(record[key])
        return record

    def get_node(self, channel: str, node_id: int) -> dict | None:
        table, column = CHANNELS[channel]
        cursor = self.connection.execute(f"SELECT * FROM {table} WHERE {column}=?", (node_id,))
        row = cursor.fetchone()
        return None if row is None else self._record(dict(zip([c[0] for c in cursor.description], row)))

    def get_edges(self, channel: str, node_id: int, limit: int = 50, offset: int = 0) -> list[dict]:
        cursor = self.connection.execute("SELECT * FROM graph_edges WHERE source_type=? AND source_id=? "
                                         "ORDER BY sequence_id, position, target_id LIMIT ? OFFSET ?", (channel, node_id, limit, offset))
        return [dict(zip([c[0] for c in cursor.description], row)) for row in cursor.fetchall()]

    def adjacent_activity(self, event_id: int, direction: str = "after") -> int | None:
        source, target = ("source_id", "target_id") if direction == "after" else ("target_id", "source_id")
        row = self.connection.execute(f"SELECT {target} FROM graph_edges WHERE source_type='activity' "
                                       f"AND relation='next' AND {source}=?", (event_id,)).fetchone()
        return row[0] if row else None

    def setting(self, key: str, default=None):
        row = self.connection.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    def set_setting(self, key: str, value):
        with self.transaction():
            self.connection.execute("INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                                    (key, json.dumps(value)))

    def cached_embedding(self, event_id: int, text: str, fingerprint: str) -> np.ndarray | None:
        row = self.connection.execute("SELECT vector FROM event_embeddings WHERE event_id=? AND text=? AND fingerprint=?",
                                      (event_id, text, fingerprint)).fetchone()
        return None if row is None else np.frombuffer(row[0], dtype=np.float32).copy()

    def store_event_embedding(self, event_id: int, text: str, fingerprint: str, vector: np.ndarray):
        vector = np.asarray(vector, dtype=np.float32)
        with self.transaction():
            self.connection.execute("INSERT INTO event_embeddings VALUES (?,?,?,?,?) ON CONFLICT(event_id) "
                "DO UPDATE SET text=excluded.text,fingerprint=excluded.fingerprint,dimension=excluded.dimension,vector=excluded.vector",
                (event_id, text, fingerprint, vector.size, vector.tobytes()))

    def replace_profile_embeddings(self, rows: list[tuple]):
        with self.transaction():
            self.connection.execute("DELETE FROM profile_embeddings")
            self.connection.executemany("INSERT INTO profile_embeddings VALUES (?,?,?,?)",
                [(channel, id_, len(vector), np.asarray(vector, dtype=np.float32).tobytes()) for channel, id_, vector in rows])

    def vectors(self, channel: str) -> tuple[list[int], list[np.ndarray]]:
        if channel == "activity":
            rows = self.connection.execute("SELECT event_id,vector FROM event_embeddings ORDER BY event_id").fetchall()
        else:
            rows = self.connection.execute("SELECT node_id,vector FROM profile_embeddings WHERE channel=? ORDER BY node_id", (channel,)).fetchall()
        return [r[0] for r in rows], [np.frombuffer(r[1], dtype=np.float32).copy() for r in rows]

    def close(self):
        self.connection.close()
