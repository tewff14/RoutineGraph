import sqlite3

CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY, --assigned by sqlite (1, 2, ...)
    participant TEXT NOT NULL,
    day TEXT NOT NULL,
    event_index INTEGER NOT NULL,
    label TEXT NOT NULL,
    description TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    cluster_id INTEGER,
    UNIQUE (participant, day, event_index) --same event of a re-ingested json keeps its event_id
)
"""

CREATE_SEQUENCES_TABLE = """
CREATE TABLE IF NOT EXISTS sequences (
    sequence_id INTEGER PRIMARY KEY, --assigned by sqlite (1, 2, ...)
    participant TEXT NOT NULL,
    day TEXT NOT NULL,
    start_position INTEGER NOT NULL,
    length INTEGER NOT NULL,
    event_ids TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    cluster_id INTEGER
)
"""

CREATE_HABITS_TABLE = """
CREATE TABLE IF NOT EXISTS habits (
    habit_id INTEGER PRIMARY KEY, --habit cluster id (= events.cluster_id)
    label TEXT NOT NULL,
    description TEXT NOT NULL,
    event_ids TEXT NOT NULL --json list of event_id in the habit
)
"""

CREATE_ROUTINES_TABLE = """
CREATE TABLE IF NOT EXISTS routines (
    routine_id INTEGER PRIMARY KEY, --routine cluster id (= sequences.cluster_id)
    label TEXT NOT NULL,
    description TEXT NOT NULL,
    sequence_ids TEXT NOT NULL --json list of sequence_id in the routine
)
"""

class Database:
    def __init__(self, db_path: str):
        self.connection = sqlite3.connect(db_path) #kept open until close()

        with self.connection: #commit (or roll back on error)
            self.connection.execute(CREATE_EVENTS_TABLE)
            self.connection.execute(CREATE_SEQUENCES_TABLE)
            self.connection.execute(CREATE_HABITS_TABLE)
            self.connection.execute(CREATE_ROUTINES_TABLE)

    def insert_events(self, rows: list[tuple]):
        """Insert (participant, day, event_index, label, description, start_time, end_time) rows

        Same (participant, day, event_index) updates the row in place, so it keeps its event_id
        """
        with self.connection:
            self.connection.executemany("INSERT INTO events (participant, day, event_index, label, description, start_time, end_time) "
                                        "VALUES (?, ?, ?, ?, ?, ?, ?) "
                                        "ON CONFLICT (participant, day, event_index) DO UPDATE SET "
                                        "label = excluded.label, description = excluded.description, "
                                        "start_time = excluded.start_time, end_time = excluded.end_time", rows)

    def get_events(self) -> list[tuple]:
        """Return every row of the events table"""
        return self.connection.execute("SELECT * FROM events").fetchall()

    def replace_sequences(self, rows: list[tuple]):
        """Replace the whole sequences table with (participant, day, start_position, length, event_ids, start_time, end_time) rows

        sequence_id restarts from 1 on every replace
        """
        with self.connection:
            self.connection.execute("DELETE FROM sequences")
            self.connection.executemany("INSERT INTO sequences (participant, day, start_position, length, event_ids, start_time, end_time) "
                                        "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)

    def get_sequences(self) -> list[tuple]:
        """Return every row of the sequences table"""
        return self.connection.execute("SELECT * FROM sequences").fetchall()

    def update_event_clusters(self, id_to_cluster: dict[int, int]):
        """Set cluster_id of each event from {event_id: cluster id} (e.g. HabitCluster.get_id_to_cluster(), -1 is noise)"""
        with self.connection:
            self.connection.executemany("UPDATE events SET cluster_id = ? WHERE event_id = ?",
                                        [(cluster_id, event_id) for event_id, cluster_id in id_to_cluster.items()])

    def update_sequence_clusters(self, id_to_cluster: dict[int, int]):
        """Set cluster_id of each sequence from {sequence_id: cluster id} (e.g. RoutineCluster.get_id_to_cluster(), -1 is noise)"""
        with self.connection:
            self.connection.executemany("UPDATE sequences SET cluster_id = ? WHERE sequence_id = ?",
                                        [(cluster_id, sequence_id) for sequence_id, cluster_id in id_to_cluster.items()])

    def replace_habits(self, rows: list[tuple]):
        """Replace the whole habits table with (habit_id, label, description, event_ids (json list)) rows"""
        with self.connection:
            self.connection.execute("DELETE FROM habits")
            self.connection.executemany("INSERT INTO habits (habit_id, label, description, event_ids) VALUES (?, ?, ?, ?)", rows)

    def get_habits(self) -> list[tuple]:
        """Return every row of the habits table"""
        return self.connection.execute("SELECT * FROM habits").fetchall()

    def replace_routines(self, rows: list[tuple]):
        """Replace the whole routines table with (routine_id, label, description, sequence_ids (json list)) rows"""
        with self.connection:
            self.connection.execute("DELETE FROM routines")
            self.connection.executemany("INSERT INTO routines (routine_id, label, description, sequence_ids) VALUES (?, ?, ?, ?)", rows)

    def get_routines(self) -> list[tuple]:
        """Return every row of the routines table"""
        return self.connection.execute("SELECT * FROM routines").fetchall()

    def close(self):
        self.connection.close()
