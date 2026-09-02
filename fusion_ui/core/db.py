"""The app's own SQLite database: schema, migrations, connections.

Everything Shot Explorer generates lives in this one file -- parameter sets,
run provenance, scalars, saved presets and the shot index. One file to back up,
one file to delete for a clean slate. The ``experimental_database`` discharge
descriptor is never written here; it stays hand-curated and read-only.

Schema changes go in :data:`MIGRATIONS`, never by editing :data:`SCHEMA` in
place: the list index is the schema version it produces, and ``PRAGMA
user_version`` records where a given database file has got to.
"""

import os
import sqlite3
from pathlib import Path

from fusion_ui import config

# ---------------------------------------------------------------------------
# Schema (version 1)
# ---------------------------------------------------------------------------

SCHEMA = """
-- what a params_hash actually means. Written once per distinct parameter set.
CREATE TABLE param_sets (
    hash         TEXT PRIMARY KEY,   -- sha1 of canonical JSON
    plot         TEXT NOT NULL,
    params_json  TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

-- one row per (what was computed, on what). The provenance record.
CREATE TABLE runs (
    id           INTEGER PRIMARY KEY,
    machine      TEXT    NOT NULL,    -- 'cmod', 'w7x', ...
    shot         INTEGER NOT NULL,
    diagnostic   TEXT    NOT NULL,
    plot         TEXT    NOT NULL,
    params_hash  TEXT    NOT NULL REFERENCES param_sets(hash),
    blob_path    TEXT,                -- netCDF on disk; NULL if scalars-only
    status       TEXT    NOT NULL,    -- ok | failed
    error        TEXT,
    seconds      REAL,
    code_version TEXT,                -- git describe: fusion_ui + imaging_methods
    created_at   TEXT    NOT NULL,
    UNIQUE (machine, shot, diagnostic, plot, params_hash)
);

-- x = y = -1 means a shot-level scalar. Sentinel, not NULL: SQLite allows
-- NULLs in a non-INTEGER primary key, which would silently break uniqueness.
CREATE TABLE scalars (
    run_id  INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    x       INTEGER NOT NULL DEFAULT -1,
    y       INTEGER NOT NULL DEFAULT -1,
    name    TEXT    NOT NULL,         -- 'vx_c', 'taud_psd', 'number_events'
    value   REAL,
    PRIMARY KEY (run_id, x, y, name)
);

-- named parameter sets, saved from the UI and shared across the group
CREATE TABLE presets (
    name         TEXT PRIMARY KEY,
    plot         TEXT NOT NULL,
    params_hash  TEXT NOT NULL REFERENCES param_sets(hash),
    note         TEXT,
    created_at   TEXT NOT NULL
);

-- the catalog: refreshed by a scan job, never walked per request
CREATE TABLE shots (
    machine      TEXT    NOT NULL,
    shot         INTEGER NOT NULL,
    diagnostic   TEXT    NOT NULL,
    preprocessed INTEGER NOT NULL DEFAULT 0,
    path         TEXT    NOT NULL,
    bytes        INTEGER,
    mtime        TEXT,
    has_metadata INTEGER NOT NULL DEFAULT 0,  -- present in the discharge DB?
    PRIMARY KEY (machine, shot, diagnostic, preprocessed)
);

CREATE INDEX idx_runs_shot     ON runs (machine, shot);
CREATE INDEX idx_runs_plot     ON runs (plot, params_hash);
CREATE INDEX idx_scalars_name  ON scalars (name);
CREATE INDEX idx_shots_shot    ON shots (shot);
"""


def _migrate_to_1(conn):
    conn.executescript(SCHEMA)


# Index = the schema version the migration produces. Append, never rewrite:
# a database file already at version N only runs MIGRATIONS[N:].
MIGRATIONS = [
    None,  # version 0 is "empty file"
    _migrate_to_1,
]

SCHEMA_VERSION = len(MIGRATIONS) - 1


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


def connect(path=None):
    """Open the app database, creating its parent directory if needed.

    WAL so the ``rescan`` cron job and the Streamlit process can work at the
    same time without blocking each other on reads.
    """
    if path is None:
        path = config.UI_DB_PATH
    path = os.path.expanduser(str(path))
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    if path != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def schema_version(conn):
    return conn.execute("PRAGMA user_version").fetchone()[0]


def init_db(conn):
    """Bring ``conn`` up to :data:`SCHEMA_VERSION`. Idempotent."""
    version = schema_version(conn)
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database is at schema version {version}, but this fusion_ui only "
            f"knows up to {SCHEMA_VERSION}. Upgrade the app, or point "
            f"FUSION_UI_DB at a different file."
        )
    for target in range(version + 1, SCHEMA_VERSION + 1):
        with conn:
            MIGRATIONS[target](conn)
            # PRAGMA does not accept a bound parameter.
            conn.execute(f"PRAGMA user_version = {target:d}")
    return SCHEMA_VERSION


def open_db(path=None):
    """``connect`` plus ``init_db`` -- what every entry point actually wants."""
    conn = connect(path)
    init_db(conn)
    return conn
