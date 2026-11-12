"""The app's own SQLite database: schema, migrations, connections.

Everything the app generates lives in this one file: parameter sets, runs,
scalars, presets and the shot index.

A schema change appends a function to :data:`MIGRATIONS` and never edits an
existing one: the list index is the schema version it produces, and ``PRAGMA
user_version`` records which version a database file is at.
"""

import os
import sqlite3
from pathlib import Path

from fusion_ui import config
from fusion_ui.core import shared

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


# v2: `runs.preprocessed`, in the unique key. The raw and the preprocessed file
# are different data, so their results must not share a row.
#
# SQLite cannot add a column to a UNIQUE constraint, so the table is rebuilt.
# `scalars` is parked in a copy without constraints first: it carries ON DELETE
# CASCADE, and dropping `runs` would take the scalars with it.
_MIGRATE_2 = """
CREATE TABLE scalars_backup AS SELECT * FROM scalars;
DROP TABLE scalars;

CREATE TABLE runs_new (
    id           INTEGER PRIMARY KEY,
    machine      TEXT    NOT NULL,
    shot         INTEGER NOT NULL,
    diagnostic   TEXT    NOT NULL,
    preprocessed INTEGER NOT NULL DEFAULT 0,
    plot         TEXT    NOT NULL,
    params_hash  TEXT    NOT NULL REFERENCES param_sets(hash),
    blob_path    TEXT,
    status       TEXT    NOT NULL,
    error        TEXT,
    seconds      REAL,
    code_version TEXT,
    created_at   TEXT    NOT NULL,
    UNIQUE (machine, shot, diagnostic, preprocessed, plot, params_hash)
);
INSERT INTO runs_new (id, machine, shot, diagnostic, preprocessed, plot,
                      params_hash, blob_path, status, error, seconds,
                      code_version, created_at)
     SELECT id, machine, shot, diagnostic, 0, plot, params_hash, blob_path,
            status, error, seconds, code_version, created_at
       FROM runs;
DROP TABLE runs;
ALTER TABLE runs_new RENAME TO runs;

CREATE TABLE scalars (
    run_id  INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    x       INTEGER NOT NULL DEFAULT -1,
    y       INTEGER NOT NULL DEFAULT -1,
    name    TEXT    NOT NULL,
    value   REAL,
    PRIMARY KEY (run_id, x, y, name)
);
INSERT INTO scalars (run_id, x, y, name, value)
     SELECT run_id, x, y, name, value FROM scalars_backup;
DROP TABLE scalars_backup;

CREATE INDEX idx_runs_shot    ON runs (machine, shot);
CREATE INDEX idx_runs_plot    ON runs (plot, params_hash);
CREATE INDEX idx_scalars_name ON scalars (name);
"""


def _migrate_to_2(conn):
    conn.executescript(_MIGRATE_2)


# v3: `shots.dt`, the file's frame interval in seconds, measured off its time
# axis by `fusion-ui backfill-dt`. Only phantom rows carry one: the camera's
# frame rate varies from shot to shot. A rescan neither fills nor clears it.
_MIGRATE_3 = """
ALTER TABLE shots ADD COLUMN dt REAL;
"""


def _migrate_to_3(conn):
    conn.executescript(_MIGRATE_3)


# v4: what a stored result was computed from, for `store.stale_runs`.
# `runs.input_mtime` is the input file's mtime as `shots.mtime` had it when the
# run was written (the same string, so the two compare directly), and
# `runs.upstream_run_id` is the run a chained result was computed on. Deleting
# that run sets the link NULL. Older rows keep NULL in both.
#
# It runs in one explicit transaction with `init_db`'s version bump (sqlite3
# would otherwise run each DDL statement on its own), and each step checks
# first, so a half-applied migration is finished by the next `init-db`.
def _migrate_to_4(conn):
    if not conn.in_transaction:
        conn.execute("BEGIN")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    if "input_mtime" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN input_mtime TEXT")
    if "upstream_run_id" not in columns:
        conn.execute(
            "ALTER TABLE runs ADD COLUMN upstream_run_id INTEGER"
            " REFERENCES runs(id) ON DELETE SET NULL"
        )
    # Deleting a run looks up the rows that point at it.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_upstream ON runs (upstream_run_id)"
    )


# The index is the schema version the migration produces. A database file at
# version N runs MIGRATIONS[N + 1:].
MIGRATIONS = [
    None,  # version 0 is an empty file
    _migrate_to_1,
    _migrate_to_2,
    _migrate_to_3,
    _migrate_to_4,
]

SCHEMA_VERSION = len(MIGRATIONS) - 1


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


def connect(path=None):
    """Open the app database, creating its parent directory if needed.

    In WAL mode, so that ``rescan`` or ``precompute`` and the Streamlit process
    do not block each other's reads.
    """
    if path is None:
        path = config.UI_DB_PATH
    path = os.path.expanduser(str(path))
    if path != ":memory:":
        shared.makedirs(Path(path).parent)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    if path != ":memory:":
        # Before the WAL pragma: SQLite creates -wal and -shm with the mode of
        # the database file itself, so relaxing it here relaxes all three.
        shared.share_file(path)
        conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def open_readonly(path=None):
    """Open the app database for reading only: no directory, no migration, no write.

    For readers outside the app (``fusion_ui.api``), who may not own the file.
    A database at another schema version is refused, not migrated.
    """
    if path is None:
        path = config.UI_DB_PATH
    path = Path(os.path.expanduser(str(path)))
    if not path.exists():
        raise FileNotFoundError(f"no app database at {path}")
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    version = schema_version(conn)
    if version != SCHEMA_VERSION:
        conn.close()
        raise RuntimeError(
            f"{path} is at schema v{version}, this code reads v{SCHEMA_VERSION}:"
            " run `fusion-ui init-db` with the app's checkout"
        )
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
    """``connect`` plus ``init_db``: what every entry point wants."""
    conn = connect(path)
    init_db(conn)
    return conn
