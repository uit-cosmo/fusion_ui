import sqlite3

import pytest

from fusion_ui.core import db

TABLES = {"param_sets", "runs", "scalars", "presets", "shots"}


def table_names(conn):
    return {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def test_schema_created(conn):
    assert TABLES <= table_names(conn)
    assert db.schema_version(conn) == db.SCHEMA_VERSION


def test_init_db_is_idempotent(conn):
    db.init_db(conn)
    db.init_db(conn)
    assert TABLES <= table_names(conn)


def test_wal_and_foreign_keys(tmp_path):
    conn = db.connect(tmp_path / "nested" / "app.sqlite")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert (tmp_path / "nested").is_dir()


def test_refuses_a_newer_schema(conn):
    conn.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION + 1}")
    with pytest.raises(RuntimeError, match="schema version"):
        db.init_db(conn)


def _make_run(conn):
    with conn:
        conn.execute(
            "INSERT INTO param_sets VALUES ('abc', 'velocity_2dca', '{}', '2026-01-01')"
        )
        cursor = conn.execute(
            "INSERT INTO runs (machine, shot, diagnostic, plot, params_hash,"
            " status, created_at) VALUES ('cmod', 1, 'apd', 'velocity_2dca',"
            " 'abc', 'ok', '2026-01-01')"
        )
    return cursor.lastrowid


def test_scalars_cascade_with_their_run(conn):
    run_id = _make_run(conn)
    with conn:
        conn.execute(
            "INSERT INTO scalars (run_id, x, y, name, value) VALUES (?, 3, 4, 'vx_c', 1.0)",
            (run_id,),
        )
    with conn:
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    assert conn.execute("SELECT COUNT(*) FROM scalars").fetchone()[0] == 0


def test_shot_level_scalar_sentinel_enforces_uniqueness(conn):
    """The x = y = -1 sentinel is the point: NULLs would not collide."""
    run_id = _make_run(conn)
    with conn:
        conn.execute(
            "INSERT INTO scalars (run_id, name, value) VALUES (?, 'taud_psd', 1.0)",
            (run_id,),
        )
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute(
                "INSERT INTO scalars (run_id, name, value) VALUES (?, 'taud_psd', 2.0)",
                (run_id,),
            )
    assert conn.execute("SELECT x, y FROM scalars").fetchone()[:] == (-1, -1)


def test_runs_unique_per_params(conn):
    _make_run(conn)
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute(
                "INSERT INTO runs (machine, shot, diagnostic, plot, params_hash,"
                " status, created_at) VALUES ('cmod', 1, 'apd', 'velocity_2dca',"
                " 'abc', 'ok', '2026-01-02')"
            )


def test_run_requires_a_known_params_hash(conn):
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute(
                "INSERT INTO runs (machine, shot, diagnostic, plot, params_hash,"
                " status, created_at) VALUES ('cmod', 1, 'apd', 'raw',"
                " 'no-such-hash', 'ok', '2026-01-02')"
            )
