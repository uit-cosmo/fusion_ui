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


def test_runs_distinguish_the_raw_file_from_the_preprocessed_one(conn):
    """Added in schema v2. Without it the preprocessed variant of a shot
    returned the raw one's cached result under its own label."""
    _make_run(conn)
    with conn:
        conn.execute(
            "INSERT INTO runs (machine, shot, diagnostic, preprocessed, plot,"
            " params_hash, status, created_at) VALUES ('cmod', 1, 'apd', 1,"
            " 'velocity_2dca', 'abc', 'ok', '2026-01-01')"
        )
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2


def test_a_v1_database_migrates_forward_without_losing_rows(tmp_path):
    """The v1 -> v2 rebuild drops and recreates both `runs` and `scalars`;
    anything already recorded has to survive it."""
    path = tmp_path / "old.sqlite"
    conn = db.connect(path)
    with conn:
        db.MIGRATIONS[1](conn)
        conn.execute("PRAGMA user_version = 1")
        conn.execute(
            "INSERT INTO param_sets VALUES ('abc', 'velocity_2dca', '{}', '2026-01-01')"
        )
        conn.execute(
            "INSERT INTO runs (id, machine, shot, diagnostic, plot, params_hash,"
            " status, created_at) VALUES (7, 'cmod', 1160616027, 'apd',"
            " 'velocity_2dca', 'abc', 'ok', '2026-01-01')"
        )
        conn.execute("INSERT INTO scalars VALUES (7, 3, 4, 'vx_c', 566.6)")
    conn.close()

    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    run = conn.execute("SELECT * FROM runs").fetchone()
    assert run["id"] == 7 and run["shot"] == 1160616027
    assert run["preprocessed"] == 0, "existing rows describe the raw file"
    assert conn.execute("SELECT value FROM scalars").fetchone()[0] == 566.6
    assert "scalars_backup" not in table_names(conn)

    # And the cascade still works on the rebuilt tables.
    with conn:
        conn.execute("DELETE FROM runs WHERE id = 7")
    assert conn.execute("SELECT COUNT(*) FROM scalars").fetchone()[0] == 0
    conn.close()


def test_a_v2_database_gains_the_dt_column_without_losing_shots(tmp_path):
    """Schema v3 is one nullable column: old rows survive, new ones measure."""
    path = tmp_path / "old.sqlite"
    conn = db.connect(path)
    with conn:
        db.MIGRATIONS[1](conn)
        db.MIGRATIONS[2](conn)
        conn.execute("PRAGMA user_version = 2")
        conn.execute(
            "INSERT INTO shots (machine, shot, diagnostic, preprocessed, path)"
            " VALUES ('cmod', 1160616027, 'phantom', 0,"
            " '/data/phantom_1160616027.nc')"
        )
    conn.close()

    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    row = conn.execute("SELECT shot, dt FROM shots").fetchone()
    assert row["shot"] == 1160616027 and row["dt"] is None
    with conn:
        conn.execute("UPDATE shots SET dt = 2.5e-6 WHERE shot = 1160616027")
    assert conn.execute("SELECT dt FROM shots").fetchone()[0] == 2.5e-6
    conn.close()


# ---------------------------------------------------------------------------
# Schema v4: what a run was computed from. It runs against the server's live
# file, so the test starts from a v3 file holding real-looking runs -- a 2DCA
# average, a result chained on it, their scalars -- and migrates a copy.
# ---------------------------------------------------------------------------


def _v3_file(path):
    conn = db.connect(path)
    with conn:
        for version in (1, 2, 3):
            db.MIGRATIONS[version](conn)
        conn.execute("PRAGMA user_version = 3")
        conn.executemany(
            "INSERT INTO param_sets VALUES (?, ?, '{}', '2026-09-08T11:52:12+00:00')",
            [("h_avg", "two_dca"), ("h_vel", "velocity_contour")],
        )
        conn.executemany(
            "INSERT INTO runs (id, machine, shot, diagnostic, preprocessed, plot,"
            " params_hash, blob_path, status, error, seconds, code_version,"
            " created_at) VALUES (?, 'cmod', 1160616027, 'apd', 1, ?, ?, ?,"
            " 'ok', NULL, 21.0, 'fusion_ui=dc292ea imaging_methods=unknown', ?)",
            [
                (7, "two_dca", "h_avg", "/cache/a.nc", "2026-09-08T11:52:12+00:00"),
                (
                    8,
                    "velocity_contour",
                    "h_vel",
                    "/cache/v.nc",
                    "2026-09-08T11:52:40+00:00",
                ),
            ],
        )
        conn.execute("INSERT INTO scalars VALUES (8, 5, 7, 'vx_c', 471.0)")
        conn.execute(
            "INSERT INTO shots (machine, shot, diagnostic, preprocessed, path,"
            " bytes, mtime, has_metadata) VALUES ('cmod', 1160616027, 'apd', 1,"
            " '/data/apd_1160616027_preprocessed.nc', 1, "
            " '2026-10-05T09:00:00+00:00', 1)"
        )
    conn.close()
    return path


def _columns(conn, table):
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def test_a_v3_file_migrates_to_v4_keeping_every_run(tmp_path):
    import shutil

    original = _v3_file(tmp_path / "live.sqlite")
    copy = tmp_path / "copy.sqlite"
    shutil.copy(original, copy)

    conn = db.open_db(copy)
    assert db.schema_version(conn) == db.SCHEMA_VERSION >= 4
    assert {"input_mtime", "upstream_run_id"} <= set(_columns(conn, "runs"))
    runs = {row["id"]: row for row in conn.execute("SELECT * FROM runs")}
    assert sorted(runs) == [7, 8]
    assert runs[8]["plot"] == "velocity_contour"
    assert runs[8]["created_at"] == "2026-09-08T11:52:40+00:00"
    assert runs[8]["code_version"] == "fusion_ui=dc292ea imaging_methods=unknown"
    # Pre-v4 rows say nothing about what they were computed from.
    assert all(r["input_mtime"] is None for r in runs.values())
    assert all(r["upstream_run_id"] is None for r in runs.values())
    assert conn.execute("SELECT value FROM scalars").fetchone()[0] == 471.0
    indexes = {r["name"] for r in conn.execute("PRAGMA index_list(runs)")}
    assert "idx_runs_upstream" in indexes
    conn.close()

    # The file it was copied from is untouched.
    conn = db.connect(original)
    assert db.schema_version(conn) == 3
    assert "input_mtime" not in _columns(conn, "runs")
    conn.close()


def test_the_v4_link_is_set_null_when_its_upstream_run_is_deleted(tmp_path):
    """The foreign key added by ALTER TABLE acts on the migrated table."""
    conn = db.open_db(_v3_file(tmp_path / "live.sqlite"))
    with conn:
        conn.execute("UPDATE runs SET upstream_run_id = 7 WHERE id = 8")
    with conn:
        conn.execute("DELETE FROM runs WHERE id = 7")
    row = conn.execute("SELECT upstream_run_id FROM runs WHERE id = 8").fetchone()
    assert row["upstream_run_id"] is None
    # And the downstream's own scalars are not cascaded away with it.
    assert conn.execute("SELECT COUNT(*) FROM scalars").fetchone()[0] == 1
    conn.close()


def test_a_half_applied_v4_migration_is_finished_not_failed(tmp_path):
    """One column already there, the version not yet bumped: init-db again."""
    path = _v3_file(tmp_path / "live.sqlite")
    conn = db.connect(path)
    with conn:
        conn.execute("ALTER TABLE runs ADD COLUMN input_mtime TEXT")
    conn.close()

    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    assert {"input_mtime", "upstream_run_id"} <= set(_columns(conn, "runs"))
    conn.close()


def test_a_failed_v4_migration_leaves_the_file_at_v3(tmp_path, monkeypatch):
    """Columns and version move together, or not at all."""
    path = _v3_file(tmp_path / "live.sqlite")

    def interrupted(conn):
        real_v4(conn)
        raise sqlite3.OperationalError("disk I/O error")

    real_v4 = db.MIGRATIONS[4]
    migrations = list(db.MIGRATIONS)
    migrations[4] = interrupted
    monkeypatch.setattr(db, "MIGRATIONS", migrations)
    conn = db.connect(path)
    with pytest.raises(sqlite3.OperationalError):
        db.init_db(conn)
    conn.close()

    conn = db.connect(path)
    assert db.schema_version(conn) == 3
    assert "upstream_run_id" not in _columns(conn, "runs")
    conn.close()
