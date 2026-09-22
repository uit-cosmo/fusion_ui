import math

import pytest

from fusion_ui.core import catalog


PHANTOM_DT = 2.5e-6


def _phantom_tree(tmp_path, shot=1160616027, dt=PHANTOM_DT, n=200):
    """A data tree with one real (tiny) phantom file plus an empty APD decoy."""
    import numpy as np
    import xarray as xr

    root = tmp_path / "alcator"
    folder = root / "phantom"
    folder.mkdir(parents=True)
    ds = xr.Dataset(
        {"frames": (["time", "y", "x"], np.zeros((n, 4, 5)))},
        coords={"time": ("time", np.arange(n) * dt)},
        attrs={"shot_number": shot},
    )
    ds.to_netcdf(folder / f"phantom_{shot}.nc")
    (root / "apd").mkdir(exist_ok=True)
    (root / "apd" / f"apd_{shot}.nc").write_bytes(b"x")
    return root


def rescan(conn, data_folder, discharge_db, machine="cmod"):
    return catalog.rescan(conn, str(data_folder), machine, str(discharge_db))


def rows(conn):
    return {
        (row["shot"], row["diagnostic"], row["preprocessed"]): row
        for row in conn.execute("SELECT * FROM shots")
    }


def test_indexes_raw_and_preprocessed(conn, data_folder, discharge_db):
    stats = rescan(conn, data_folder, discharge_db)
    assert stats.seen == 5  # 4 apd + 1 asp; notes.txt and apd_broken.nc are not
    assert stats.inserted == 5
    assert stats.skipped == 2
    assert stats.missing_folders == ("phantom",)

    indexed = rows(conn)
    assert (1160616027, "apd", 0) in indexed
    assert (1160616027, "apd", 1) in indexed
    assert (1150618021, "asp", 0) in indexed
    row = indexed[(1160616027, "apd", 0)]
    assert row["path"].endswith("apd/apd_1160616027.nc")
    assert row["bytes"] > 0 and row["mtime"].endswith("+00:00")


def test_has_metadata_flags_uncurated_shots(conn, data_folder, discharge_db):
    rescan(conn, data_folder, discharge_db)
    indexed = rows(conn)
    assert indexed[(1160616027, "apd", 0)]["has_metadata"] == 1
    assert indexed[(1150618021, "apd", 0)]["has_metadata"] == 0


def test_rescan_is_idempotent(conn, data_folder, discharge_db):
    rescan(conn, data_folder, discharge_db)
    stats = rescan(conn, data_folder, discharge_db)
    assert (stats.inserted, stats.updated, stats.removed) == (0, 0, 0)
    assert stats.unchanged == 5


def test_deleted_file_drops_its_row(conn, data_folder, discharge_db):
    rescan(conn, data_folder, discharge_db)
    (data_folder / "apd" / "apd_1150618021.nc").unlink()
    stats = rescan(conn, data_folder, discharge_db)
    assert stats.removed == 1
    assert (1150618021, "apd", 0) not in rows(conn)
    assert (1150618021, "asp", 0) in rows(conn)


def test_changed_file_is_updated(conn, data_folder, discharge_db):
    rescan(conn, data_folder, discharge_db)
    (data_folder / "asp" / "asp_1150618021.nc").write_bytes(b"y" * 999)
    stats = rescan(conn, data_folder, discharge_db)
    assert stats.updated == 1
    assert rows(conn)[(1150618021, "asp", 0)]["bytes"] == 999


def test_machines_do_not_clobber_each_other(conn, data_folder, discharge_db):
    rescan(conn, data_folder, discharge_db, machine="cmod")
    rescan(conn, data_folder, discharge_db, machine="w7x")
    assert conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == 10
    empty = data_folder.parent / "empty"
    (empty / "apd").mkdir(parents=True)
    catalog.rescan(conn, str(empty), "w7x", str(discharge_db))
    # Rescanning w7x must never touch cmod's rows. And a diagnostic whose
    # folder is missing right now (asp/phantom/...) must not be wiped by a
    # transient mount loss: only the present-but-empty apd folder drops rows.
    machines = {row["machine"] for row in conn.execute("SELECT machine FROM shots")}
    assert machines == {"cmod", "w7x"}
    assert conn.execute(
        "SELECT COUNT(*) FROM shots WHERE machine = 'cmod'"
    ).fetchone()[0] == 5
    # The w7x apd rows are gone (folder seen empty); the w7x asp row survives
    # (folder missing, not empty).
    assert conn.execute(
        "SELECT COUNT(*) FROM shots WHERE machine = 'w7x' AND diagnostic = 'asp'"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM shots WHERE machine = 'w7x' AND diagnostic = 'apd'"
    ).fetchone()[0] == 0


def test_rescan_without_a_discharge_db(conn, data_folder):
    stats = catalog.rescan(conn, str(data_folder), "cmod", None)
    assert stats.seen == 5
    assert all(row["has_metadata"] == 0 for row in rows(conn).values())


def test_shot_table_is_one_row_per_shot(conn, data_folder, discharge_db):
    rescan(conn, data_folder, discharge_db)
    table = catalog.shot_table(conn, str(discharge_db))

    assert list(table["shot"]) == [1110201007, 1150618021, 1160616027]
    by_shot = table.set_index("shot")
    assert by_shot.loc[1160616027, "apd"] == "R+P"
    assert by_shot.loc[1110201007, "apd"] == "R"
    assert by_shot.loc[1150618021, "apd"] == "R"
    assert by_shot.loc[1150618021, "asp"] == "R"
    assert by_shot.loc[1160616027, "asp"] == ""
    assert list(table.columns) == catalog.TABLE_COLUMNS


def test_shot_table_carries_metadata_and_the_missing_flag(
    conn, data_folder, discharge_db
):
    rescan(conn, data_folder, discharge_db)
    by_shot = catalog.shot_table(conn, str(discharge_db)).set_index("shot")

    curated = by_shot.loc[1160616027]
    assert curated["has_metadata"] and curated["mode"] == "IWL"
    assert (curated["t_start"], curated["t_end"]) == (1.3, 1.6)
    assert curated["f_GW"] == 0.72 and curated["f_GW_source"] == "db"

    uncurated = by_shot.loc[1150618021]
    assert not uncurated["has_metadata"]
    assert uncurated["mode"] == "" and math.isnan(uncurated["f_GW"])


def test_f_gw_falls_back_to_the_derived_value(conn, data_folder, discharge_db):
    rescan(conn, data_folder, discharge_db)
    row = catalog.shot_table(conn, str(discharge_db)).set_index("shot").loc[1110201007]
    assert row["f_GW_source"] == "derived"
    assert row["f_GW"] == 1.10 * math.pi * 0.22**2 / 0.93


def test_greenwald_fraction_never_divides_by_zero(discharge_db):
    """I_p = 0 is common in the descriptor's placeholder rows."""
    zero_current = catalog.load_discharges(str(discharge_db))[1090813019]
    value, source = catalog.greenwald_fraction(zero_current)
    assert math.isnan(value) and source == ""


def test_duplicate_descriptor_entries_are_deduped(discharge_db):
    discharges = catalog.load_discharges(str(discharge_db))
    assert sorted(discharges) == [1090813019, 1110201007, 1160616027]


def test_empty_index_gives_an_empty_table(conn, discharge_db):
    table = catalog.shot_table(conn, str(discharge_db))
    assert table.empty and list(table.columns) == catalog.TABLE_COLUMNS


def test_fingerprint_changes_with_the_index(conn, data_folder, discharge_db):
    before = catalog.index_fingerprint(conn)
    rescan(conn, data_folder, discharge_db)
    assert catalog.index_fingerprint(conn) != before


def test_frame_dt_reads_the_median_interval(tmp_path):
    root = _phantom_tree(tmp_path)
    value = catalog.frame_dt(str(root / "phantom" / "phantom_1160616027.nc"))
    assert value == pytest.approx(PHANTOM_DT)


@pytest.mark.parametrize("name", ["missing.nc", "empty.nc", "no_time.nc"])
def test_frame_dt_returns_none_for_broken_files(tmp_path, name):
    import numpy as np
    import xarray as xr

    path = tmp_path / name
    if name == "empty.nc":
        path.write_bytes(b"not a dataset")
    elif name == "no_time.nc":
        xr.Dataset({"frames": (["y", "x"], np.zeros((4, 5)))}).to_netcdf(path)
    assert catalog.frame_dt(str(path)) is None


def test_backfill_dt_fills_phantom_rows_only(conn, tmp_path):
    root = _phantom_tree(tmp_path)
    catalog.rescan(conn, str(root), "cmod", None)

    stats = catalog.backfill_dt(conn, "cmod")
    assert (stats.considered, stats.measured, stats.skipped, stats.failed) == (1, 1, 0, 0)

    rows = {
        row["diagnostic"]: row["dt"]
        for row in conn.execute("SELECT diagnostic, dt FROM shots")
    }
    assert rows["phantom"] == pytest.approx(PHANTOM_DT)
    assert rows["apd"] is None, "the APD row is not this command's business"


def test_backfill_dt_skips_filled_rows_unless_forced(conn, tmp_path):
    root = _phantom_tree(tmp_path)
    catalog.rescan(conn, str(root), "cmod", None)

    catalog.backfill_dt(conn, "cmod")
    second = catalog.backfill_dt(conn, "cmod")
    assert (second.measured, second.skipped) == (0, 1)

    forced = catalog.backfill_dt(conn, "cmod", force=True)
    assert (forced.measured, forced.skipped) == (1, 0)


def test_backfill_dt_counts_a_vanished_file_as_failed(conn, tmp_path):
    root = _phantom_tree(tmp_path)
    catalog.rescan(conn, str(root), "cmod", None)
    (root / "phantom" / "phantom_1160616027.nc").unlink()

    stats = catalog.backfill_dt(conn, "cmod")
    assert (stats.measured, stats.failed) == (0, 1)
    assert conn.execute("SELECT dt FROM shots").fetchone()[0] is None


def test_rescan_preserves_a_backfilled_dt(conn, tmp_path):
    root = _phantom_tree(tmp_path)
    catalog.rescan(conn, str(root), "cmod", None)
    catalog.backfill_dt(conn, "cmod")

    (root / "apd" / "apd_1160616027.nc").write_bytes(b"y" * 999)
    stats = catalog.rescan(conn, str(root), "cmod", None)
    assert stats.updated == 1
    assert conn.execute(
        "SELECT dt FROM shots WHERE diagnostic = 'phantom'"
    ).fetchone()[0] == pytest.approx(PHANTOM_DT)


def test_shot_table_reports_phantom_dt_in_seconds(conn, tmp_path):
    root = _phantom_tree(tmp_path)
    catalog.rescan(conn, str(root), "cmod", None)

    before = catalog.shot_table(conn, None).set_index("shot")
    assert math.isnan(before.loc[1160616027, "phantom_dt"])

    catalog.backfill_dt(conn, "cmod")
    after = catalog.shot_table(conn, None).set_index("shot")
    assert after.loc[1160616027, "phantom_dt"] == pytest.approx(PHANTOM_DT)


def test_fingerprint_moves_on_backfill(conn, tmp_path):
    root = _phantom_tree(tmp_path)
    catalog.rescan(conn, str(root), "cmod", None)
    before = catalog.index_fingerprint(conn)
    catalog.backfill_dt(conn, "cmod")
    assert catalog.index_fingerprint(conn) != before
