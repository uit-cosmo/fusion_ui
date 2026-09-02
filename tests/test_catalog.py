import math

from fusion_ui.core import catalog


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
    machines = {row["machine"] for row in conn.execute("SELECT machine FROM shots")}
    assert machines == {"cmod"}


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
