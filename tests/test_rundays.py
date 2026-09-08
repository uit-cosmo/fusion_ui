"""The run-day overview: parsing the hand-written file, and the join to disk."""

import pytest

from fusion_ui.core import db, rundays

SAMPLE = """# Run days

Preamble, which has no heading of its own and must not become an entry.

## 1160616

**MP800 — Scrape-off layer fluctuation statistic in ohmic L- and EDA H-modes**

An ohmic L-mode density scan to high Greenwald fraction.

Second paragraph.

## 1110201

**MP644 — Edge profiles and fluctuations in EDA and ELM-free H-Modes**

Edge of H-mode plasmas.

## 1090813

Half-written: no miniproposal line yet, so no entry.
"""


def test_parse_keeps_the_verbatim_title_and_the_whole_summary():
    days = rundays.parse(SAMPLE)
    assert set(days) == {"1160616", "1110201"}
    entry = days["1160616"]
    assert entry.mp == "MP800"
    # The upstream title carries a typo ("statistic"); it is quoted, not fixed.
    assert entry.title == (
        "Scrape-off layer fluctuation statistic in ohmic L- and EDA H-modes"
    )
    assert entry.summary.startswith("An ohmic L-mode density scan")
    assert entry.summary.endswith("Second paragraph.")
    assert entry.date == "2016-06-16"
    assert entry.url == (
        "https://www-internal.psfc.mit.edu/research/alcator/miniproposals/800.pdf"
    )


def test_a_shot_maps_onto_its_run_day():
    assert rundays.day_of(1160616027) == "1160616"
    assert rundays.day_to_date("1090813") == "2009-08-13"


def test_the_shipped_file_covers_every_curated_run_day(discharge_db):
    """Every day in the throwaway descriptor is one the real file describes --
    the fixture shots are real shot numbers, so a missing entry here means the
    file has fallen behind the discharge database."""
    days = rundays.load()
    assert len(days) >= 22
    for shot in (1160616027, 1110201007, 1090813019):
        assert rundays.day_of(shot) in days


def test_the_table_joins_curated_days_to_what_is_on_disk(
    conn, data_folder, discharge_db
):
    from fusion_ui.core import catalog

    catalog.rescan(conn, str(data_folder), "cmod", str(discharge_db))
    table = rundays.run_day_table(conn, str(discharge_db)).set_index("day")

    # Three curated days, plus 1150618 which is on disk but uncurated.
    assert list(table.index) == ["1090813", "1110201", "1150618", "1160616"]
    assert table.loc["1160616", "shots"] == 1
    assert table.loc["1160616", "on_disk"] == 1
    assert table.loc["1160616", "diagnostics"] == "apd"
    # Curated but no files here: still listed, so the campaign is visible.
    assert (table.loc["1090813", "shots"], table.loc["1090813", "on_disk"]) == (1, 0)
    # Files but nobody curated it: listed too, the way the browser lists it.
    assert (table.loc["1150618", "shots"], table.loc["1150618", "on_disk"]) == (0, 1)
    assert table.loc["1150618", "diagnostics"] == "apd asp"
    assert table.loc["1160616", "mp"] == "MP800"
    assert table.loc["1160616", "mp_url"].endswith("/miniproposals/800.pdf")
    # A day with no section gets no link rather than a link to nowhere.
    assert table.loc["1150618", "mp_url"].endswith("/miniproposals/761.pdf")


def test_the_table_survives_a_missing_discharge_database(conn, data_folder):
    from fusion_ui.core import catalog

    catalog.rescan(conn, str(data_folder), "cmod", None)
    table = rundays.run_day_table(conn, None)
    assert set(table["day"]) == {"1110201", "1150618", "1160616"}
    assert (table["shots"] == 0).all()


def test_a_missing_run_day_file_is_not_an_error(tmp_path):
    assert rundays.load(tmp_path / "nothing.md") == {}
