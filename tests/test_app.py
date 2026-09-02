"""Smoke tests: the pages render without raising, against a temporary tree."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from fusion_ui import config
from fusion_ui.core import catalog, db

REPO_ROOT = Path(__file__).resolve().parent.parent
APP = str(REPO_ROOT / "fusion_ui" / "app.py")
BROWSER = str(REPO_ROOT / "fusion_ui" / "pages" / "1_shot_browser.py")
SINGLE_SHOT = str(REPO_ROOT / "fusion_ui" / "pages" / "2_single_shot.py")


@pytest.fixture
def deployment(monkeypatch, tmp_path, data_folder, discharge_db):
    """Point the whole app at the throwaway tree, index it, clear the caches."""
    import streamlit as st

    database = tmp_path / "state" / "shot_explorer.sqlite"
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(data_folder))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(discharge_db))
    monkeypatch.setenv("FUSION_UI_DB", str(database))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")

    conn = db.open_db(database)
    catalog.rescan(conn, str(data_folder), "cmod", str(discharge_db))
    conn.close()

    st.cache_data.clear()
    st.cache_resource.clear()
    yield database
    st.cache_data.clear()
    st.cache_resource.clear()


def run(path):
    app = AppTest.from_file(path, default_timeout=60).run()
    assert not app.exception, app.exception
    return app


def test_landing_page_renders(deployment):
    app = run(APP)
    assert "Shot Explorer" in app.title[0].value
    assert [m.value for m in app.metric if m.label == "Shots"] == ["3"]
    assert not app.error


def test_landing_page_survives_a_missing_data_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(tmp_path / "gone"))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(tmp_path / "gone.json"))
    monkeypatch.setenv("FUSION_UI_DB", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()
    app = run(APP)
    # It reports the broken paths rather than dying on them.
    assert app.error
    st.cache_data.clear()
    st.cache_resource.clear()


def test_shot_browser_lists_every_shot(deployment):
    app = run(BROWSER)
    table = app.dataframe[0].value
    assert sorted(table["shot"]) == [1110201007, 1150618021, 1160616027]
    assert set(table["meta"]) == {"✓", "⚠"}
    assert "3 of 3 shots" in app.caption[1].value


def test_shot_browser_filters_on_missing_metadata(deployment):
    app = run(BROWSER)
    app.sidebar.radio[0].set_value("Missing only").run()
    assert not app.exception
    assert list(app.dataframe[0].value["shot"]) == [1150618021]


def test_selecting_a_shot_writes_the_selection_contract(deployment):
    app = run(BROWSER)
    app.button[1].click().run()  # "Use these N shots for multi-shot"
    assert app.session_state["shot_selection"] == [
        1110201007,
        1150618021,
        1160616027,
    ]


def test_browser_says_so_when_the_index_is_empty(monkeypatch, tmp_path, discharge_db):
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(tmp_path / "empty"))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(discharge_db))
    monkeypatch.setenv("FUSION_UI_DB", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()
    app = run(BROWSER)
    assert app.warning
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture
def single_shot_deployment(monkeypatch, tmp_path, apd_dataset_path, asp_dataset_path):
    """A data tree with one real (tiny) APD file and one real ASP file.

    Unlike ``deployment``, no discharge DB is set up -- the point is to also
    exercise the single-shot page's "no metadata yet" default-window path.
    """
    import streamlit as st

    data_folder = apd_dataset_path.parent.parent  # .../alcator, shared with asp
    database = tmp_path / "state" / "shot_explorer.sqlite"
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(data_folder))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(tmp_path / "no_such_discharges.json"))
    monkeypatch.setenv("FUSION_UI_DB", str(database))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")

    conn = db.open_db(database)
    catalog.rescan(conn, str(data_folder), "cmod", None)
    conn.close()

    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def test_single_shot_frame_view_renders(single_shot_deployment):
    # Shot 1234 (apd) sorts before 5678 (asp), so it is the default pick with
    # no need to touch the sidebar -- exercises the "standalone" path where no
    # browser selection has been made yet.
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60).run()
    assert app.session_state["selection"] is None
    assert not app.exception
    assert "Window" in app.caption[0].value
    assert "no discharge-DB entry" in app.caption[0].value


def test_single_shot_probe_view_renders(single_shot_deployment):
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60)
    app.session_state["selection"] = {
        "machine": "cmod",
        "shot": 5678,
        "diagnostic": "asp",
        "preprocessed": False,
    }
    app.run()
    assert not app.exception
    assert app.selectbox[0].label == "Quantity"


def test_single_shot_click_moves_the_selected_pixel(single_shot_deployment):
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60).run()
    assert not app.exception
    app.session_state["pixel_cmod_1234_apd"] = (2, 3)
    app.run()
    assert not app.exception
    assert "y=2, x=3" in app.caption[2].value


def test_connections_are_not_shared_between_threads(deployment):
    """Streamlit runs each session on a pooled thread; sqlite3 objects are not
    shareable across them, so a process-wide cached connection breaks as soon as
    a second browser tab connects."""
    import threading

    from fusion_ui import ui

    results = []

    def query():
        try:
            results.append(
                ui.get_connection().execute("SELECT COUNT(*) FROM shots").fetchone()[0]
            )
        except Exception as error:  # noqa: BLE001 - the assertion is the report
            results.append(error)

    threads = [threading.Thread(target=query) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [5, 5, 5], results
