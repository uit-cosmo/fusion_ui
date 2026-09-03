"""Smoke tests: the pages render without raising, against a temporary tree."""

import dataclasses
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import fusion_ui.plots  # noqa: F401 - registers the specs the page offers
from fusion_ui import config
from fusion_ui.core import catalog, db, registry, store

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
    yield database
    st.cache_data.clear()
    st.cache_resource.clear()


def widget(app, kind, label):
    """One widget by label rather than by position.

    The single-shot page is assembled from the registry now, so which widgets
    exist depends on which spec is selected; indexing into ``app.selectbox[0]``
    would make every one of these tests a hostage to widget ordering.
    """
    matches = [w for w in getattr(app, kind) if w.label == label]
    assert (
        matches
    ), f"no {kind} labelled {label!r}: {[w.label for w in getattr(app, kind)]}"
    return matches[0]


def captions(app):
    return [c.value for c in app.caption]


def test_single_shot_frame_view_renders(single_shot_deployment):
    # Shot 1234 (apd) sorts before 5678 (asp), so it is the default pick with
    # no need to touch the sidebar -- exercises the "standalone" path where no
    # browser selection has been made yet.
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60).run()
    assert app.session_state["selection"] is None
    assert not app.exception
    window = [c for c in captions(app) if c.startswith("Window")]
    assert window and "no discharge-DB entry" in window[0]


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
    assert widget(app, "selectbox", "quantity").options == ["Vf", "ne"]
    # A probe file has no shared time axis, so there is no window to report.
    assert not [c for c in captions(app) if c.startswith("Window")]


def test_single_shot_click_moves_the_selected_pixel(single_shot_deployment):
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60).run()
    assert not app.exception
    app.session_state["pixel.cmod_1234_apd_r"] = (2, 3)
    app.run()
    assert not app.exception
    assert any("y=2, x=3" in c for c in captions(app))


def test_the_plot_picker_offers_only_specs_for_this_diagnostic(single_shot_deployment):
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60).run()
    assert widget(app, "selectbox", "Plot").options == [
        "Frames and pixel trace",
        "Duration time (PSD fit)",
        "Conditional average (2DCA)",
        "Blob velocity (contour tracking)",
        "Blob size (FWHM)",
    ]

    app.session_state["selection"] = {
        "machine": "cmod",
        "shot": 5678,
        "diagnostic": "asp",
        "preprocessed": False,
    }
    app.run()
    assert widget(app, "selectbox", "Plot").options == ["Probe trace"]


def test_a_cached_spec_does_not_run_until_compute_is_pressed(single_shot_deployment):
    """A nudged widget must not be able to start a long analysis."""
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=120)
    app.session_state["spec.apd"] = registry.get("taud_psd")
    app.run()
    assert not app.exception
    assert app.info, "expected the 'press Compute' prompt"

    conn = db.connect(single_shot_deployment)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    conn.close()


def test_computing_a_cached_spec_records_a_run_and_its_scalars(single_shot_deployment):
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=120)
    app.session_state["spec.apd"] = registry.get("taud_psd")
    app.run()
    widget(app, "button", "Compute").click().run()
    assert not app.exception
    assert not app.error

    conn = db.connect(single_shot_deployment)
    run = conn.execute("SELECT * FROM runs").fetchone()
    assert run["plot"] == "taud_psd"
    assert run["status"] == "ok"
    assert run["preprocessed"] == 0
    assert os.path.exists(run["blob_path"])

    scalars = {
        (r["x"], r["y"], r["name"]) for r in conn.execute("SELECT * FROM scalars")
    }
    # The tiny fixture array is 4x5, so the default reference pixel (6, 6) is
    # clamped by the spec's choices to what actually exists.
    assert {name for _, _, name in scalars} == {"taud_psd", "lambda_psd"}
    conn.close()

    assert any("Computed" in c for c in captions(app))


def test_a_failed_run_is_shown_rather_than_raised(single_shot_deployment, monkeypatch):
    """A recorded failure must render as an error with a Retry button, not as
    a traceback on every rerun."""
    conn = db.open_db(single_shot_deployment)
    target = registry.Target(
        machine="cmod",
        shot=1234,
        diagnostic="apd",
        preprocessed=False,
        path="unused",
        t_start=1.0,
        t_end=1.02,
    )
    spec = registry.get("taud_psd")
    params = spec.params(refx=0, refy=0)
    store.compute_and_store(
        conn,
        dataclasses.replace(spec, compute=_boom),
        target,
        params,
        ds=None,
    )
    conn.close()

    app = AppTest.from_file(SINGLE_SHOT, default_timeout=120)
    app.session_state["spec.apd"] = spec
    app.session_state["params.taud_psd.refx"] = 0
    app.session_state["params.taud_psd.refy"] = 0
    app.session_state["ready.taud_psd.cmod_1234_apd_r"] = True
    app.run()

    assert not app.exception
    assert app.error and "deliberate" in app.error[0].value
    assert widget(app, "button", "Retry")


def _boom(ds, params):
    raise ValueError("deliberate test failure")


@pytest.fixture
def blob_deployment(monkeypatch, tmp_path, blob_dataset_path):
    """One indexed shot whose data has blobs in it, so the 2DCA chain can run
    through the page rather than only through ``store.result``."""
    import streamlit as st

    data_folder = blob_dataset_path.parent.parent
    database = tmp_path / "state" / "shot_explorer.sqlite"
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(data_folder))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(tmp_path / "none.json"))
    monkeypatch.setenv("FUSION_UI_DB", str(database))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")

    conn = db.open_db(database)
    catalog.rescan(conn, str(data_folder), "cmod", None)
    conn.close()

    st.cache_data.clear()
    st.cache_resource.clear()
    yield database
    st.cache_data.clear()
    st.cache_resource.clear()


def test_a_chained_spec_renders_and_stores_both_links(blob_deployment):
    """The whole phase-03 shape through the page: the deepest parameter form in
    the app, one Compute, and two ledger rows -- the derived plot and the
    conditional average it was built on."""
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=180)
    app.session_state["spec.apd"] = registry.get("velocity_contour")
    # The APD default reference pixel is (8, 8), a corner of this 9x9 fixture
    # that no blob crosses. Parameters live under the spec key, not the target,
    # so they survive moving to the next shot.
    app.session_state["params.velocity_contour.two_dca.refx"] = 4
    app.session_state["params.velocity_contour.two_dca.refy"] = 4
    app.run()
    assert not app.exception
    # Four nested dataclasses' worth of widgets, including the two string
    # fields that only exist as prose upstream.
    assert widget(app, "selectbox", "estimator").options == ["central_diff", "lsq"]
    assert widget(app, "selectbox", "window type").options[0] == "hann"
    assert widget(app, "checkbox", "require within boundaries") is not None

    widget(app, "button", "Compute").click().run()
    assert not app.exception
    assert not app.error, [e.value for e in app.error]

    conn = db.connect(blob_deployment)
    runs = {r["plot"]: r for r in conn.execute("SELECT * FROM runs")}
    assert set(runs) == {"two_dca", "velocity_contour"}
    assert all(r["status"] == "ok" for r in runs.values())
    assert all(os.path.exists(r["blob_path"]) for r in runs.values())
    velocity = {
        r["name"]: r["value"]
        for r in conn.execute(
            "SELECT s.name, s.value FROM scalars s WHERE s.run_id = ?",
            (runs["velocity_contour"]["id"],),
        )
    }
    conn.close()
    assert velocity["vx_c"] == pytest.approx(400.0, rel=0.05)


def test_the_fwhm_spec_renders_and_stores_both_links(blob_deployment):
    """The other chained port, through the page: one Compute, two ledger
    rows -- the FWHM sizes and the conditional average they were built on."""
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=180)
    app.session_state["spec.apd"] = registry.get("fwhm_sizes")
    # Same corner-pixel trap as the contour spec: the APD default (8, 8) is a
    # corner of this 9x9 fixture that no blob crosses.
    app.session_state["params.fwhm_sizes.two_dca.refx"] = 4
    app.session_state["params.fwhm_sizes.two_dca.refy"] = 4
    app.run()
    assert not app.exception

    widget(app, "button", "Compute").click().run()
    assert not app.exception
    assert not app.error, [e.value for e in app.error]

    conn = db.connect(blob_deployment)
    runs = {r["plot"]: r for r in conn.execute("SELECT * FROM runs")}
    assert set(runs) == {"two_dca", "fwhm_sizes"}
    assert all(r["status"] == "ok" for r in runs.values())
    assert all(os.path.exists(r["blob_path"]) for r in runs.values())
    sizes = {
        r["name"]: r["value"]
        for r in conn.execute(
            "SELECT s.name, s.value FROM scalars s WHERE s.run_id = ?",
            (runs["fwhm_sizes"]["id"],),
        )
    }
    conn.close()
    assert sizes["lr"] == pytest.approx(sizes["lz"], rel=0.05)


def test_a_cached_spec_may_draw_its_own_widgets(blob_deployment):
    """The conditional average is a short movie, so it draws a field picker and
    a lag slider instead of returning one figure. Those are view state: they
    must not appear in the parameter form or in the hash."""
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=180)
    app.session_state["spec.apd"] = registry.get("two_dca")
    app.session_state["params.two_dca.two_dca.refx"] = 4
    app.session_state["params.two_dca.two_dca.refy"] = 4
    app.run()
    widget(app, "button", "Compute").click().run()
    assert not app.exception
    assert not app.error, [e.value for e in app.error]

    assert widget(app, "selectbox", "Field").options == [
        "conditional average",
        "conditional representativeness",
        "cross-correlation",
    ]
    assert any("20 events" in c for c in captions(app))

    conn = db.connect(blob_deployment)
    (before,) = conn.execute("SELECT COUNT(*) FROM param_sets").fetchone()
    conn.close()

    widget(app, "slider", "Lag").set_value(3).run()
    assert not app.exception
    conn = db.connect(blob_deployment)
    (after,) = conn.execute("SELECT COUNT(*) FROM param_sets").fetchone()
    (runs,) = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
    conn.close()
    assert (after, runs) == (before, 1)
