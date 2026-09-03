"""Smoke tests for the multi-shot page."""

import dataclasses
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import fusion_ui.plots  # noqa: F401
from fusion_ui.core import catalog, db, registry, store

REPO_ROOT = Path(__file__).resolve().parent.parent
MULTI_SHOT = str(REPO_ROOT / "fusion_ui" / "pages" / "3_multi_shot.py")


@dataclasses.dataclass
class _Params:
    source: str = "test"


@pytest.fixture
def deployment(monkeypatch, tmp_path, data_folder, discharge_db):
    import streamlit as st

    database = tmp_path / "state" / "shot_explorer.sqlite"
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(data_folder))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(discharge_db))
    monkeypatch.setenv("FUSION_UI_DB", str(database))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")

    conn = db.open_db(database)
    catalog.rescan(conn, str(data_folder), "cmod", str(discharge_db))
    _write_scalars(conn)
    conn.close()

    st.cache_data.clear()
    st.cache_resource.clear()
    yield database
    st.cache_data.clear()
    st.cache_resource.clear()


def _write_scalars(conn, machine="cmod", scalars=None):
    """Two shots with stored ``vx_c`` values, one source, per-pixel."""
    for shot, values in scalars or [
        (1160616027, {(6, 6): 400.0, (6, 5): 380.0}),
        (1110201007, {(6, 6): 100.0}),
    ]:
        target = registry.Target(machine, shot, "apd", True, "", 0.0, 0.0, "none")
        params_hash, _ = store.record_params(conn, "synthetic", _Params())
        run = store.record_run(
            conn,
            target,
            "synthetic",
            params_hash,
            blob_path=None,
            status="ok",
            error=None,
            seconds=None,
            code_version="test",
        )
        mapping = {(x, y, "vx_c"): v for (x, y), v in values.items()}
        store.write_scalars(conn, run["id"], mapping)


def run(path, default_timeout=60):
    app = AppTest.from_file(path, default_timeout=default_timeout).run()
    assert not app.exception, app.exception
    return app


def widget(app, kind, label):
    matches = [w for w in getattr(app, kind) if w.label == label]
    assert (
        matches
    ), f"no {kind} labelled {label!r}: {[w.label for w in getattr(app, kind)]}"
    return matches[0]


def test_an_empty_store_explains_itself(
    monkeypatch, tmp_path, data_folder, discharge_db
):
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

    app = run(MULTI_SHOT)
    assert app.warning
    st.cache_data.clear()
    st.cache_resource.clear()


def test_the_page_offers_the_stored_scalar_and_renders(deployment):
    app = run(MULTI_SHOT)
    assert widget(app, "selectbox", "Scalar").options == ["vx_c"]
    # Both shots have a stored value, so the scatter carries two points.
    assert any("2 shots" in c.value for c in app.caption)


def test_a_carried_over_selection_restricts_the_scatter(deployment):
    app = AppTest.from_file(MULTI_SHOT, default_timeout=60)
    app.session_state["shot_selection"] = [1110201007]
    app.run()
    assert not app.exception
    assert any("1 shot" in c.value for c in app.caption)


def test_the_aggregate_picker_offers_each_collapse(deployment):
    app = run(MULTI_SHOT)
    options = widget(app, "selectbox", "Aggregate").options
    assert options == [
        "mean over pixels",
        "median over pixels",
        "maximum over pixels",
        "fixed pixel",
    ]


def test_another_machines_scalars_are_left_out(deployment):
    """`rescan` only deletes its own machine's rows, so an index can hold two
    machines at once -- and shot numbers are not unique across them."""
    conn = db.open_db(deployment)
    _write_scalars(conn, machine="aug", scalars=[(1160616027, {(6, 6): -1.0})])
    conn.close()

    app = run(MULTI_SHOT)
    assert any("2 shots on `cmod`" in c.value for c in app.caption)
