"""A spec that needs the whole record gets it, from the single-shot page and from precompute alike.

Every other spec computes on the file cut to the discharge window, which keeps a 500 MB record out of memory. A spec
that declares ``whole_record`` finds something against the *start* of the record, and is handed all of it, with the
discharge window it would have been cut to in the dataset's attributes. ``loader.input_for`` decides, and both callers
ask it; what is tested here is that they do, that the default is unchanged, and, with the dead-pixel view on records
with a gas puff in them, that the page and ``fusion-ui precompute`` end at the analysis window of the whole record,
which is the one preprocessing makes.

The toys (``tests/precompute_toys``) record what they were given; the real spec runs on ``tests/puff_fixtures``.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from streamlit.testing.v1 import AppTest

import fusion_ui.plots  # noqa: F401 - registers the real specs
from fusion_ui.core import catalog, db, fusion_scripts, loader, precompute, registry, store
from fusion_ui.plots import dead_pixels
from tests import precompute_toys as toys
from tests import puff_fixtures as pf

REPO_ROOT = Path(__file__).resolve().parent.parent
SINGLE_SHOT = str(REPO_ROOT / "fusion_ui" / "pages" / "2_single_shot.py")

#: The tiny APD fixture (``apd_dataset_path``) is 200 samples over 1.00-1.02 s; its discharge window is this.
WINDOW = (1.005, 1.01)


@pytest.fixture(autouse=True, scope="module")
def registered_toys():
    """The toys, for these tests only: a page under test elsewhere lists every registered spec."""
    toys.register()
    yield
    toys.unregister()


def deployment(monkeypatch, tmp_path, data_folder, shot, window):
    """Point the whole app at ``data_folder``, whose discharge DB gives ``shot`` this ``window``; index it."""
    import streamlit as st

    discharges = tmp_path / "plasma_discharges.json"
    discharges.write_text(json.dumps([pf.discharge_entry(shot, window)]))
    database = tmp_path / "state" / "shot_explorer.sqlite"
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(data_folder))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(discharges))
    monkeypatch.setenv("FUSION_UI_DB", str(database))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")
    conn = db.open_db(database)
    catalog.rescan(conn, str(data_folder), "cmod", str(discharges))
    st.cache_data.clear()
    st.cache_resource.clear()
    return conn


@pytest.fixture
def tiny(monkeypatch, tmp_path, apd_dataset_path):
    """The tiny APD file, shot 1234, indexed, with a discharge window narrower than the record."""
    conn = deployment(monkeypatch, tmp_path, apd_dataset_path.parent.parent, 1234, WINDOW)
    yield SimpleNamespace(conn=conn, path=apd_dataset_path)
    import streamlit as st

    conn.close()
    st.cache_data.clear()
    st.cache_resource.clear()


def blob_attrs(conn, plot):
    (row,) = conn.execute("SELECT * FROM runs WHERE plot = ?", (plot,)).fetchall()
    assert row["status"] == "ok", row["error"]
    with xr.open_dataset(row["blob_path"]) as blob:
        return dict(blob.attrs)


def press_compute(spec_key):
    """The single-shot page on the first shot, with ``spec_key`` picked and Compute pressed."""
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=120)
    app.session_state["spec.apd"] = registry.get(spec_key)
    app.run()
    (compute,) = [b for b in app.button if b.label == "Compute"]
    compute.click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    return app


def in_window(path, window):
    with xr.open_dataset(path) as ds:
        return int(ds.sizes["time"]), int(ds.sel(time=slice(*window)).sizes["time"])


# ---------------------------------------------------------------------------
# The single-shot page
# ---------------------------------------------------------------------------


def test_the_page_cuts_a_spec_to_the_discharge_window_as_it_always_did(tiny):
    press_compute("toy_mean")
    attrs = blob_attrs(tiny.conn, "toy_mean")
    total, inside = in_window(tiny.path, WINDOW)
    assert 0 < inside < total
    assert attrs["samples"] == inside
    assert "window_start" not in attrs


def test_the_page_hands_a_whole_record_spec_the_whole_record_and_its_window(tiny):
    press_compute("toy_whole")
    attrs = blob_attrs(tiny.conn, "toy_whole")
    total, inside = in_window(tiny.path, WINDOW)
    assert attrs["samples"] == total > inside
    assert (attrs["window_start"], attrs["window_end"]) == WINDOW
    assert (attrs["first"], attrs["last"]) == (1.0, 1.02)  # the record from its start


def test_the_page_asks_for_the_whole_record_only_for_the_spec_that_declares_it(tiny, monkeypatch):
    """The cut stays lazy and small; only the spec that declared it is handed the whole."""
    seen = []
    real = loader.input_for

    def spy(ds, t_start, t_end, whole=False):
        seen.append((t_start, t_end, whole))
        return real(ds, t_start, t_end, whole)

    monkeypatch.setattr(loader, "input_for", spy)
    press_compute("toy_mean")  # the page runs twice: once to draw the form, once with Compute pressed
    assert seen and set(seen) == {(*WINDOW, False)}
    seen.clear()
    press_compute("toy_whole")
    assert seen and set(seen) == {(*WINDOW, True)}


def test_the_window_caption_is_still_the_discharge_dbs_for_a_raw_file(tiny):
    app = press_compute("toy_whole")
    captions = [c.value for c in app.caption]
    assert "Window 1.0050–1.0100 s — from the discharge DB." in captions


# ---------------------------------------------------------------------------
# Precompute: in this process, and the record a pool worker loads
# ---------------------------------------------------------------------------


def fill_in_process(conn, key):
    spec = registry.get(key)
    targets = precompute.targets_for(conn, spec, "cmod")
    return precompute.run(conn, spec, targets, spec.params())


def test_precompute_cuts_a_spec_to_the_discharge_window_as_it_always_did(tiny):
    assert fill_in_process(tiny.conn, "toy_mean").computed == 1
    attrs = blob_attrs(tiny.conn, "toy_mean")
    assert attrs["samples"] == in_window(tiny.path, WINDOW)[1]
    assert "window_start" not in attrs


def test_precompute_hands_a_whole_record_spec_the_whole_record_and_its_window(tiny):
    assert fill_in_process(tiny.conn, "toy_whole").computed == 1
    attrs = blob_attrs(tiny.conn, "toy_whole")
    total, inside = in_window(tiny.path, WINDOW)
    assert attrs["samples"] == total > inside
    assert (attrs["window_start"], attrs["window_end"]) == WINDOW
    assert (attrs["first"], attrs["last"]) == (1.0, 1.02)


def job_for(tiny):
    target = registry.Target("cmod", 1234, "apd", False, str(tiny.path), *WINDOW)
    window = SimpleNamespace(t_start=WINDOW[0], t_end=WINDOW[1])
    return precompute.Job(target=target, index=1, steps=(), window=window)


@pytest.mark.parametrize("order", [(False, True), (True, False)])
def test_a_job_whose_steps_want_both_reads_each_once_whichever_comes_first(tiny, order):
    """What a pool worker does: it loads the record into memory, once per target and once per kind of input."""
    record = precompute._Record(job_for(tiny), load=True, interrupts=precompute._Interrupts())
    got = {whole: record.get(whole=whole) for whole in order}
    total, inside = in_window(tiny.path, WINDOW)
    assert got[False].sizes["time"] == inside and got[True].sizes["time"] == total
    assert all(ds["frames"].variable._in_memory for ds in got.values())  # loaded: a worker has no file to go back to
    assert loader.WINDOW_ATTR not in got[False].attrs
    assert loader.discharge_window(got[True]) == WINDOW
    assert record.get() is got[False] and record.get(whole=True) is got[True]  # not read again
    record.close()


def test_a_job_whose_steps_all_want_the_cut_never_loads_the_whole_record(tiny, monkeypatch):
    seen = []
    real = loader.input_for

    def spy(ds, t_start, t_end, whole=False):
        seen.append(whole)
        return real(ds, t_start, t_end, whole)

    monkeypatch.setattr(loader, "input_for", spy)
    record = precompute._Record(job_for(tiny), load=True, interrupts=precompute._Interrupts())
    record.get()
    record.get()
    assert seen == [False]
    record.close()


def test_a_record_that_cannot_be_opened_fails_every_step_of_the_job_the_same_way(tiny, tmp_path):
    """The error is kept, as it always was: the steps record it, and the file is not tried again for each."""
    job = job_for(tiny)
    missing = registry.Target("cmod", 1234, "apd", False, str(tmp_path / "gone.nc"), *WINDOW)
    record = precompute._Record(
        precompute.Job(target=missing, index=1, steps=(), window=job.window), False, precompute._Interrupts()
    )
    with pytest.raises(Exception) as first:
        record.get(whole=True)
    with pytest.raises(Exception) as second:
        record.get()
    assert second.value is first.value


# ---------------------------------------------------------------------------
# The dead-pixel view, which is why there is a flag: the gas puff is found against the dark level at the record's
# start. The page and precompute must both end at the whole record's analysis window.
# ---------------------------------------------------------------------------

NAMES = list(pf.SCENARIOS)


@pytest.fixture
def raw_record(monkeypatch, tmp_path, request):
    """One raw record of ``tests/puff_fixtures`` as shot 1234, indexed, with the scenario's discharge window."""
    import streamlit as st

    case = pf.SCENARIOS[request.param]
    folder = tmp_path / "alcator" / "apd"
    folder.mkdir(parents=True)
    ds = pf.record(case)
    ds.attrs["shot_number"] = 1234
    ds.to_netcdf(folder / "apd_1234.nc")
    conn = deployment(monkeypatch, tmp_path, tmp_path / "alcator", 1234, case.window)
    yield SimpleNamespace(conn=conn, case=case, record=ds, name=request.param)
    conn.close()
    st.cache_data.clear()
    st.cache_resource.clear()


def whole_and_cut(raw):
    fusion_scripts.import_config()
    from density_scan import puff

    window = raw.case.window
    cut = raw.record.sel(time=slice(*window))
    return puff.puff_window(raw.record, window), puff.puff_window(cut, window)


def stored(conn):
    spec = registry.get("dead_pixels")
    target = registry.Target("cmod", 1234, "apd", False, "unused", 0.0, 0.0)
    result, run = store.lookup(conn, spec, target, spec.params())
    assert run is not None and run["status"] == "ok", run and run["error"]
    return result


@pytest.mark.parametrize("raw_record", NAMES, indirect=True)
def test_the_page_and_precompute_end_at_the_analysis_window_of_the_whole_record(raw_record):
    whole, cut = whole_and_cut(raw_record)
    ours = tuple(map(float, whole.analysis_window))

    app = press_compute("dead_pixels")
    from_page = stored(raw_record.conn)
    assert dead_pixels.puff_of(from_page).analysis_window == ours
    assert dead_pixels.puff_of(from_page).discharge_window == raw_record.case.window
    # The page draws the figure of that cut, and says it under it.
    assert len(app.get("plotly_chart")) == 2
    (line,) = [c.value for c in app.caption if c.value.startswith("Analysis window")]
    assert line.startswith(f"Analysis window {dead_pixels.seconds(ours)} · discharge window ")

    # The record cut to the discharge window first -- as the page cut every dataset before this flag -- would have
    # ended elsewhere, except on the control: a window that starts with the record.
    assert (tuple(cut.analysis_window) != ours) is (raw_record.name != "from the start")

    spec = registry.get("dead_pixels")
    targets = precompute.targets_for(raw_record.conn, spec, "cmod")
    report = precompute.run(raw_record.conn, spec, targets, spec.params(), force=True)
    assert report.computed == 1
    from_precompute = stored(raw_record.conn)
    assert dead_pixels.puff_of(from_precompute).analysis_window == ours
    for variable in ("dead", "evidence", "red_ratio", "gain", "pdf", "pdf_volts", "puff_signal"):
        np.testing.assert_array_equal(from_page[variable].values, from_precompute[variable].values, err_msg=variable)
    assert from_page.attrs["puff_rule"] == from_precompute.attrs["puff_rule"] == whole.rule


@pytest.mark.parametrize("raw_record", NAMES[:1], indirect=True)
def test_a_cached_result_without_the_puff_window_renders_with_its_note_and_recompute_replaces_it(raw_record):
    """The 111 results cached on the server, until ``precompute dead_pixels --force``: opened from the cache, with
    the grid drawn and a note in place of the figure of the cut; Recompute, under the grid, makes it."""
    app = press_compute("dead_pixels")
    (run,) = raw_record.conn.execute("SELECT * FROM runs WHERE plot = 'dead_pixels'").fetchall()
    with xr.open_dataset(run["blob_path"]) as blob:
        aged = pf.before_the_puff_window(blob.load())
    aged.to_netcdf(run["blob_path"])

    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    assert [i.value for i in app.info] == [dead_pixels.PREDATES]
    assert len(app.get("plotly_chart")) == 1  # the grid: nothing was stored of the cut to draw
    assert not [c for c in app.caption if c.value.startswith("Analysis window")]
    assert "Recompute" in [b.label for b in app.button]

    [b for b in app.button if b.label == "Recompute"][0].click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.info and len(app.get("plotly_chart")) == 2
    assert [c.value for c in app.caption if c.value.startswith("Analysis window")]
    (replaced,) = raw_record.conn.execute("SELECT * FROM runs WHERE plot = 'dead_pixels'").fetchall()
    assert replaced["status"] == "ok" and replaced["created_at"] != run["created_at"]
    assert dead_pixels.puff_of(stored(raw_record.conn)) is not None


@pytest.mark.parametrize("raw_record", NAMES[:1], indirect=True)
def test_a_raw_files_window_caption_is_the_discharge_dbs_and_the_cut_is_under_the_figure(raw_record):
    app = press_compute("dead_pixels")
    captions = [c.value for c in app.caption]
    assert "Window 0.0350–0.1150 s — from the discharge DB." in captions
    assert any(c.startswith("Analysis window 0.0350–0.1150 s · discharge window 0.0350–0.1150 s") for c in captions)
