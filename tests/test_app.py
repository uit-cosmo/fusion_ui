"""Smoke tests: the pages render without raising, against a temporary tree."""

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from streamlit.testing.v1 import AppTest

import fusion_ui.plots  # noqa: F401 - registers the specs the page offers
from fusion_ui import config
from fusion_ui.core import catalog, db, registry, store
from fusion_ui.plots import dead_pixels

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


def test_landing_page_lists_every_run_day_with_its_purpose(deployment):
    app = run(APP)
    table = app.dataframe[0].value
    # Three curated days plus 1150618, which has files but no discharge-DB
    # entry -- listed rather than hidden, like the browser's uncurated shots.
    assert list(table["day"]) == ["1090813", "1110201", "1150618", "1160616"]
    assert list(table["on_disk"]) == [0, 1, 1, 1]
    assert (
        table.set_index("day")
        .loc["1160616", "mp_url"]
        .endswith("/miniproposals/800.pdf")
    )
    assert [m.value for m in app.metric if m.label == "Run days"] == ["4"]

    # The summaries are hard-wrapped in the file, so a phrase can straddle a
    # newline; compare against the text with its whitespace collapsed.
    prose = " ".join(" ".join(m.value.split()) for m in app.markdown)
    assert "MP800" in prose and "density scan to high Greenwald fraction" in prose
    # 1150618 has no curated shots at all, only files -- its purpose still shows.
    assert "MP761" in prose


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
        "Dead pixels (PDF and spectrum of every pixel)",
        "Duration time (PSD fit)",
        "Blob velocity (TDE, whole record)",
        "Conditional average (2DCA)",
        "Blob velocity (contour tracking)",
        "Blob size (FWHM)",
        "Blob size (Gaussian fit)",
        "Blob velocity (2DCA time delay)",
        "Tracked trajectories (2DCA)",
        "Two-sided exponential fits (2DCA cuts)",
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


# ---------------------------------------------------------------------------
# Batch-only specs on the page: shown from cache, or the command that fills
# them -- never computed in the process that serves everyone.
# ---------------------------------------------------------------------------

#: What the toy specs computed, in order. AppTest runs the page in this same
#: process, so this is how a test sees whether the page computed anything.
TOY_CALLS = []


@dataclasses.dataclass
class ToyBankParams:
    window: int = 60


@dataclasses.dataclass
class ToyFieldsParams:
    bank: ToyBankParams = dataclasses.field(default_factory=ToyBankParams)
    scale: float = 2.0


def _toy_bank(ds, params):
    TOY_CALLS.append("bank")
    return xr.Dataset({"y": ("t", np.arange(4.0))})


def _toy_fields(ds, params, upstream):
    TOY_CALLS.append("fields")
    return xr.Dataset({"y": upstream["y"] * params.scale})


def _say_total(name):
    def render(result, params, target):
        import streamlit as st

        st.markdown(f"{name} total {float(result['y'].sum()):g}")

    return render


@pytest.fixture
def toy_bank(single_shot_deployment):
    """A batch-only spec and a spec built on it, registered for one test."""
    TOY_CALLS.clear()
    bank = registry.register(
        registry.PlotSpec(
            key="toy_bank",
            label="Toy bank",
            diagnostics=("apd",),
            params=ToyBankParams,
            render=_say_total("toy bank"),
            compute=_toy_bank,
            batch_only=True,
        )
    )
    fields = registry.register(
        registry.PlotSpec(
            key="toy_fields",
            label="Toy fields",
            diagnostics=("apd",),
            params=ToyFieldsParams,
            render=_say_total("toy fields"),
            compute=_toy_fields,
            requires="toy_bank",
            upstream_params=lambda params: params.bank,
        )
    )
    yield bank, fields
    registry.REGISTRY.pop("toy_fields", None)
    registry.REGISTRY.pop("toy_bank", None)
    TOY_CALLS.clear()


def fill_toy_bank(database, bank):
    """What `fusion-ui precompute toy_bank --shot 1234` leaves behind."""
    conn = db.open_db(database)
    target = registry.Target(
        machine="cmod",
        shot=1234,
        diagnostic="apd",
        preprocessed=False,
        path="unused",
        t_start=1.0,
        t_end=1.02,
    )
    _, run = store.result(conn, bank, target, ToyBankParams(), ds=None, batch=True)
    conn.close()
    TOY_CALLS.clear()
    return run


def ledger(database):
    conn = db.connect(database)
    rows = {r["plot"]: dict(r) for r in conn.execute("SELECT * FROM runs")}
    conn.close()
    return rows


def markdown(app):
    return [m.value for m in app.markdown]


def test_a_batch_only_spec_is_shown_from_cache_without_computing(
    single_shot_deployment, toy_bank
):
    bank, _ = toy_bank
    seeded = fill_toy_bank(single_shot_deployment, bank)

    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60)
    app.session_state["spec.apd"] = bank
    app.run()  # no button pressed: a lookup never computes, so nothing to wait for
    assert not app.exception, app.exception
    assert not app.error, [e.value for e in app.error]

    assert "toy bank total 6" in markdown(app)
    assert TOY_CALLS == []
    assert ledger(single_shot_deployment) == {"toy_bank": dict(seeded)}
    # Its form commits with Show, and there is no Recompute to delete a
    # result this page could not compute again -- the command instead.
    buttons = [b.label for b in app.button]
    assert "Show" in buttons and "Compute" not in buttons
    assert "Recompute" not in buttons
    assert any(
        "fusion-ui precompute toy_bank --shot 1234 --force" in c for c in captions(app)
    )
    # Nor is it offered on many pixels.
    assert not [r for r in app.sidebar.radio if r.label == "Pixels"]


def test_a_missing_batch_only_spec_names_its_command_and_computes_nothing(
    single_shot_deployment, toy_bank
):
    bank, _ = toy_bank
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60)
    app.session_state["spec.apd"] = bank
    app.run()
    assert not app.exception, app.exception
    assert [c.value for c in app.code] == ["fusion-ui precompute toy_bank --shot 1234"]
    assert app.warning

    widget(app, "button", "Show").click().run()
    assert not app.exception, app.exception
    assert TOY_CALLS == []
    assert ledger(single_shot_deployment) == {}


def test_a_batch_only_spec_with_other_parameters_shows_them_to_save(
    single_shot_deployment, toy_bank
):
    bank, _ = toy_bank
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60)
    app.session_state["spec.apd"] = bank
    app.session_state["params.toy_bank.window"] = 30
    app.run()
    assert not app.exception, app.exception
    command, saved = [c.value for c in app.code]
    assert command == (
        "fusion-ui precompute toy_bank --shot 1234 --params-json params.json"
    )
    assert '"window": 30' in saved and '"plot": "toy_bank"' in saved


def test_a_spec_built_on_a_missing_batch_only_result_computes_nothing(
    single_shot_deployment, toy_bank
):
    _, fields = toy_bank
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60)
    app.session_state["spec.apd"] = fields
    app.run()
    assert not app.exception, app.exception
    assert [c.value for c in app.code] == ["fusion-ui precompute toy_bank --shot 1234"]

    widget(app, "button", "Compute").click().run()
    assert not app.exception, app.exception
    assert [c.value for c in app.code] == ["fusion-ui precompute toy_bank --shot 1234"]
    assert TOY_CALLS == []
    assert ledger(single_shot_deployment) == {}


def test_an_unreadable_batch_only_blob_is_not_recomputed_by_the_page(
    single_shot_deployment, toy_bank
):
    """The blob exists, so the page's check passes; the store must refuse
    anyway, and the page must name the command that replaces it."""
    bank, fields = toy_bank
    seeded = fill_toy_bank(single_shot_deployment, bank)
    with open(seeded["blob_path"], "wb") as broken:
        broken.write(b"not netCDF")

    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60)
    app.session_state["spec.apd"] = fields
    app.run()
    widget(app, "button", "Compute").click().run()
    assert not app.exception, app.exception
    assert [c.value for c in app.code] == [
        "fusion-ui precompute toy_bank --shot 1234 --force"
    ]
    assert TOY_CALLS == []
    assert ledger(single_shot_deployment) == {"toy_bank": dict(seeded)}


def test_a_spec_built_on_a_cached_batch_only_result_computes_inline(
    single_shot_deployment, toy_bank
):
    bank, fields = toy_bank
    seeded = fill_toy_bank(single_shot_deployment, bank)

    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60)
    app.session_state["spec.apd"] = fields
    app.run()
    assert not app.exception, app.exception
    assert app.info, "a derived spec still waits for Compute"
    assert TOY_CALLS == []

    widget(app, "button", "Compute").click().run()
    assert not app.exception, app.exception
    assert not app.error, [e.value for e in app.error]
    assert "toy fields total 12" in markdown(app)
    assert TOY_CALLS == ["fields"], "only the derived part is computed"

    runs = ledger(single_shot_deployment)
    assert runs["toy_bank"] == dict(seeded)
    assert runs["toy_fields"]["upstream_run_id"] == seeded["id"]
    conn = db.connect(single_shot_deployment)
    (mtime,) = conn.execute(
        "SELECT mtime FROM shots WHERE shot = 1234 AND diagnostic = 'apd'"
    ).fetchone()
    conn.close()
    assert runs["toy_fields"]["input_mtime"] == mtime


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


# ---------------------------------------------------------------------------
# Two machines in one index
# ---------------------------------------------------------------------------


@pytest.fixture
def two_machine_deployment(single_shot_deployment):
    """The same shot numbers indexed under two machines.

    ``rescan`` runs for one machine and only deletes that machine's rows, so
    repointing ``FUSION_MACHINE`` leaves the previous machine indexed for good
    -- there is no "switch machines" that also forgets the old one. The
    fixture reuses the real tree so both machines carry shots 1234 and 5678,
    which is the collision the picker has to survive.
    """
    import streamlit as st

    conn = db.open_db(single_shot_deployment)
    catalog.rescan(conn, str(config.DATA_FOLDER), "other", None)
    conn.close()
    st.cache_data.clear()
    st.cache_resource.clear()
    return single_shot_deployment


def test_a_shot_number_on_two_machines_still_picks_exactly_one_row(
    two_machine_deployment,
):
    """Keyed on the shot alone, ``table.set_index("shot").loc[1234]`` returns
    both machines' rows and ``available_targets`` then asks for the truth
    value of a Series -- so the page dies on a ValueError rather than showing
    the wrong shot."""
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60).run()
    assert not app.exception, app.exception

    machine = widget(app, "selectbox", "Machine")
    assert machine.options == ["cmod", "other"]
    assert machine.value == "cmod"
    # Each machine offers its own shots, not the union counted twice.
    # AppTest serialises selectbox options as strings; what matters is that
    # each machine offers its own two shots, not the union counted twice.
    assert widget(app, "selectbox", "Shot").options == ["1234", "5678"]


def test_switching_machine_keeps_the_target_on_that_machine(two_machine_deployment):
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60).run()
    # Target.key names the cache blob and every session-state entry, so a
    # picker that dropped the machine would give both machines one key and let
    # them overwrite each other's view state. Proven here through the pixel
    # the frame view reports: it is read out of session state under that key,
    # so it only comes back if the target really moved to "other".
    app.session_state["pixel.other_1234_apd_r"] = (2, 3)
    widget(app, "selectbox", "Machine").set_value("other").run()
    assert not app.exception, app.exception
    assert widget(app, "selectbox", "Machine").value == "other"
    assert any("y=2, x=3" in c for c in captions(app))


def test_a_browser_selection_on_the_other_machine_is_honoured(two_machine_deployment):
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60)
    app.session_state["selection"] = {
        "machine": "other",
        "shot": 5678,
        "diagnostic": "asp",
        "preprocessed": False,
    }
    app.run()
    assert not app.exception, app.exception
    assert widget(app, "selectbox", "Machine").value == "other"
    assert widget(app, "selectbox", "Shot").value == 5678
    assert widget(app, "selectbox", "Diagnostic").value == "asp"


def test_the_dead_pixel_view_draws_its_method_and_grid(single_shot_deployment):
    """Its render draws into Streamlit itself: the method, a view toggle and one panel per pixel."""
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=120)
    app.session_state["spec.apd"] = registry.get("dead_pixels")
    app.run()
    widget(app, "button", "Compute").click().run()
    assert not app.exception
    assert not app.error
    assert [e.label for e in app.expander] == ["How dead pixels are found"]
    assert widget(app, "radio", "Show").options == list(dead_pixels.VIEWS)

    # The plain summary is always shown, directly above the technical text in the expander.
    column = list(app.main.children.values())
    above = column[[element.type for element in column].index("expander") - 1]
    assert above.type == "markdown" and above.value == dead_pixels.plain_summary()

    conn = db.connect(single_shot_deployment)
    names = {r["name"] for r in conn.execute("SELECT name FROM scalars")}
    conn.close()
    assert names == {"dead", "psd_ratio", "number_dead"}
