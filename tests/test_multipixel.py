"""Many-pixel mode: eligibility, pixel stamping, estimates, overlays."""

import numpy as np
import pytest
import xarray as xr

import fusion_ui.plots  # noqa: F401 - registers every spec
from fusion_ui.core import multipixel, precompute, registry, store
from fusion_ui.plots import spectra

CACHED_ELIGIBLE = {
    "taud_psd",
    "velocity_tde",
    "two_dca",
    "velocity_contour",
    "fwhm_sizes",
    "gaussian_sizes",
    "velocity_2dca_tde",
    "trajectories",
    "two_sided_exp",
}


def test_supported_matches_the_per_pixel_specs_and_the_live_trace():
    assert {
        key
        for key, spec in registry.REGISTRY.items()
        if multipixel.supported(spec) and spec.cached
    } == CACHED_ELIGIBLE
    # The frame viewer is live -- nothing to stamp, nothing to cache -- and is
    # eligible through its overlay, which reads each trace off the open file.
    assert multipixel.supported(registry.get("raw_frames"))
    assert not multipixel.supported(registry.get("probe_trace"))
    assert not multipixel.supported(registry.get("velocity_field"))


def test_with_pixel_stamps_a_top_level_pair_without_mutating():
    params = spectra.TaudPsdParams(refx=6, refy=6)
    stamped = multipixel.with_pixel(params, 2, 3)
    assert (stamped.refx, stamped.refy) == (2, 3)
    assert (params.refx, params.refy) == (6, 6)


def test_with_pixel_stamps_a_nested_pair_without_mutating():
    spec = registry.get("velocity_contour")
    params = spec.params()
    before = (params.two_dca.refx, params.two_dca.refy)
    stamped = multipixel.with_pixel(params, 1, 2)
    assert (stamped.two_dca.refx, stamped.two_dca.refy) == (1, 2)
    assert (params.two_dca.refx, params.two_dca.refy) == before


def test_precompute_still_stamps_the_pixel_after_the_refactor():
    spec = registry.get("taud_psd")
    params = precompute.default_params(spec, pixel=(2, 3))
    assert (params.refx, params.refy) == (2, 3)
    derived = precompute.default_params(registry.get("velocity_contour"), pixel=(1, 2))
    assert (derived.two_dca.refx, derived.two_dca.refy) == (1, 2)


def test_pixels_from_points_reads_customdata_deduplicates_and_sorts():
    event = [
        {"customdata": [2, 3]},
        {"customdata": [0, 1]},
        {"customdata": [2, 3]},  # dragged twice
        {"nonsense": True},
        {"customdata": ["bad", None]},
    ]
    assert multipixel.pixels_from_points(event) == [(0, 1), (2, 3)]
    assert multipixel.pixels_from_points([]) == []
    assert multipixel.pixels_from_points(None) == []


@pytest.fixture
def target():
    return registry.Target(
        machine="cmod",
        shot=4321,
        diagnostic="apd",
        preprocessed=True,
        path="unused",
        t_start=0.0,
        t_end=1.6e-3,
    )


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))


def _seed_ok(conn, target, spec, params, seconds):
    params_hash, _ = store.record_params(conn, spec.key, params)
    return store.record_run(
        conn,
        target,
        spec.key,
        params_hash,
        blob_path="/tmp/gone.nc",
        status="ok",
        error=None,
        seconds=seconds,
    )


def test_estimate_counts_cached_pixels_from_the_ledger(conn, cache, target):
    spec = registry.get("taud_psd")
    params = spec.params(refx=0, refy=0)
    _seed_ok(conn, target, spec, multipixel.with_pixel(params, 1, 1), 10.0)
    _seed_ok(conn, target, spec, multipixel.with_pixel(params, 2, 2), 20.0)

    est = multipixel.estimate(
        conn, spec, target, params, [(1, 1), (2, 2), (3, 3)]
    )
    assert (est.total, est.cached, est.to_compute) == (3, 2, 1)
    # Median of the seeded 10 s and 20 s runs prices the one miss.
    assert est.seconds_per_pixel == pytest.approx(15.0)
    assert est.eta_seconds == pytest.approx(15.0)


def test_estimate_of_a_chained_spec_includes_its_upstream(conn, cache, target):
    spec = registry.get("velocity_contour")
    upstream = registry.get("two_dca")
    params = multipixel.with_pixel(spec.params(), 1, 1)

    _seed_ok(conn, target, upstream, spec.upstream_params(params), 21.0)
    _seed_ok(conn, target, spec, params, 4.0)

    # The pixel whose rows exist costs nothing further.
    assert multipixel.estimate(conn, spec, target, spec.params(), [(1, 1)]).eta_seconds == 0.0

    # A fresh pixel pays both links' medians, not just the downstream's.
    est = multipixel.estimate(conn, spec, target, spec.params(), [(5, 5)])
    assert est.to_compute == 1
    assert est.eta_seconds == pytest.approx(25.0)


def test_estimate_with_no_history_has_no_eta(conn, cache, target):
    spec = registry.get("taud_psd")
    est = multipixel.estimate(conn, spec, target, spec.params(), [(0, 0)])
    assert est.to_compute == 1
    assert est.seconds_per_pixel is None
    assert est.eta_seconds is None


def _psd_result(x, y, taud=3.1e-5):
    omega = np.logspace(3, 6, 24)
    return xr.Dataset(
        {
            "psd": ("omega", np.abs(np.random.default_rng(x * 10 + y).normal(size=24)) + 1.0),
            "psd_fit": ("omega", np.full(24, 2.0)),
            "taud": float(taud),
            "lam": 0.5,
            "refx": int(x),
            "refy": int(y),
        },
        coords={"omega": omega},
    )


def test_spectra_overlay_puts_every_pixel_on_one_log_axis(target):
    params = spectra.TaudPsdParams()
    items = [((6, 4), _psd_result(6, 4)), ((1, 2), _psd_result(1, 2, 4.2e-5))]
    figure = spectra.overlay(items, params, target)

    assert len(figure.data) == 4  # PSD + fit per pixel
    groups = [t.legendgroup for t in figure.data]
    assert groups[0] == groups[1] != groups[2] == groups[3]
    assert figure.data[0].name == "(x=6, y=4)  τ_d=3.1e-05 s"
    assert figure.data[1].showlegend is False
    assert figure.layout.xaxis.type == "log"
    assert figure.layout.yaxis.type == "log"
    assert "2 pixels" in figure.layout.title.text


def _contour_result(x, y, vx=400.0):
    return xr.Dataset(
        {
            "vx": float(vx + x),
            "vy": 10.0,
            "area": 1e-4,
            "refx": int(x),
            "refy": int(y),
        }
    )


def test_scalar_figure_falls_back_to_a_scalar_across_pixels(target):
    spec = registry.get("velocity_contour")
    assert spec.overlay is None
    items = [((0, 0), _contour_result(0, 0)), ((1, 2), _contour_result(1, 2))]
    figure = multipixel.scalar_figure(items, spec, "vx_c")

    assert len(figure.data) == 1
    assert list(figure.data[0].y) == pytest.approx([400.0, 401.0])
    assert figure.data[0].customdata[1] == [1, 2]
    assert "vx_c" in figure.layout.title.text


def test_scalar_data_skips_shot_level_scalars(target):
    spec = registry.get("velocity_contour")
    items = [((0, 0), _contour_result(0, 0))]
    data = multipixel.scalar_data(items, spec)
    assert set(data) == {"vx_c", "vy_c", "area_c"}
    assert data["vx_c"] == [(0, 0, 400.0)]


# ---------------------------------------------------------------------------
# End to end through the store, in the test_plots_roundtrip.py style
# ---------------------------------------------------------------------------


@pytest.fixture
def blobs(blob_dataset_path):
    with xr.open_dataset(blob_dataset_path) as ds:
        yield ds.load()


def test_four_pixels_share_the_cache_both_ways(conn, cache, blobs, target):
    """Four runs rows, eight scalar rows, and a second pass computes nothing."""
    spec = registry.get("taud_psd")
    base = spec.params()
    pixels = [(3, 3), (4, 4), (5, 5), (4, 3)]

    first_ids = []
    for x, y in pixels:
        _, run = store.result(conn, spec, target, multipixel.with_pixel(base, x, y), blobs)
        assert run["status"] == "ok", f"pixel {(x, y)}: {run['error']}"
        first_ids.append(run["id"])
    assert len(set(first_ids)) == 4

    (runs,) = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
    assert runs == 4
    (scalars,) = conn.execute("SELECT COUNT(*) FROM scalars").fetchone()
    assert scalars == 8  # taud_psd + lambda_psd per pixel

    # A second pass over the same rectangle is all hits.
    for (x, y), old_id in zip(pixels, first_ids):
        _, run = store.result(conn, spec, target, multipixel.with_pixel(base, x, y), blobs)
        assert run["id"] == old_id

    # And one of those pixels opened singly hits the same row.
    _, single = store.result(
        conn, spec, target, multipixel.with_pixel(base, 4, 4), blobs
    )
    assert single["id"] == first_ids[1]


def _frames_dataset(n_time=6000, n_y=4, n_x=5):
    rng = np.random.default_rng(3)
    time = np.linspace(1.0, 1.02, n_time)
    frames = rng.normal(size=(n_y, n_x, n_time))
    return xr.Dataset(
        {"frames": (["y", "x", "time"], frames)},
        coords={"time": ("time", time)},
    )


def test_raw_overlay_draws_one_decimated_trace_per_pixel(target):
    from fusion_ui.plots import raw

    ds = _frames_dataset()
    items = [((0, 0), ds), ((2, 3), ds)]
    figure = raw.overlay(items, raw.RawFramesParams(), target)

    assert [t.name for t in figure.data] == ["(x=0, y=0)", "(x=2, y=3)"]
    # 6000 samples per trace would kill Plotly three times over uncut.
    assert all(len(t.x) <= 4000 for t in figure.data)
    assert figure.layout.xaxis.title.text == "time [s]"
    assert "2 pixels" in figure.layout.title.text


# ---------------------------------------------------------------------------
# Smoke test: Many mode on the single-shot page
# ---------------------------------------------------------------------------


def test_many_mode_renders_the_selector(monkeypatch, tmp_path, apd_dataset_path):
    import streamlit as st
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    from fusion_ui.core import catalog, db

    data_folder = apd_dataset_path.parent.parent  # .../alcator
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
    try:
        single_shot = str(
            Path(__file__).resolve().parent.parent / "fusion_ui" / "pages"
            / "2_single_shot.py"
        )
        app = AppTest.from_file(single_shot, default_timeout=60)
        app.session_state["spec.apd"] = registry.get("taud_psd")
        app.run()
        assert not app.exception, app.exception

        radios = [w for w in app.sidebar.radio if w.label == "Pixels"]
        assert radios, "expected the One/Many pixel-mode radio"
        radios[0].set_value("Many").run()
        assert not app.exception, app.exception
        assert app.get("plotly_chart"), "expected the pixel-map selector to render"
    finally:
        st.cache_data.clear()
        st.cache_resource.clear()


def test_many_mode_on_the_live_trace_draws_selector_and_overlay(
    monkeypatch, tmp_path, apd_dataset_path
):
    """The pixel trace needs no run and no button: selector plus overlay."""
    import streamlit as st
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    from fusion_ui.core import catalog, db

    data_folder = apd_dataset_path.parent.parent  # .../alcator
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
    try:
        single_shot = str(
            Path(__file__).resolve().parent.parent / "fusion_ui" / "pages"
            / "2_single_shot.py"
        )
        app = AppTest.from_file(single_shot, default_timeout=60)
        # raw_frames is the default spec for apd: no need to pick one.
        app.session_state["pixels.cmod_1234_apd_r"] = [(0, 0), (1, 1)]
        app.run()
        assert not app.exception, app.exception

        radios = [w for w in app.sidebar.radio if w.label == "Pixels"]
        assert radios, "expected the One/Many pixel-mode radio"
        radios[0].set_value("Many").run()
        assert not app.exception, app.exception
        # Selector plus the two-trace overlay, drawn with no Compute in sight.
        assert len(app.get("plotly_chart")) >= 2
        labels = [b.label for b in app.button]
        assert not any(
            label.startswith("Run on") or label == "Compute" for label in labels
        ), f"live mode must offer no run button, found {labels}"

        conn = db.connect(database)
        (runs,) = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        conn.close()
        assert runs == 0, "a live overlay must not write ledger rows"
    finally:
        st.cache_data.clear()
        st.cache_resource.clear()
