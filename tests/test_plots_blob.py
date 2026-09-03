"""The 2DCA chain, against blobs whose velocity is known in advance.

These are the only tests in the suite that run a real ``imaging_methods``
analysis. They are worth the few seconds: a velocity estimator that silently
returns something plausible is exactly the failure this tool exists to avoid,
and the synthetic dataset makes "plausible" checkable.
"""

import numpy as np
import pytest
import xarray as xr

from fusion_ui.core import registry, store
from fusion_ui.plots import fwhm_sizes, two_dca, velocity_contour

#: What the fixture plants: 20 blobs at 400 m/s radially outward.
EVENTS, VX = 20, 400.0
CENTRE = 4  # the reference pixel on the 9x9 grid
#: The planted Gaussian's FWHM, in metres: sigma is 0.4 cm and both the
#: radial and poloidal cuts through the peak-lag average are that same
#: Gaussian, so lr and lz should agree with this and with each other.
BLOB_FWHM = 2 * np.sqrt(2 * np.log(2)) * 0.4 / 100


@pytest.fixture
def blobs(blob_dataset_path):
    with xr.open_dataset(blob_dataset_path) as ds:
        yield ds.load()


@pytest.fixture
def two_dca_params():
    params = two_dca.TwoDcaSpecParams()
    params.two_dca.refx = params.two_dca.refy = CENTRE
    return params


@pytest.fixture
def contour_params():
    params = velocity_contour.ContourVelocityParams()
    params.two_dca.refx = params.two_dca.refy = CENTRE
    return params


@pytest.fixture
def fwhm_params():
    params = fwhm_sizes.FwhmSizeParams()
    params.two_dca.refx = params.two_dca.refy = CENTRE
    return params


@pytest.fixture
def target(tmp_path):
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


# ---------------------------------------------------------------------------
# two_dca
# ---------------------------------------------------------------------------


def test_the_conditional_average_finds_every_planted_blob(blobs, two_dca_params):
    average = two_dca.compute(blobs, two_dca_params)
    assert int(average["number_events"]) == EVENTS
    assert set(average.data_vars) >= {"cond_av", "cond_repr", "cross_corr"}


def test_the_average_peaks_at_the_reference_pixel_and_at_zero_lag(
    blobs, two_dca_params
):
    average = two_dca.compute(blobs, two_dca_params)
    cond_av = average["cond_av"]
    peak = np.unravel_index(int(cond_av.argmax()), cond_av.shape)
    assert (peak[0], peak[1]) == (CENTRE, CENTRE)  # (y, x, time)
    assert float(cond_av["time"][peak[2]]) == pytest.approx(0.0, abs=1e-9)


def test_a_threshold_nothing_reaches_is_a_readable_failure(blobs, two_dca_params):
    """It has to say why rather than store an empty dataset -- an empty blob
    would surface as a confusing error inside whatever consumed it."""
    two_dca_params.two_dca.threshold = 50.0
    with pytest.raises(ValueError, match="no events survived"):
        two_dca.compute(blobs, two_dca_params)


def test_the_scalar_is_written_at_the_reference_pixel(blobs, two_dca_params):
    average = two_dca.compute(blobs, two_dca_params)
    assert two_dca.scalars(average) == {(CENTRE, CENTRE, "number_events"): 20.0}


# ---------------------------------------------------------------------------
# velocity_contour
# ---------------------------------------------------------------------------


def test_the_contour_track_recovers_the_planted_velocity(
    blobs, two_dca_params, contour_params
):
    average = two_dca.compute(blobs, two_dca_params)
    result = velocity_contour.compute(blobs, contour_params, average)

    assert float(result["vx"]) == pytest.approx(VX, rel=0.05)
    # Purely radial motion: the poloidal component must come back at zero, not
    # merely small next to vx.
    assert abs(float(result["vy"])) < 0.05 * VX
    assert result["tracked"].values.sum() > 2


def test_the_least_squares_estimator_agrees_with_the_centred_difference(
    blobs, two_dca_params, contour_params
):
    """Two reductions of the same track. On a straight track they must agree;
    the estimator choice is a knob, not a different physical quantity."""
    average = two_dca.compute(blobs, two_dca_params)
    central = velocity_contour.compute(blobs, contour_params, average)
    contour_params.velocity.estimator = "lsq"
    lsq = velocity_contour.compute(blobs, contour_params, average)
    assert float(lsq["vx"]) == pytest.approx(float(central["vx"]), rel=0.1)


def test_the_scalars_are_the_three_density_scan_reports(
    blobs, two_dca_params, contour_params
):
    average = two_dca.compute(blobs, two_dca_params)
    result = velocity_contour.compute(blobs, contour_params, average)
    scalars = velocity_contour.scalars(result)
    assert set(scalars) == {
        (CENTRE, CENTRE, "vx_c"),
        (CENTRE, CENTRE, "vy_c"),
        (CENTRE, CENTRE, "area_c"),
    }
    assert scalars[(CENTRE, CENTRE, "area_c")] > 0


def test_compute_never_looks_at_the_raw_dataset(blobs, two_dca_params, contour_params):
    """Its input is the upstream result. If that stops being true, the store's
    chaining is no longer the only path to this analysis."""
    average = two_dca.compute(blobs, two_dca_params)
    assert velocity_contour.compute(None, contour_params, average) is not None


# ---------------------------------------------------------------------------
# Through the store, which is how the page reaches it
# ---------------------------------------------------------------------------


def test_the_chain_round_trips_through_the_store(
    conn, cache, blobs, contour_params, target
):
    spec = registry.get("velocity_contour")
    result, run = store.result(conn, spec, target, contour_params, blobs)

    assert run["status"] == "ok", run["error"]
    # A netCDF round trip is where a bool array or a ragged dimension would
    # have been quietly rewritten, so assert on what came back off disk.
    assert float(result["vx"]) == pytest.approx(VX, rel=0.05)
    assert result["tracked"].dtype == np.int8
    assert velocity_contour.render(result, contour_params, target) is not None

    names = {
        (r["plot"], r["name"])
        for r in conn.execute(
            "SELECT r.plot, s.name FROM scalars s JOIN runs r ON r.id = s.run_id"
        )
    }
    assert names == {
        ("two_dca", "number_events"),
        ("velocity_contour", "vx_c"),
        ("velocity_contour", "vy_c"),
        ("velocity_contour", "area_c"),
    }


# ---------------------------------------------------------------------------
# fwhm_sizes
# ---------------------------------------------------------------------------


def test_the_fwhm_recovers_the_planted_blob_width(blobs, two_dca_params, fwhm_params):
    average = two_dca.compute(blobs, two_dca_params)
    result = fwhm_sizes.compute(blobs, fwhm_params, average)

    assert float(result["lr"]) == pytest.approx(BLOB_FWHM, rel=0.05)
    # The planted blob is a symmetric Gaussian, so the radial and poloidal
    # cuts through it must agree with each other, not just with the analytic
    # width -- a bug that mixed up the two axes would still pass the first
    # assertion by coincidence but fail this one.
    assert float(result["lz"]) == pytest.approx(float(result["lr"]), rel=0.05)


def test_the_scalars_are_the_two_density_scan_reports(blobs, two_dca_params, fwhm_params):
    average = two_dca.compute(blobs, two_dca_params)
    result = fwhm_sizes.compute(blobs, fwhm_params, average)
    assert fwhm_sizes.scalars(result) == {
        (CENTRE, CENTRE, "lr"): pytest.approx(float(result["lr"])),
        (CENTRE, CENTRE, "lz"): pytest.approx(float(result["lz"])),
    }


def test_compute_never_looks_at_the_raw_dataset_fwhm(blobs, two_dca_params, fwhm_params):
    """Its input is the upstream result. If that stops being true, the store's
    chaining is no longer the only path to this analysis."""
    average = two_dca.compute(blobs, two_dca_params)
    assert fwhm_sizes.compute(None, fwhm_params, average) is not None


def test_the_fwhm_chain_round_trips_through_the_store(conn, cache, blobs, fwhm_params, target):
    spec = registry.get("fwhm_sizes")
    result, run = store.result(conn, spec, target, fwhm_params, blobs)

    assert run["status"] == "ok", run["error"]
    assert float(result["lr"]) == pytest.approx(BLOB_FWHM, rel=0.05)
    assert fwhm_sizes.render(result, fwhm_params, target) is not None

    names = {
        (r["plot"], r["name"])
        for r in conn.execute(
            "SELECT r.plot, s.name FROM scalars s JOIN runs r ON r.id = s.run_id"
        )
    }
    assert names == {
        ("two_dca", "number_events"),
        ("fwhm_sizes", "lr"),
        ("fwhm_sizes", "lz"),
    }
