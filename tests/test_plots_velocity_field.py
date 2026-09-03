"""The whole-field 2DCA velocity, against blobs whose velocity is known.

Measured wall time on this machine: ~7.4 s for the full 9x9 = 81-pixel field
(``test_the_full_field_recovers_the_planted_velocity``, marked ``slow``) and
~0.05 s for the forced-failure subset below. The full-field test is the only
expensive one in this file -- everything else either uses a small subset or
reuses that one result -- so ``pytest -m "not slow"`` still exercises render,
scalars and the netCDF round trip.
"""

import dataclasses

import numpy as np
import pytest
import xarray as xr

from fusion_ui.core import registry
from fusion_ui.plots import velocity_field

#: What the fixture plants: 20 blobs at 400 m/s radially outward, crossing
#: only the centre row (y=4) of the 9x9 grid -- see blob_dataset_path's own
#: docstring for why (the planted Gaussian's poloidal width falls below
#: threshold one pixel off that row).
VX = 400.0
CENTRE = 4


@pytest.fixture
def blobs(blob_dataset_path):
    with xr.open_dataset(blob_dataset_path) as ds:
        yield ds.load()


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


# ---------------------------------------------------------------------------
# The one expensive test: the real double loop, on the real fixture.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_full_field_recovers_the_planted_velocity(blobs, target, tmp_path):
    """Runs the actual double loop once and checks everything off that one
    result, so the O(pixels^2) cost is paid a single time in this file."""
    params = velocity_field.VelocityFieldParams()
    result = velocity_field.compute(blobs, params)

    # Shape and coords.
    assert result["vr"].dims == ("y", "x")
    assert result["vr"].shape == (blobs.sizes["y"], blobs.sizes["x"])
    assert result["R"].dims == ("y", "x")
    assert result["Z"].dims == ("y", "x")

    # The middle of the row the blobs actually cross recovers the planted
    # velocity; the poloidal component must come back at zero, not merely
    # small next to it. Columns near the array edge are excluded here on
    # purpose -- their tracks visibly perturb near the boundary, which is a
    # real property of the field this test is not about.
    for x in (3, 4, 5):
        assert float(result["vr"].isel(y=CENTRE, x=x)) == pytest.approx(VX, rel=0.05)
        assert abs(float(result["vz"].isel(y=CENTRE, x=x))) < 0.05 * VX

    # A row the blobs never reach: no events survive, so NaN, and the pixel
    # loop kept going rather than stopping.
    assert np.isnan(float(result["vr"].isel(y=0, x=4)))
    assert int(result["nevents"].isel(y=0, x=4)) == 0
    assert int(result["nlags"].isel(y=0, x=4)) == 0  # present, not omitted

    vr = result["vr"].values
    finite = np.isfinite(vr)
    assert int(finite.sum()) == 9  # all nine pixels of the centre row

    # min_lags is applied by render, not by compute: compute() stored a
    # velocity for every pixel that had an event regardless of how many lags
    # its mask rests on, so raising min_lags past some of those pixels' own
    # lag counts changes what render draws without touching the stored array.
    nlags = result["nlags"].values
    low_cut = int(nlags[finite].min()) + 1

    def n_arrows(fig):
        tip = next((t for t in fig.data if t.name == "arrow tip"), None)
        return len(tip.x) if tip is not None else 0

    # min_lags is not a parameter at all -- it is a slider read in render and
    # passed to figure(), so it cannot mint a param_sets row. Driving figure()
    # directly is also what keeps this assertion runnable without a Streamlit
    # runtime.
    assert not any(f.name == "min_lags" for f in dataclasses.fields(params))

    lenient = velocity_field.figure(result, velocity_field.DEFAULT_MIN_LAGS)
    strict = velocity_field.figure(result, low_cut)
    assert n_arrows(strict) < n_arrows(lenient)
    # compute()'s own array is untouched by whichever cut the figure drew at.
    assert int(np.isfinite(result["vr"].values).sum()) == 9

    # scalars(): every pixel, not just the reference -- more than one pixel
    # carries all four names, and NaN pixels are written through rather than
    # skipped (see velocity_field.scalars's docstring).
    written = velocity_field.scalars(result)
    names = {"vx_field", "vy_field", "number_events_field", "nlags_field"}
    assert {name for _, _, name in written} == names
    assert len(written) == blobs.sizes["y"] * blobs.sizes["x"] * len(names)
    assert written[(CENTRE, CENTRE, "vx_field")] == pytest.approx(VX, rel=0.05)
    assert np.isnan(written[(0, 0, "vx_field")])  # an unfitted pixel, kept as NaN

    # netCDF round trip -- the actual check for any array that silently
    # widened or dropped a NaN through encoding.
    path = tmp_path / "field.nc"
    result.to_netcdf(path)
    with xr.open_dataset(path) as reloaded:
        reloaded = reloaded.load()
    assert float(reloaded["vr"].isel(y=CENTRE, x=4)) == pytest.approx(VX, rel=0.05)
    assert np.isnan(float(reloaded["vr"].isel(y=0, x=4)))
    assert velocity_field.figure(reloaded) is not None


# ---------------------------------------------------------------------------
# Cheap: a forced failure, on a 3x3 corner of the same fixture.
# ---------------------------------------------------------------------------


def test_a_threshold_nothing_reaches_comes_back_nan_and_does_not_crash_the_loop(
    blobs,
):
    """Forced, not hoped for: threshold=100 standard deviations is far above
    anything the planted blobs (amplitude ~3) or the noise floor reach, at
    every one of the nine pixels in this corner -- so every pixel takes the
    "no events survived" path in compute(), and the loop must still return a
    plain, entirely-NaN field rather than raising."""
    corner = blobs.isel(y=slice(0, 3), x=slice(0, 3))
    params = velocity_field.VelocityFieldParams(threshold=100.0)
    result = velocity_field.compute(corner, params)

    assert np.isnan(result["vr"].values).all()
    assert np.isnan(result["vz"].values).all()
    assert (result["nevents"].values == 0).all()
    assert (result["nlags"].values == 0).all()  # present, not omitted
    # The summary is no longer stored (it depended on the min_lags cut, which
    # is now view state), so an all-NaN field draws as one rather than
    # carrying a precomputed count of zero.
    assert "nfitted" not in result
    assert int(result["npixels"]) == 9


SPEC = registry.get("velocity_field")


def test_the_spec_is_registered_as_an_unchained_cached_spec():
    """Not a chained spec: two_dca's reference pixel cannot help here, since
    this analysis needs the 2DCA run once per pixel -- see the module
    docstring."""
    assert SPEC.cached
    assert SPEC.requires is None
    assert SPEC.upstream_params is None
    assert "apd" in SPEC.diagnostics
