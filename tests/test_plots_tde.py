"""``velocity_tde``, against blobs whose velocity is known in advance.

Unlike the 2DCA chain in ``test_plots_blob.py``, this estimator reads the raw
frames directly and conditions on nothing, so there is no ``two_dca`` fixture
to run first -- ``compute`` takes the dataset straight from
``blob_dataset_path``.
"""

import numpy as np
import pytest

from fusion_ui.core import params_ui, registry, store
from fusion_ui.plots import velocity_tde as vt

#: What the fixture plants: 20 blobs at 400 m/s radially outward, crossing a
#: 9x9 grid. See tests/conftest.py:blob_dataset_path.
VX_PLANTED = 400.0
CENTRE = 4  # the reference pixel on the 9x9 grid


@pytest.fixture
def blobs(blob_dataset_path):
    import xarray as xr

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


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))


# ---------------------------------------------------------------------------
# The mask-shape trap
# ---------------------------------------------------------------------------


def test_the_dead_pixel_mask_is_9_wide_by_10_tall():
    """The literal copied from density_scan/dead_pixel_mask.py -- if this
    ever drifts from the real camera's shape, every test below that relies on
    the mismatch path silently stops exercising it."""
    mask = vt._dead_pixel_mask()
    assert mask.shape == (10, 9)  # (y, x)


def test_the_9x9_fixture_takes_the_mask_mismatch_path(blobs):
    """The synthetic grid is 9x9; the real camera mask is 9x10. Applying a
    mask to an array it does not fit would mask the wrong pixels and produce
    a plausible wrong answer -- the exact failure this spec exists to avoid.
    So on a mismatch the mask must not be applied, and that has to be visible
    on the result, not just silently correct."""
    params = vt.TdeVelocityParams(refx=CENTRE, refy=CENTRE)
    result = vt.compute(blobs, params)

    assert bool(result["mask_shape_mismatch"])
    assert not bool(result["mask_applied"])
    # No pixel is drawn as dead: the mask genuinely was not applied, not
    # applied-and-happens-to-mark-nothing.
    assert int(result["dead_pixels"].sum()) == 0

    figure = vt.render(result, params, None)
    texts = [a.text for a in figure.layout.annotations]
    assert any("mask" in t.lower() and "not applied" in t.lower() for t in texts)


def test_mask_dead_pixels_false_reports_no_mismatch(blobs):
    """Turning the mask off is a different thing from it failing to apply --
    the mismatch flag is about a mask that was *wanted* but did not fit."""
    params = vt.TdeVelocityParams(refx=CENTRE, refy=CENTRE, mask_dead_pixels=False)
    result = vt.compute(blobs, params)
    assert not bool(result["mask_shape_mismatch"])
    assert not bool(result["mask_applied"])


# ---------------------------------------------------------------------------
# Recovering the planted velocity
# ---------------------------------------------------------------------------


def test_the_default_ccf_min_lag_finds_no_estimate_on_pure_radial_motion(blobs):
    """Upstream's own default (ccf_min_lag=1) requires the poloidal neighbour
    to show a *measurable* delay. The fixture moves purely radially, so the
    poloidal cross-correlation peaks at exactly zero lag -- correctly reported
    as "no usable neighbour", not silently as vy=0. This is a property of the
    estimator on this fixture, not a bug in the port: worth asserting so a
    future change to the estimator or the fixture surfaces here rather than
    only in a confusing NaN downstream."""
    params = vt.TdeVelocityParams(refx=CENTRE, refy=CENTRE)
    result = vt.compute(blobs, params)

    assert np.isnan(float(result["vx"]))
    assert np.isnan(float(result["vy"]))
    assert not bool(result["estimate_failed"])
    assert not bool(result["is_dead"])

    figure = vt.render(result, params, None)
    texts = [a.text for a in figure.layout.annotations]
    assert any("no estimate" in t.lower() for t in texts)
    assert "no velocity estimate" in figure.layout.title.text


def test_ccf_min_lag_zero_recovers_the_planted_velocity(blobs):
    """With ccf_min_lag=0, a neighbour only has to be alive, not show a
    nonzero delay -- which is exactly what a purely radial blob needs to let
    the poloidal pair contribute (correctly) a near-zero delay rather than
    being discarded outright. This averages the whole record through a single
    conditional-average time delay per neighbour pair, a much coarser
    estimator than contour tracking's frame-by-frame track, so 10% is the
    tolerance that this fixture's noise floor actually supports (observed:
    ~393 m/s against 400 planted)."""
    params = vt.TdeVelocityParams(refx=CENTRE, refy=CENTRE, ccf_min_lag=0)
    result = vt.compute(blobs, params)

    assert float(result["vx"]) == pytest.approx(VX_PLANTED, rel=0.1)
    # Purely radial motion: vy must come back at zero, not merely small next
    # to vx -- here, an order of magnitude below it.
    assert abs(float(result["vy"])) < 0.05 * VX_PLANTED

    figure = vt.render(result, params, None)
    assert "no estimate" not in figure.layout.title.text
    assert len(figure.layout.annotations) >= 1  # the velocity arrow


def test_the_neighbours_used_are_marked(blobs):
    params = vt.TdeVelocityParams(refx=CENTRE, refy=CENTRE, ccf_min_lag=0)
    result = vt.compute(blobs, params)
    used = result["neighbours_used"].values
    assert used.sum() > 0
    # The reference pixel itself is never its own neighbour.
    assert used[CENTRE, CENTRE] == 0


# ---------------------------------------------------------------------------
# Params: the trap this port exists to avoid
# ---------------------------------------------------------------------------


def test_the_upstream_options_classes_have_no_dataclass_fields():
    """EstimationOptions, CAOptions and NeighbourOptions all carry @dataclass
    but declare their own __init__, so fields() on each is empty -- the exact
    trap params_ui guards against. Asserted directly so a future upstream
    release that fixes this (and makes a bare EstimationOptions field
    suddenly hash something real) is noticed here rather than nowhere."""
    import velocity_estimation as ve
    import dataclasses

    assert dataclasses.fields(ve.EstimationOptions) == ()
    assert dataclasses.fields(ve.CAOptions) == ()
    assert dataclasses.fields(ve.NeighbourOptions) == ()


def test_max_threshold_none_hashes_the_same_as_a_value_and_does_not_collide():
    """np.inf is not valid JSON and must never reach the hash; None has to
    stand in for it without becoming an empty or colliding cache key."""
    default_hash, _ = params_ui.hash_params(
        "velocity_tde", vt.TdeVelocityParams(max_threshold=None)
    )
    other_hash, _ = params_ui.hash_params(
        "velocity_tde", vt.TdeVelocityParams(max_threshold=10.0)
    )
    again_hash, _ = params_ui.hash_params(
        "velocity_tde", vt.TdeVelocityParams(max_threshold=None)
    )
    assert default_hash == again_hash
    assert default_hash != other_hash


def test_compute_treats_np_inf_and_none_max_threshold_the_same(blobs):
    """None means "no upper threshold" and must convert to np.inf inside
    compute -- the same physical setting upstream spells as a literal
    np.inf, just one that survives the JSON round trip that a real np.inf
    would not."""
    params_none = vt.TdeVelocityParams(refx=CENTRE, refy=CENTRE, ccf_min_lag=0)
    params_explicit = vt.TdeVelocityParams(
        refx=CENTRE, refy=CENTRE, ccf_min_lag=0, max_threshold=1e12
    )
    result_none = vt.compute(blobs, params_none)
    result_explicit = vt.compute(blobs, params_explicit)
    assert float(result_none["vx"]) == pytest.approx(float(result_explicit["vx"]))
    assert float(result_none["vy"]) == pytest.approx(float(result_explicit["vy"]))


# ---------------------------------------------------------------------------
# scalars
# ---------------------------------------------------------------------------


def test_scalars_writes_both_names_at_the_reference_pixel(blobs):
    params = vt.TdeVelocityParams(refx=CENTRE, refy=CENTRE, ccf_min_lag=0)
    result = vt.compute(blobs, params)
    assert vt.scalars(result) == {
        (CENTRE, CENTRE, "vx_tde"): pytest.approx(float(result["vx"])),
        (CENTRE, CENTRE, "vy_tde"): pytest.approx(float(result["vy"])),
    }


def test_scalars_are_written_even_when_the_estimate_is_nan(blobs):
    """A NaN is still the answer for this pixel and belongs in the store --
    dropping it silently would look like the plot was never run."""
    params = vt.TdeVelocityParams(refx=CENTRE, refy=CENTRE)  # default ccf_min_lag=1
    result = vt.compute(blobs, params)
    scalars = vt.scalars(result)
    assert set(scalars) == {(CENTRE, CENTRE, "vx_tde"), (CENTRE, CENTRE, "vy_tde")}
    assert all(np.isnan(v) for v in scalars.values())


# ---------------------------------------------------------------------------
# Through the store
# ---------------------------------------------------------------------------


def test_the_result_round_trips_through_the_store(conn, cache, blobs, target):
    spec = registry.get("velocity_tde")
    params = vt.TdeVelocityParams(refx=CENTRE, refy=CENTRE, ccf_min_lag=0)
    result, run = store.result(conn, spec, target, params, blobs)

    assert run["status"] == "ok", run["error"]
    assert float(result["vx"]) == pytest.approx(VX_PLANTED, rel=0.1)
    # A netCDF round trip is where an int8 flag or a ragged shape would have
    # been quietly rewritten -- assert on what actually came back off disk.
    assert result["neighbours_used"].dtype == np.int8
    assert result["dead_pixels"].dtype == np.int8
    assert vt.render(result, params, target) is not None

    names = {
        (r["plot"], r["name"])
        for r in conn.execute(
            "SELECT r.plot, s.name FROM scalars s JOIN runs r ON r.id = s.run_id"
        )
    }
    assert names == {("velocity_tde", "vx_tde"), ("velocity_tde", "vy_tde")}


def test_the_missing_axis_is_named_when_there_is_no_estimate(blobs):
    """The fixture moves purely radially, so at the upstream default the
    poloidal neighbour's cross-correlation peaks at exactly lag 0 and is
    rejected -- leaving a radial pair but no vertical one.

    Every estimate pairs one neighbour from each axis, so the honest message
    names the axis that came up empty. The generic "no combination of
    neighbours" fallback fires only when both axes have candidates and none
    of them combine, which is a different failure.
    """
    params = vt.TdeVelocityParams(refx=CENTRE, refy=CENTRE)
    result = vt.compute(blobs, params)

    assert not np.isfinite(float(result["vx"]))
    assert int(result["n_horizontal"]) > 0, "the radial pair should be found"
    assert int(result["n_vertical"]) == 0, "the poloidal pair should be rejected"

    reason = vt._reason(result)
    assert "poloidal" in reason
    assert "ccf_min_lag" in reason
    assert "no combination" not in reason
