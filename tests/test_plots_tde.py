"""``velocity_tde``, against blobs whose velocity is known in advance."""

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


def test_the_dead_pixel_mask_is_experimental_databases_1160616_mask():
    """9 wide by 10 tall: the tests below rely on it not fitting the 9x9 fixture."""
    from experimental_database import hand_made_dead_pixel_mask

    mask = vt._dead_pixel_mask()
    assert mask.shape == (10, 9)  # (y, x)
    np.testing.assert_array_equal(mask, hand_made_dead_pixel_mask("1160616").values)


def test_the_9x9_fixture_takes_the_mask_mismatch_path(blobs):
    """The synthetic grid is 9x9; the camera mask is 9x10. A mask applied to an
    array it does not fit would mask the wrong pixels, so on a mismatch it is
    not applied, and the result and the figure say so."""
    params = vt.TdeVelocityParams(refx=CENTRE, refy=CENTRE)
    result = vt.compute(blobs, params)

    assert bool(result["mask_shape_mismatch"])
    assert not bool(result["mask_applied"])
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
    """The default (ccf_min_lag=1) requires the poloidal neighbour to show a
    measurable delay. The fixture moves purely radially, so the poloidal
    cross-correlation peaks at zero lag and that neighbour is rejected: no
    estimate, not vy=0, and the message names the axis that came up empty."""
    params = vt.TdeVelocityParams(refx=CENTRE, refy=CENTRE)
    result = vt.compute(blobs, params)

    assert np.isnan(float(result["vx"]))
    assert np.isnan(float(result["vy"]))
    assert not bool(result["estimate_failed"])
    assert not bool(result["is_dead"])
    assert int(result["n_horizontal"]) > 0, "the radial pair should be found"
    assert int(result["n_vertical"]) == 0, "the poloidal pair should be rejected"

    reason = vt._reason(result)
    assert "poloidal" in reason and "ccf_min_lag" in reason
    assert "no combination" not in reason

    figure = vt.render(result, params, None)
    texts = [a.text for a in figure.layout.annotations]
    assert any("no estimate" in t.lower() for t in texts)
    assert "no velocity estimate" in figure.layout.title.text


def test_ccf_min_lag_zero_recovers_the_planted_velocity(blobs):
    """With ccf_min_lag=0 a neighbour only has to be alive, so the poloidal
    pair contributes its near-zero delay. One conditional-average delay per
    neighbour pair over the whole record is a coarse estimator: 10% is what
    this fixture supports (about 393 m/s against 400 planted)."""
    params = vt.TdeVelocityParams(refx=CENTRE, refy=CENTRE, ccf_min_lag=0)
    result = vt.compute(blobs, params)

    assert float(result["vx"]) == pytest.approx(VX_PLANTED, rel=0.1)
    # Purely radial motion: vy an order of magnitude below vx.
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
# Params
# ---------------------------------------------------------------------------


def test_the_upstream_options_classes_have_no_dataclass_fields():
    """EstimationOptions, CAOptions and NeighbourOptions carry @dataclass but
    declare their own __init__, so fields() on each is empty and none may be a
    params field. Asserted so that an upstream release that changes it is
    noticed."""
    import dataclasses

    import velocity_estimation as ve

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
    """None means "no upper threshold" and becomes np.inf inside compute."""
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
    """A NaN is still the answer for this pixel and belongs in the store."""
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

