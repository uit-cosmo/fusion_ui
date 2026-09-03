"""``trajectories`` -- two trackers on one conditional average, against blobs
whose velocity is known in advance (see ``test_plots_blob.py``)."""

import numpy as np
import pytest
import xarray as xr

from fusion_ui.core import registry, store
from fusion_ui.plots import trajectories, two_dca

#: What the fixture plants: 20 blobs at 400 m/s, purely radially outward.
EVENTS, VX = 20, 400.0
CENTRE = 4  # the reference pixel on the 9x9 grid


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
def traj_params():
    params = trajectories.TrajectoryParams()
    params.two_dca.refx = params.two_dca.refy = CENTRE
    return params


@pytest.fixture
def average(blobs, two_dca_params):
    return two_dca.compute(blobs, two_dca_params)


def test_both_tracks_recover_the_planted_radial_velocity(average, traj_params):
    result = trajectories.compute(None, traj_params, average)

    # The tolerance is generous next to velocity_contour's 5%: the maximum
    # tracker on cross_corr has no contour-level averaging to smooth its
    # sub-pixel estimate, so its lsq slope carries more scatter than the
    # contour centroid's -- 15% is comfortably inside what was observed while
    # writing this test and still tight enough to catch a broken tracker.
    assert float(result["slope_r_cond_av"]) == pytest.approx(VX, rel=0.05)
    assert float(result["slope_r_cross_corr"]) == pytest.approx(VX, rel=0.15)


def test_purely_radial_motion_gives_zero_poloidal_slope(average, traj_params):
    result = trajectories.compute(None, traj_params, average)

    # Not merely small next to the radial slope: a bug that mixed up R and Z
    # would still look "small" by that standard.
    assert float(result["slope_z_cond_av"]) == pytest.approx(0.0, abs=0.05 * VX)
    assert float(result["slope_z_cross_corr"]) == pytest.approx(0.0, abs=0.05 * VX)


def test_tracks_may_differ_in_length_and_survive_a_netcdf_round_trip(
    average, traj_params, tmp_path
):
    result = trajectories.compute(None, traj_params, average)

    assert result["valid_cond_av"].dtype == np.int8
    assert result["mask_cond_av"].dtype == np.int8
    assert result["valid_cross_corr"].dtype == np.int8
    assert result["mask_cross_corr"].dtype == np.int8

    path = tmp_path / "trajectories.nc"
    result.to_netcdf(path)
    with xr.open_dataset(path) as reloaded:
        reloaded = reloaded.load()

    assert reloaded.sizes["lag_cond_av"] > 0
    assert reloaded.sizes["lag_cross_corr"] > 0
    assert float(reloaded["slope_r_cond_av"]) == pytest.approx(VX, rel=0.05)
    figure = trajectories.render(reloaded, traj_params, target=None)
    assert figure is not None


def test_the_cross_corr_override_changes_only_its_own_mask(average, traj_params):
    baseline = trajectories.compute(None, traj_params, average)

    traj_params.cross_corr.mask_signal_factor = 0.95
    tightened = trajectories.compute(None, traj_params, average)

    assert not np.array_equal(
        tightened["mask_cross_corr"].values, baseline["mask_cross_corr"].values
    )
    assert np.array_equal(
        tightened["mask_cond_av"].values, baseline["mask_cond_av"].values
    )


def test_scalars_writes_all_four_new_names_at_the_reference_pixel(
    average, traj_params
):
    result = trajectories.compute(None, traj_params, average)
    scalars = trajectories.scalars(result)

    assert set(scalars) == {
        (CENTRE, CENTRE, "vx_2dca_lsq"),
        (CENTRE, CENTRE, "vy_2dca_lsq"),
        (CENTRE, CENTRE, "vx_ccf_lsq"),
        (CENTRE, CENTRE, "vy_ccf_lsq"),
    }
    assert scalars[(CENTRE, CENTRE, "vx_2dca_lsq")] == pytest.approx(VX, rel=0.05)


def test_a_failed_track_comes_back_nan_without_taking_the_other_down(
    average, traj_params, monkeypatch
):
    """If the contour tracker raises for cond_av (e.g. no contour anywhere),
    the cross_corr track must still compute normally."""

    def _boom(*args, **kwargs):
        raise RuntimeError("no contour ever closes")

    monkeypatch.setattr(trajectories.im, "get_contour_evolution", _boom)
    result = trajectories.compute(None, traj_params, average)

    assert not np.isfinite(result["slope_r_cond_av"])
    assert np.isfinite(result["slope_r_cross_corr"])
    figure = trajectories.render(result, traj_params, target=None)
    assert figure is not None


# ---------------------------------------------------------------------------
# The figure has to be readable, and the chain has to go through the store
# ---------------------------------------------------------------------------


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


def test_two_tracks_sharing_a_corner_do_not_stack_their_labels(
    average, traj_params, target
):
    """Both slopes are positive here, so both velocity labels pick the same
    corner. Placed at the same y they land exactly on top of each other and
    neither is readable -- the figure then shows two velocities as one smear.
    """
    result = trajectories.compute(None, traj_params, average)
    figure = trajectories.render(result, traj_params, target)
    positions = [
        (str(a.xref), str(a.yref), round(float(a.x), 6), round(float(a.y), 6))
        for a in figure.layout.annotations
        if a.text and "m/s" in a.text
    ]
    assert len(positions) >= 2, "expected a velocity label per track per panel"
    assert len(positions) == len(set(positions)), f"labels overlap: {positions}"


def test_the_chain_round_trips_through_the_store(
    conn, cache, blobs, traj_params, target
):
    """The store, not ``compute``, is what the page actually calls.

    Going through it is the only thing that exercises ``upstream_params``: the
    2DCA is resolved out of this spec's own parameters, and both links get
    their own ledger row.
    """
    spec = registry.get("trajectories")
    stored, run = store.result(conn, spec, target, traj_params, blobs)

    assert run["status"] == "ok", run["error"]
    assert float(stored["slope_r_cond_av"]) == pytest.approx(VX, rel=0.05)

    upstream_hash, _ = store.record_params(
        conn, "two_dca", trajectories.upstream_params(traj_params)
    )
    upstream = store.find_run(conn, target, "two_dca", upstream_hash)
    assert upstream is not None and upstream["status"] == "ok"
