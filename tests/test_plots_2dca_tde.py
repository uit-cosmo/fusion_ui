"""The 2DCA time-delay velocity, against blobs whose velocity is known in advance.

Mirrors ``tests/test_plots_blob.py``'s fixture style: the ``blob_dataset_path``
fixture plants 20 Gaussians crossing a 9x9 grid radially at 400 m/s.
"""

import numpy as np
import pytest
import xarray as xr

from fusion_ui.core import registry, store
from fusion_ui.plots import two_dca, velocity_2dca_tde

#: What the fixture plants: 20 blobs at 400 m/s radially outward.
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
def tde_params():
    params = velocity_2dca_tde.Velocity2dcaTdeParams()
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


def test_the_estimator_recovers_the_planted_velocity(blobs, two_dca_params, tde_params):
    average = two_dca.compute(blobs, two_dca_params)
    result = velocity_2dca_tde.compute(blobs, tde_params, average)

    # The fixture gives back vx ~= 411 m/s against a planted 400 -- a ~3%
    # error, comfortably inside the same 5% test_plots_blob.py holds the
    # contour-tracking estimator to.
    assert float(result["vx"]) == pytest.approx(VX, rel=0.05)
    # Purely radial motion: the poloidal component must come back at zero,
    # not merely small next to vx -- 5% of vx, the same bound
    # test_plots_blob.py holds the contour estimator to.
    assert abs(float(result["vy"])) < 0.05 * VX


def test_a_reference_pixel_on_the_edge_still_yields_a_velocity(blobs):
    # refy stays at the centre: the planted blob is a 2D Gaussian narrow in Z
    # (sigma 0.4 cm), so a reference pixel away from the centre row sees no
    # signal at all regardless of x -- only x is pushed to the array edge.
    for edge, missing in ((0, "left"), (8, "right")):
        two_dca_params = two_dca.TwoDcaSpecParams()
        two_dca_params.two_dca.refx, two_dca_params.two_dca.refy = edge, CENTRE
        average = two_dca.compute(blobs, two_dca_params)

        tde_params = velocity_2dca_tde.Velocity2dcaTdeParams()
        tde_params.two_dca.refx, tde_params.two_dca.refy = edge, CENTRE
        result = velocity_2dca_tde.compute(blobs, tde_params, average)

        assert np.isfinite(float(result["vx"]))
        assert np.isfinite(float(result["vy"]))
        # The one neighbour off the array has no located maximum at all.
        assert int(result[f"edge_{missing}"]) == -1
        assert np.isnan(float(result[f"tau_{missing}"]))


def test_the_scalars_are_the_two_density_scan_reports(blobs, two_dca_params, tde_params):
    average = two_dca.compute(blobs, two_dca_params)
    result = velocity_2dca_tde.compute(blobs, tde_params, average)
    assert velocity_2dca_tde.scalars(result) == {
        (CENTRE, CENTRE, "vx_2dca_tde"): pytest.approx(float(result["vx"])),
        (CENTRE, CENTRE, "vy_2dca_tde"): pytest.approx(float(result["vy"])),
    }


def test_upstream_params_carries_a_changed_two_dca_through(tde_params):
    tde_params.two_dca.threshold = 3.5
    upstream = velocity_2dca_tde.upstream_params(tde_params)
    assert upstream.two_dca is tde_params.two_dca
    assert upstream.two_dca.threshold == 3.5


def test_compute_never_looks_at_the_raw_dataset(blobs, two_dca_params, tde_params):
    """Its input is the upstream result. If that stops being true, the store's
    chaining is no longer the only path to this analysis."""
    average = two_dca.compute(blobs, two_dca_params)
    assert velocity_2dca_tde.compute(None, tde_params, average) is not None


def test_the_chain_round_trips_through_the_store(conn, cache, blobs, tde_params, target):
    spec = registry.get("velocity_2dca_tde")
    result, run = store.result(conn, spec, target, tde_params, blobs)

    assert run["status"] == "ok", run["error"]
    assert float(result["vx"]) == pytest.approx(VX, rel=0.05)
    assert result["edge_ref"].dtype == np.int8
    assert velocity_2dca_tde.render(result, tde_params, target) is not None

    names = {
        (r["plot"], r["name"])
        for r in conn.execute(
            "SELECT r.plot, s.name FROM scalars s JOIN runs r ON r.id = s.run_id"
        )
    }
    assert names == {
        ("two_dca", "number_events"),
        ("velocity_2dca_tde", "vx_2dca_tde"),
        ("velocity_2dca_tde", "vy_2dca_tde"),
    }
