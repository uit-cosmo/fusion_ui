"""The Gaussian-fit blob size, against a blob whose width is known in advance.

Mirrors ``tests/test_plots_blob.py``: the ``blob_dataset_path`` fixture plants
20 identical, isotropic Gaussians (sigma = 0.4 cm) crossing a 9x9 grid, so the
fitted ellipse has an answer to check against.
"""

import numpy as np
import pytest
import xarray as xr

from fusion_ui.core import registry, store
from fusion_ui.plots import gaussian_sizes, two_dca

CENTRE = 4  # the reference pixel on the 9x9 grid
#: The planted Gaussian's sigma, in metres.
#:
#: The fitted size is not the planted width: it is that width as the penalties
#: leave it. Measured on this fixture, the unpenalised least-squares optimum is
#: lx = ly = 0.00885 m, and GaussFitParams' default size_penalty=5 pulls it down
#: to 0.00394 m -- a factor of 2.2, which happens to land near sigma itself.
#:
#: The tempting analytic anchor, matching exp(-(x/lx)^2) to exp(-x^2/(2 sigma^2))
#: at lx = sqrt(2) sigma = 0.00566 m, does NOT hold here: fit_ellipse's model has
#: fixed unit amplitude while the conditional average peaks near 3, so the
#: pointwise-match argument assumes an amplitude equality the data does not have.
#: Hence the test targets the observed scale and leans on the lx == ly isotropy
#: cross-check, which is amplitude- and penalty-independent, to catch an axis or
#: unit mix-up.
BLOB_SIGMA = 0.4 / 100
BLOB_L = BLOB_SIGMA


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
def gauss_params():
    params = gaussian_sizes.GaussianSizeParams()
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


def test_the_fit_recovers_the_planted_blob_size(blobs, two_dca_params, gauss_params):
    average = two_dca.compute(blobs, two_dca_params)
    result = gaussian_sizes.compute(blobs, gauss_params, average)

    # 15% tolerance on the scale: the size penalty pulls the fit in from the
    # unpenalised analytic width (see BLOB_L above), and the exact amount
    # depends on the amplitude and the other two penalty factors -- 15% is
    # comfortably wider than the ~1.6% seen with the default GaussFitParams,
    # while still failing on a mixed-up unit (100x) or a fit that ran away.
    assert float(result["lx"]) == pytest.approx(BLOB_L, rel=0.15)
    assert float(result["ly"]) == pytest.approx(BLOB_L, rel=0.15)
    # The planted blob is isotropic, so the two semi-axes must agree with each
    # other tightly, not just with the analytic width within 15% -- a bug that
    # mixed up the two axes would still pass the assertions above by
    # coincidence but fail this one.
    assert float(result["lx"]) == pytest.approx(float(result["ly"]), rel=0.02)


def test_the_scalars_are_the_three_density_scan_reports(
    blobs, two_dca_params, gauss_params
):
    average = two_dca.compute(blobs, two_dca_params)
    result = gaussian_sizes.compute(blobs, gauss_params, average)
    scalars = gaussian_sizes.scalars(result)
    assert set(scalars) == {
        (CENTRE, CENTRE, "lx_f"),
        (CENTRE, CENTRE, "ly_f"),
        (CENTRE, CENTRE, "theta_f"),
    }
    assert scalars[(CENTRE, CENTRE, "lx_f")] == pytest.approx(float(result["lx"]))
    assert scalars[(CENTRE, CENTRE, "ly_f")] == pytest.approx(float(result["ly"]))
    assert scalars[(CENTRE, CENTRE, "theta_f")] == pytest.approx(float(result["theta"]))


def test_compute_never_looks_at_the_raw_dataset(blobs, two_dca_params, gauss_params):
    """Its input is the upstream result. If that stops being true, the store's
    chaining is no longer the only path to this analysis."""
    average = two_dca.compute(blobs, two_dca_params)
    assert gaussian_sizes.compute(None, gauss_params, average) is not None


def test_upstream_params_carries_a_changed_two_dca_through(gauss_params):
    """Reading the field out, rather than defaulting it, is what keeps the two
    cache keys in step: change the threshold here and the average this fit is
    built on has to change too."""
    gauss_params.two_dca.threshold = 4.0
    upstream = gaussian_sizes.upstream_params(gauss_params)
    assert upstream.two_dca.threshold == 4.0
    assert upstream.two_dca is gauss_params.two_dca


def test_the_chain_round_trips_through_the_store(conn, cache, blobs, gauss_params, target):
    spec = registry.get("gaussian_sizes")
    result, run = store.result(conn, spec, target, gauss_params, blobs)

    assert run["status"] == "ok", run["error"]
    assert float(result["lx"]) == pytest.approx(BLOB_L, rel=0.15)
    assert gaussian_sizes.render(result, gauss_params, target) is not None

    names = {
        (r["plot"], r["name"])
        for r in conn.execute(
            "SELECT r.plot, s.name FROM scalars s JOIN runs r ON r.id = s.run_id"
        )
    }
    assert names == {
        ("two_dca", "number_events"),
        ("gaussian_sizes", "lx_f"),
        ("gaussian_sizes", "ly_f"),
        ("gaussian_sizes", "theta_f"),
    }
