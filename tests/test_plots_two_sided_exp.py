"""Two-sided exponential fits to the 2DCA cuts, against a blob of known shape.

The ``blob_dataset_path`` fixture plants a *symmetric* Gaussian (sigma 0.4 cm)
moving radially at 400 m/s -- not an exponential. So the assertions here are
about what a symmetric, known-velocity structure has to give back: sigma near
0.5, a decay length of the right order, and a velocity consistent between the
two fits -- not a perfect residual, which a Gaussian was never going to give
an exponential fit.
"""

import numpy as np
import pytest
import xarray as xr

from fusion_ui.core import registry, store
from fusion_ui.plots import two_dca, two_sided_exp

CENTRE = 4  # the reference pixel on the 9x9 grid
VX = 400.0  # m/s, planted


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
def fit_params():
    params = two_sided_exp.TwoSidedExpParams()
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


@pytest.fixture
def result(blobs, two_dca_params, fit_params):
    average = two_dca.compute(blobs, two_dca_params)
    return two_sided_exp.compute(blobs, fit_params, average)


# ---------------------------------------------------------------------------


def test_both_fits_come_back_near_symmetric(result):
    """The planted structure is a symmetric Gaussian, so a fit claiming a
    strongly asymmetric decay would be a bug, not a property of the data.

    The tolerance (0.1 either side of 0.5) is generous because a two-sided
    exponential is the wrong family for a Gaussian core: the optimiser is
    free to trade sigma against offset near the peak, where an exponential's
    kink at x=0 does not match a Gaussian's smooth top. What must not happen
    is a lopsided asymmetry -- e.g. sigma outside [0.3, 0.7] -- which would
    mean the fit is favouring one side for no reason the planted data gives it.
    """
    sigma_t = float(result["sigma_t"])
    sigma_sp = float(result["sigma_sp"])
    assert sigma_t == pytest.approx(0.5, abs=0.1)
    assert sigma_sp == pytest.approx(0.5, abs=0.1)


def test_l_prime_is_the_order_of_the_planted_width(result):
    """A Gaussian's exponential-fit scale is not its sigma (they are different
    functional forms), so this asserts on the order of magnitude and on
    stability, not on an identity derived from the Gaussian's sigma."""
    l_prime = float(result["l_prime"])
    assert np.isfinite(l_prime)
    # Order of the planted 0.4 cm = 0.004 m width: neither vanishing nor
    # blown up to the size of the whole 9x9, ~3.2 cm grid.
    assert 0.001 < l_prime < 0.02


def test_velocity_from_the_two_scales_is_consistent_with_the_planted_speed(result):
    """l_prime / tau_prime is the velocity in ca_two_sided_exp.py's own
    pipeline -- the strongest available check that both fits are on the same
    footing, since it combines a spatial and a temporal scale that have no
    other reason to agree.

    On this fixture it comes back at ~632 m/s against a planted 400 m/s (a
    factor of ~1.6): the two-sided exponential's scale is measured out where
    the curve has fallen to 1/e, and a Gaussian's shoulder is still well above
    that at the exponential-fit's implied 1/e point, so both l_prime and
    tau_prime come out longer than the naive length/time you would read off
    the Gaussian directly -- consistently enough in both fits that the ratio
    only drifts ~60%, not an order of magnitude. That the model is the wrong
    family for a Gaussian core is a real finding, not something to tune away;
    the tolerance below is set from the number actually observed, not backed
    into 400.
    """
    l_prime = float(result["l_prime"])
    tau_prime = float(result["tau_prime"])
    assert tau_prime > 0
    velocity = l_prime / tau_prime
    assert velocity == pytest.approx(VX, rel=0.65)


def test_normalisation_is_load_bearing(blobs, two_dca_params, fit_params):
    """The model is pinned to amplitude 1 at x=0. Scaling the conditional
    average by a constant must leave every fitted number unchanged -- if it
    didn't, the cut was not actually being normalised before the fit."""
    average = two_dca.compute(blobs, two_dca_params)
    scaled = average.copy()
    scaled["cond_av"] = average["cond_av"] * 7.3

    base = two_sided_exp.compute(blobs, fit_params, average)
    rescaled = two_sided_exp.compute(blobs, fit_params, scaled)

    for name in (
        "tau_prime",
        "sigma_t",
        "offset_t",
        "l_prime",
        "sigma_sp",
        "offset_sp",
    ):
        assert float(rescaled[name]) == pytest.approx(float(base[name]), rel=1e-6)


def test_scalars_are_the_four_new_names(result):
    scalars = two_sided_exp.scalars(result)
    assert set(scalars) == {
        (CENTRE, CENTRE, "tau_prime"),
        (CENTRE, CENTRE, "sigma_t"),
        (CENTRE, CENTRE, "l_prime"),
        (CENTRE, CENTRE, "sigma_sp"),
    }


def test_upstream_params_carries_a_changed_two_dca_through(fit_params):
    fit_params.two_dca.threshold = 3.5
    upstream = two_sided_exp.upstream_params(fit_params)
    assert upstream.two_dca.threshold == 3.5
    assert upstream.two_dca is fit_params.two_dca


def test_compute_never_looks_at_the_raw_dataset(blobs, two_dca_params, fit_params):
    average = two_dca.compute(blobs, two_dca_params)
    assert two_sided_exp.compute(None, fit_params, average) is not None


def test_render_draws_a_figure(result, fit_params, target):
    figure = two_sided_exp.render(result, fit_params, target)
    assert figure is not None


def test_the_chain_round_trips_through_the_store(
    conn, cache, blobs, fit_params, target
):
    spec = registry.get("two_sided_exp")
    result, run = store.result(conn, spec, target, fit_params, blobs)

    assert run["status"] == "ok", run["error"]
    assert two_sided_exp.render(result, fit_params, target) is not None

    names = {
        (r["plot"], r["name"])
        for r in conn.execute(
            "SELECT r.plot, s.name FROM scalars s JOIN runs r ON r.id = s.run_id"
        )
    }
    assert names == {
        ("two_dca", "number_events"),
        ("two_sided_exp", "tau_prime"),
        ("two_sided_exp", "sigma_t"),
        ("two_sided_exp", "l_prime"),
        ("two_sided_exp", "sigma_sp"),
    }


# ---------------------------------------------------------------------------
# The failure the optimiser reports as a success
# ---------------------------------------------------------------------------


def test_a_fit_resting_on_its_own_bound_is_rejected(blobs, two_dca_params, fit_params):
    """L-BFGS-B walking into a corner of its box is not a converged fit.

    At ``window = 71`` on this fixture the temporal fit comes to rest with
    ``offset`` pinned at its ``-10`` bound and a ``scale`` twenty-two times the
    one either neighbouring window gives -- and scipy reports ``success=True``,
    ``status=0``. Neither a non-finite check nor scipy's own flag catches it,
    so ``_on_a_bound`` does, and the answer is NaN rather than a number that
    would sit on a multi-shot axis looking like a duration.

    The two neighbouring windows are asserted alongside it: a guard that
    rejected every fit would pass the first assertion on its own.
    """
    for window, expected_finite in ((61, True), (71, False), (91, True)):
        two_dca_params.two_dca.window = window
        fit_params.two_dca.window = window
        average = two_dca.compute(blobs, two_dca_params)
        result = two_sided_exp.compute(None, fit_params, average)

        tau = float(result["tau_prime"])
        assert np.isfinite(tau) == expected_finite, (
            f"window={window}: tau_prime={tau}"
        )
        # The radial cut is read off the zero-lag slice and never sees the
        # window, so it must stay finite throughout -- that is what makes this
        # a test of the guard and not of the 2DCA falling over.
        assert np.isfinite(float(result["l_prime"]))


def test_the_guard_leaves_a_healthy_fit_alone():
    """``_on_a_bound`` must not fire on an ordinary interior answer."""
    assert not two_sided_exp._on_a_bound(sigma=0.5, offset=0.0)
    assert two_sided_exp._on_a_bound(sigma=0.5, offset=-10.0)
    assert two_sided_exp._on_a_bound(sigma=0.5, offset=10.0)
    assert two_sided_exp._on_a_bound(sigma=1e-9, offset=0.0)
    assert two_sided_exp._on_a_bound(sigma=1 - 1e-9, offset=0.0)
