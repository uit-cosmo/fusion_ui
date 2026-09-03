"""Asymmetric two-sided exponential fits to the cuts through the 2DCA average.

Ported from ``waveform_analysis.fit_two_sided`` as
``individual_events/ca_two_sided_exp.py:estimate_parameters`` drives it, with
one deliberate change of subject: upstream only ever fits synthetic 1D pulse
superpositions. Here the same fit is applied to the two cuts through a real
(or synthetic-fixture) 2D conditional average -- the temporal cut at the
reference pixel, and the radial cut at zero lag -- because that is the shape
``density_scan`` results come in. The model
(``offset + (1-offset)*exp(-|x|/lam)``, split into ``lam_left``/``lam_right``
by ``sigma`` about ``x=0``) is pinned to amplitude 1 at the origin, so both
cuts are normalised by the conditional average's value at the reference pixel
/ zero lag before fitting -- exactly as ``estimate_parameters`` normalises by
``average_ds.frames.isel(time=t_middle, x=x_middle)``. Fitting an
un-normalised cut is the one way this silently returns nonsense.

Chained the same way as ``fwhm_sizes``: ``requires="two_dca"``, no parameters
of its own beyond the ``two_dca`` field every derived spec needs, and
``compute`` never looks at ``ds``.

**These four scalars are not comparable across different settings**, and that
is a property of the estimator, not a defect here. The fitted scale depends on
how far out the cut it is fitted to extends, because an exponential fitted to a
non-exponential core reads a different decay length over ``+-1.5`` widths than
over ``+-4``. The two cuts do not extend equally far: the radial one spans the
pixel array, the temporal one spans ``TwoDcaParams.window``. Measured on the
synthetic fixture, holding everything else fixed and widening ``window`` alone
moves ``tau_prime`` by a factor of three (2.3e-5 s at the default 61 frames,
6.8e-5 s at 131) while ``l_prime`` does not move at all -- so their ratio,
which is a velocity, sweeps from 632 m/s down through the planted 400 to 215.

So: ``tau_prime`` is a function of the 2DCA window, ``l_prime`` of the array
geometry, and neither may be put on a multi-shot axis against values computed
at another window or on another camera without saying so. Comparing them
within one parameter set is fine, which is what the store's cache key already
guarantees.
"""

from dataclasses import dataclass, field

import imaging_methods as im
import numpy as np
import plotly.graph_objects as go
import waveform_analysis as wf
import xarray as xr
from imaging_methods.method_parameters import TwoDcaParams
from plotly.subplots import make_subplots

from fusion_ui.core import registry
from fusion_ui.plots import two_dca

#: R is stored in centimetres; the group reports sizes in metres -- convert
#: once, here, after the fit itself (which runs on the native cm coordinate).
CM = 100.0


def _apd_two_dca_defaults():
    return im.get_default_apd_method_params().two_dca


@dataclass
class TwoSidedExpParams:
    #: Identifies the upstream conditional average -- see ``upstream_params``.
    two_dca: TwoDcaParams = field(default_factory=_apd_two_dca_defaults)


def upstream_params(params):
    """The 2DCA parameters, lifted out of this spec's own.

    Reading them out rather than defaulting them is what keeps the two cache
    keys in step: change the threshold here and both the average and this fit
    get a new entry.
    """
    return two_dca.TwoDcaSpecParams(two_dca=params.two_dca)


#: The box ``waveform_analysis.fit_two_sided`` optimises inside, mirrored here
#: so a result resting on one of its walls can be recognised. Kept as a literal
#: rather than imported because upstream does not export it; if upstream ever
#: widens the box this stays merely conservative, never wrong.
FIT_EPS = 1e-6
OFFSET_BOUND = 10.0

#: How close to a wall still counts as on it. Loose enough that the optimiser's
#: own tolerance cannot land just inside and read as a real answer.
BOUND_TOLERANCE = 1e-3


def _on_a_bound(sigma, offset):
    """True if the fit came to rest against one of its own box constraints."""
    return (
        abs(offset) >= OFFSET_BOUND - BOUND_TOLERANCE
        or sigma <= FIT_EPS + BOUND_TOLERANCE
        or sigma >= 1 - FIT_EPS - BOUND_TOLERANCE
    )


def _fit_normalized(coords, values, peak, ref_index):
    """Normalise ``values`` by ``peak`` and fit ``wf.fit_two_sided`` to it.

    The model is pinned to amplitude 1 at ``x=0``, so the caller's ``peak`` --
    the conditional average at the reference pixel / zero lag -- has to be
    divided out first. ``ref_index`` is where the model's origin sits in
    ``coords``: if the normalised data does not itself peak there, the two-
    sided model's central assumption is false for this cut, and the honest
    answer is NaN, not a fit optimised into whatever shape happens to fall
    out. The optimiser itself never raises (L-BFGS-B always returns *some*
    ``x``), so no exception marks a bad fit and neither does scipy's own
    ``success`` flag -- a fit that walks into a corner of the box and stops
    reports ``success=True`` with ``status=0``. What does mark one is the
    answer sitting *on* a bound: ``fit_two_sided`` boxes ``offset`` into
    ``(-10, 10)`` and ``sigma`` into ``(eps, 1-eps)``, and on normalised data
    that should live in roughly ``[0, 1]`` an offset pinned at ``-10`` is the
    optimiser having failed, not having found something. Observed on the
    synthetic fixture at ``window=71``: ``offset`` at the bound and a ``scale``
    twenty-two times its neighbours', with scipy reporting convergence.

    So three things are rejected, all as NaN rather than as a plausible number:
    a normalised cut that does not peak at ``ref_index`` (the model's central
    assumption is false for it), a non-finite result, and a fit resting on any
    of its bounds.

    Returns ``(scale, sigma, offset, normalised_values, y_fit)``.
    """
    coords = np.asarray(coords, dtype=float)
    normalized = np.asarray(values, dtype=float) / peak
    nan_fit = np.full_like(normalized, np.nan)
    failed = (float("nan"), float("nan"), float("nan"), normalized, nan_fit)

    if int(np.argmax(normalized)) != ref_index:
        return failed

    popt, y_fit = wf.fit_two_sided(coords, normalized)
    scale, sigma, offset = (float(v) for v in popt)
    if not np.isfinite((scale, sigma, offset)).all():
        return failed
    if _on_a_bound(sigma, offset):
        return failed
    return scale, sigma, offset, normalized, np.asarray(y_fit, dtype=float)


def compute(ds, params, upstream):
    """Fit the temporal cut at the reference pixel and the radial cut at zero lag."""
    average = upstream
    refx, refy = int(average["refx"]), int(average["refy"])
    cond_av = average["cond_av"]

    time = np.asarray(cond_av["time"].values, dtype=float)
    t_index = int(np.argmin(np.abs(time)))

    r_ref = average.R.isel(x=refx, y=refy).item()
    radial_pos = average.R.isel(y=refy).values - r_ref

    peak = float(cond_av.isel(x=refx, y=refy, time=t_index).item())

    temporal_cut = cond_av.isel(x=refx, y=refy).values
    tau_prime, sigma_t, offset_t, temporal_data, temporal_fit = _fit_normalized(
        time, temporal_cut, peak, t_index
    )

    radial_cut = cond_av.isel(time=t_index, y=refy).values
    l_prime_cm, sigma_sp, offset_sp, radial_data, radial_fit = _fit_normalized(
        radial_pos, radial_cut, peak, refx
    )
    l_prime = l_prime_cm / CM

    # scale splits into lam_left = scale*(1-sigma), lam_right = scale*sigma --
    # store the split lengths too, since they are what a reader compares.
    lam_left_t = tau_prime * (1 - sigma_t)
    lam_right_t = tau_prime * sigma_t
    lam_left_sp = l_prime * (1 - sigma_sp)
    lam_right_sp = l_prime * sigma_sp

    return xr.Dataset(
        {
            "lag": ("time", time),
            "temporal_data": ("time", np.asarray(temporal_data)),
            "temporal_fit": ("time", np.asarray(temporal_fit)),
            "radial_pos": ("x", np.asarray(radial_pos)),
            "radial_data": ("x", np.asarray(radial_data)),
            "radial_fit": ("x", np.asarray(radial_fit)),
            "tau_prime": float(tau_prime),
            "sigma_t": float(sigma_t),
            "offset_t": float(offset_t),
            "lam_left_t": float(lam_left_t),
            "lam_right_t": float(lam_right_t),
            "l_prime": float(l_prime),
            "sigma_sp": float(sigma_sp),
            "offset_sp": float(offset_sp),
            "lam_left_sp": float(lam_left_sp),
            "lam_right_sp": float(lam_right_sp),
            "refx": refx,
            "refy": refy,
            "number_events": int(average["number_events"]),
        }
    )


# ---------------------------------------------------------------------------


def _panel_annotation(scale, sigma, lam_left, lam_right, unit):
    if not np.isfinite(scale):
        return "fit did not converge, or the cut's peak is off the reference pixel"
    return (
        f"scale = {scale:.3g} {unit} · σ = {sigma:.2f}<br>"
        f"λ_left = {lam_left:.3g} {unit} · λ_right = {lam_right:.3g} {unit}"
    )


def render(result, params, target):
    """The two cuts, data as markers and the fit as a line, side by side.

    An asymmetric exponential either describes the waveform or visibly does
    not -- the residual against the markers is the whole reason this view
    exists, so the fit is never drawn alone.
    """
    lag_us = result["lag"].values * 1e6
    radial_pos = result["radial_pos"].values

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("temporal cut at the reference pixel", "radial cut at zero lag"),
    )

    figure.add_trace(
        go.Scatter(
            x=lag_us,
            y=result["temporal_data"].values,
            mode="markers",
            name="Φ(t) / peak",
            marker=dict(color="royalblue", size=5),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=lag_us,
            y=result["temporal_fit"].values,
            mode="lines",
            name="fit",
            line=dict(color="black"),
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=radial_pos,
            y=result["radial_data"].values,
            mode="markers",
            name="Φ(R−R*) / peak",
            marker=dict(color="seagreen", size=5),
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=radial_pos,
            y=result["radial_fit"].values,
            mode="lines",
            name="fit",
            line=dict(color="black"),
        ),
        row=1,
        col=2,
    )

    figure.add_annotation(
        text=_panel_annotation(
            float(result["tau_prime"]) * 1e6,
            float(result["sigma_t"]),
            float(result["lam_left_t"]) * 1e6,
            float(result["lam_right_t"]) * 1e6,
            "µs",
        ),
        xref="x domain",
        yref="y domain",
        x=0.02,
        y=0.98,
        showarrow=False,
        align="left",
        row=1,
        col=1,
    )
    figure.add_annotation(
        text=_panel_annotation(
            float(result["l_prime"]) * CM,
            float(result["sigma_sp"]),
            float(result["lam_left_sp"]) * CM,
            float(result["lam_right_sp"]) * CM,
            "cm",
        ),
        xref="x domain",
        yref="y domain",
        x=0.02,
        y=0.98,
        showarrow=False,
        align="left",
        row=1,
        col=2,
    )

    figure.update_xaxes(title_text="lag [µs]", row=1, col=1)
    figure.update_xaxes(title_text="R−R* [cm]", row=1, col=2)
    figure.update_yaxes(title_text="normalised Φ", row=1, col=1)
    figure.update_yaxes(title_text="normalised Φ", row=1, col=2)
    figure.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
        title=f"{int(result['number_events'])} events",
    )
    return figure


def scalars(result):
    """The four new names this estimator reports, at the reference pixel.

    ``tau_prime``, ``sigma_t``, ``l_prime`` and ``sigma_sp`` are not among the
    seeded ``density_scan`` names -- this estimator was never run upstream.
    """
    x, y = int(result["refx"]), int(result["refy"])
    return {
        (x, y, "tau_prime"): float(result["tau_prime"]),
        (x, y, "sigma_t"): float(result["sigma_t"]),
        (x, y, "l_prime"): float(result["l_prime"]),
        (x, y, "sigma_sp"): float(result["sigma_sp"]),
    }


SPEC = registry.register(
    registry.PlotSpec(
        key="two_sided_exp",
        label="Two-sided exponential fits (2DCA cuts)",
        diagnostics=("apd", "phantom"),
        params=TwoSidedExpParams,
        render=render,
        compute=compute,
        scalars=scalars,
        choices=two_dca.choices,
        requires="two_dca",
        upstream_params=upstream_params,
        description=(
            "Asymmetric two-sided exponential fits to the temporal cut at the "
            "reference pixel and the radial cut at zero lag. Emits tau_prime, "
            "sigma_t, l_prime and sigma_sp."
        ),
    )
)
