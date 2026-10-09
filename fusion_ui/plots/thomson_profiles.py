"""The Thomson scattering profiles view: n_e, T_e and p_e against ρ_pol or R_mid.

A *live* spec: nothing is computed and nothing is cached. ``thomson_scattering_<shot>.nc`` is a few hundred kilobytes
that already hold the measurements and the fit, and this view only draws them. The file is made by fusion_scripts'
``tools/get_thomson_scattering.py`` (the module docstring of ``tools/mfe/get_thomson_scattering.py`` is the reference
for the layout); nothing here imports from fusion_scripts, the view reads the file and nothing else.

What the file holds, and what this view reads of it (netCDF3, ``time`` in seconds)
    ``<sys>_ne``, ``<sys>_Te`` and their ``_err``, by ``(time, <sys>_channel)``, for the edge and the core system
        The tree's values over the whole shot, in SI units with T_e in eV. **A 0 is no measurement**: n_e and T_e
        are almost always 0 together, and the errors there are junk, so only the points with n_e > 0 are drawn.
    ``<sys>_rho_pol`` and ``<sys>_R_mid``, by ``(time, <sys>_channel)``
        Where each point is: ρ_pol = √ψ_N and the outboard midplane radius of the same flux surface, from the EFIT tree
        the attribute ``efit_tree`` names, at the point's own pulse. NaN outside the EFIT time base.
    ``<sys>_R``, ``<sys>_Z`` by channel
        Where the channel looks, for the hover text.
    ``rho_pol`` and, on it, ``fit_ne``, ``fit_Te``, ``fit_pe`` and ``fit_R_mid``
        The fit over the discharge window, as ``get_cmod_kin_profs`` makes it. **Only where the fit worked**: a file
        whose fit failed holds the raw points and says why in ``fit_status``.
    ``R_mid_sep``, ``Te_sep_2pm``, ``lambda_q_2pm``
        Shot-level numbers, when they could be computed. The first is the separatrix on the R_mid axis.
    attributes
        ``window_t_start`` and ``window_t_end`` (the fit's window), ``fit_status``, ``fit_check`` (run_db.py's own test
        of whether the fit silently failed), ``fit_note``, ``fit_settings`` and ``caveat``, among others.

What is drawn
    One row each for n_e, T_e and p_e, on one shared axis, ρ_pol or R_mid (a choice): the edge and core points in the
    page's window, told apart by marker and colour, with the tree's ±1σ bars; the stored fit, over its whole range
    and **without a band**; and the separatrix. The file holds no p_e of the measurements, so its points are e·n_e·T_e,
    with the errors propagated, beside the fit's own ``fit_pe``. A band around the fit is left out on purpose (the
    user, 2026-10-09):
    with ``frac_err`` the fit's n_e and T_e errors are 20% of the fit by construction, and ``fit_pe_err`` adds a
    density to a temperature, so it is no error of p_e. A fit that fails ``fit_check`` gets a warning that quotes it,
    and the diagnostician's caveat is under every figure.

Everything but :func:`render` is pure: it takes an opened file, or what :func:`read` made of one, and returns data, a
figure or text, so each can be tested without Streamlit.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fusion_ui.core import loader, params_ui, registry

SYSTEMS = ("edge", "core")

#: One electron-volt in joules: n_e [m^-3] times T_e [eV] times this is p_e [Pa].
ELEMENTARY_CHARGE = 1.602176634e-19

#: The scalars the file may hold, read for the separatrix and the "about this file" list.
SCALARS = ("R_mid_sep", "Te_sep_2pm", "lambda_q_2pm")

#: The caveat of the C-Mod_Analysis README on the Thomson scattering data, in the words ``get_thomson_scattering.py``
#: writes into every file's ``caveat``. A file that lacks the attribute is shown under this one: it is never left
#: out of the caption.
CAVEAT = (
    "From the C-Mod_Analysis README: it is highly recommended to talk with the diagnostician who collected the data "
    "before using it in a public forum. The Thomson scattering data were collected by Jerry Hughes "
    "(jwhughes@psfc.mit.edu)."
)

#: How each system is drawn. Colour and marker both say which it is, so that the distinction survives a colour-blind
#: reader and a greyscale print; the three colours (with the fit's) are the data-viz palette's first three slots, in
#: the steps that clear 3:1 on a light and on a dark page alike.
LOOKS = {
    "edge": dict(label="edge TS", colour="#2a78d6", symbol="circle"),
    "core": dict(label="core TS", colour="#d95926", symbol="diamond"),
}
FIT_COLOUR = "#199e70"
SEPARATRIX_COLOUR = "#898781"

#: The two things the profiles can be drawn against: the option, its axis title and what the hover calls it.
AXES = {
    "rho_pol": ("ρ<sub>pol</sub>", "ρ_pol"),
    "R_mid": ("R<sub>mid</sub> [m]", "R_mid"),
}

#: The three rows: the file's name for it, how the hover reads it, the axis title, the factor from SI to what is drawn,
#: and its unit.
QUANTITIES = (
    ("ne", "n_e", "n<sub>e</sub> [10<sup>20</sup> m<sup>-3</sup>]", 1e-20, "10²⁰ m⁻³"),
    ("Te", "T_e", "T<sub>e</sub> [eV]", 1.0, "eV"),
    ("pe", "p_e", "p<sub>e</sub> [kPa]", 1e-3, "kPa"),
)


@dataclass
class ThomsonProfilesParams:
    """
    axis: What the profiles are drawn against. rho_pol is √ψ_N, the normalised poloidal flux coordinate, which is 1 at
        the separatrix. R_mid is the major radius at which the flux surface crosses the outboard midplane.
    """

    axis: str = "rho_pol"


params_ui.CHOICES[("ThomsonProfilesParams", "axis")] = tuple(AXES)


# ---------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------


@dataclass(eq=False)  # arrays do not compare as a whole: identity, not a generated __eq__ that raises
class Points:
    """One system's measurements in the window: every channel of every pulse that read something.

    ``pulses`` and ``readings`` count what the window held, ``time`` and the arrays after it what is drawn: a reading of
    0 is no measurement, so it is counted and not kept. Every array has one entry per kept point, in pulse order.
    """

    system: str
    pulses: int
    readings: int
    time: np.ndarray  # s
    channel: np.ndarray
    R: np.ndarray  # m, where the channel looks
    Z: np.ndarray
    rho_pol: np.ndarray  # NaN where the pulse is outside the EFIT time base
    R_mid: np.ndarray  # m
    ne: np.ndarray  # m^-3
    ne_err: np.ndarray  # NaN where the tree's error is missing, or is not positive
    Te: np.ndarray  # eV, NaN where it reads 0 though n_e does not
    Te_err: np.ndarray

    @property
    def drawn(self):
        return int(self.time.size)

    def pe(self):
        """p_e in Pa, from the points: e·n_e·T_e."""
        return ELEMENTARY_CHARGE * self.ne * self.Te

    def pe_err(self):
        """The error of :meth:`pe`, ``e·sqrt((T_e σ_n)² + (n_e σ_T)²)``: NaN where either error is."""
        return ELEMENTARY_CHARGE * np.hypot(self.Te * self.ne_err, self.ne * self.Te_err)

    def series(self, quantity):
        """``(values, errors)`` of ``"ne"``, ``"Te"`` or ``"pe"`` at the kept points, in SI units."""
        if quantity == "ne":
            return self.ne, self.ne_err
        if quantity == "Te":
            return self.Te, self.Te_err
        return self.pe(), self.pe_err()

    def on(self, axis):
        """Where the points are on ``axis``: ρ_pol, or R_mid in metres."""
        return self.rho_pol if axis == "rho_pol" else self.R_mid


@dataclass(eq=False)
class Fit:
    """The stored fit: one profile for the whole window, on a ρ_pol grid.

    ``pe`` and ``R_mid`` are ``None`` where the file lacks them, and then there is no pressure fit, or nothing to draw
    the fit against on the R_mid axis.
    """

    rho_pol: np.ndarray
    ne: np.ndarray  # m^-3
    Te: np.ndarray  # eV
    pe: Optional[np.ndarray]  # Pa
    R_mid: Optional[np.ndarray]  # m, at the window's middle time
    window: Optional[tuple]  # (start, end) in seconds, the window the fit was made over

    def on(self, axis):
        return self.rho_pol if axis == "rho_pol" else self.R_mid

    def values(self, quantity):
        return {"ne": self.ne, "Te": self.Te, "pe": self.pe}[quantity]


@dataclass(eq=False)
class Profiles:
    """What a Thomson scattering file says, cut to the window it was opened over."""

    #: ``{"edge": Points, "core": Points}``, whichever the file holds.
    points: dict
    fit: Optional[Fit]
    #: Laser pulses in the window, as the file's ``time`` axis has them.
    pulses: int
    #: ``R_mid_sep``, ``Te_sep_2pm`` and ``lambda_q_2pm``, those the file holds.
    scalars: dict
    #: The file's attributes, as plain Python: text for text.
    attrs: dict

    def has(self, quantity, axis):
        """Whether the row of ``quantity`` has anything to draw on ``axis``: a point with a place on it, or the fit."""
        for found in self.points.values():
            if np.any(np.isfinite(found.on(axis)) & np.isfinite(found.series(quantity)[0])):
                return True
        fit = self.fit
        return fit is not None and fit.values(quantity) is not None and fit.on(axis) is not None


def _plain(value):
    """An attribute as text, a number or a list: what a netCDF3 file's attributes arrive as, bytes and numpy
    scalars included."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _grid(ds, name, channel):
    """``ds[name]`` as a ``(time, channel)`` float array, or ``None`` when the file lacks it or it is shaped otherwise."""
    if name not in ds:
        return None
    array = ds[name]
    if loader.TIME_DIM not in array.dims or channel not in array.dims:
        return None
    return np.asarray(array.transpose(loader.TIME_DIM, channel).values, dtype=float)


def _vector(ds, name, channel, size):
    """``ds[name]`` as one float per channel, NaN where the file does not say."""
    if name in ds and ds[name].dims == (channel,):
        return np.asarray(ds[name].values, dtype=float)
    return np.full(size, np.nan)


def _points(ds, system):
    """One system's :class:`Points`, or ``None`` when the file holds no ``<system>_ne`` and ``<system>_Te``."""
    channel = f"{system}_channel"
    ne, Te = _grid(ds, f"{system}_ne", channel), _grid(ds, f"{system}_Te", channel)
    if ne is None or Te is None:
        return None
    shape = ne.shape

    def grid(name):
        found = _grid(ds, f"{system}_{name}", channel)
        return np.full(shape, np.nan) if found is None else found

    ne_err, Te_err = grid("ne_err"), grid("Te_err")
    times = np.asarray(ds[loader.TIME_DIM].values, dtype=float)
    channels = np.asarray(ds[channel].values) if channel in ds.coords else np.arange(shape[1])
    with np.errstate(invalid="ignore"):  # NaN compares false, which is the intent
        measured = ne > 0
        Te_measured = measured & (Te > 0)
        ne_err_ok = np.isfinite(ne_err) & (ne_err > 0)
        Te_err_ok = np.isfinite(Te_err) & (Te_err > 0)
    keep = measured.ravel()

    def kept(array):
        return np.broadcast_to(array, shape).ravel()[keep]

    return Points(
        system=system,
        pulses=shape[0],
        readings=int(np.isfinite(ne).sum()),
        time=kept(times[:, None]),
        channel=kept(channels[None, :]),
        R=kept(_vector(ds, f"{system}_R", channel, shape[1])[None, :]),
        Z=kept(_vector(ds, f"{system}_Z", channel, shape[1])[None, :]),
        rho_pol=kept(grid("rho_pol")),
        R_mid=kept(grid("R_mid")),
        ne=kept(ne),
        ne_err=kept(np.where(measured & ne_err_ok, ne_err, np.nan)),
        Te=kept(np.where(Te_measured, Te, np.nan)),
        Te_err=kept(np.where(Te_measured & Te_err_ok, Te_err, np.nan)),
    )


def _window(attrs):
    """The fit's ``(start, end)`` from ``window_t_start`` and ``window_t_end``, or ``None``."""
    try:
        start, end = float(attrs["window_t_start"]), float(attrs["window_t_end"])
    except (KeyError, TypeError, ValueError):
        return None
    return (start, end) if np.isfinite([start, end]).all() else None


def _fit(ds, attrs):
    """The stored fit, or ``None`` for a file whose fit failed (it then has no ``fit_ne``, and no ``rho_pol``)."""
    if "rho_pol" not in ds or "fit_ne" not in ds or "fit_Te" not in ds:
        return None

    def on_grid(name):
        if name in ds and ds[name].dims == ("rho_pol",):
            return np.asarray(ds[name].values, dtype=float)
        return None

    return Fit(
        rho_pol=np.asarray(ds["rho_pol"].values, dtype=float),
        ne=on_grid("fit_ne"),
        Te=on_grid("fit_Te"),
        pe=on_grid("fit_pe"),
        R_mid=on_grid("fit_R_mid"),
        window=_window(attrs),
    )


def read(ds):
    """The :class:`Profiles` of an open Thomson scattering file, cut to whatever window ``ds`` was sliced to.

    Everything is optional but one of the systems' points or the fit: a file that holds only the core, or only the
    edge, or whose fit failed, draws what it has.
    """
    attrs = {name: _plain(value) for name, value in ds.attrs.items()}
    points = {}
    for system in SYSTEMS:
        found = _points(ds, system)
        if found is not None:
            points[system] = found
    return Profiles(
        points=points,
        fit=_fit(ds, attrs),
        pulses=int(ds.sizes.get(loader.TIME_DIM, 0)),
        scalars={name: float(ds[name].values) for name in SCALARS if name in ds},
        attrs=attrs,
    )


# ---------------------------------------------------------------------------
# What there is to say about it
# ---------------------------------------------------------------------------

#: ``fit_check`` ends in ``passed`` or ``failed: <reasons>``; the description before it says "silently failed'" and so
#: never has a colon after its ``failed``.
_FAILED = re.compile(r"(?:^|\s)failed:\s*(.*)$", re.S)


def check_failure(text):
    """The reasons a ``fit_check`` attribute gives for a fit that failed run_db.py's own test.

    ``None`` for a fit that passed, for a file with no fit (``"no fit"``) and for no attribute at all.
    """
    if not text:
        return None
    found = _FAILED.search(str(text))
    return None if found is None else (found.group(1).strip() or "no reason given")


def warning(profiles):
    """The warning above the figure of a fit that fails ``fit_check``, quoting it, or ``None``."""
    check = profiles.attrs.get("fit_check")
    reasons = check_failure(check)
    if reasons is None or profiles.fit is None:
        return None
    return (
        f"**This fit fails run_db.py's own check: {reasons}.** A fit that fails it is not to be trusted in the core, "
        "where it is often an extrapolation: read the points there, not the curve.\n\n"
        f"The file's `fit_check`: “{check}”"
    )


def _settings(attrs):
    """``fit_settings`` as a dict: the file stores it as JSON text."""
    try:
        found = json.loads(attrs.get("fit_settings") or "")
    except (TypeError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def separatrix(profiles, axis):
    """Where the separatrix is on ``axis``, or ``None`` when the file cannot say.

    ρ_pol = 1 by definition. On the R_mid axis it is the file's ``R_mid_sep``, the separatrix's outboard midplane radius
    at the window's middle time, which the fit's own R_mid reaches at ρ_pol = 1 as well; the fit's is used when the
    file holds no scalar.
    """
    if axis == "rho_pol":
        return 1.0
    sep = profiles.scalars.get("R_mid_sep")
    if sep is not None and np.isfinite(sep):
        return float(sep)
    fit = profiles.fit
    if fit is not None and fit.R_mid is not None and fit.rho_pol.min() <= 1.0 <= fit.rho_pol.max():
        return float(np.interp(1.0, fit.rho_pol, fit.R_mid))
    return None


def _span(window):
    return f"{window[0]:g}–{window[1]:g} s"


def _differs(window, target):
    """Whether the window the fit was made over is not the one the page cuts the points to."""
    if window is None or target is None:
        return False
    start, end = getattr(target, "t_start", np.nan), getattr(target, "t_end", np.nan)
    if not np.isfinite([start, end]).all():
        return False
    return not (np.isclose(window[0], start, rtol=0, atol=1e-6) and np.isclose(window[1], end, rtol=0, atol=1e-6))


def notes(profiles, target, axis):
    """The lines under the figure, as markdown, one paragraph each: the points, the fit, the axis, the caveat.

    ``target`` supplies the page's window; ``None`` leaves out whatever depends on it.
    """
    attrs = profiles.attrs
    lines = []

    # The points.
    held = []
    for system in SYSTEMS:
        found = profiles.points.get(system)
        held.append(
            f"no {system} TS in this file" if found is None else f"{system} {found.drawn} of {found.readings} readings"
        )
    unmapped = sum(int(np.isnan(found.on(axis)).sum()) for found in profiles.points.values())
    if not unmapped:
        why = ""
    elif attrs.get("efit_tree") == "none":
        why = f", since {attrs.get('efit') or 'no EFIT tree could be opened'}: they are not drawn"
    else:
        why = " (the pulse is outside the EFIT time base) and are not drawn"
    lines.append(
        f"Points: the {profiles.pulses} laser pulses in the window, " + ", ".join(held) + ". "
        "A reading of 0 is no measurement and is not drawn. The bars are the tree's ±1σ, and the p_e points are "
        "e·n_e·T_e with the errors propagated."
        + (f" {unmapped} of the points have no {AXES[axis][1]}{why}." if unmapped else "")
    )

    # The fit.
    fit = profiles.fit
    if fit is None:
        status = attrs.get("fit_status")
        lines.append(f"No fit in this file ({status})." if status else "No fit in this file.")
    else:
        what = attrs.get("fit_function") or "the file's fit"
        span = (
            f"over {_span(fit.window)}, one profile for the whole window"
            if fit.window is not None
            else "one profile for the whole window"
        )
        text = (
            f"Fit: `{what}` of C-Mod_Analysis, {span}, drawn over its whole range "
            f"(ρ_pol {fit.rho_pol.min():g}–{fit.rho_pol.max():g})."
        )
        if axis == "R_mid" and fit.R_mid is None:
            text += " The file has no R_mid for the fit, so it is not drawn on this axis."
        if _differs(fit.window, target):
            text += (
                f" The page's window is {target.t_start:g}–{target.t_end:g} s, so the points and the fit are not over "
                "the same pulses."
            )
        if _settings(attrs).get("frac_err") is True:
            text += (
                " No band is drawn around it: its n_e and T_e errors are 20% of the fit by construction (`frac_err`), "
                "and `fit_pe_err` adds a density to a temperature, so it is no error of p_e."
            )
        else:
            text += " No band is drawn around it."
        lines.append(text)

    # The axis and the separatrix.
    sep = separatrix(profiles, axis)
    if axis == "rho_pol":
        tree = attrs.get("efit_tree")
        text = (
            "ρ_pol = √ψ_N of each point's channel at its own pulse"
            + (f", from the EFIT tree `{tree}`" if tree and tree != "none" else "")
            + ". The separatrix is ρ_pol = 1."
        )
    else:
        text = (
            "R_mid is the major radius at which each point's flux surface crosses the outboard midplane, from EFIT at the "
            "point's own pulse; the fit and the separatrix are at the window's middle time, so a plasma that moves "
            "within the window shows as scatter against them."
            + (f" The separatrix is at R_mid = {sep:.4f} m." if sep is not None else " The file gives no separatrix.")
        )
    lines.append(text)

    lines.append(str(attrs.get("caveat") or CAVEAT))
    return lines


#: The attributes the "about this file" list quotes, in order: how the file was made and what its fit is.
ABOUT = (
    "created",
    "script_commit",
    "efit",
    "time_alignment",
    "fit_function",
    "fit_settings",
    "fit_reg",
    "fit_note",
    "fit_check",
    "cmod_analysis_commit",
    "cmod_analysis_local_changes",
    "two_point_model",
)


def about(profiles):
    """``[(attribute, text), …]``: what the file says of how it was made, in its own words, for the expander."""
    rows = [(name, str(profiles.attrs[name])) for name in ABOUT if profiles.attrs.get(name) not in (None, "")]
    for name, unit, scale in (("Te_sep_2pm", "eV", 1.0), ("lambda_q_2pm", "mm", 1e3), ("R_mid_sep", "m", 1.0)):
        if name in profiles.scalars:
            rows.append((name, f"{profiles.scalars[name] * scale:.4g} {unit}"))
    return rows


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------


def _rgba(colour, alpha):
    """``#rrggbb`` as an ``rgba(...)`` string."""
    r, g, b = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _hover(points, quantity, scale, unit, values, errors):
    """One hover text per point: which system, when, where, and the value with its error."""
    texts = []
    for i in range(points.drawn):
        value = f"{values[i] * scale:.3g}"
        if np.isfinite(errors[i]):
            value += f" ± {errors[i] * scale:.2g}"
        texts.append(
            f"{LOOKS[points.system]['label']} · t = {points.time[i]:.4f} s · channel {int(points.channel[i])}<br>"
            f"ρ_pol = {points.rho_pol[i]:.3f} · R_mid = {points.R_mid[i]:.4f} m<br>"
            f"R = {points.R[i]:.4f} m, Z = {points.Z[i]:+.4f} m<br>"
            f"{quantity} = {value} {unit}"
        )
    return texts


def _message(text, height=200):
    """An empty figure that says ``text``: what is drawn when there is nothing to draw."""
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def _fit_label(fit):
    return "fit" if fit.window is None else f"fit, {_span(fit.window)}"


def figure(profiles, axis="rho_pol"):
    """n_e, T_e and p_e on one shared ``axis`` (``"rho_pol"`` or ``"R_mid"``), one row each.

    The edge and core points, with their bars; the stored fit over its whole range, without a band; and the separatrix.
    Only what there is is drawn: a row with no point to place and no fit is left out, a system with none is left out
    of the legend, and a file with nothing at all gives a figure that says so. A legend entry appears once, however
    many rows hold its trace, and hides it in all of them.
    """
    if axis not in AXES:
        raise ValueError(f"axis is one of {', '.join(AXES)}, not {axis!r}")
    rows = [quantity for quantity in QUANTITIES if profiles.has(quantity[0], axis)]
    if not rows:
        measured = sum(found.drawn for found in profiles.points.values())
        if measured:
            return _message(
                f"None of the {measured} measured points can be placed on {AXES[axis][1]}, and there is no fit to draw."
            )
        return _message("No Thomson scattering measurement in this window, and no fit in this file.")

    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.05)
    sep, fit = separatrix(profiles, axis), profiles.fit
    for row, (name, label, title, scale, unit) in enumerate(rows, start=1):
        for system in SYSTEMS:
            found = profiles.points.get(system)
            if found is None:
                continue
            values, errors = found.series(name)
            if not np.any(np.isfinite(found.on(axis)) & np.isfinite(values)):
                continue
            look = LOOKS[system]
            fig.add_trace(
                go.Scatter(
                    x=found.on(axis),
                    y=values * scale,
                    mode="markers",
                    name=look["label"],
                    legendgroup=system,
                    showlegend=row == 1,
                    marker=dict(symbol=look["symbol"], size=8, color=look["colour"], opacity=0.85),
                    error_y=dict(
                        type="data",
                        array=errors * scale,
                        visible=True,
                        thickness=1,
                        width=0,
                        color=_rgba(look["colour"], 0.45),
                    ),
                    hovertext=_hover(found, label, scale, unit, values, errors),
                    hoverinfo="text",
                ),
                row=row,
                col=1,
            )
        if fit is not None and fit.values(name) is not None and fit.on(axis) is not None:
            fig.add_trace(
                go.Scatter(
                    x=fit.on(axis),
                    y=fit.values(name) * scale,
                    mode="lines",
                    name=_fit_label(fit),
                    legendgroup="fit",
                    showlegend=row == 1,
                    line=dict(color=FIT_COLOUR, width=2),
                    hovertemplate=(
                        f"{_fit_label(fit)}<br>{AXES[axis][1]} = %{{x:.4g}}<br>{label} = %{{y:.3g}} {unit}<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )
        if sep is not None:
            fig.add_vline(x=sep, line=dict(color=SEPARATRIX_COLOUR, width=1, dash="dash"), row=row, col=1)
        # From zero, and no lower: a bar that reaches below zero ends at the axis, and does not drag the range negative.
        fig.update_yaxes(title_text=title, autorangeoptions=dict(minallowed=0), row=row, col=1)

    if sep is not None:
        # Shapes have no legend entry of their own.
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="separatrix",
                legendgroup="separatrix",
                line=dict(color=SEPARATRIX_COLOUR, width=1, dash="dash"),
            ),
            row=1,
            col=1,
        )
    fig.update_xaxes(title_text=AXES[axis][0], row=len(rows), col=1)
    fig.update_layout(
        height=250 * len(rows) + 110,
        margin=dict(l=70, r=20, t=40, b=60),
        legend=dict(orientation="h", x=0, y=1.0, xanchor="left", yanchor="bottom"),
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def _escape(text):
    """``text`` with markdown's own characters escaped, for attribute text that is not markdown."""
    return re.sub(r"([\\`*_{}\[\]<>$])", r"\\\1", text)


def render(ds, params, target):
    """The warning if the fit fails its check, the figure, the lines under it and what the file says of itself.

    Draws into Streamlit and returns ``None``: the warning goes above the figure and the notes below it.
    """
    import streamlit as st

    profiles = read(ds)
    if not profiles.points and profiles.fit is None:
        st.warning("This file holds no Thomson scattering data to draw.", icon="⚠️")
        return None
    axis = params.axis if params.axis in AXES else "rho_pol"

    text = warning(profiles)
    if text:
        st.warning(text, icon="⚠️")
    st.plotly_chart(figure(profiles, axis), width="stretch")
    for line in notes(profiles, target, axis):
        st.caption(line)
    rows = about(profiles)
    if rows:
        with st.expander("About this file"):
            st.markdown("\n".join(f"- `{name}`: {_escape(value)}" for name, value in rows))
    return None


SPEC = registry.register(
    registry.PlotSpec(
        key="thomson_profiles",
        label="Thomson scattering profiles",
        diagnostics=("thomson_scattering",),
        params=ThomsonProfilesParams,
        render=render,
        preprocessed=False,
        description=(
            "n_e, T_e and p_e of the edge and core Thomson scattering systems against ρ_pol or R_mid: the points in "
            "the window, and the fit the file stores. Read off the file: nothing is computed."
        ),
    )
)
