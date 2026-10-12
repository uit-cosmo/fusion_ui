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
    user, 2026-10-09): with ``frac_err`` the fit's n_e and T_e errors are 20% of the fit by construction, and
    ``fit_pe_err`` adds a density to a temperature, so it is no error of p_e.

    **A row's range is set by the points, not by the fit** (:func:`y_limits`, the rule of the paper's PDF of the density
    scan, ported): it opens on ``[0, top]``, and a fit that runs away leaves through the top instead of squashing the
    points flat against zero. Autoscale returns to that range, and the points above it are counted in the row's corner.

    A fit that does not pass ``fit_check`` gets a warning above the figure that quotes it, and **anything but the
    outcome** ``passed`` **is warned about**: a verdict spelt another way, an outcome nobody wrote a rule for, an empty
    attribute. Its advice follows the clause that failed: the temperature on axis says the core is an extrapolation,
    the temperature at the grid's outer end that the outer end is not credible. A ``fit_note`` that calls the core an
    extrapolation gets a caption of its own, passed or not. The diagnostician's caveat is under every figure.

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
    """The fit's ``(start, end)`` from ``window_t_start`` and ``window_t_end``, or ``None``: a window as the single-shot
    page reads a preprocessed file's (:func:`fusion_ui.core.loader.as_window`)."""
    return loader.as_window([attrs.get("window_t_start"), attrs.get("window_t_end")])


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

#: ``fit_check`` is the rule's description, a colon, then the outcome, ``passed`` or ``failed: <reasons>``. The description
#: ends in a quote ("... must have silently failed'"), so the last quote and colon are where the outcome begins.
_OUTCOME_AFTER = "': "

#: What a failed check says of the fit, by the clause of run_db.py's test that failed, as its reasons name it: a
#: temperature or density too low "on axis", or a temperature too high at the grid's outer end ("rho_pol 1.10").
CHECK_MEANING = (
    (
        "on axis",
        "Its core is an extrapolation, not a description of the core points: read the points there, not the curve.",
    ),
    (
        "rho_pol",
        "Its outer end is not credible: its T_e there can lie several times above the points, or run on past the last "
        "one. Read the points, not the curve, near the separatrix and beyond.",
    ),
)
#: What it says of an outcome that names neither clause.
CHECK_UNRECOGNISED = "It may have failed silently: read the points, not the curve."


def check_outcome(text):
    """What a ``fit_check`` attribute records of run_db.py's own test: ``"passed"``, ``"failed: <reasons>"`` as the
    file writes it, or whatever else it says.

    ``""`` for no attribute, an empty one and ``"no fit"``, the words of a file that has none. A file with a fit has
    one of the first two, so for it ``""`` means that no outcome was recorded.
    """
    text = "" if text is None else str(text).strip()
    if text in ("", "no fit"):
        return ""
    return text.rsplit(_OUTCOME_AFTER, 1)[-1].strip()


def check_failed(profiles):
    """Whether the fit that is drawn did not pass its check.

    **Anything but the outcome** ``passed`` **counts**: a verdict spelt another way ("FAILED: ..."), an outcome this page
    does not know, an empty or missing attribute. A check that cannot be read must not pass for one that was.
    """
    return profiles.fit is not None and check_outcome(profiles.attrs.get("fit_check")) != "passed"


def check_meaning(outcome):
    """What a failed check says of the fit, in sentences of advice: one for each clause that failed, or that it may have
    failed silently when the outcome is not a verdict ``failed: <reasons>`` that names one.

    Only the reasons of a verdict are read for clauses: the rule's description, which a file may record without an
    outcome, says "on axis" too.
    """
    reasons = outcome if outcome.lower().startswith("failed") else ""
    said = [text for clause, text in CHECK_MEANING if clause in reasons]
    return " ".join(said) or CHECK_UNRECOGNISED


def warning(profiles):
    """The warning above the figure of a fit that did not pass ``fit_check`` (:func:`check_failed`), quoting it, or ``None``.

    What it advises follows the clause that failed (:func:`check_meaning`): a core that is an extrapolation, or an outer
    end that is not credible. A fit whose outcome is empty, or is not one this page knows, is warned about too.
    """
    if not check_failed(profiles):
        return None
    check = profiles.attrs.get("fit_check")
    outcome = check_outcome(check)
    verb = "fails" if outcome.lower().startswith("failed") else "does not pass"
    said = f"“{outcome}”" if outcome else "the file records no result"
    text = f"**This fit {verb} run_db.py's own check: {said}.** {check_meaning(outcome)}\n\n"
    if check is None:
        return text + "The file has no `fit_check`."
    if not str(check).strip():
        return text + "The file's `fit_check` is empty."
    return text + f"The file's `fit_check`: “{check}”"


def extrapolation(attrs):
    """The sentence of ``fit_note`` that calls the fit's core an extrapolation, or ``""`` when it does not.

    The file says so when it finds no core T_e of 1 keV or more inside ρ_pol 0.2: the fit's pre-filter drops the lower
    ones, so the fit has none to rest on there, whether or not its check passed.
    """
    sentences = re.split(r"(?<=\.)\s+", str(attrs.get("fit_note") or ""))
    return " ".join(sentence for sentence in sentences if "extrapolation" in sentence)


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
        found = float(np.interp(1.0, fit.rho_pol, fit.R_mid))
        return found if np.isfinite(found) else None  # a fit_R_mid that is NaN there says nothing
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
        why = f", since {_escape(str(attrs.get('efit') or 'no EFIT tree could be opened'))}: they are not drawn"
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
        status = attrs.get("fit_status")  # an exception's text, which may hold markdown's own characters
        lines.append(f"No fit in this file ({_escape(str(status))})." if status else "No fit in this file.")
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
        # Whether or not the check passed: a core with no T_e of 1 keV or more has none for the fit to rest on.
        core = extrapolation(attrs)
        if core:
            lines.append(
                f"The fit's core is an extrapolation: the file's `fit_note` says “{core}” Read the points there, not "
                "the curve."
            )

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
# The range of a row
# ---------------------------------------------------------------------------

#: How the top of a row is set, and how far a fit may stray from the points beside it before it stops counting. **This is
#: the rule of the paper's PDF of the density scan, ported**: ``pool``, ``plausible`` and ``y_limits`` of fusion_scripts'
#: ``density_scan/thomson_profiles.py`` (J12c), with the constants it names the same way, so that the page and the PDF
#: draw the same range. It is not imported, since nothing here imports from fusion_scripts: a change to the rule is made
#: in both. The PDF's ``EDGE_PERCENTILE`` is not ported, because the page has no row zoomed on the edge.
#:
#: ``WELL_MEASURED``
#:     A point is measured to better than 2σ when its error is under this fraction of its value.
#: ``MIN_WELL``
#:     Only the well-measured points set the scale, unless fewer than this many are: then all of them do.
#: ``WHOLE_PERCENTILE``
#:     The percentile of those points that sets the scale, so that a stray point does not.
#: ``NEAREST``, ``FIT_FACTOR``
#:     A sample of the fit counts only where it is within ``FIT_FACTOR`` times the median of the ``NEAREST`` points
#:     nearest to it on the axis drawn.
#: ``MARGIN``
#:     The top is this many times what sets it.
WELL_MEASURED, MIN_WELL, WHOLE_PERCENTILE = 0.5, 5, 98
NEAREST, FIT_FACTOR, MARGIN = 10, 3.0, 1.1


def pool(profiles, quantity, axis, scale=1.0):
    """All the points of the row of ``quantity`` on ``axis``, edge and core together: ``(x, value, error)``.

    Those the figure draws: a point with a place on ``axis`` and a value. The values and errors are in the row's own unit,
    the SI ones times ``scale``, and ``error`` is NaN where the tree gives none (:class:`Points`). Empty arrays when
    there is no such point.
    """
    parts = []
    for system in SYSTEMS:
        found = profiles.points.get(system)
        if found is None:
            continue
        values, errors = found.series(quantity)
        x = found.on(axis)
        shown = np.isfinite(x) & np.isfinite(values)
        parts.append((x[shown], values[shown] * scale, errors[shown] * scale))
    if not parts:
        return np.empty(0), np.empty(0), np.empty(0)
    return tuple(np.concatenate(array) for array in zip(*parts))


def plausible(fit_x, fit, x, value):
    """Which samples of a fit stay within ``FIT_FACTOR`` times the median of the ``NEAREST`` points nearest to them on the
    axis: a fit is held to the data beside it, wherever the data are. All of them when there are no points."""
    if not x.size:
        return np.ones(fit.shape, dtype=bool)
    k = min(NEAREST, x.size)
    nearest = np.argpartition(np.abs(fit_x[:, None] - x[None, :]), k - 1, axis=1)[:, :k]
    return np.abs(fit) <= FIT_FACTOR * np.abs(np.median(value[nearest], axis=1))


def y_limits(profiles, quantity, axis, scale=1.0):
    """``(0.0, top)`` of the row of ``quantity`` on ``axis``, in the row's own unit (the SI values times ``scale``), or
    ``None`` when nothing in the row is above zero.

    ``top`` is ``MARGIN`` times the larger of two things. One is the ``WHOLE_PERCENTILE`` percentile of the points
    measured to better than 2σ (error under ``WELL_MEASURED`` of the value), or of all of them when fewer than
    ``MIN_WELL`` are. The other is the largest value of the fit where :func:`plausible` says it stays near the points, or
    of the whole fit when the row has none. So a fit that runs away leaves through the top instead of rescaling the row,
    and the points and the fit are judged on ``axis``: the same rule gives another top on R_mid than on ρ_pol, where the
    points beside a sample of the fit are other points. The bottom is zero, as n_e, T_e and p_e are not negative.
    """
    x, value, error = pool(profiles, quantity, axis, scale)
    values = []
    fit = profiles.fit
    if fit is not None and fit.values(quantity) is not None and fit.on(axis) is not None:
        fit_x, fit_y = fit.on(axis), fit.values(quantity) * scale
        shown = np.isfinite(fit_x) & np.isfinite(fit_y)  # a sample with no place or no value is not drawn
        fit_x, fit_y = fit_x[shown], fit_y[shown]
        values += list(fit_y[plausible(fit_x, fit_y, x, value)])
    if x.size:
        with np.errstate(invalid="ignore"):  # a point with no error is not well measured: NaN compares false
            good = error < WELL_MEASURED * value
        values.append(np.percentile(value[good] if good.sum() >= MIN_WELL else value, WHOLE_PERCENTILE))
    top = MARGIN * max([0.0, *values])
    return (0.0, float(top)) if top > 0 else None


def points_above(profiles, quantity, axis, top, scale=1.0):
    """How many of the row's drawn points lie above ``top``, which its range leaves out: the ones the row says it counts."""
    value = pool(profiles, quantity, axis, scale)[1]
    return int((value > top).sum())


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------


def _rgba(colour, alpha):
    """``#rrggbb`` as an ``rgba(...)`` string."""
    r, g, b = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _number(value, digits):
    """``value`` to ``digits`` significant digits, and no exponent below a million: a core T_e of 2140 eV reads ``2140``,
    not ``2.14e+03``, and its error of 234 reads ``234``, not ``2.3e+02``."""
    text = f"{value:.{digits}g}"
    if "e" in text and 1 <= abs(value) < 1e6:
        return f"{value:.0f}"
    return text


def _hover(points, quantity, scale, unit, values, errors):
    """One hover text per point: which system, when, where, and the value with its error."""
    texts = []
    for i in range(points.drawn):
        value = _number(values[i] * scale, 3)
        if np.isfinite(errors[i]):
            value += f" ± {_number(errors[i] * scale, 2)}"
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
    many rows hold its trace or its line (the separatrix is a line in every row), and hides it in all of them.

    Every row runs from zero, as n_e, T_e and p_e are not negative, to a top that **the points set and a fit that runs
    away cannot** (:func:`y_limits`, the rule of the density scan's PDF). The top is 1.1 times the larger of the 98th
    percentile of the row's points measured to better than 2σ (all of them when fewer than 5 are) and the fit's largest
    value where it stays within 3 times the median of the 10 points nearest it on the axis drawn, so a runaway fit
    leaves through the top instead of squashing the points against zero. That is why the range is not left to
    autorange: one such fit (1091216034's n_e reaches 1.3e23 m⁻³, 1110201014's T_e 51 keV) set the range and left the
    points flat against zero. The rows open on that range, and autoscale (a double-click, the modebar) returns to it. A
    row says how many of its points lie above its top ("▲ 3 points above") in its corner, and says nothing when none
    does. Error bars are not part of the rule, as in the PDF, so a bar can leave through the top as well. A bar that
    reaches below zero ends at the axis.
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
                        f"{_fit_label(fit)}<br>{AXES[axis][1]} = %{{x:.4g}}<br>{label} = %{{y:.4~g}} {unit}<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )
        if sep is not None:
            # A line of each row, one legend entry for them all: clicking it hides the separatrix everywhere.
            fig.add_vline(
                x=sep,
                line=dict(color=SEPARATRIX_COLOUR, width=1, dash="dash"),
                name="separatrix",
                legendgroup="separatrix",
                showlegend=row == 1,
                row=row,
                col=1,
            )
        limits = y_limits(profiles, name, axis, scale)
        if limits is None:
            # Nothing above zero to set a top by. From zero, and no lower: a bar that reaches below zero ends at the axis,
            # and does not drag the range negative.
            fig.update_yaxes(title_text=title, autorangeoptions=dict(minallowed=0), row=row, col=1)
            continue
        # From zero to the top the points set, which a fit that runs away cannot move. The range is given outright, so the
        # row opens on it, and as ``autorangeoptions`` too, so that autoscale (a double-click, the modebar) returns to it
        # and not to the runaway fit: ``minallowed`` and ``maxallowed`` are used exactly as the autorange's ends.
        bottom, top = limits
        fig.update_yaxes(
            title_text=title,
            range=[bottom, top],
            autorangeoptions=dict(minallowed=bottom, maxallowed=top),
            row=row,
            col=1,
        )
        above = points_above(profiles, name, axis, top, scale)
        if above:  # nothing is left out silently: the points the top leaves out are counted, in the row's corner
            fig.add_annotation(
                text=f"▲ {above} point{'s' * (above > 1)} above",
                xref="paper",
                yref=f"{'y' if row == 1 else f'y{row}'} domain",
                x=0.99,
                y=0.985,
                xanchor="right",
                yanchor="top",
                showarrow=False,
                font=dict(size=11),
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
