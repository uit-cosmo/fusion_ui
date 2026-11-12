"""The Thomson scattering profiles view: n_e, T_e and p_e against ρ_pol or R_mid.

A live spec: ``thomson_scattering_<shot>.nc`` already holds the measurements and the fit, and this view only draws
them. What matters about the file (netCDF3, ``time`` in seconds):

- ``<sys>_ne``, ``<sys>_Te`` and their ``_err`` by ``(time, <sys>_channel)`` for the edge and the core system, with
  ``<sys>_rho_pol`` and ``<sys>_R_mid`` of each point from EFIT at its own pulse (NaN outside the EFIT time base).
  **A 0 is no measurement**, and its error is junk: only the points with n_e > 0 are drawn.
- ``rho_pol`` and, on it, ``fit_ne``, ``fit_Te``, ``fit_pe`` and ``fit_R_mid``: the fit over the discharge window,
  only where it worked. It is drawn over its whole range and without a band (its stored errors are 20% of the fit by
  construction, and ``fit_pe_err`` is no error of p_e).
- ``fit_rho_shift``: how far the fit, and its own copy of the points, were moved in ρ_pol so that its T_e is
  ``Te_sep_fit`` at ρ_pol 1. Where it is not 0 the points are drawn moved with the fit, where the fit saw them
  (:meth:`Points.on`), and the separatrix is the fit's (:func:`separatrix`). A file without it draws every point
  where EFIT puts it.

A row's range is set by the points, not by the fit (:func:`y_limits`), so a fit that runs away leaves through the top.
The warnings (:func:`warning`, :func:`points_warning`) and the lines under the figure (:func:`notes`) quote what the
file says of its own fit.

On request the Mirror Langmuir Probe's profiles are drawn over these (:func:`overlay`), from the shot's ASP file,
which the spec names as a companion. The probe's data are read only while the overlay is on.

Everything but :func:`render` and :func:`mlp_controls` is pure.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fusion_ui.core import loader, mlp, params_ui, registry

SYSTEMS = ("edge", "core")

#: One electron-volt in joules: n_e [m^-3] times T_e [eV] times this is p_e [Pa].
ELEMENTARY_CHARGE = 1.602176634e-19

#: The scalars the file may hold. ``Te_sep_fit`` is only in a file whose fit was moved to a separatrix it found.
SCALARS = ("R_mid_sep", "Te_sep_2pm", "Te_sep_fit", "lambda_q_2pm")

#: The caveat shown under a file that has no ``caveat`` attribute of its own: it is never left out.
CAVEAT = (
    "From the C-Mod_Analysis README: it is highly recommended to talk with the diagnostician who collected the data "
    "before using it in a public forum. The Thomson scattering data were collected by Jerry Hughes "
    "(jwhughes@psfc.mit.edu)."
)

#: How each system is drawn: colour and marker both say which it is. ``bar`` is the colour of its error bars.
LOOKS = {
    "edge": dict(label="edge TS", colour="#2a78d6", bar="rgba(42,120,214,0.45)", symbol="circle"),
    "core": dict(label="core TS", colour="#d95926", bar="rgba(217,89,38,0.45)", symbol="diamond"),
}
FIT_COLOUR = "#199e70"
SEPARATRIX_COLOUR = "#898781"

#: The two axes: the option, its axis title and what the hover calls it.
AXES = {
    "rho_pol": ("ρ<sub>pol</sub>", "ρ_pol"),
    "R_mid": ("R<sub>mid</sub> [m]", "R_mid"),
}

#: The three rows: the file's name, the hover's name, the axis title, the factor from SI to what is drawn, the unit.
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


@dataclass(eq=False)  # arrays do not compare as a whole
class Points:
    """One system's measurements in the window, one entry per kept point (n_e > 0), in pulse order.

    ``pulses`` and ``readings`` count what the window held. ``rho_pol`` and ``R_mid`` are where EFIT puts each point;
    :meth:`on` says where it is drawn.
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
    rho_shift: float = 0.0  # how far the fit moved the points in ρ_pol (``fit_rho_shift``)
    R_mid_moved: Optional[np.ndarray] = None  # m, the fit's own R_mid of the moved ρ_pol; None for points not moved

    @property
    def drawn(self):
        return int(self.time.size)

    @property
    def moved(self):
        return bool(self.rho_shift)

    def pe(self):
        """p_e in Pa, from the points: e·n_e·T_e."""
        return ELEMENTARY_CHARGE * self.ne * self.Te

    def pe_err(self):
        """The propagated error of :meth:`pe`: NaN where either error is."""
        return ELEMENTARY_CHARGE * np.hypot(self.Te * self.ne_err, self.ne * self.Te_err)

    def series(self, quantity):
        """``(values, errors)`` of ``"ne"``, ``"Te"`` or ``"pe"`` at the kept points, in SI units."""
        if quantity == "ne":
            return self.ne, self.ne_err
        if quantity == "Te":
            return self.Te, self.Te_err
        return self.pe(), self.pe_err()

    def on(self, axis):
        """Where the points are drawn on ``axis``: where EFIT puts them, or, moved with the fit, where the fit saw them
        (ρ_pol plus the shift, and the fit's own R_mid of that)."""
        if not self.moved:
            return self.rho_pol if axis == "rho_pol" else self.R_mid
        if axis == "rho_pol":
            return self.rho_pol + self.rho_shift
        return self.R_mid_moved if self.R_mid_moved is not None else np.full(self.rho_pol.shape, np.nan)


@dataclass(eq=False)
class Fit:
    """The stored fit: one profile for the whole window, on a ρ_pol grid. ``pe`` and ``R_mid`` are ``None`` where the
    file lacks them."""

    rho_pol: np.ndarray
    ne: np.ndarray  # m^-3
    Te: np.ndarray  # eV
    pe: Optional[np.ndarray]  # Pa
    R_mid: Optional[np.ndarray]  # m, at the window's middle time
    window: Optional[tuple]  # (start, end) in seconds, the window the fit was made over
    #: The file's ``fit_rho_shift``: ``rho_pol`` is then the moved grid. ``None`` in a file without one.
    rho_shift: Optional[float] = None

    @property
    def moved(self):
        return self.rho_shift is not None and bool(np.isfinite(self.rho_shift)) and self.rho_shift != 0.0

    def on(self, axis):
        return self.rho_pol if axis == "rho_pol" else self.R_mid

    def values(self, quantity):
        return {"ne": self.ne, "Te": self.Te, "pe": self.pe}[quantity]

    def R_mid_of(self, rho_pol):
        """The fit's own R_mid of ``rho_pol`` (``fit_R_mid``, interpolated): NaN outside its grid or without one."""
        rho_pol = np.asarray(rho_pol, dtype=float)
        if self.R_mid is None:
            return np.full(rho_pol.shape, np.nan)
        return np.interp(rho_pol, self.rho_pol, self.R_mid, left=np.nan, right=np.nan)


@dataclass(eq=False)
class Profiles:
    """What a Thomson scattering file says, cut to the window it was opened over."""

    points: dict  # {"edge": Points, "core": Points}, whichever the file holds
    fit: Optional[Fit]
    pulses: int  # laser pulses in the window
    scalars: dict  # those of SCALARS the file holds
    attrs: dict  # the file's attributes, as plain Python

    def has(self, quantity, axis):
        """Whether the row of ``quantity`` has anything to draw on ``axis``: a point with a place on it, or the fit."""
        for found in self.points.values():
            if np.any(np.isfinite(found.on(axis)) & np.isfinite(found.series(quantity)[0])):
                return True
        fit = self.fit
        return fit is not None and fit.values(quantity) is not None and fit.on(axis) is not None


def _plain(value):
    """An attribute as text, a number or a list, whatever netCDF3 hands back."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _grid(ds, name, channel):
    """``ds[name]`` as a ``(time, channel)`` float array, or ``None``."""
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


def _fit(ds, attrs):
    """The stored fit, or ``None`` for a file whose fit failed."""
    if "rho_pol" not in ds or "fit_ne" not in ds or "fit_Te" not in ds:
        return None

    def on_grid(name):
        if name in ds and ds[name].dims == ("rho_pol",):
            return np.asarray(ds[name].values, dtype=float)
        return None

    shift = ds["fit_rho_shift"] if "fit_rho_shift" in ds else None
    return Fit(
        rho_pol=np.asarray(ds["rho_pol"].values, dtype=float),
        ne=on_grid("fit_ne"),
        Te=on_grid("fit_Te"),
        pe=on_grid("fit_pe"),
        R_mid=on_grid("fit_R_mid"),
        window=loader.as_window([attrs.get("window_t_start"), attrs.get("window_t_end")]),
        rho_shift=float(shift.values) if shift is not None and shift.ndim == 0 else None,
    )


def read(ds):
    """The :class:`Profiles` of an open Thomson scattering file, over whatever window ``ds`` was cut to. Either system
    and the fit are optional. Where the fit was moved to a separatrix, the points are moved with it."""
    attrs = {name: _plain(value) for name, value in ds.attrs.items()}
    fit = _fit(ds, attrs)
    points = {}
    for system in SYSTEMS:
        found = _points(ds, system)
        if found is not None:
            if fit is not None and fit.moved:
                found.rho_shift = float(fit.rho_shift)
                found.R_mid_moved = fit.R_mid_of(found.rho_pol + found.rho_shift)
            points[system] = found
    return Profiles(
        points=points,
        fit=fit,
        pulses=int(ds.sizes.get(loader.TIME_DIM, 0)),
        scalars={name: float(ds[name].values) for name in SCALARS if name in ds},
        attrs=attrs,
    )


# ---------------------------------------------------------------------------
# What there is to say about it
# ---------------------------------------------------------------------------

#: ``fit_check`` is the rule's description, a colon, then the outcome, ``passed`` or ``failed: <reasons>``. The
#: description ends in a quote, so the last quote and colon are where the outcome begins.
_OUTCOME_AFTER = "': "

#: What a failed check says of the fit, by the clause its reasons name.
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
    """The outcome a ``fit_check`` attribute records: ``"passed"``, ``"failed: <reasons>"``, or whatever else follows
    the description. ``""`` for no attribute, an empty one and ``"no fit"``."""
    text = "" if text is None else str(text).strip()
    if text in ("", "no fit"):
        return ""
    return text.rsplit(_OUTCOME_AFTER, 1)[-1].strip()


def warning(profiles):
    """The warning above the figure of a fit that did not pass ``fit_check``, quoting it, or ``None``.

    Anything but the outcome ``passed`` is warned about, an empty or unreadable one too: a check that cannot be read
    must not pass for one that was. The advice follows the clauses a ``failed`` verdict names; the description alone
    says "on axis" too, so only a verdict's reasons are read for them.
    """
    check = profiles.attrs.get("fit_check")
    outcome = check_outcome(check)
    if profiles.fit is None or outcome == "passed":
        return None
    failed = outcome.lower().startswith("failed")
    advice = " ".join(text for clause, text in CHECK_MEANING if failed and clause in outcome) or CHECK_UNRECOGNISED
    said = f"“{outcome}”" if outcome else "the file records no result"
    text = f"**This fit {'fails' if failed else 'does not pass'} run_db.py's own check: {said}.** {advice}\n\n"
    if check is None:
        return text + "The file has no `fit_check`."
    if not str(check).strip():
        return text + "The file's `fit_check` is empty."
    return text + f"The file's `fit_check`: “{check}”"


def points_warning(profiles):
    """The warning above the figure when ``fit_points_check`` does not begin with ``equal``, quoting it, or ``None``.

    The attribute says whether the edge points the fit was made on are this file's. A file without the attribute, or
    without a fit, gets no warning.
    """
    if profiles.fit is None or "fit_points_check" not in profiles.attrs:
        return None
    said = str(profiles.attrs["fit_points_check"] or "").strip()
    if said.startswith("equal"):
        return None
    quoted = f"says “{_escape(said)}”" if said else "is empty"
    return (
        "**The points drawn may not be where the fit saw them.** The file's `fit_points_check`, which compares the "
        f"edge points the fit was made on with this file's, {quoted}."
    )


def _sentences(text, word):
    """The sentences of ``text`` that hold ``word``, joined, or ``""``."""
    sentences = re.split(r"(?<=\.)\s+", str(text or ""))
    return " ".join(sentence for sentence in sentences if word in sentence)


def extrapolation(attrs):
    """The sentence of ``fit_note`` that calls the fit's core an extrapolation, or ``""``. The file says so when the
    fit had no core T_e to rest on, whether or not its check passed."""
    return _sentences(attrs.get("fit_note"), "extrapolation")


def move_note(profiles):
    """The caption of a fit moved to a separatrix, or ``""``: by how much, to which T_e,sep, and the file's own
    ``fit_separatrix``, quoted."""
    fit = profiles.fit
    if fit is None or not fit.moved:
        return ""
    te_sep = profiles.scalars.get("Te_sep_fit")
    to = (
        f"so that its T_e is T_e,sep = {te_sep:.1f} eV (`Te_sep_fit`) at ρ_pol = 1"
        if te_sep is not None and np.isfinite(te_sep)
        else "though the file records no T_e,sep that it was moved to"
    )
    text = (
        "The fit was moved to a separatrix, and the points are drawn moved with it, where the fit saw them: "
        f"C-Mod_Analysis moved the fit, and its own copy of the points, by {fit.rho_shift:+.4f} in ρ_pol, {to}."
    )
    said = profiles.attrs.get("fit_separatrix")
    if said:
        text += f" The file's `fit_separatrix`: “{_escape(str(said))}”"
    return text


def held_note(attrs):
    """The caption of a fit whose outer side fictitious points hold (``force_to_zero`` in ``fit_settings``), quoting what
    ``fit_note`` says of them, or ``""``."""
    if _settings(attrs).get("force_to_zero") is not True:
        return ""
    text = (
        "Fictitious points hold the outer side of the n_e and T_e fits (`force_to_zero`), so there the fit says nothing "
        "measured. The p_e fit is not held, and can run away in the SOL."
    )
    said = _sentences(attrs.get("fit_note"), "force_to_zero")
    if said:
        text += f" The file's `fit_note` says “{_escape(said)}”"
    return text


def _settings(attrs):
    """``fit_settings`` as a dict: the file stores it as JSON text."""
    try:
        found = json.loads(attrs.get("fit_settings") or "")
    except (TypeError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def separatrix(profiles, axis):
    """Where the separatrix is on ``axis``, or ``None`` when the file cannot say.

    ρ_pol = 1 by definition. On R_mid it is the file's ``R_mid_sep``, else the fit's own R_mid at ρ_pol 1. For a fit
    moved to a separatrix the order is the other way round: the fit is drawn on its own axis, so its own R_mid at its
    ρ_pol 1 comes first, where its T_e is the T_e,sep it was moved to.
    """
    if axis == "rho_pol":
        return 1.0
    fit = profiles.fit
    if fit is not None and fit.moved:
        found = float(fit.R_mid_of(1.0))
        if np.isfinite(found):
            return found
    sep = profiles.scalars.get("R_mid_sep")
    if sep is not None and np.isfinite(sep):
        return float(sep)
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
    """The lines under the figure, as markdown, one paragraph each: the points, the fit (with what the file says of
    its core, its move and its fictitious points), the axis, the caveat.

    The points with no place on ``axis`` are counted, with what the file says of why. ``target`` supplies the page's
    window; ``None`` leaves out whatever depends on it.
    """
    attrs = profiles.attrs
    lines = []
    fit = profiles.fit
    moved = fit is not None and fit.moved

    # The points.
    held = []
    for system in SYSTEMS:
        found = profiles.points.get(system)
        held.append(
            f"no {system} TS in this file" if found is None else f"{system} {found.drawn} of {found.readings} readings"
        )
    beyond = gap = 0
    if moved:
        # A moved point has a place wherever EFIT gave it a ρ_pol, but on R_mid only where the fit's own R_mid reaches.
        unmapped = sum(int(np.isnan(found.rho_pol).sum()) for found in profiles.points.values())
        if axis == "R_mid" and fit.R_mid is not None:
            low, high = np.nanmin(fit.rho_pol), np.nanmax(fit.rho_pol)
            for found in profiles.points.values():
                lost = np.isfinite(found.rho_pol) & np.isnan(found.on(axis))
                outside = (found.on("rho_pol") < low) | (found.on("rho_pol") > high)
                beyond += int((lost & outside).sum())
                gap += int((lost & ~outside).sum())
    else:
        unmapped = sum(int(np.isnan(found.on(axis)).sum()) for found in profiles.points.values())
    failure = attrs.get("mapping_failure")  # only in a file whose mapping of the points failed somewhere
    if not unmapped:
        why = ""
    elif attrs.get("efit_tree") == "none":
        why = f", since {_escape(str(attrs.get('efit') or 'no EFIT tree could be opened'))}: they are not drawn"
    elif failure not in (None, ""):
        why = f" and are not drawn: the file's `mapping_failure` says “{_escape(str(failure))}”"
    else:
        why = " (the pulse is outside the EFIT time base) and are not drawn"
    lines.append(
        f"Points: the {profiles.pulses} laser pulses in the window, " + ", ".join(held) + ". "
        "A reading of 0 is no measurement and is not drawn. The bars are the tree's ±1σ, and the p_e points are "
        "e·n_e·T_e with the errors propagated."
        + (f" {unmapped} of the points have no {AXES[axis][1]}{why}." if unmapped else "")
        + (
            f" {beyond} of the points moved with the fit lie beyond its grid, where `fit_R_mid` gives them no R_mid, "
            "and are not drawn."
            if beyond
            else ""
        )
        + (f" {gap} of the points moved with the fit fall where `fit_R_mid` is NaN, and are not drawn." if gap else "")
    )

    # The fit.
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
            text += (
                " The file has no R_mid for the fit, so neither it nor the points moved with it are drawn on this axis."
                if moved
                else " The file has no R_mid for the fit, so it is not drawn on this axis."
            )
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
        core = extrapolation(attrs)
        if core:
            lines.append(
                f"The fit's core is an extrapolation: the file's `fit_note` says “{core}” Read the points there, not "
                "the curve."
            )
        lines += [line for line in (move_note(profiles), held_note(attrs)) if line]

    # The axis and the separatrix.
    sep = separatrix(profiles, axis)
    if axis == "rho_pol":
        tree = attrs.get("efit_tree")
        text = (
            "ρ_pol = √ψ_N of each point's channel at its own pulse"
            + (f", from the EFIT tree `{tree}`" if tree and tree != "none" else "")
            + (
                f", moved by {fit.rho_shift:+.4f} with the fit. The separatrix is the fit's ρ_pol = 1."
                if moved
                else ". The separatrix is ρ_pol = 1."
            )
        )
    elif moved:
        text = (
            "R_mid is the major radius at which a flux surface crosses the outboard midplane. The points are moved with "
            "the fit, so each is drawn at the fit's own R_mid of its moved ρ_pol (`fit_R_mid`, from EFIT at the window's "
            "middle time), and not at its own pulse."
            + (
                f" The separatrix, the fit's ρ_pol = 1, is at R_mid = {sep:.4f} m."
                if sep is not None
                else " The file gives no separatrix."
            )
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
    "fit_separatrix",
    "separatrix_code",
)

#: The file's numbers the list gives, with the unit and the factor each is shown in.
NUMBERS = (("Te_sep_2pm", "eV", 1.0), ("Te_sep_fit", "eV", 1.0), ("lambda_q_2pm", "mm", 1e3), ("R_mid_sep", "m", 1.0))


def separatrix_code(attrs):
    """What the "about" list says of the code that looked for the separatrix: its folder, that no commit pins it, and
    whether the file keeps the sha256 of its files (``separatrix_code_sha256``)."""
    try:
        files = json.loads(attrs.get("separatrix_code_sha256") or "")
    except (TypeError, ValueError):
        files = None
    kept = (
        f"the file keeps the sha256 of each of the {len(files)} files of it that were loaded, in separatrix_code_sha256"
        if isinstance(files, dict) and files
        else "the file records no sha256 of it"
    )
    return f"{attrs.get('separatrix_code')}: outside C-Mod_Analysis' repository, so no commit pins it; {kept}."


def about(profiles):
    """``[(name, text), …]`` for the expander: what the file says of how it was made, then its numbers."""
    attrs = profiles.attrs
    rows = [
        (name, separatrix_code(attrs) if name == "separatrix_code" else str(attrs[name]))
        for name in ABOUT
        if attrs.get(name) not in (None, "")
    ]
    for name, unit, scale in NUMBERS:
        if name in profiles.scalars:
            rows.append((name, f"{profiles.scalars[name] * scale:.4g} {unit}"))
    fit = profiles.fit
    if fit is not None and fit.rho_shift is not None:
        shift = fit.rho_shift
        rows.append(("fit_rho_shift", f"{shift:+.4g} in ρ_pol" if shift else "0: the fit was not moved"))
    return rows


# ---------------------------------------------------------------------------
# The range of a row
# ---------------------------------------------------------------------------

#: The rule that sets the top of a row (:func:`y_limits`).
#: A point is well measured when its error is under ``WELL_MEASURED`` of its value (better than 2σ). Only those set
#: the scale, by their ``WHOLE_PERCENTILE`` percentile, unless fewer than ``MIN_WELL`` are: then all the points do.
#: A sample of the fit counts only where it is within ``FIT_FACTOR`` times the median of the ``NEAREST`` points nearest
#: to it on the axis drawn. The top is ``MARGIN`` times what sets it.
WELL_MEASURED, MIN_WELL, WHOLE_PERCENTILE = 0.5, 5, 98
NEAREST, FIT_FACTOR, MARGIN = 10, 3.0, 1.1


def pool(profiles, quantity, axis, scale=1.0):
    """``(x, value, error)`` of the points the row of ``quantity`` draws on ``axis``, edge and core together, in the
    row's own unit (SI times ``scale``)."""
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
    """Which samples of a fit stay within ``FIT_FACTOR`` times the median of the ``NEAREST`` points nearest to them on
    the axis. All of them when there are no points."""
    if not x.size:
        return np.ones(fit.shape, dtype=bool)
    k = min(NEAREST, x.size)
    nearest = np.argpartition(np.abs(fit_x[:, None] - x[None, :]), k - 1, axis=1)[:, :k]
    return np.abs(fit) <= FIT_FACTOR * np.abs(np.median(value[nearest], axis=1))


def y_limits(profiles, quantity, axis, scale=1.0):
    """``(0.0, top)`` of the row of ``quantity`` on ``axis``, in the row's own unit, or ``None`` when nothing in the
    row is above zero.

    ``top`` is ``MARGIN`` times the larger of the percentile of the well-measured points and the largest value of the
    fit where it stays near the points (:func:`plausible`). So a fit that runs away leaves through the top instead of
    squashing the points against zero. Both are judged on ``axis``, where the points beside a sample of the fit can be
    other points than on the other axis.
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
    """How many of the row's drawn points lie above ``top``."""
    value = pool(profiles, quantity, axis, scale)[1]
    return int((value > top).sum())


# ---------------------------------------------------------------------------
# The Mirror Langmuir Probe's profiles, drawn over the Thomson scattering ones
# ---------------------------------------------------------------------------

#: The diagnostic whose file the spec draws beside its own.
COMPANION = "asp"

#: How the MLP's points are drawn, with the markers of the MLP profiles plot: a triangle pointing the way the probe
#: ran, filled on the way in and open on the way out. The colour is near the core's, but the two never meet: the core
#: system reaches no further out than ρ_pol 0.9.
MLP_LOOK = dict(label="MLP", colour="#c98500", symbols={"in": "triangle-left", "out": "triangle-right-open"})

#: What the MLP's profile can be moved to, by the file's own name for the temperature.
MOVES = (
    ("none", "not moved: EFIT's mapping"),
    ("Te_sep_2pm", "the two-point model's"),
    ("Te_sep_fit", "the fit's"),
)


def separatrix_choices(profiles):
    """``[(name, te_sep, label), …]``: ``none`` (0 eV, not moved) first, then ``Te_sep_2pm`` and ``Te_sep_fit``, each
    where the file holds it as a positive number."""
    found = [("none", 0.0, MOVES[0][1])]
    for name, said in MOVES[1:]:
        value = profiles.scalars.get(name)
        if value is not None and np.isfinite(value) and value > 0:
            found.append((name, float(value), f"{name} = {value:.1f} eV, {said}"))
    return found


def rho_pol_of(fit, r_mid):
    """The ρ_pol whose R_mid is ``r_mid`` (m), through ``fit_R_mid`` inverted: NaN where it does not reach, or without one.

    Only the outermost stretch of the grid over which ``fit_R_mid`` grows is inverted, where the probe is: a fit moved
    inward has a few samples near the axis where it does not grow.
    """
    r_mid = np.asarray(r_mid, dtype=float)
    found = np.full(r_mid.shape, np.nan)
    if fit is None or fit.R_mid is None:
        return found
    rho, grid_r = np.asarray(fit.rho_pol, dtype=float), np.asarray(fit.R_mid, dtype=float)
    finite = np.flatnonzero(np.isfinite(grid_r) & np.isfinite(rho))
    if finite.size < 2:
        return found
    rho, grid_r = rho[: finite[-1] + 1], grid_r[: finite[-1] + 1]
    with np.errstate(invalid="ignore"):
        grows = np.isfinite(grid_r[1:]) & np.isfinite(grid_r[:-1]) & (np.diff(grid_r) > 0)
    stops = np.flatnonzero(~grows)
    first = int(stops[-1]) + 1 if stops.size else 0
    if grid_r.size - first < 2:
        return found
    placed = np.isfinite(r_mid)
    found[placed] = np.interp(r_mid[placed], grid_r[first:], rho[first:], left=np.nan, right=np.nan)
    return found


@dataclass(eq=False)
class Overlay:
    """The MLP's profile on the Thomson scattering axes, one entry per point of ``found.drawn``.

    ``rho`` is the probe's distance to the separatrix at the outboard midplane (m), moved where a move was asked for
    and reached. A point is drawn on R_mid at ``separatrix_R_mid`` + ``rho`` and on ρ_pol at the ρ_pol of that
    (:func:`rho_pol_of`): NaN where it has no place.
    """

    found: mlp.Profiles
    source: str  # what the profile was asked to be moved to: "none", "Te_sep_2pm" or "Te_sep_fit"
    te_sep: float  # its value, 0 for none
    time: np.ndarray  # s, the middle of the point's window
    plunge: np.ndarray
    stroke: np.ndarray  # "in" or "out"
    rho: np.ndarray  # m
    ne: np.ndarray  # m^-3
    Te: np.ndarray  # eV
    R_mid: np.ndarray  # m
    rho_pol: np.ndarray
    separatrix_R_mid: Optional[float]  # the R_mid of ρ = 0: the separatrix the figure draws, or None

    @property
    def size(self):
        return int(self.time.size)

    @property
    def shift(self):
        """How far the profile was moved, in metres of ρ: 0 where it was not."""
        return self.found.shift.by

    def on(self, axis):
        return self.rho_pol if axis == "rho_pol" else self.R_mid

    def series(self, quantity):
        """``(values, errors)`` as :meth:`Points.series` gives them: a 1 ms mean has no error bar, so the errors are NaN."""
        values = {"ne": self.ne, "Te": self.Te, "pe": ELEMENTARY_CHARGE * self.ne * self.Te}[quantity]
        return values, np.full(values.shape, np.nan)

    def has(self, quantity, axis):
        return bool(np.any(np.isfinite(self.on(axis)) & np.isfinite(self.series(quantity)[0])))


def overlay(profiles, asp, t_start, t_end, source="none"):
    """The MLP's profile from the ASP file ``asp`` over ``[t_start, t_end]``, placed on the axes of ``profiles``: the
    one place this view reads the probe's data. The points are those the MLP profiles plot draws.

    ``source`` names what to move it to (:func:`separatrix_choices`); one the file does not hold moves nothing.
    """
    te_sep = {name: value for name, value, _ in separatrix_choices(profiles)}.get(source, 0.0)
    found = mlp.read(asp, t_start, t_end, te_sep)
    drawn = found.drawn

    def values(name):
        return np.asarray(drawn[name].values, dtype=float) if name in drawn else np.full(drawn.sizes["window"], np.nan)

    rho = values("rho") + found.shift.by
    sep = separatrix(profiles, "R_mid")
    r_mid = rho + sep if sep is not None else np.full(rho.shape, np.nan)
    return Overlay(
        found=found,
        source=source if te_sep else "none",
        te_sep=float(te_sep),
        time=np.asarray(drawn["time"].values, dtype=float),
        plunge=np.asarray(drawn["plunge"].values, dtype=int),
        stroke=np.asarray(drawn["stroke"].values, dtype=str),
        rho=rho,
        ne=values("ne"),
        Te=values("Te"),
        R_mid=r_mid,
        rho_pol=rho_pol_of(profiles.fit, r_mid),
        separatrix_R_mid=sep,
    )


def _strokes_said(pairs):
    """``[(3, "in"), (3, "out")]`` as ``plunge 3, in and out``."""
    by_plunge = {}
    for number, stroke in pairs:
        by_plunge.setdefault(number, []).append(stroke)
    return "; ".join(f"plunge {number}, {' and '.join(ways)}" for number, ways in by_plunge.items())


def mlp_mode_hint(target):
    """The discharge DB's ``mlp_mode`` for the shot, quoted: a hint beside the offer."""
    discharge = getattr(target, "discharge", None)
    if discharge is None:
        return "The shot has no discharge-DB entry."
    mode = str(getattr(discharge, "mlp_mode", "") or "").strip()
    return f"The discharge DB's mlp_mode: “{mode}”." if mode else "The discharge DB gives no mlp_mode."


def _held_depth(found_track, t_start, t_end):
    """The probe's depth over ``[t_start, t_end]`` in words, from its track, or ``""``."""
    time, depth = found_track
    inside = (time >= t_start) & (time <= t_end)
    if not inside.any():
        return ""
    low, high = float(depth[inside].min()) * 100, float(depth[inside].max()) * 100
    return f" at a depth of {low:.1f} cm" if high - low < 0.05 else f" between depths of {low:.1f} and {high:.1f} cm"


def offer(asp, target):
    """``(strokes, text)``: the strokes the probe runs in the page's window, and the words the sidebar says.

    Judged from the probe's depth in the file's attributes alone, so no variable is read. With no stroke the text says
    why. It ends with the discharge DB's ``mlp_mode``, a hint the judgement does not rest on.
    """
    hint = mlp_mode_hint(target)
    if not mlp.layout(asp):
        return [], (
            "No MLP profiles to draw over these: the ASP file holds no MLP quantity (ne, Te, Js, Is, Vp or Vf on a "
            f"pin). {hint}"
        )
    strokes = mlp.strokes_in(asp, target.t_start, target.t_end)
    if strokes:
        return strokes, (
            f"The probe plunges in the window: {_strokes_said(strokes)}, judged from its depth in the ASP file. {hint}"
        )
    found_track = mlp.track(asp)
    if found_track is None:
        why = "the ASP file has no plunge depth (`plunge_time`, `plunge_depth`), so its plunges cannot be found"
    else:
        record = mlp.plunges(*found_track)
        if any(p.start <= target.t_end and p.end >= target.t_start for p in record):
            why = "the probe holds still in the window" + _held_depth(found_track, target.t_start, target.t_end)
        elif record:
            tips = ", ".join(f"{p.tip:.3f}" for p in record)
            why = f"its plunges are outside the window (their tips at {tips} s)"
        else:
            why = "the probe does not plunge in this shot"
    return [], f"No MLP profiles to draw over these: {why}. {hint}"


def overlay_warnings(points):
    """The warnings the overlay calls for: a move that was asked for and not made, and a window with no MLP point."""
    said = []
    found = points.found
    if points.te_sep and not found.shift.ok:
        said.append(
            f"The MLP profile is not moved to T_e = {points.te_sep:.1f} eV ({points.source}): {found.shift.reason}. It is "
            "drawn on EFIT's mapping."
        )
    if not points.size:
        recorded, partial = mlp.recording(found.means)
        if not recorded:
            why = "the MLP recorded nothing while the probe moved"
        elif partial == recorded:
            why = f"in none of the {recorded} ms the MLP recorded while the probe moved did every pin record"
        else:
            why = "the file holds no ρ for the milliseconds the MLP recorded, so none of them can be placed"
        said.append(f"The probe plunges in the window, but there is no MLP point to draw: {why}.")
    return said


def overlay_notes(points, profiles, axis):
    """The lines under the figure for the MLP's profile: what its points are and how they were placed."""
    fit = profiles.fit
    found = points.found
    pins = found.means.sizes.get("pin", 0)
    strokes = _strokes_said(mlp.strokes(found.drawn))
    lines = [
        f"MLP: {points.size} 1 ms means of the Mirror Langmuir Probe in the window ({strokes}), each the mean of its "
        f"{pins} pins, as the MLP profiles plot reduces them: ◀ on the way in, ▷ on the way out. Their p_e is e·n_e·T_e "
        "of those means. The probe's file names no unit: the Documentation page says how each was established."
    ]
    sep = points.separatrix_R_mid
    if sep is None:
        placed = (
            "The Thomson scattering file gives no separatrix on R_mid, so the MLP points, whose ρ is a distance to the "
            "separatrix, have no place on either axis and are not drawn."
        )
    else:
        source = "`R_mid_sep`" if fit is None or not fit.moved else "the fit's R_mid at its ρ_pol 1"
        placed = (
            "Placed by ρ, the probe's distance to the separatrix mapped to the outboard midplane, in metres: on R_mid at "
            f"R_mid,sep + ρ, with R_mid,sep = {sep:.4f} m ({source}, the separatrix drawn), and on ρ_pol at the ρ_pol of "
            "that R_mid through the fit's own R_mid (`fit_R_mid`, from EFIT at the window's middle time)."
        )
        if axis == "rho_pol":
            if fit is None or fit.R_mid is None:
                placed += " The file has no `fit_R_mid`, so the MLP points are not drawn on this axis."
            else:
                lost = int((np.isfinite(points.R_mid) & np.isnan(points.rho_pol)).sum())
                if lost:
                    placed += (
                        f" {lost} of the {points.size} MLP points lie where `fit_R_mid` does not reach (it ends at "
                        f"ρ_pol {np.nanmax(fit.rho_pol):g}) and are not drawn on this axis."
                    )
    if not points.te_sep:
        moved = (
            "Not moved: the MLP's ρ and the Thomson scattering points' ρ_pol both come from EFIT's mapping."
            if fit is None or not fit.moved
            else (
                f"Not moved: the MLP's ρ comes from EFIT's mapping, while the Thomson scattering points are drawn moved "
                f"with the fit, by {fit.rho_shift:+.4f} in ρ_pol. Moved to Te_sep_fit, the MLP is on the fit's "
                "separatrix too, where the probe's T_e reaches it."
            )
        )
    elif found.shift.ok:
        moved = (
            f"Moved by {points.shift * 100:+.2f} cm, as the MLP profiles plot moves its profile, so that the MLP's T_e is "
            f"{points.te_sep:.1f} eV ({points.source}) at ρ = 0: an exponential fit of the {found.shift.points} 1 ms "
            f"means of T_e within a factor {mlp.BAND:g} of it puts it at {-points.shift * 100:+.2f} cm of the file's ρ."
        )
    else:
        moved = f"Not moved to {points.te_sep:.1f} eV ({points.source}): {found.shift.reason}."
    lines.append(f"{placed} {moved}")
    return lines


def _mlp_hover(points, quantity, scale, unit, values):
    """One hover text per MLP point: which stroke, when, where on both axes and in ρ, and the value."""
    texts = []
    for i in range(points.size):
        rho = f"ρ = {points.rho[i] * 100:.2f} cm"
        if points.shift:
            rho += f" (file: {(points.rho[i] - points.shift) * 100:.2f} cm)"
        r_mid = "–" if not np.isfinite(points.R_mid[i]) else f"{points.R_mid[i]:.4f} m"
        rho_pol = "–" if not np.isfinite(points.rho_pol[i]) else f"{points.rho_pol[i]:.3f}"
        value = "–" if not np.isfinite(values[i]) else _number(values[i] * scale, 3)
        texts.append(
            f"MLP · plunge {int(points.plunge[i])}, {points.stroke[i]} · t = {points.time[i]:.4f} s<br>"
            f"{rho} · R_mid = {r_mid} · ρ_pol = {rho_pol}<br>"
            f"{quantity} = {value} {unit}"
        )
    return texts


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------


def _number(value, digits):
    """``value`` to ``digits`` significant digits, with no exponent below a million: 2140, not 2.14e+03."""
    text = f"{value:.{digits}g}"
    if "e" in text and 1 <= abs(value) < 1e6:
        return f"{value:.0f}"
    return text


def _hover(points, quantity, scale, unit, values, errors):
    """One hover text per point: which system, when, where, and the value with its error. A point moved with the fit
    says where it is drawn, and where EFIT puts it."""
    texts = []
    rho, r_mid = points.on("rho_pol"), points.on("R_mid")
    for i in range(points.drawn):
        value = _number(values[i] * scale, 3)
        if np.isfinite(errors[i]):
            value += f" ± {_number(errors[i] * scale, 2)}"
        where = f"ρ_pol = {points.rho_pol[i]:.3f} · R_mid = {points.R_mid[i]:.4f} m<br>"
        if points.moved:
            where = (
                f"ρ_pol = {rho[i]:.3f} · R_mid = {r_mid[i]:.4f} m, moved with the fit ({points.rho_shift:+.4f})<br>"
                f"EFIT: ρ_pol = {points.rho_pol[i]:.3f} · R_mid = {points.R_mid[i]:.4f} m<br>"
            )
        texts.append(
            f"{LOOKS[points.system]['label']} · t = {points.time[i]:.4f} s · channel {int(points.channel[i])}<br>"
            f"{where}"
            f"R = {points.R[i]:.4f} m, Z = {points.Z[i]:+.4f} m<br>"
            f"{quantity} = {value} {unit}"
        )
    return texts


def _message(text, height=200):
    """An empty figure that says ``text``."""
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def _fit_label(fit):
    return "fit" if fit.window is None else f"fit, {_span(fit.window)}"


def _mlp_traces(fig, mlp_points, axis, row, quantity, label, scale, unit, seen):
    """The MLP's points of one row: a trace for each way the probe ran, both in one legend group. ``seen`` holds the
    names already in the legend."""
    values = mlp_points.series(quantity)[0]
    texts = _mlp_hover(mlp_points, label, scale, unit, values)
    for stroke, symbol in MLP_LOOK["symbols"].items():
        mask = mlp_points.stroke == stroke
        x, y = mlp_points.on(axis)[mask], values[mask]
        if not np.any(np.isfinite(x) & np.isfinite(y)):
            continue
        name = f"{MLP_LOOK['label']}, {stroke}"
        colour = MLP_LOOK["colour"]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y * scale,
                mode="markers",
                name=name,
                legendgroup="mlp",
                showlegend=name not in seen,
                marker=dict(symbol=symbol, size=8, color=colour, opacity=0.85, line=dict(width=1.5, color=colour)),
                hovertext=[text for text, keep in zip(texts, mask) if keep],
                hoverinfo="text",
            ),
            row=row,
            col=1,
        )
        seen.add(name)


def figure(profiles, axis="rho_pol", mlp_points=None):
    """n_e, T_e and p_e on one shared ``axis`` (``"rho_pol"`` or ``"R_mid"``), one row each: the edge and core points
    with their bars, the stored fit over its whole range, and the separatrix. Only what there is is drawn.

    Every row opens on zero to the top the points set (:func:`y_limits`), and autoscale returns to it: left to
    autorange, one runaway fit sets the range and leaves the points flat against zero. A row says in its corner how
    many of its points lie above its top.

    ``mlp_points``, an :class:`Overlay`, draws the probe's profile over these. It does not move a row's range; its
    points above the top are counted in the corner as well.
    """
    if axis not in AXES:
        raise ValueError(f"axis is one of {', '.join(AXES)}, not {axis!r}")
    rows = [
        quantity
        for quantity in QUANTITIES
        if profiles.has(quantity[0], axis) or (mlp_points is not None and mlp_points.has(quantity[0], axis))
    ]
    if not rows:
        measured = sum(found.drawn for found in profiles.points.values())
        if measured:
            return _message(
                f"None of the {measured} measured points can be placed on {AXES[axis][1]}, and there is no fit to draw."
            )
        return _message("No Thomson scattering measurement in this window, and no fit in this file.")

    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.05)
    sep, fit = separatrix(profiles, axis), profiles.fit
    in_legend = set()  # the MLP's entries, which may first appear in any row
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
                        color=look["bar"],
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
        if mlp_points is not None:
            _mlp_traces(fig, mlp_points, axis, row, name, label, scale, unit, in_legend)
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
            # Nothing above zero to set a top by: autorange, but from zero.
            fig.update_yaxes(title_text=title, autorangeoptions=dict(minallowed=0), row=row, col=1)
            continue
        # The range is given outright, so the row opens on it, and as ``autorangeoptions`` too, so that autoscale
        # returns to it and not to a runaway fit.
        bottom, top = limits
        fig.update_yaxes(
            title_text=title,
            range=[bottom, top],
            autorangeoptions=dict(minallowed=bottom, maxallowed=top),
            row=row,
            col=1,
        )
        above = points_above(profiles, name, axis, top, scale)
        said = [f"▲ {above} point{'s' * (above > 1)} above"] if above else []
        if mlp_points is not None:
            values = mlp_points.series(name)[0] * scale
            mlp_above = int((np.isfinite(mlp_points.on(axis)) & (np.nan_to_num(values, nan=-np.inf) > top)).sum())
            if mlp_above:
                said.append(f"▲ {mlp_above} MLP point{'s' * (mlp_above > 1)} above")
        if said:
            fig.add_annotation(
                text="<br>".join(said),
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


def mlp_controls(asp, profiles, target):
    """The sidebar's offer of the MLP's profiles: the name of the separatrix temperature to move the profile to
    (:func:`separatrix_choices`), or ``None`` while the overlay is off or not offered.

    Where the probe does not plunge in the window, only the words that say why. Both widgets are view state, keyed
    off the target, so a shot opens with the overlay off.
    """
    import streamlit as st

    if asp is None:
        return None
    strokes, said = offer(asp, target)
    if not strokes:
        st.sidebar.caption(said)
        return None
    on = st.sidebar.toggle(
        "MLP profiles",
        key=f"thomson_profiles.mlp.{target.key}",
        help=(
            "Draw the Mirror Langmuir Probe's profiles from the shot's ASP file on the n_e, T_e and p_e rows: the 1 ms "
            "means of its plunges in the window, as the MLP profiles plot draws them. Its data are read only while "
            "this is on."
        ),
    )
    st.sidebar.caption(said)
    if not on:
        return None
    choices = separatrix_choices(profiles)
    names, labels = [name for name, _, _ in choices], {name: label for name, _, label in choices}
    key = f"thomson_profiles.mlp_moved_to.{target.key}"
    if key in st.session_state and st.session_state[key] not in names:
        del st.session_state[key]  # a remembered choice this file does not offer: back to not moving it
    return st.sidebar.selectbox(
        "Move the MLP profile to",
        names,
        format_func=lambda name: labels[name],
        key=key,
        help=(
            "A separatrix temperature from this file, to move the MLP's profile to as the MLP profiles plot moves it: "
            "so that its T_e is that at ρ = 0. Not moved, its ρ and the Thomson scattering points' ρ_pol both come "
            "from EFIT's mapping."
        ),
    )


def render(ds, params, target, companions=None):
    """The warnings, the figure, the lines under it and what the file says of itself, with the offer of the MLP's
    profiles in the sidebar. Draws into Streamlit and returns ``None``.

    ``companions`` is ``{"asp": the shot's ASP file, opened lazily, or None}``. Its data are read only while the
    overlay is on.
    """
    import streamlit as st

    profiles = read(ds)
    if not profiles.points and profiles.fit is None:
        st.warning("This file holds no Thomson scattering data to draw.", icon="⚠️")
        return None
    axis = params.axis if params.axis in AXES else "rho_pol"

    mlp_points, problems = None, []
    asp = (companions or {}).get(COMPANION)
    moved_to = mlp_controls(asp, profiles, target)
    if moved_to is not None:
        try:
            mlp_points = overlay(profiles, asp, target.t_start, target.t_end, moved_to)
        except Exception as error:  # noqa: BLE001 - the probe's file must not take the Thomson scattering view down
            problems.append(f"The MLP profiles could not be read from the ASP file: {type(error).__name__}: {error}")
        else:
            problems += overlay_warnings(mlp_points)

    for text in (warning(profiles), points_warning(profiles), *problems):
        if text:
            st.warning(text, icon="⚠️")
    st.plotly_chart(figure(profiles, axis, mlp_points), width="stretch")
    lines = notes(profiles, target, axis)
    if mlp_points is not None and mlp_points.size:
        lines[-1:-1] = overlay_notes(mlp_points, profiles, axis)  # before the caveat, which ends the notes
    for line in lines:
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
        companions=(COMPANION,),
        description=(
            "n_e, T_e and p_e of the edge and core Thomson scattering systems against ρ_pol or R_mid: the points in "
            "the window, and the fit the file stores; and, where the probe plunges in the window, the Mirror Langmuir "
            "Probe's profiles over them on request. Read off the files: nothing is computed."
        ),
    )
)
