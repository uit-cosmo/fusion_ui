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
    ``fit_rho_shift``, ``Te_sep_fit``, and the attributes ``fit_separatrix`` and ``separatrix_code``
        Only in a file made with C-Mod_Analysis 24e1499, whose fit looks for the separatrix in its own T_e profile and,
        where it finds one, moves itself, and its own copy of the points, by ``fit_rho_shift`` in ρ_pol, so that its
        T_e is ``Te_sep_fit`` at ρ_pol 1: ``rho_pol`` is then the moved grid. The raw points stay as EFIT maps them,
        and ``fit_rho_shift`` is 0 where the fit was not moved. ``separatrix_code`` is the folder of the code that
        looks, which is outside C-Mod_Analysis. A file made with d6a6466 has none of them.
    the attributes ``fit_points_check`` and ``mapping_failure``
        Only in a file made by the script's later versions. ``fit_points_check`` says whether the edge points the fit
        was made on are the file's: ``equal``, or a sentence beginning ``differ:``. ``mapping_failure``, there only
        where mapping the points failed, is a sentence beginning ``mapping failed:`` that says which system and
        coordinate, at which pulses, and the error; the points' ρ_pol or R_mid are NaN where it failed.

What is drawn
    One row each for n_e, T_e and p_e, on one shared axis, ρ_pol or R_mid (a choice): the edge and core points in the
    page's window, told apart by marker and colour, with the tree's ±1σ bars; the stored fit, over its whole range
    and **without a band**; and the separatrix. The file holds no p_e of the measurements, so its points are e·n_e·T_e,
    with the errors propagated, beside the fit's own ``fit_pe``. A band around the fit is left out on purpose (the
    user, 2026-10-09): with ``frac_err`` the fit's n_e and T_e errors are 20% of the fit by construction, and
    ``fit_pe_err`` adds a density to a temperature, so it is no error of p_e.

    **A fit moved to a separatrix takes its points with it.** Where ``fit_rho_shift`` is not 0, the points are drawn
    where the fit saw them: at their ρ_pol plus the shift, and on R_mid at the fit's own R_mid of that ρ_pol
    (``fit_R_mid``, interpolated). The separatrix is the fit's ρ_pol 1, and its R_mid there, and a caption says by how
    much they were moved and to which T_e,sep, quoting ``fit_separatrix``. Every other file, a d6a6466 one among them,
    draws each point at its own pulse's ρ_pol and R_mid, as EFIT puts it. A fit whose outer side fictitious points
    hold (``force_to_zero``) gets a caption that says so, and that the p_e fit is not held. A file whose
    ``fit_points_check`` is anything but ``equal`` gets a warning above the figure, quoting it: the points drawn may
    then not be where the fit saw them. A point with no place on the axis drawn is left out and counted, with the
    file's ``mapping_failure`` quoted where it has one.

    **A row's range is set by the points, not by the fit** (:func:`y_limits`, the rule of the paper's PDF of the density
    scan, ported): it opens on ``[0, top]``, and a fit that runs away leaves through the top instead of squashing the
    points flat against zero. Autoscale returns to that range, and the points above it are counted in the row's corner.
    The points are judged where they are drawn, moved or not.

    A fit that does not pass ``fit_check`` gets a warning above the figure that quotes it, and **anything but the
    outcome** ``passed`` **is warned about**: a verdict spelt another way, an outcome nobody wrote a rule for, an empty
    attribute. Its advice follows the clause that failed: the temperature on axis says the core is an extrapolation,
    the temperature at the grid's outer end that the outer end is not credible. A ``fit_note`` that calls the core an
    extrapolation gets a caption of its own, passed or not. The diagnostician's caveat is under every figure.

The Mirror Langmuir Probe's profiles, on request
    The spec names the ASP file as a companion (``companions=("asp",)``), so the single-shot page hands it the shot's
    ASP file beside its own, opened lazily: nothing but its attributes is read until this view reads something. Where
    the probe plunges within the page's window, which :func:`fusion_ui.core.mlp.strokes_in` judges from its depth in
    the attributes alone, the sidebar offers the MLP's profiles, with the discharge DB's ``mlp_mode`` beside the offer
    as a hint; where it does not, the sidebar says why. **The probe's data are read only while the overlay is on**
    (:func:`overlay`), with the reduction of the MLP profiles plot (:func:`fusion_ui.core.mlp.read`): the 1 ms means of
    the four pins combined, over the strokes in the window. They are drawn on the n_e, T_e and p_e rows (p_e =
    e·n_e·T_e), placed by ρ, the probe's distance to the separatrix mapped to the outboard midplane, in metres: on R_mid
    at the separatrix drawn plus ρ, and on ρ_pol at the ρ_pol of that R_mid through the fit's own ``fit_R_mid``. **Not
    moved by default**, so that both sets are on EFIT's mapping; the sidebar offers the move of the MLP profiles plot
    to the file's ``Te_sep_2pm``, or ``Te_sep_fit`` where it has one. A caption says how the points were placed. Each
    row's range stays the one the Thomson scattering points set, and the MLP points above it are counted in the row's
    corner. **Without the overlay, the view is drawn exactly as it was without the companion.**

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

from fusion_ui.core import loader, mlp, params_ui, registry

SYSTEMS = ("edge", "core")

#: One electron-volt in joules: n_e [m^-3] times T_e [eV] times this is p_e [Pa].
ELEMENTARY_CHARGE = 1.602176634e-19

#: The scalars the file may hold, read for the separatrix, the captions and the "about this file" list. ``Te_sep_fit``
#: is only in a file whose fit was moved to a separatrix it found (C-Mod_Analysis 24e1499).
SCALARS = ("R_mid_sep", "Te_sep_2pm", "Te_sep_fit", "lambda_q_2pm")

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

    ``rho_pol`` and ``R_mid`` are where EFIT puts each point at its own pulse, as the file holds them. A fit that was
    moved to a separatrix took its own copy of the points with it, and :meth:`on` says where they are drawn: at
    ``rho_pol`` plus ``rho_shift``, and on R_mid at ``R_mid_moved``, the fit's own R_mid of that ρ_pol.
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
    #: How far the fit moved the points in ρ_pol (the file's ``fit_rho_shift``): 0 where it did not move them.
    rho_shift: float = 0.0
    #: Where the moved points are on the R_mid axis, in metres: the fit's own R_mid of their moved ρ_pol, NaN where it does
    #: not reach. ``None`` for points that were not moved.
    R_mid_moved: Optional[np.ndarray] = None

    @property
    def drawn(self):
        return int(self.time.size)

    @property
    def moved(self):
        """Whether these points are drawn moved with the fit, away from where EFIT puts them."""
        return bool(self.rho_shift)

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
        """Where the points are drawn on ``axis``: ρ_pol, or R_mid in metres. Points the fit moved are where the fit saw
        them, all others where EFIT puts them."""
        if not self.moved:
            return self.rho_pol if axis == "rho_pol" else self.R_mid
        if axis == "rho_pol":
            return self.rho_pol + self.rho_shift
        return self.R_mid_moved if self.R_mid_moved is not None else np.full(self.rho_pol.shape, np.nan)


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
    #: How far C-Mod_Analysis moved the fit, and its own copy of the points, in ρ_pol (the file's ``fit_rho_shift``), so
    #: that its T_e is the separatrix T_e it found at ρ_pol 1: ``rho_pol`` is then the moved grid. 0 where it did not
    #: move it, and ``None`` in a file whose code never moves its fit (C-Mod_Analysis d6a6466).
    rho_shift: Optional[float] = None

    @property
    def moved(self):
        """Whether the fit was moved to a separatrix: a shift that is a number, and not 0."""
        return self.rho_shift is not None and bool(np.isfinite(self.rho_shift)) and self.rho_shift != 0.0

    def on(self, axis):
        return self.rho_pol if axis == "rho_pol" else self.R_mid

    def values(self, quantity):
        return {"ne": self.ne, "Te": self.Te, "pe": self.pe}[quantity]

    def R_mid_of(self, rho_pol):
        """The fit's own R_mid of ``rho_pol`` on its axis (``fit_R_mid``, interpolated), in metres: NaN outside its grid,
        and everywhere when the file has no ``fit_R_mid``."""
        rho_pol = np.asarray(rho_pol, dtype=float)
        if self.R_mid is None:
            return np.full(rho_pol.shape, np.nan)
        return np.interp(rho_pol, self.rho_pol, self.R_mid, left=np.nan, right=np.nan)


@dataclass(eq=False)
class Profiles:
    """What a Thomson scattering file says, cut to the window it was opened over."""

    #: ``{"edge": Points, "core": Points}``, whichever the file holds.
    points: dict
    fit: Optional[Fit]
    #: Laser pulses in the window, as the file's ``time`` axis has them.
    pulses: int
    #: Those of :data:`SCALARS` the file holds.
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

    shift = ds["fit_rho_shift"] if "fit_rho_shift" in ds else None
    return Fit(
        rho_pol=np.asarray(ds["rho_pol"].values, dtype=float),
        ne=on_grid("fit_ne"),
        Te=on_grid("fit_Te"),
        pe=on_grid("fit_pe"),
        R_mid=on_grid("fit_R_mid"),
        window=_window(attrs),
        rho_shift=float(shift.values) if shift is not None and shift.ndim == 0 else None,
    )


def _move(points, fit):
    """Draw ``points`` where the moved ``fit`` saw them: C-Mod_Analysis moved its own copy of them with it, by its shift
    in ρ_pol, and on the R_mid axis they go to the fit's own R_mid of their moved ρ_pol."""
    points.rho_shift = float(fit.rho_shift)
    points.R_mid_moved = fit.R_mid_of(points.rho_pol + points.rho_shift)


def read(ds):
    """The :class:`Profiles` of an open Thomson scattering file, cut to whatever window ``ds`` was sliced to.

    Everything is optional but one of the systems' points or the fit: a file that holds only the core, or only the
    edge, or whose fit failed, draws what it has. Where the fit was moved to a separatrix, the points are moved with it
    (:class:`Points`).
    """
    attrs = {name: _plain(value) for name, value in ds.attrs.items()}
    fit = _fit(ds, attrs)
    points = {}
    for system in SYSTEMS:
        found = _points(ds, system)
        if found is not None:
            if fit is not None and fit.moved:
                _move(found, fit)
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


def points_warning(profiles):
    """The warning above the figure when the file does not say that the fit's own points are its own, or ``None``.

    A file made by the script's later versions records in ``fit_points_check`` whether the edge points
    ``get_cmod_kin_profs`` fitted are this file's, as EFIT maps them and moved by any ``fit_rho_shift``: ``equal``, or a
    sentence beginning ``differ:`` that says by how much. **Anything that does not begin with** ``equal`` **is warned
    about**, quoting it, since the points drawn may then not be where the fit saw them. A file without the attribute,
    every older one among them, and a file with no fit get none.
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
    """The sentences of ``text`` that hold ``word``, joined, or ``""``: the attributes write whole sentences."""
    sentences = re.split(r"(?<=\.)\s+", str(text or ""))
    return " ".join(sentence for sentence in sentences if word in sentence)


def extrapolation(attrs):
    """The sentence of ``fit_note`` that calls the fit's core an extrapolation, or ``""`` when it does not.

    The file says so when it finds no core T_e of 1 keV or more inside ρ_pol 0.2: the fit's pre-filter drops the lower
    ones, so the fit has none to rest on there, whether or not its check passed.
    """
    return _sentences(attrs.get("fit_note"), "extrapolation")


def move_note(profiles):
    """The caption of a fit moved to a separatrix, or ``""`` for any other: by how much the fit and the points were
    moved, to which T_e,sep, and the file's own ``fit_separatrix``, quoted. Where the points are on the axis drawn is the
    axis's own line (:func:`notes`)."""
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
    ``fit_note`` says of them, or ``""`` for any other fit.

    C-Mod_Analysis 24e1499's ``run_db.py`` sets it: before each fit it adds fictitious T_e and n_e points beyond the
    separatrix, which hold the n_e and T_e fits there but not the p_e fit.
    """
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

    ρ_pol = 1 by definition. On the R_mid axis it is the file's ``R_mid_sep``, the separatrix's outboard midplane radius
    at the window's middle time, which the fit's own R_mid reaches at ρ_pol = 1 as well; the fit's is used when the
    file holds no scalar.

    A fit moved to a separatrix is drawn on its own axis, so the separatrix is its ρ_pol = 1, and on R_mid its own R_mid
    there (``fit_R_mid``, interpolated), where its T_e is the T_e,sep it was moved to. ``fit_R_mid`` maps the moved
    ρ_pol through EFIT, so that is ``R_mid_sep`` as well, to the grid's interpolation, and ``R_mid_sep`` stands in when
    the fit's R_mid does not reach ρ_pol 1.
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
    """The lines under the figure, as markdown, one paragraph each: the points, the fit, the axis, the caveat.

    A fit that was moved to a separatrix adds a line on the move (:func:`move_note`), and one whose outer side
    fictitious points hold a line on them (:func:`held_note`), both after the fit's.

    The points with no place on ``axis`` are counted with what the file says of why: no EFIT tree could be opened, its
    ``mapping_failure`` (quoted), or else a pulse outside the EFIT time base. Points moved with the fit have a place on
    R_mid only where the fit's own R_mid reaches: those beyond its grid, and those where it is NaN, are counted apart.

    ``target`` supplies the page's window; ``None`` leaves out whatever depends on it.
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
        # Whether or not the check passed: a core with no T_e of 1 keV or more has none for the fit to rest on.
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
)

#: What the list quotes after those of a file made by code that looks for the separatrix in its own fit (C-Mod_Analysis
#: 24e1499), which no other file has: what that code found and did, and where it is.
SEPARATRIX_ABOUT = ("fit_separatrix", "separatrix_code")

#: The file's numbers the list gives, with the unit and the factor each is shown in: the separatrix T_e of the fit beside
#: the two-point model's.
NUMBERS = (("Te_sep_2pm", "eV", 1.0), ("Te_sep_fit", "eV", 1.0), ("lambda_q_2pm", "mm", 1e3), ("R_mid_sep", "m", 1.0))


def separatrix_code(attrs):
    """What the "about" list says of the code that looked for the separatrix: its folder, that no commit pins it, and
    that the file keeps the sha256 of each of its files that were loaded (``separatrix_code_sha256``)."""
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
    """``[(attribute, text), …]``: what the file says of how it was made, in its own words, for the expander, then its
    numbers. The separatrix code is said in the list's words (:func:`separatrix_code`)."""
    rows = [(name, str(profiles.attrs[name])) for name in ABOUT if profiles.attrs.get(name) not in (None, "")]
    for name in SEPARATRIX_ABOUT:
        if profiles.attrs.get(name) not in (None, ""):
            text = separatrix_code(profiles.attrs) if name == "separatrix_code" else str(profiles.attrs[name])
            rows.append((name, text))
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
# The Mirror Langmuir Probe's profiles, drawn over the Thomson scattering ones
# ---------------------------------------------------------------------------

#: The diagnostic whose file the spec draws beside its own (:attr:`fusion_ui.core.registry.PlotSpec.companions`).
COMPANION = "asp"

#: How the MLP's points are drawn: the data-viz palette's fourth slot, yellow, in the step that clears 3:1 on a dark page
#: (2.99:1 on a light one, which the legend and the hover text relieve), with the markers of the MLP profiles plot, which
#: no Thomson scattering system uses: a triangle pointing the way the probe ran, filled on the way in and open on the
#: way out. Against the edge points and the fit, among which it is drawn, the colour clears the palette's checks for a
#: scatter; it is near the core's orange, but the two never meet: the core system reaches no further out than ρ_pol 0.9,
#: and the probe no further in than about a centimetre inside the separatrix.
MLP_LOOK = dict(label="MLP", colour="#c98500", symbols={"in": "triangle-left", "out": "triangle-right-open"})

#: What the MLP's profile can be moved to, by the file's own name for the temperature (:func:`separatrix_choices`): not at
#: all, or the separatrix temperature of the two-point model, or the one the fit was moved to.
MOVES = (
    ("none", "not moved: EFIT's mapping"),
    ("Te_sep_2pm", "the two-point model's"),
    ("Te_sep_fit", "the fit's"),
)


def separatrix_choices(profiles):
    """``[(name, te_sep, label), …]``: what the MLP's profile can be moved to, as the MLP profiles plot moves it (its
    ``te_sep``). ``none`` (0 eV, not moved) first, the default; then ``Te_sep_2pm`` and ``Te_sep_fit``, each where the file
    holds it as a positive number. The label says what each is, with its value."""
    found = [("none", 0.0, MOVES[0][1])]
    for name, said in MOVES[1:]:
        value = profiles.scalars.get(name)
        if value is not None and np.isfinite(value) and value > 0:
            found.append((name, float(value), f"{name} = {value:.1f} eV, {said}"))
    return found


def rho_pol_of(fit, r_mid):
    """The ρ_pol whose R_mid is ``r_mid`` (m), through the fit's own R_mid (``fit_R_mid``) inverted: NaN where it does not
    reach, and everywhere when the file has no fit or no ``fit_R_mid``.

    Only the outermost stretch of the grid over which ``fit_R_mid`` grows is inverted: the separatrix and the SOL, where
    the probe is. A fit moved inward (a negative ``fit_rho_shift``) has a few samples near the axis where it does not
    grow, and they cannot be inverted.
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


@dataclass(eq=False)  # arrays do not compare as a whole
class Overlay:
    """The MLP's profile on the Thomson scattering axes, one entry per point: the combined 1 ms means of the four pins on
    a stroke in the page's window (``found.drawn``, :func:`fusion_ui.core.mlp.read`).

    ``rho`` is the file's ρ plus the move to ``te_sep``, where one was asked for and reached (``found.shift``): the
    probe's distance to the separatrix at the outboard midplane, in metres. Each point is drawn on R_mid at ``R_mid`` =
    ``separatrix_R_mid`` + ``rho``, and on ρ_pol at ``rho_pol``, the ρ_pol of that R_mid through the fit's own R_mid
    (:func:`rho_pol_of`). NaN where it has no place.
    """

    found: mlp.Profiles
    #: What the profile was asked to be moved to: ``"none"``, ``"Te_sep_2pm"`` or ``"Te_sep_fit"``, and its value (0 eV for
    #: none).
    source: str
    te_sep: float
    time: np.ndarray  # s, the middle of the point's window
    plunge: np.ndarray
    stroke: np.ndarray  # "in" or "out"
    rho: np.ndarray  # m
    ne: np.ndarray  # m^-3
    Te: np.ndarray  # eV
    R_mid: np.ndarray  # m
    rho_pol: np.ndarray
    #: The R_mid of ρ = 0, the separatrix the figure draws on its R_mid axis (:func:`separatrix`), or ``None``.
    separatrix_R_mid: Optional[float]

    @property
    def size(self):
        return int(self.time.size)

    @property
    def shift(self):
        """How far the profile was moved, in metres of ρ: 0 where it was not."""
        return self.found.shift.by if self.te_sep and self.found.shift.ok else 0.0

    def on(self, axis):
        return self.rho_pol if axis == "rho_pol" else self.R_mid

    def series(self, quantity):
        """``(values, errors)`` of ``"ne"``, ``"Te"`` or ``"pe"`` in SI units, as :meth:`Points.series` gives them: a 1 ms
        mean has no error bar, so the errors are NaN."""
        values = {"ne": self.ne, "Te": self.Te, "pe": ELEMENTARY_CHARGE * self.ne * self.Te}[quantity]
        return values, np.full(values.shape, np.nan)

    def has(self, quantity, axis):
        """Whether a point of ``quantity`` has a place on ``axis``."""
        return bool(np.any(np.isfinite(self.on(axis)) & np.isfinite(self.series(quantity)[0])))


def overlay(profiles, asp, t_start, t_end, source="none"):
    """The MLP's profile from the ASP file ``asp``, over the page's window ``[t_start, t_end]``, placed on the axes of the
    Thomson scattering ``profiles``: an :class:`Overlay`. **The one place this view reads the probe's data.**

    ``source`` names what to move it to (:func:`separatrix_choices`); one the file does not hold moves nothing. The
    reduction and the move are the MLP profiles plot's (:func:`fusion_ui.core.mlp.read`), so the points are those it
    draws.
    """
    te_sep = {name: value for name, value, _ in separatrix_choices(profiles)}.get(source, 0.0)
    found = mlp.read(asp, t_start, t_end, te_sep)
    drawn = found.drawn

    def values(name):
        return np.asarray(drawn[name].values, dtype=float) if name in drawn else np.full(drawn.sizes["window"], np.nan)

    shift = found.shift.by if te_sep and found.shift.ok else 0.0
    rho = values("rho") + shift
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
    """What the discharge DB says of the probe for the shot, as a hint beside the offer: its ``mlp_mode``, quoted."""
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
    """``(strokes, text)``: whether the sidebar offers the MLP's profiles, and the words it says beside the offer.

    ``strokes`` is :func:`fusion_ui.core.mlp.strokes_in` over the page's window, judged from the probe's depth in the
    file's attributes alone, so no variable is read: empty where the probe does not plunge in the window, or the file
    holds no MLP quantity, and the text then says why. The text ends with the discharge DB's ``mlp_mode``
    (:func:`mlp_mode_hint`), a hint the judgement does not rest on.
    """
    hint = mlp_mode_hint(target)
    if not mlp.layout(asp):  # the names alone: a probe file of another kind has no MLP quantity to draw
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
    """The warnings above the figure that the overlay calls for: a move to a separatrix temperature that was asked for
    and not made, and a window with no MLP point to draw. Empty for an overlay drawn as asked."""
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
    """The lines under the figure for the MLP's profile: what its points are, how they were placed on the axes, and how
    many have no place on ``axis``."""
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
    """An empty figure that says ``text``: what is drawn when there is nothing to draw."""
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def _fit_label(fit):
    return "fit" if fit.window is None else f"fit, {_span(fit.window)}"


def _mlp_traces(fig, mlp_points, axis, row, quantity, label, scale, unit, seen):
    """The MLP's points of one row on ``axis``: a trace for each way the probe ran, both in one legend group, so that a
    click on either entry hides the MLP in every row. ``seen`` holds the names already in the legend."""
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
    """n_e, T_e and p_e on one shared ``axis`` (``"rho_pol"`` or ``"R_mid"``), one row each.

    The edge and core points, with their bars; the stored fit over its whole range, without a band; and the separatrix.
    Only what there is is drawn: a row with no point to place and no fit is left out, a system with none is left out
    of the legend, and a file with nothing at all gives a figure that says so. A legend entry appears once, however
    many rows hold its trace or its line (the separatrix is a line in every row), and hides it in all of them.

    The points are drawn where the fit saw them. C-Mod_Analysis 24e1499 moves its fit, and its own copy of the points,
    so that the fit's T_e is the separatrix T_e it found at ρ_pol 1 (``fit_rho_shift``, ``Te_sep_fit``); drawn where
    EFIT puts them, the points would sit beside the fit by that shift, so they are drawn moved with it (:class:`Points`),
    and the separatrix is the fit's ρ_pol 1 and its R_mid there (:func:`separatrix`). Every other file, among them all
    those of d6a6466, which never moves its fit, draws each point where EFIT puts it at its own pulse.

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

    ``mlp_points``, an :class:`Overlay`, draws the Mirror Langmuir Probe's profile over these on its own place on
    ``axis``, in yellow, a triangle pointing the way the probe ran. It does not move a row's range, which the Thomson
    scattering points set: those of its points above the top are counted in the row's corner as well ("▲ 2 MLP points
    above"). Without it, the figure is what it always was.
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
        said = [f"▲ {above} point{'s' * (above > 1)} above"] if above else []
        if mlp_points is not None:  # the probe's points do not set the range, but they are not left out silently either
            values = mlp_points.series(name)[0] * scale
            mlp_above = int((np.isfinite(mlp_points.on(axis)) & (np.nan_to_num(values, nan=-np.inf) > top)).sum())
            if mlp_above:
                said.append(f"▲ {mlp_above} MLP point{'s' * (mlp_above > 1)} above")
        if said:  # nothing is left out silently: the points the top leaves out are counted, in the row's corner
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
    """The sidebar's offer of the MLP's profiles, drawn below the plot's parameters: the name of the separatrix
    temperature to move the profile to (:func:`separatrix_choices`), or ``None`` while the overlay is off or not offered.

    Nothing without an ASP file. Where the probe does not plunge in the window, only the words that say why
    (:func:`offer`); else a toggle with those words under it and, while it is on, the choice of what to move the profile
    to, which starts at not moving it. Both widgets are view state, keyed off the target as the page's are, so a shot
    opens with the overlay off.
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
    """The warnings, if the fit fails its check or its own points are not known to be the file's, the figure, the lines
    under it and what the file says of itself; and, where the probe plunges in the window, the offer of the MLP's
    profiles over these, in the sidebar.

    Draws into Streamlit and returns ``None``: the warnings go above the figure, the notes below it, and the MLP's
    controls below the plot's parameters. ``companions`` is what the single-shot page hands the spec beside its own
    file: ``{"asp": the shot's ASP file, opened lazily, or None}``. Its data are read only while the overlay is on
    (:func:`overlay`), and without the overlay the view is what it was without the file.
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
