"""The MLP profiles view: every quantity of the Mirror Langmuir Probe against ρ, over the probe's plunges in the window.

A *live* spec: nothing is cached. The reduction is the paper's, O E Garcia, J M Losada et al., Plasma Phys. Control.
Fusion 68 (2026) 055034, section 3 (the 1 ms means, against ρ), and it lives in :mod:`fusion_ui.core.mlp`, which says
what the file holds and how each step works. This module draws it.

What is drawn
    One row each for n_e, T_e, J_s, I_s, V_p and V_f, on one shared axis, ρ in cm: the 1 ms means of the four pins
    combined, over the plunges in the page's window. **Each plunge, in and out, is told apart**: a colour for the
    plunge, by its number in the shot, so that plunge 3 has the same colour on every shot, and a marker pointing the
    way the probe ran, filled on the way in (◀) and open on the way out (▷). The four pins are combined, as the paper
    draws one profile per discharge and as the group's own code combines them; how they differ is the probe trace's to
    show, pin by pin. The rows of n_e, T_e, J_s and I_s are logarithmic, as the paper's Fig. 2 is, and those of the
    potentials linear.

    The far-SOL e-folding lengths of n_e and T_e (:func:`fusion_ui.core.mlp.far_sol`) are drawn dashed over the points
    they were fitted to, from the first to the last within the range, and given under the figure. The separatrix is a
    dotted line at ρ = 0.

    **ρ is the file's**, the distance to the separatrix that the tree's equilibrium mapping gives, unless a separatrix
    temperature is given (``te_sep``): the profiles are then moved, as the paper moves its own, so that T_e is that
    temperature at ρ = 0 (:func:`fusion_ui.core.mlp.shift_to`). The fit range stays on the file's ρ, so a shift moves the
    axis and not the fit. The file holds nothing to work the temperature out from.

    **No plunge in the window**, as when the probe dwells, is said in words, and the one chart is then the probe's
    depth over the shot, with the window marked, rather than an empty profile.

Everything but :func:`render` is pure: it takes :func:`fusion_ui.core.mlp.read`'s profiles and returns a figure or text,
so each can be tested without Streamlit.
"""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fusion_ui.core import decimate, mlp, registry

#: The rows: the file's name, how the hover reads it, the axis title, the factor from SI to what is drawn, the unit drawn,
#: and whether the axis is logarithmic. n_e as the Thomson scattering view draws it.
ROWS = (
    ("ne", "n_e", "n<sub>e</sub> [10<sup>20</sup> m<sup>-3</sup>]", 1e-20, "10²⁰ m⁻³", True),
    ("Te", "T_e", "T<sub>e</sub> [eV]", 1.0, "eV", True),
    ("Js", "J_s", "J<sub>s</sub> [kA m<sup>-2</sup>]", 1e-3, "kA m⁻²", True),
    ("Is", "I_s", "I<sub>s</sub> [mA]", 1e3, "mA", True),
    ("Vp", "V_p", "V<sub>p</sub> [V]", 1.0, "V", False),
    ("Vf", "V_f", "V<sub>f</sub> [V]", 1.0, "V", False),
)

#: ρ is drawn in cm, as the paper draws it.
CM = 100.0

#: A plunge's colour, by its number in the shot: the data-viz palette's first three slots, in the steps that clear 3:1 on
#: a light and on a dark page alike, and pass its checks for a scatter in both. A shot plunges at most three times; a
#: fourth and later are drawn grey rather than in a fourth hue, which a scatter could not tell from the third.
PLUNGE_COLOURS = {1: "#2a78d6", 2: "#d95926", 3: "#199e70"}
OTHER_COLOUR = "#898781"
#: The marker of each stroke points the way the probe ran, in towards the core and out towards the wall, and is filled on
#: the way in and open on the way out, so that the two read apart at a glance as well.
STROKE_SYMBOLS = {"in": "triangle-left", "out": "triangle-right-open"}
#: The far-SOL fits, the separatrix and the depth chart's line, in the neutral the Thomson scattering view gives its
#: separatrix.
NEUTRAL = "#898781"


@dataclass
class MlpProfilesParams:
    """
    te_sep: The electron temperature at the separatrix, in eV, to move the profiles to, as the paper does: ρ is shifted
        so that T_e reaches it at ρ = 0. 0 draws ρ as the file holds it. The file holds nothing to work it out from: the
        paper takes it from SOL power balance.
    fit_from: Where the far SOL begins, in cm of ρ as the file holds it, before any shift: the inner end of the
        exponential fits that give L_n and L_T.
    fit_to: Where the far SOL ends, in cm of ρ as the file holds it: the outer end of the fits, where the limiter shadow
        begins.
    """

    te_sep: float = 0.0
    fit_from: float = round(mlp.FAR_SOL[0] * CM, 9)
    fit_to: float = round(mlp.FAR_SOL[1] * CM, 9)


# ---------------------------------------------------------------------------------------------------------------------
# What there is to say
# ---------------------------------------------------------------------------------------------------------------------


def _span(start, end):
    return f"{start:.3f}–{end:.3f} s"


def _window(profiles):
    """The page's window in words, or ``None`` when there is none (the whole record is drawn)."""
    if not np.isfinite([profiles.t_start, profiles.t_end]).all():
        return None
    return _span(profiles.t_start, profiles.t_end)


def stroke_label(plunge, stroke):
    return f"plunge {plunge}, {stroke}"


def drawn_strokes(profiles):
    """``[(plunge, stroke, start, end, points), …]`` of the strokes the profile draws, in the order the probe ran them:
    the middle of their first and last window (s), and how many 1 ms means they hold."""
    drawn = profiles.drawn
    found = []
    for number, stroke in mlp.strokes(drawn):
        mask = (drawn["plunge"].values == number) & (drawn["stroke"].values == stroke)
        times = drawn["time"].values[mask]
        found.append((number, stroke, float(times.min()), float(times.max()), int(mask.sum())))
    return found


def held_still(profiles):
    """What the probe did in the window when no plunge is drawn: ``(depth_min, depth_max, rho_min, rho_max)`` in m over
    the window, NaN where the file does not say."""
    combined = profiles.combined
    depth, rho = combined["depth"].values, combined["rho"].values

    def ends(values):
        values = values[np.isfinite(values)]
        return (float(values.min()), float(values.max())) if values.size else (np.nan, np.nan)

    return (*ends(depth), *ends(rho))


def no_plunge_text(profiles):
    """The words for a window with no profile to draw, or ``None`` when there is one."""
    if profiles.drawn.sizes["window"]:
        return None
    window = _window(profiles)
    where = f"in the window {window}" if window else "in this record"
    if not profiles.means.attrs.get("track"):
        return (
            f"**No profile {where}:** the file has no plunge depth (`plunge_time`, `plunge_depth`), so the probe's "
            "plunges cannot be found."
        )
    moving = profiles.combined["stroke"].values != ""
    if moving.any():
        recorded, partial = mlp.recording(profiles.means)
        if not recorded:
            why = "the MLP recorded nothing while it moved"
        elif partial == recorded:
            why = f"in none of the {recorded} ms the MLP recorded while it moved did every pin record"
        else:
            why = "the file holds no ρ for the milliseconds the MLP recorded, so none of them can be placed"
        return f"**No profile {where}:** the probe plunged, but {why}."
    depth_min, depth_max, rho_min, rho_max = held_still(profiles)
    if np.isfinite(depth_min):
        at = (
            f"at a depth of {depth_min * CM:.1f} cm"
            if abs(depth_max - depth_min) < 0.0005
            else f"between depths of {depth_min * CM:.1f} and {depth_max * CM:.1f} cm"
        )
        if not np.isfinite(rho_min):
            rho = ""
        elif rho_max - rho_min < 0.00005:
            rho = f", where ρ was {rho_min * CM:.2f} cm"
        else:
            rho = f", where ρ ran from {rho_min * CM:.2f} to {rho_max * CM:.2f} cm as the plasma moved"
        return (
            f"**No plunge {where}:** the probe held still {at}{rho}, so there is no profile to draw. The chart is its "
            "depth over the shot, with the window shaded."
        )
    return f"**No plunge {where}:** the probe's depth is not known there, so there is no profile to draw."


def shift_note(profiles, params):
    """The sentence on ρ's shift: how far the profiles were moved and to which T_e, or why they were not."""
    shift = profiles.shift
    if not params.te_sep:
        return (
            "ρ is drawn as the file holds it. The paper moves its profiles so that T_e at ρ = 0 is the separatrix "
            "temperature it works out from SOL power balance, which needs the power into the SOL and the heat-flux "
            "width; the file holds neither. Give one as te_sep to move them."
        )
    if not shift.ok:
        return f"Not moved to T_e = {params.te_sep:g} eV at the separatrix: {shift.reason}."
    return (
        f"Moved by {shift.by * CM:+.2f} cm, as the paper moves its profiles, so that T_e is {shift.te_sep:g} eV at ρ = 0: "
        f"an exponential fit of the {shift.points} 1 ms means of T_e within a factor {mlp.BAND:g} of it puts it at "
        f"{-shift.by * CM:+.2f} cm of the file's ρ."
    )


def fit_note(fits, shift):
    """The sentence that gives the far-SOL e-folding lengths, with the range and the points they were fitted on."""
    any_fit = next(iter(fits.values()))
    lo, hi = any_fit.lo * CM, any_fit.hi * CM
    moved = f" ({lo + shift * CM:.2f} to {hi + shift * CM:.2f} cm on the shifted axis)" if shift else ""
    said = []
    for quantity, label in (("ne", "L_n"), ("Te", "L_T")):
        fit = fits.get(quantity)
        if fit is None:
            continue
        said.append(f"{label} = {fit.length * CM:.2f} cm ({fit.points} points)" if fit.ok else f"no {label}: {fit.reason}")
    return (
        "Far-SOL e-folding lengths (the dashed lines), least-squares fits of an exponential to the 1 ms means between "
        f"ρ = {lo:g} and {hi:g} cm of the file's ρ{moved}: " + "; ".join(said) + "."
    )


def notes(profiles, params):
    """The lines under the figure, as markdown: the points, ρ, the fits and the probe."""
    lines = []
    window = _window(profiles)
    found = drawn_strokes(profiles)
    strokes_said = ", ".join(
        f"{stroke_label(number, stroke)} ({_span(start, end)}, {points})" for number, stroke, start, end, points in found
    )
    recorded, partial = mlp.recording(profiles.means)
    pins = ", ".join(mlp.PIN_NAMES.get(int(pin), str(pin)) for pin in profiles.means["pin"].values)
    over = f"in the window {window}" if window else "over the whole record, which has no discharge-DB window"
    text = (
        f"Points: the 1 ms means of the MLP over the plunges {over}: {strokes_said}. Each is the mean of the "
        f"{profiles.means.sizes['pin']} pins ({pins}) over one millisecond, as the paper reduces its data and the group's "
        "code combines the pins."
    )
    if partial:
        text += (
            f" Of the {recorded} ms the MLP recorded while the probe moved, the {partial} in which not every pin "
            "recorded are left out."
        )
    lines.append(text)
    lines.append(
        "ρ is each pin's distance to the separatrix, mapped to the outboard midplane through the equilibrium, as the "
        "tree's RHO holds it. The file names no unit for it or for any quantity: those on the axes were established "
        "from the data, as the Documentation page says. " + shift_note(profiles, params)
    )
    fits = mlp.far_sol(profiles.drawn, params.fit_from / CM, params.fit_to / CM)
    lines.append(fit_note(fits, profiles.shift.by))
    where = profiles.origin
    lines.append(
        (profiles.probe_type or "A Langmuir probe")
        + " on the horizontal scanning probe"
        + (f", at R = {where[0]:.3f} m and Z = {where[1]:.3f} m (`probe_origin`)." if where else ".")
    )
    return lines


# ---------------------------------------------------------------------------------------------------------------------
# The figures
# ---------------------------------------------------------------------------------------------------------------------


def _number(value, digits=3):
    return "–" if not np.isfinite(value) else f"{value:.{digits}g}"


def _hover(part, row, x):
    """One hover text per point of a stroke: when, where, the probe's depth and the value."""
    name, label, _, scale, unit, _ = row
    values = part[name].values * scale
    texts = []
    for i in range(part.sizes["window"]):
        texts.append(
            f"{stroke_label(int(part['plunge'].values[i]), str(part['stroke'].values[i]))} · "
            f"t = {part['time'].values[i]:.4f} s<br>"
            f"ρ = {x[i]:.2f} cm (file: {part['rho'].values[i] * CM:.2f} cm) · "
            f"depth {part['depth'].values[i] * CM:.2f} cm<br>"
            f"{label} = {_number(values[i])} {unit}"
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


def figure(profiles, params=None):
    """The six quantities against ρ (cm), one row each, sharing that axis: the combined 1 ms means of each stroke, a colour
    for each plunge and a marker for each way, the far-SOL fits of n_e and T_e dashed, and the separatrix dotted.

    A row the file holds no quantity for is left out. The x axis is ρ as the file holds it, moved by the profiles'
    shift when there is one (:attr:`fusion_ui.core.mlp.Profiles.shift`).
    """
    params = params or MlpProfilesParams()
    drawn = profiles.drawn
    rows = [row for row in ROWS if row[0] in drawn and np.isfinite(drawn[row[0]].values).any()]
    if not drawn.sizes["window"] or not rows:
        return _message("No MLP profile in this window.")
    shift = profiles.shift.by if profiles.shift.ok else 0.0
    fits = mlp.far_sol(drawn, params.fit_from / CM, params.fit_to / CM)

    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.025)
    seen = set()
    for number, stroke in mlp.strokes(drawn):
        mask = (drawn["plunge"].values == number) & (drawn["stroke"].values == stroke)
        part = drawn.isel(window=np.flatnonzero(mask))
        x = (part["rho"].values + shift) * CM
        colour = PLUNGE_COLOURS.get(number, OTHER_COLOUR)
        label = stroke_label(number, stroke)
        for k, row in enumerate(rows, start=1):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=part[row[0]].values * row[3],
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    showlegend=label not in seen,
                    marker=dict(
                        symbol=STROKE_SYMBOLS.get(stroke, "circle"),
                        size=8,
                        color=colour,
                        opacity=0.85,
                        line=dict(width=1.5, color=colour),
                    ),
                    hovertext=_hover(part, row, x),
                    hoverinfo="text",
                ),
                row=k,
                col=1,
            )
            seen.add(label)

    for k, row in enumerate(rows, start=1):
        fit = fits.get(row[0])
        if fit is not None and fit.ok:
            rho = np.linspace(fit.first, fit.last, 50)  # over the points it rests on, not the range asked for
            fig.add_trace(
                go.Scatter(
                    x=(rho + shift) * CM,
                    y=fit.at(rho) * row[3],
                    mode="lines",
                    name="far-SOL fit",
                    legendgroup="fit",
                    showlegend="fit" not in seen,
                    line=dict(color=NEUTRAL, width=3, dash="dash"),
                    hovertemplate=f"far-SOL fit of {row[1]}: e-folding length {fit.length * CM:.2f} cm<extra></extra>",
                ),
                row=k,
                col=1,
            )
            seen.add("fit")
        fig.add_vline(
            x=0.0,
            line=dict(color=NEUTRAL, width=1, dash="dot"),
            name="separatrix",
            legendgroup="separatrix",
            showlegend=k == 1,
            row=k,
            col=1,
        )
        # A logarithmic row labels 1, 2 and 5 of each decade, not every minor tick.
        log = dict(type="log", dtick="D2") if row[5] else dict(type="linear")
        fig.update_yaxes(title_text=row[2], row=k, col=1, **log)

    title = "ρ [cm]" if not shift else f"ρ [cm], moved by {shift * CM:+.2f} cm to T<sub>e</sub> = {profiles.shift.te_sep:g} eV"
    fig.update_xaxes(title_text=title, row=len(rows), col=1)
    fig.update_layout(
        height=190 * len(rows) + 120,
        margin=dict(l=70, r=20, t=40, b=60),
        legend=dict(orientation="h", x=0, y=1.0, xanchor="left", yanchor="bottom"),
        hovermode="closest",
    )
    return fig


def depth_figure(profiles, track):
    """The probe's depth over the shot (cm), with the page's window shaded: the one chart of a window with no plunge.

    ``track`` is :func:`fusion_ui.core.mlp.track` of the file. Decimated to its envelope, so a spike survives.
    """
    if track is None:
        return _message("The file has no plunge depth.")
    time, depth = track
    time, depth = decimate.envelope(time, depth * CM, max_points=2000)
    fig = go.Figure(
        go.Scatter(
            x=time,
            y=depth,
            mode="lines",
            name="depth",
            line=dict(color=NEUTRAL, width=2),
            hovertemplate="t = %{x:.4f} s<br>depth %{y:.2f} cm<extra></extra>",
        )
    )
    if np.isfinite([profiles.t_start, profiles.t_end]).all():
        fig.add_vrect(
            x0=profiles.t_start,
            x1=profiles.t_end,
            fillcolor=PLUNGE_COLOURS[1],
            opacity=0.12,
            line_width=0,
            annotation_text="window",
            annotation_position="top left",
        )
    fig.update_xaxes(title_text="time [s]")
    fig.update_yaxes(title_text="probe depth [cm]")
    fig.update_layout(height=320, margin=dict(l=70, r=20, t=30, b=50), showlegend=False)
    return fig


# ---------------------------------------------------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------------------------------------------------


def render(ds, params, target):
    """The profiles and the lines under them, or, with no plunge in the window, the words that say so and the probe's
    depth over the shot. Draws into Streamlit and returns ``None``."""
    import streamlit as st

    if not mlp.layout(ds):
        st.warning("This file holds no MLP quantity to draw (ne, Te, Js, Is, Vp or Vf on a pin).", icon="⚠️")
        return None
    profiles = mlp.read(ds, target.t_start, target.t_end, params.te_sep)
    said = no_plunge_text(profiles)
    if said:
        st.info(said, icon="ℹ️")
        st.plotly_chart(depth_figure(profiles, mlp.track(ds)), width="stretch")
        return None
    if params.te_sep and not profiles.shift.ok:
        st.warning(f"Not moved to T_e = {params.te_sep:g} eV: {profiles.shift.reason}.", icon="⚠️")
    st.plotly_chart(figure(profiles, params), width="stretch")
    for line in notes(profiles, params):
        st.caption(line)
    return None


SPEC = registry.register(
    registry.PlotSpec(
        key="mlp_profiles",
        label="MLP profiles",
        diagnostics=("asp",),
        params=MlpProfilesParams,
        render=render,
        description=(
            "n_e, T_e, J_s, I_s, V_p and V_f of the Mirror Langmuir Probe against ρ, as 1 ms means over each plunge in "
            "the window, with the far-SOL e-folding lengths of n_e and T_e. Read off the file: nothing is cached."
        ),
    )
)
