"""The dead-pixel mask a preprocessed file was made with, and the evidence it rests on.

Preprocessing (``density_scan.preprocess`` in fusion_scripts) interpolates the dead pixels away, so the preprocessed
frames show nothing to judge them by: that is what ``dead_pixels`` is for, on the raw file. What the preprocessed file
does keep is the verdict, stored beside the frames, and this view draws it. A live spec: those variables are a few
hundred bytes each, so there is nothing to compute and nothing to cache.

``dead`` (y, x)
    The mask preprocessing used: the run day's (dead in at least a third of the day's shots), or the hand-made one
    on 1160616.
``dead_shot`` (y, x)
    This shot's own verdict, before the run day's rule.
``dead_evidence`` and ``dead_psd_ratio`` (y, x)
    What that verdict rests on: ``density_scan.dead_pixels``' evidence codes (-1 no data, 0 dead, 1 live through a
    neighbour, 2 live by a red spectrum) and the PSD ratio the red-spectrum test compares with its threshold.
attributes
    ``dead_mask_source`` says where the mask came from, in words. ``analysis_window`` and ``discharge_window`` (each
    ``[start, end]`` in seconds) and ``puff_rule`` say which part of the record preprocessing kept and why;
    ``dead_thresholds``, ``preprocess_radius``, ``fusion_scripts_commit`` and ``created`` say how the file was made.

A file made before masks were stored has none of it. On 1160616 the analyses read the hand-made mask for such a file
(``decorrelation.pipeline.dead_mask``), and so does this view, and says so. Anywhere else the mask the file was made
with is unknown, possibly another run day's, and the view says that instead of drawing a guess.

Every variable and attribute is optional but ``dead``: a file that stores only part of it draws what it has.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import plotly.graph_objects as go

from fusion_ui.core import loader, registry
from fusion_ui.plots import dead_pixels

# Why a pixel is dead or live. The first four are density_scan.dead_pixels' evidence codes, the codes the file stores
# in dead_evidence; the other three are classes of this view's own.
NO_DATA, DEAD, NEIGHBOUR, RED = -1, 0, 1, 2
LIVE = 3  # live, and the file does not say how it was found
DEAD_BY_DAY = 4  # dead in the mask, though this shot's own verdict said live
LIVE_BY_DAY = 5  # live in the mask, though this shot's own verdict said dead

#: In legend order. The evidence classes keep the words and the colours of the dead-pixel view, so what is red,
#: orange or dark there is the same here.
LABELS = {
    **dead_pixels.LABELS,
    LIVE: "live",
    DEAD_BY_DAY: "dead by the day's mask, live on this shot",
    LIVE_BY_DAY: "live by the day's mask, dead on this shot",
}
COLOURS = {
    **dead_pixels.COLOURS,
    LIVE: dead_pixels.COLOURS[RED],
    DEAD_BY_DAY: "#9467bd",
    LIVE_BY_DAY: "#1f77b4",
}

#: What the page says about a file that stores no mask and has no hand-made one. ``{day}`` is its run day.
PREDATES = (
    "This file predates stored masks. It records no dead-pixel mask, and none can be assumed for run day {day}: it "
    "may have been made with another run day's. It must be preprocessed again (`density_scan.preprocess` in "
    "fusion_scripts) before anything computed from it is trusted."
)

#: The attributes the line on how the file was made quotes, in its order, with how each reads.
MADE = (
    ("created", "created {}"),
    ("fusion_scripts_commit", "fusion_scripts {}"),
    ("preprocess_radius", "running-normalisation radius {}"),
    ("dead_thresholds", "dead-pixel thresholds {}"),
)


@dataclass
class StoredMaskParams:
    """
    show_ratio: Print each pixel's PSD ratio in its cell, the number the red-spectrum test compares with its threshold.
    """

    show_ratio: bool = True


@dataclass(eq=False)  # arrays do not compare as a whole: identity, not a generated __eq__ that raises
class StoredMask:
    """What a preprocessed file says about its dead pixels. Everything but ``dead`` may be ``None``."""

    #: ``(y, x)`` bool, the mask the file was made with; ``None`` when the file stores none and none can stand in.
    dead: Optional[np.ndarray] = None
    #: ``"stored"`` in the file, ``"hand-made"`` for a 1160616 file that stores none, or ``"missing"``.
    origin: str = "missing"
    #: Where the mask came from, in words.
    source: Optional[str] = None
    #: This shot's own verdict, its evidence codes and PSD ratio, each ``(y, x)``.
    dead_shot: Optional[np.ndarray] = None
    evidence: Optional[np.ndarray] = None
    ratio: Optional[np.ndarray] = None
    #: ``(y, x)`` pixel positions in centimetres, for the hover text.
    R: Optional[np.ndarray] = None
    Z: Optional[np.ndarray] = None
    #: How the file was cropped, as the lines under the map read them.
    analysis_window: Optional[str] = None
    discharge_window: Optional[str] = None
    puff_rule: Optional[str] = None
    made: Optional[str] = None


# ---------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------


def _text(value):
    """An attribute as one line of text: arrays and lists joined, floats short."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, (list, tuple, np.ndarray)):
        return ", ".join(_text(v) for v in np.asarray(value).ravel().tolist())
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    return str(value)


#: ``(start, end)`` in seconds from a ``[start, end]`` attribute (an array, a list or its JSON), else ``None``. The
#: single-shot page reads a file's ``analysis_window`` the same way (``loader.stored_window``).
window = loader.as_window


def _window_text(value):
    """A window attribute as ``"1.0734–1.4000 s"``, or as it is stored when it is not a ``[start, end]``."""
    found = window(value)
    return dead_pixels.seconds(found) if found else _text(value)


def _grid(ds, name):
    """``ds[name]`` as a ``(y, x)`` array, or ``None`` when the file does not store it."""
    return ds[name].transpose("y", "x").values if name in ds else None


def read(ds, target):
    """The :class:`StoredMask` of an open preprocessed file.

    ``target`` names the shot, which decides the fallback: a file that stores no ``dead`` is on the hand-made mask
    for 1160616, as ``decorrelation.pipeline.dead_mask`` reads it, and has none anywhere else.
    """
    out = StoredMask()
    attrs = ds.attrs
    dead = _grid(ds, "dead")
    if dead is not None:
        out.dead = dead.astype(bool)
        out.origin = "stored"
        # The API's own wording for a mask that does not say where it came from.
        out.source = str(attrs.get("dead_mask_source") or "stored in the preprocessed file")
    else:
        by_hand = dead_pixels.hand_made(target)
        if by_hand is not None and by_hand.shape == (ds.sizes["y"], ds.sizes["x"]):
            out.dead = by_hand.astype(bool)
            out.origin = "hand-made"
            out.source = f"hand-made mask, {target.shot // 1000}"

    shot = _grid(ds, "dead_shot")
    out.dead_shot = None if shot is None else shot.astype(bool)
    evidence = _grid(ds, "dead_evidence")
    out.evidence = None if evidence is None else evidence.astype(float)
    ratio = _grid(ds, "dead_psd_ratio")
    out.ratio = None if ratio is None else ratio.astype(float)
    if "R" in ds.coords and "Z" in ds.coords and ds["R"].dims == ("y", "x") and ds["Z"].dims == ("y", "x"):
        out.R, out.Z = ds["R"].values, ds["Z"].values

    if "analysis_window" in attrs:
        out.analysis_window = _window_text(attrs["analysis_window"])
    if "discharge_window" in attrs:
        out.discharge_window = _window_text(attrs["discharge_window"])
    if attrs.get("puff_rule"):
        out.puff_rule = _text(attrs["puff_rule"])
    parts = []
    for name, wording in MADE:
        text = _text(attrs[name]) if name in attrs else ""
        if text:
            parts.append(wording.format(text))
    out.made = " · ".join(parts) or None
    return out


# ---------------------------------------------------------------------------
# What there is to say about it
# ---------------------------------------------------------------------------


def classify(dead, dead_shot=None, evidence=None):
    """The class of every pixel, ``(y, x)`` ints from :data:`LABELS`.

    ``dead`` is the mask the file was made with. Where it disagrees with ``dead_shot``, the shot's own verdict, the
    run day's mask overrode the shot: ``DEAD_BY_DAY`` or ``LIVE_BY_DAY``. Elsewhere the class is the evidence code
    when there is one, and plain ``DEAD`` or ``LIVE`` when there is not. A shot's verdict missing from the file is
    its evidence's (dead at 0 or below), and with neither no pixel counts as overridden.
    """
    dead = np.asarray(dead, dtype=bool)
    if evidence is not None:
        evidence = np.asarray(evidence, dtype=float)
    if dead_shot is None:
        dead_shot = dead if evidence is None else np.where(np.isfinite(evidence), evidence <= 0, dead)
    dead_shot = np.asarray(dead_shot, dtype=bool)

    classes = np.where(dead, DEAD, LIVE)
    if evidence is not None:
        classes = np.where(dead & (evidence == NO_DATA), NO_DATA, classes)
        classes = np.where(~dead & (evidence == NEIGHBOUR), NEIGHBOUR, classes)
        classes = np.where(~dead & (evidence == RED), RED, classes)
    classes = np.where(dead & ~dead_shot, DEAD_BY_DAY, classes)
    classes = np.where(~dead & dead_shot, LIVE_BY_DAY, classes)
    return classes.astype(int)


def classes_of(mask):
    return classify(mask.dead, mask.dead_shot, mask.evidence)


def summary(mask):
    """The counts and the pixels behind them, in one line, as ``dead_pixels.summary`` does for a raw file."""
    classes = classes_of(mask)
    parts = [f"{int(mask.dead.sum())} of {mask.dead.size} pixels dead: {dead_pixels.cells(mask.dead)}"]
    through = classes == NEIGHBOUR
    if through.any():
        parts.append(f"{int(through.sum())} live only through a neighbour: {dead_pixels.cells(through)}")
    if mask.dead_shot is not None or mask.evidence is not None:
        dead_here, live_here = classes == DEAD_BY_DAY, classes == LIVE_BY_DAY
        if dead_here.any() or live_here.any():
            said = []
            if dead_here.any():
                said.append(f"dead in the mask, live on this shot at {dead_pixels.cells(dead_here)}")
            if live_here.any():
                said.append(f"live in the mask, dead on this shot at {dead_pixels.cells(live_here)}")
            parts.append("the run day's mask overrides this shot's own verdict: " + "; ".join(said))
        else:
            parts.append("the mask agrees with this shot's own verdict at every pixel")
    return " · ".join(parts)


def source_line(mask):
    """The line under the map: where the mask came from."""
    if mask.origin == "hand-made":
        return (
            f"Mask source: **{mask.source}**. The file stores no mask, so the hand-made one for this run day stands "
            "in, as the analyses read it."
        )
    return f"Mask source: **{mask.source}**."


def window_line(mask):
    """The analysis window beside the discharge window and the puff rule that chose it, in the file's words.

    ``None`` when the file records none of the three: a file made before the puff was looked for.
    """
    return dead_pixels.window_line(mask.analysis_window, mask.discharge_window, mask.puff_rule)


def made_line(mask):
    """How the file was made, from what it records of that, or ``None`` when it records nothing."""
    return None if mask.made is None else f"Preprocessing: {mask.made}"


# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------


def _ink(fill):
    """Text that reads on ``fill``, a ``#rrggbb`` colour: white on a dark cell, near-black on a light one."""
    r, g, b = (int(fill[i : i + 2], 16) for i in (1, 3, 5))
    return "#ffffff" if 0.299 * r + 0.587 * g + 0.114 * b < 140 else "#1f1f1f"


def _ratio_text(ratio):
    if not np.isfinite(ratio):
        return "–"
    return f"{ratio:.1f}" if ratio < 10 else f"{ratio:.0f}"


def _steps(colours):
    """A colour scale of equal steps, one colour each: a categorical heatmap."""
    n = len(colours)
    scale = []
    for i, colour in enumerate(colours):
        scale += [[i / n, colour], [(i + 1) / n, colour]]
    return scale


def figure(mask, show_ratio=True):
    """One cell per pixel as the array is laid out (Z up, R to the right), coloured by what the mask says of it.

    Every class has its colour in every figure, so a colour means the same thing whichever classes a file has, and
    the legend lists only those present. Hovering gives the pixel's position, the verdict and the PSD ratio.
    """
    classes = classes_of(mask)
    ny, nx = classes.shape
    order = list(LABELS)
    z = np.vectorize(order.index)(classes)
    has_ratio = mask.ratio is not None

    hover, annotations = [], []
    for y in range(ny):
        row = []
        for x in range(nx):
            lines = [f"pixel (y={y}, x={x})"]
            if mask.R is not None:
                lines[0] += f" · R {float(mask.R[y, x]):.2f} cm, Z {float(mask.Z[y, x]):.2f} cm"
            lines.append(LABELS[int(classes[y, x])])
            if has_ratio:
                lines.append(f"PSD ratio {_ratio_text(float(mask.ratio[y, x]))}")
            row.append("<br>".join(lines))
            if show_ratio and has_ratio:
                annotations.append(
                    dict(
                        text=_ratio_text(float(mask.ratio[y, x])),
                        x=x,
                        y=y,
                        xref="x",
                        yref="y",
                        showarrow=False,
                        font=dict(size=12, color=_ink(COLOURS[int(classes[y, x])])),
                    )
                )
        hover.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(range(nx)),
            y=list(range(ny)),
            text=hover,
            hoverinfo="text",
            colorscale=_steps([COLOURS[code] for code in order]),
            zmin=-0.5,
            zmax=len(order) - 0.5,
            showscale=False,
            xgap=3,
            ygap=3,
        )
    )
    for code in order:
        if (classes == code).any():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=LABELS[code],
                    # Outlined, because the dark class is all but invisible on a dark page.
                    marker=dict(symbol="square", size=12, color=COLOURS[code], line=dict(width=1, color="#9e9e9e")),
                )
            )
    fig.update_xaxes(
        title="x, R increasing to the right",
        tickmode="linear",
        dtick=1,
        showgrid=False,
        zeroline=False,
        constrain="domain",
    )
    fig.update_yaxes(
        title="y, Z increasing upward",
        tickmode="linear",
        dtick=1,
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
    )
    fig.update_layout(
        annotations=annotations,
        height=62 * ny + 150,
        margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"),
        title=dict(
            text="The mask this file was made with: one cell per pixel (y, x)"
            + ("; the number is the PSD ratio" if show_ratio and has_ratio else ""),
            font=dict(size=13),
            y=0.995,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def render(ds, params, target):
    """The counts, the plain words and the method, the map, and what the file says about it; draws into Streamlit."""
    import streamlit as st

    mask = read(ds, target)
    if mask.dead is None:
        st.warning(PREDATES.format(day=target.shot // 1000), icon="⚠️")
    else:
        st.caption(summary(mask))
    dead_pixels.explain()
    if mask.dead is not None:
        st.plotly_chart(figure(mask, params.show_ratio), use_container_width=True)
        st.caption(source_line(mask))
    for line in (window_line(mask), made_line(mask)):
        if line:
            st.caption(line)
    return None


SPEC = registry.register(
    registry.PlotSpec(
        key="stored_mask",
        label="Dead-pixel mask stored at preprocessing",
        diagnostics=("apd",),
        params=StoredMaskParams,
        render=render,
        preprocessed=True,
        description=(
            "The dead-pixel mask this preprocessed file was made with, this shot's own verdict beside it, and the "
            "window it was judged over. Read off the file: nothing is computed."
        ),
    )
)
