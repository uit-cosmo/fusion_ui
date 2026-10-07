"""Dead pixels: every pixel's PDF and spectrum, drawn against the mask estimated from them.

The mask comes from ``density_scan.dead_pixels`` (fusion_scripts), the step preprocessing uses to choose which
pixels to interpolate. This spec runs it on the raw file and draws what it looked at, laid out as the array is,
so a mask can be checked by eye the way masks used to be made: one PDF per pixel. Raw files only -- the
preprocessed file has its dead pixels interpolated from their neighbours, so nothing is left to judge there.

The PDFs are histograms aligned on the digitizer's levels. A dead pixel spans a few tens of levels, and bins
finer than a level would draw it as a comb of spikes. One bin per level draws it as the narrow Gaussian it is.
"""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

from fusion_ui.core import loader, registry

N_BINS = 120  # most PDF bins per pixel; fewer when it spans fewer digitizer levels
N_FREQ = 200  # log-spaced PSD points kept for drawing
TAIL = 5e-4  # fraction left out at either end of a PDF's range, so rare spikes do not stretch it
LEVEL_SAMPLES = 200_000  # samples the digitizer step is read off

COLOURS = {-1: "#9e9e9e", 0: "#d62728", 1: "#ff7f0e", 2: "#1f1f1f"}
LABELS = {
    -1: "no data",
    0: "dead",
    1: "live, follows a live neighbour",
    2: "live, red spectrum",
}
VIEWS = ("PDF, standardised", "PDF, volts", "Spectrum")

METHOD = """
A pixel is **live when it sees the plasma**, and the evidence is in its fluctuations, not in its PDF alone. A dim
pixel inside the separatrix has a narrow, near-Gaussian PDF just as a dead one does. A dead channel can be skewed
by rare spikes or by telegraph noise.

1. **Red spectrum.** The Welch PSD of each pixel over the discharge window gives a ratio: the median over
   1–20 kHz (the blob band) divided by the median over 300–900 kHz (digitizer noise only). The medians ignore
   the narrow pickup lines in that upper band. Noise is white, a ratio near 1. A ratio above **{red:g}** means
   the pixel is live.
2. **Neighbours.** A pixel below that is still live if its 2–20 kHz signal follows an already-live neighbour:
   its regression gain on that neighbour exceeds **{gain:g}**. This step repeats until no pixel joins. It keeps
   dim pixels whose own fluctuations are weak. A dead channel with crosstalk carries a few percent of a
   neighbour's signal and stays below the gain. The test uses the gain rather than the correlation, because dead
   channels share pickup with one another and correlate strongly while carrying nothing.
3. **Everything else is dead.**

Hardware does not change within a run day. Preprocessing therefore marks a pixel dead all day when it is dead in
at least a third of the day's shots. This view shows the single shot.

Checked on the 111 raw APD shots on the server. It reproduces the hand-made 1160616 mask on all nine shots. From
2012-02 to 2015-09 it gives the same 18 pixels on every run day; the four extra dead pixels of 2016 died between
1150916 and 1160616. The code is `density_scan/dead_pixels.py` in fusion_scripts.
"""


@dataclass
class DeadPixelParams:
    """
    red: PSD ratio, 1-20 kHz over 300-900 kHz, above which a pixel sees the plasma by itself.
    gain: Regression gain of a pixel's 2-20 kHz signal on a live neighbour's above which it follows that neighbour.
    """

    red: float = 150.0
    gain: float = 0.1


def _pdf(samples):
    """``(centres, density)`` of one pixel, one bin per digitizer level or per group of levels; NaN-padded."""
    centres = np.full(N_BINS + 2, np.nan, dtype=np.float32)
    density = np.full(N_BINS + 2, np.nan, dtype=np.float32)
    s = np.sort(samples[np.isfinite(samples)])
    if s.size < 2 or s[0] == s[-1]:
        return centres, density
    lo, hi = s[int(TAIL * (s.size - 1))], s[int((1 - TAIL) * (s.size - 1))]
    steps = np.diff(np.unique(samples[:LEVEL_SAMPLES]))
    steps = steps[steps > 0]
    level = float(np.median(np.sort(steps)[:10])) if steps.size else (hi - lo) / N_BINS
    width = level * max(1, int(np.ceil((hi - lo) / level / N_BINS)))
    edges = lo - level / 2 + width * np.arange(int(np.floor((hi - lo + level) / width)) + 2)
    counts, edges = np.histogram(s, edges)
    n = min(counts.size, N_BINS + 2)
    centres[:n] = 0.5 * (edges[1:] + edges[:-1])[:n]
    density[:n] = counts[:n] / (s.size * width)
    return centres, density


def compute(ds, params):
    """The mask and its evidence, plus each pixel's PDF and a log-spaced PSD for drawing."""
    from density_scan import dead_pixels

    variable = loader.image_variable(ds)
    record = ds[[variable]].rename({variable: "frames"}).transpose("y", "x", "time").load()
    result = dead_pixels.estimate(record, red=params.red, gain=params.gain, keep_psd=True)

    frames = record.frames.values
    ny, nx, _ = frames.shape
    centres = np.full((ny, nx, N_BINS + 2), np.nan, dtype=np.float32)
    density = np.full_like(centres, np.nan)
    for y in range(ny):
        for x in range(nx):
            centres[y, x], density[y, x] = _pdf(frames[y, x])
    live = np.isfinite(frames).any(-1)  # a pixel with no samples gets NaN moments, without a warning
    mean = np.full((ny, nx), np.nan)
    std = np.full((ny, nx), np.nan)
    mean[live] = np.nanmean(frames[live], axis=-1)
    std[live] = np.nanstd(frames[live], axis=-1)

    nf = result.frequency.size
    keep = np.unique(np.round(np.geomspace(1, nf - 1, N_FREQ)).astype(int))
    return result.isel(frequency=keep).assign(
        pdf=(("y", "x", "bin"), density),
        pdf_volts=(("y", "x", "bin"), centres),
        mean=(("y", "x"), mean),
        std=(("y", "x"), std),
    )


def _cells(mask):
    return ", ".join(f"({y}, {x})" for y, x in np.argwhere(mask)) or "none"


def summary(result):
    ev = result.evidence.values
    dead = result.dead.values
    return (
        f"{int(dead.sum())} of {dead.size} pixels dead: {_cells(dead)} · "
        f"{int((ev == 1).sum())} live only through a neighbour: {_cells(ev == 1)}"
    )


def figure(result, view=VIEWS[0]):
    """One small panel per pixel, placed as on the array (Z up, R to the right), coloured by the verdict.

    Traces, band shading and labels are handed to Plotly in one batch each: added one subplot at a time, the
    90 panels take tens of seconds to assemble.
    """
    ev = result.evidence.values
    ny, nx = ev.shape
    fig = make_subplots(
        rows=ny,
        cols=nx,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.006,
        vertical_spacing=0.006,
    )
    traces, labels, shapes = [], [], []
    for y in range(ny):
        for x in range(nx):
            row, col = ny - y, x + 1
            index = (row - 1) * nx + col
            suffix = "" if index == 1 else str(index)
            if view == "Spectrum":
                xs = result.frequency.values
                ys = result.psd.values[y, x].astype(float)
            else:
                xs = result.pdf_volts.values[y, x].astype(float)
                ys = result.pdf.values[y, x].astype(float)
                if view == "PDF, standardised":
                    std = float(result["std"].values[y, x])
                    if np.isfinite(std) and std > 0:
                        xs = (xs - float(result["mean"].values[y, x])) / std
                        ys = ys * std
            ys = np.where(ys > 0, ys, np.nan)
            ratio = float(result.red_ratio.values[y, x])
            gain = float(result.gain.values[y, x])
            colour = COLOURS[int(ev[y, x])]
            traces.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    xaxis=f"x{suffix}",
                    yaxis=f"y{suffix}",
                    mode="lines",
                    line=dict(color=colour, width=1.2),
                    showlegend=False,
                    hovertemplate=(
                        f"pixel (y={y}, x={x})<br>{LABELS[int(ev[y, x])]}<br>"
                        f"PSD ratio {ratio:.3g} · gain on a live neighbour {gain:.2f}<extra></extra>"
                    ),
                )
            )
            labels.append(
                dict(
                    text=f"{y},{x}",
                    xref=f"x{suffix} domain",
                    yref=f"y{suffix} domain",
                    x=0.02,
                    y=0.95,
                    showarrow=False,
                    font=dict(size=8, color="#9e9e9e"),
                    xanchor="left",
                    yanchor="top",
                )
            )
            labels.append(
                dict(
                    text=f"{ratio:.0f}" if np.isfinite(ratio) else "–",
                    xref=f"x{suffix} domain",
                    yref=f"y{suffix} domain",
                    x=0.98,
                    y=0.95,
                    showarrow=False,
                    font=dict(size=9, color=colour),
                    xanchor="right",
                    yanchor="top",
                )
            )
            if view == "Spectrum":
                for (f0, f1), fill in (
                    ((1e3, 20e3), "rgba(31,119,180,0.12)"),
                    ((300e3, 900e3), "rgba(127,127,127,0.12)"),
                ):
                    shapes.append(
                        dict(
                            type="rect", xref=f"x{suffix}", yref=f"y{suffix} domain",
                            x0=f0, x1=f1, y0=0, y1=1, fillcolor=fill, line_width=0, layer="below",
                        )
                    )
    for code, label in LABELS.items():
        if (ev == code).any():
            traces.append(
                go.Scatter(x=[None], y=[None], mode="lines", name=label, line=dict(color=COLOURS[code], width=3))
            )
    fig.add_traces(traces)
    # Shared axes: Plotly labels the outer row and column only.
    fig.update_yaxes(type="log", zeroline=False, tickfont=dict(size=8), exponentformat="power")
    fig.update_xaxes(zeroline=False, tickfont=dict(size=8))
    if view == "Spectrum":
        fig.update_xaxes(type="log", exponentformat="power")
    elif view == "PDF, standardised":
        # Fixed, because a pixel stuck on a few levels has a standardised density far above 1 and would push
        # every other curve to the bottom of its panel.
        fig.update_xaxes(range=[-5, 10])
        fig.update_yaxes(range=[-4.5, 0.3])
    fig.update_layout(
        annotations=labels,
        shapes=shapes,
        height=105 * ny + 110,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"),
        title=dict(
            text=f"{view}: one panel per pixel (y, x), Z up and R to the right; the number is the PSD ratio"
            + ("; shaded: the two bands it compares" if view == "Spectrum" else ""),
            font=dict(size=13),
            y=0.995,
        ),
    )
    return fig


def _hand_made(target):
    """The hand-made 1160616 mask, for that run day only."""
    if target.shot // 1000 != 1160616:
        return None
    from density_scan.dead_pixel_mask import get_dead_pixel_mask

    return get_dead_pixel_mask().values


def render(result, params, target):
    """Summary, the method, a view toggle and the grid; draws into Streamlit."""
    import streamlit as st

    st.caption(summary(result))
    hand_made = _hand_made(target)
    if hand_made is not None and hand_made.shape == result.dead.shape:
        differ = hand_made != result.dead.values
        st.caption(
            "Agrees with the hand-made 1160616 mask."
            if not differ.any()
            else f"Differs from the hand-made 1160616 mask at {_cells(differ)}."
        )
    with st.expander("How dead pixels are found"):
        st.markdown(METHOD.format(red=params.red, gain=params.gain))
    view = st.radio("Show", VIEWS, horizontal=True, key=f"dead_pixels.view.{target.key}")
    st.plotly_chart(figure(result, view), use_container_width=True)
    return None


def scalars(result):
    out = {"number_dead": float(result.dead.sum())}
    ny, nx = result.dead.shape
    for y in range(ny):
        for x in range(nx):
            out[(x, y, "dead")] = float(result.dead.values[y, x])
            out[(x, y, "psd_ratio")] = float(result.red_ratio.values[y, x])
    return out


SPEC = registry.register(
    registry.PlotSpec(
        key="dead_pixels",
        label="Dead pixels (PDF and spectrum of every pixel)",
        diagnostics=("apd",),
        params=DeadPixelParams,
        render=render,
        compute=compute,
        scalars=scalars,
        preprocessed=False,
        description=(
            "The dead-pixel mask preprocessing uses, estimated from this raw file, with every pixel's PDF and "
            "spectrum laid out as the array is, to check it by eye. Seconds per shot. Emits dead and "
            "psd_ratio at every pixel and number_dead."
        ),
    )
)
