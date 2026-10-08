"""Dead pixels: every pixel's PDF and spectrum, drawn against the mask estimated from them.

The mask comes from ``density_scan.dead_pixels`` (fusion_scripts), the step preprocessing uses to choose which
pixels to interpolate. This spec runs it on the raw file and draws what it looked at, laid out as the array is,
so a mask can be checked by eye the way masks used to be made: one PDF per pixel. Raw files only -- the
preprocessed file has its dead pixels interpolated from their neighbours, so nothing is left to judge there. What
that file does keep is the mask it was made with, which ``stored_mask`` draws.

The two views explain the method alike: a plain summary for a physicist new to the code, always shown, and the
technical ``METHOD`` under it in an expander. The summary is hand-written Markdown in ``fusion_ui/data/``, beside
``run_days.md``; ``METHOD`` is below.

The PDFs are histograms aligned on the digitizer's levels. A dead pixel spans a few tens of levels, and bins
finer than a level would draw it as a comb of spikes. One bin per level draws it as the narrow Gaussian it is.

**The window.** The mask is judged over the *analysis window*: the discharge window cut to the gas puff
(``density_scan.puff``), the window preprocessing crops its files to. On many shots the discharge window starts
before the puff, and the live pixels' dark stretch there gives bimodal PDFs and dilutes the spectra. The puff is
found against the dark level at the *start of the record*, so this spec declares ``whole_record``: it is handed the
whole record, with the discharge window in its attributes (``loader.discharge_window``), finds the puff on it, and
judges and draws its PDFs and spectra over the analysis window only. Cut to the discharge window first, as every
other spec's input is, a window that began after the puff had risen would read "already on", and the view would
differ from preprocessing. The array mean and what the search found of it are drawn above the grid, so the cut
can be checked by eye.

The result holds, beside ``dead``, ``evidence``, ``red_ratio``, ``gain``, ``psd``, ``pdf``, ``pdf_volts``, ``mean``
and ``std`` (all over the analysis window), the search for the puff: the smoothed array mean ``puff_signal`` over
``puff_time`` (the whole record, at most 4000 points), and the attributes ``analysis_window``, ``discharge_window``
and ``record_window`` (each ``[start, end]`` in seconds), ``puff_found``, ``puff_start``, ``puff_end``,
``puff_baseline``, ``puff_level``, ``puff_level_quantile``, ``puff_threshold``, ``puff_note``, ``puff_rule`` and,
when the signal dips below the threshold inside the window, ``puff_dips`` (start, end, start, end, ...). The four
that ``density_scan.dead_pixels.estimate_shot`` also gives its result carry its names. A result computed before this
has none of it: :func:`puff_of` says so, and the view says that it predates the puff window.
"""

import os
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

from fusion_ui.core import fusion_scripts, loader, registry

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

# ---------------------------------------------------------------------------------------------------------------
# The words. Both dead-pixel views show a plain summary, always, and METHOD under it in a collapsed expander: this
# view, on the raw file, and ``stored_mask``, on the preprocessed one. Both go through ``explain`` below.
#
# The plain summary is hand-written Markdown in fusion_ui/data/dead_pixels_in_plain_words.md, beside run_days.md, so
# that rewording it is an edit to that file and nothing else. It is read at every rerun: an edit shows on the next
# one, with no restart. METHOD is the technical text, Markdown with the two thresholds in force filled in as
# {red:g} and {gain:g}.
# ---------------------------------------------------------------------------------------------------------------

PLAIN_SUMMARY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dead_pixels_in_plain_words.md"
)


def plain_summary():
    """The plain summary, as ``PLAIN_SUMMARY_PATH`` holds it now."""
    with open(PLAIN_SUMMARY_PATH, encoding="utf-8") as f:
        return f.read().strip()


#: The title of the expander that holds METHOD.
METHOD_TITLE = "How dead pixels are found"

METHOD = """
A pixel is **live when it sees the plasma**, and the evidence is in its fluctuations, not in its PDF alone. A dim
pixel inside the separatrix has a narrow, near-Gaussian PDF just as a dead one does. A dead channel can be skewed
by rare spikes or by telegraph noise.

**The window.** The judgement is made over the *analysis window*: the discharge window of the discharge DB, cut to
the gas puff. On many shots the discharge window starts before the puff, while the live pixels still sit at the
digitizer's dark level, and that dark stretch gives bimodal PDFs and dilutes the spectra. The puff is found in the
array-mean light, smoothed over 1 ms: it starts where the signal crosses halfway between its dark level at the
start of the record and its median over the discharge window, and stays across for at least 5 ms. Where most of
the window is dark after a short puff, the window's 90th percentile stands for the light in place of its median.
A record that starts with the light already on, or has no clear rise, keeps the discharge window, and says why. A
dip below the halfway level inside the window, as between two puffs, is reported and not cut. The raw file's view
draws the array mean with both windows marked; a preprocessed file is cropped to the analysis window, and its view
quotes it.

1. **Red spectrum.** The Welch PSD of each pixel over the analysis window gives a ratio: the median over
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
at least a third of the day's shots. The raw file's view shows this shot's own verdict, before that rule; a
preprocessed file's view shows the mask the file was made with, which is the day's, and marks the pixels where it
overrides the shot's own verdict.

Checked on the 111 raw APD shots on the server. It reproduces the hand-made 1160616 mask on all nine shots. From
2012-02 to 2015-09 it gives the same 18 pixels on every run day; the four extra dead pixels of 2016 died between
1150916 and 1160616. Cut to the gas puff, every one of the 111 shots gets the verdict it got over the whole
discharge window, and so every run day gets the same mask. The code is `density_scan/dead_pixels.py` and
`density_scan/puff.py` in fusion_scripts.
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
    """The mask over the analysis window and its evidence, each pixel's PDF and a log-spaced PSD for drawing, and
    the search for the gas puff that chose the window.

    ``ds`` is the whole record with its discharge window in its attributes (``whole_record``): the puff is found
    on all of it, against the dark level at its start, and everything else is judged over the analysis window.
    This is ``density_scan.dead_pixels.estimate_shot(shot, window="puff")``, step for step, on a record in hand.
    """
    fusion_scripts.import_config()  # density_scan reads its settings by the bare name `config`
    from density_scan import dead_pixels, puff

    discharge = loader.discharge_window(ds)
    variable = loader.image_variable(ds)
    record = ds[[variable]].rename({variable: "frames"}).transpose("y", "x", "time").load()
    found = puff.puff_window(record, discharge)
    analysed = record.sel(time=slice(*found.analysis_window))
    result = dead_pixels.estimate(analysed, red=params.red, gain=params.gain, keep_psd=True)

    frames = analysed.frames.values
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
    out = (
        result.isel(frequency=keep)
        .assign(
            pdf=(("y", "x", "bin"), density),
            pdf_volts=(("y", "x", "bin"), centres),
            mean=(("y", "x"), mean),
            std=(("y", "x"), std),
            puff_signal=("puff_time", found.signal),
        )
        .assign_coords(puff_time=found.time)
    )
    out.attrs.update(
        analysis_window=list(map(float, found.analysis_window)),
        discharge_window=list(map(float, found.discharge_window)),
        record_window=list(map(float, found.record_window)),
        puff_found=int(found.found),
        puff_start=float(found.start),
        puff_end=float(found.end),
        puff_baseline=float(found.baseline),
        puff_level=float(found.level),
        puff_level_quantile=float(found.level_quantile),
        puff_threshold=float(found.threshold),
        puff_note=found.note,
        puff_rule=found.rule,
    )
    if found.dips:  # a netCDF attribute cannot be empty: no dips, no attribute
        out.attrs["puff_dips"] = [float(edge) for dip in found.dips for edge in dip]
    return out


# ---------------------------------------------------------------------------------------------------------------
# The search for the puff, as the result stores it
# ---------------------------------------------------------------------------------------------------------------


@dataclass(eq=False)  # arrays inside: identity, not a generated __eq__ that raises
class Puff:
    """What the result stores of the search for the gas puff, read back by :func:`puff_of`.

    Times are seconds and the signal volts. ``analysis_window`` is ``discharge_window`` whenever ``found`` is
    False, and ``note`` then says why. ``start`` and ``end`` are NaN when the puff was not found.
    """

    found: bool
    analysis_window: tuple
    discharge_window: tuple
    record_window: tuple
    start: float
    end: float
    baseline: float  # the dark level at the start of the record
    level: float  # the light: the window's median, or its ``level_quantile`` where the median is the dark level
    level_quantile: float
    threshold: float  # halfway between baseline and level
    dips: tuple  # ((start, end), ...) below the threshold inside the analysis window
    note: str  # what was decided about this record, and why
    rule: str  # the rule in words, with the note: what a preprocessed file stores as ``puff_rule``
    time: np.ndarray  # the smoothed array mean over the whole record, for drawing
    signal: np.ndarray


#: What a result has to carry to say it holds the search for the puff.
_PUFF_ATTRS = (
    "analysis_window",
    "discharge_window",
    "record_window",
    "puff_found",
    "puff_start",
    "puff_end",
    "puff_baseline",
    "puff_level",
    "puff_level_quantile",
    "puff_threshold",
    "puff_note",
    "puff_rule",
)


def puff_of(result):
    """The :class:`Puff` a result stores, or ``None`` for one that predates the puff window.

    A result computed before the view looked for the puff has neither the smoothed array mean nor the attributes.
    One that has part of it is treated the same way: the figure of a cut needs all of it.
    """
    if "puff_signal" not in result or any(name not in result.attrs for name in _PUFF_ATTRS):
        return None
    a = result.attrs
    windows = [loader.as_window(a[name]) for name in ("analysis_window", "discharge_window", "record_window")]
    if any(window is None for window in windows):
        return None
    dips = np.asarray(a.get("puff_dips", []), dtype=float).reshape(-1, 2)
    return Puff(
        found=bool(a["puff_found"]),
        analysis_window=windows[0],
        discharge_window=windows[1],
        record_window=windows[2],
        start=float(a["puff_start"]),
        end=float(a["puff_end"]),
        baseline=float(a["puff_baseline"]),
        level=float(a["puff_level"]),
        level_quantile=float(a["puff_level_quantile"]),
        threshold=float(a["puff_threshold"]),
        dips=tuple((float(start), float(end)) for start, end in dips),
        note=str(a["puff_note"]),
        rule=str(a["puff_rule"]),
        time=result["puff_time"].values.astype(float),
        signal=result["puff_signal"].values.astype(float),
    )


def cells(mask):
    """The pixels of a ``(y, x)`` bool mask as ``"(y, x), (y, x)"``, or ``"none"``."""
    return ", ".join(f"({y}, {x})" for y, x in np.argwhere(mask)) or "none"


def summary(result):
    ev = result.evidence.values
    dead = result.dead.values
    return (
        f"{int(dead.sum())} of {dead.size} pixels dead: {cells(dead)} · "
        f"{int((ev == 1).sum())} live only through a neighbour: {cells(ev == 1)}"
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


# ---------------------------------------------------------------------------------------------------------------
# The puff window, drawn
# ---------------------------------------------------------------------------------------------------------------

#: The colours of the puff figure. Chosen to read on a light and a dark page alike, and apart from the evidence
#: colours above, which belong to the grid: a pixel's colour there means a verdict.
PUFF_COLOURS = {
    "signal": "#1f77b4",
    "discharge": "#7f7f7f",
    "analysis": "#2ca02c",
    "baseline": "#7f7f7f",
    "level": "#17becf",
    "threshold": "#9467bd",
    "dip": "#d62728",
    "edge": "#ff7f0e",
}

#: What the page says about a result that holds no search for the puff: all 111 cached ones, until they are computed
#: again. The grid below it is still drawn, over the discharge window it was computed on.
PREDATES = (
    "This result predates the puff window. It was computed over the whole discharge window, dark stretch before "
    "the gas puff included, and it records no search for the puff, so there is nothing to draw of the cut and the "
    "mask below may differ from the one preprocessing now makes. Recompute it (the button under the grid) to judge "
    "it over the analysis window."
)


def seconds(window):
    """A ``(start, end)`` window in seconds as ``"1.0734–1.4000 s"``."""
    return f"{window[0]:.4f}–{window[1]:.4f} s"


def window_line(analysis_window=None, discharge_window=None, rule=None):
    """The analysis window beside the discharge window and the puff rule that chose it, in words.

    The arguments are the text of each, ``None`` for one that is not known; the result is ``None`` when none of the
    three is. Both dead-pixel views end with this line, the raw one under the figure of the cut and the preprocessed
    one from the file's own attributes (``stored_mask``), so the two read alike.
    """
    parts = []
    if analysis_window is not None:
        parts.append(f"analysis window {analysis_window}")
    if discharge_window is not None:
        parts.append(f"discharge window {discharge_window}")
    if rule is not None:
        parts.append(f"puff rule: {rule}")
    line = " · ".join(parts)
    return line[:1].upper() + line[1:] if line else None


def _ordinal(n):
    suffix = "th" if n % 100 in (11, 12, 13) else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def light_level(quantile):
    """How the light level is named: ``"median"``, or ``"90th percentile"`` where the median is the dark level."""
    return "median" if quantile == 0.5 else f"{_ordinal(round(100 * quantile))} percentile"


def _rgba(colour, alpha):
    """``"#rrggbb"`` as an ``rgba(...)`` string."""
    r, g, b = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def puff_figure(puff):
    """The smoothed array mean over the whole record, with the cut marked, so that it can be checked by eye.

    Shaded: the discharge window, and over it the analysis window, which is what the mask below was judged on; and
    each dip below the threshold that the cut leaves in. The lines are what the rule compared the signal with: its
    dark level at the start of the record, the light level of the window, and the threshold halfway between. The
    triangles mark where the puff comes on and goes off (no "off" for a puff still on when the record ends). Drawn
    the same when the puff was not found: the lines then show what the rule had to stand on, and the note under the
    figure says what it lacked.
    """
    first, last = puff.record_window
    d0, d1 = puff.discharge_window
    a0, a1 = puff.analysis_window
    colour = PUFF_COLOURS

    def band(x0, x1, key, alpha, y0=0.0, y1=1.0):
        return dict(
            type="rect",
            xref="x",
            yref="y domain",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            fillcolor=_rgba(colour[key], alpha),
            line_width=0,
            layer="below",
        )

    shapes = [band(d0, d1, "discharge", 0.16), band(a0, a1, "analysis", 0.22)]
    # A dip lies inside the analysis window: a strip along the bottom of it, since a third translucent fill over the
    # other two would only turn them brown.
    shapes += [band(start, end, "dip", 0.7, y0=0.0, y1=0.05) for start, end in puff.dips]

    fig = go.Figure(
        go.Scatter(
            x=puff.time,
            y=puff.signal,
            mode="lines",
            name="array mean, 1 ms mean",
            line=dict(color=colour["signal"], width=1.4),
            hovertemplate="%{x:.4f} s: %{y:.3f} V<extra>array mean</extra>",
        )
    )
    for name, value, key, dash in (
        ("baseline", puff.baseline, "baseline", "dash"),
        (f"level ({light_level(puff.level_quantile)})", puff.level, "level", "dash"),
        ("threshold", puff.threshold, "threshold", "dot"),
    ):
        fig.add_trace(
            go.Scatter(
                x=[first, last],
                y=[value, value],
                mode="lines",
                name=f"{name} {value:.3f} V",
                line=dict(color=colour[key], width=1.2, dash=dash),
                hovertemplate=f"{name}: {value:.3f} V<extra></extra>",
            )
        )
    if puff.found:
        # A puff still on when the record ends does not go off there: the rule's "end" is then the record's last
        # sample, which is no event, and a triangle at the edge of the axes would pass for one.
        events = [("on", puff.start, "triangle-up")]
        if puff.end < puff.time[-1]:
            events.append(("off", puff.end, "triangle-down"))
        when = [t for _, t, _ in events]
        fig.add_trace(
            go.Scatter(
                x=when,
                y=np.interp(when, puff.time, puff.signal),
                mode="markers",
                name="puff " + ", ".join(name for name, _, _ in events),
                marker=dict(symbol=[symbol for _, _, symbol in events], size=11, color=colour["edge"]),
                customdata=[name for name, _, _ in events],
                hovertemplate="puff %{customdata} at %{x:.4f} s<extra></extra>",
            )
        )
    for name, key, shown in (
        ("discharge window", "discharge", True),
        ("analysis window", "analysis", True),
        ("dip below the threshold, not cut", "dip", bool(puff.dips)),
    ):
        if shown:  # a legend entry for a band, which a shape cannot give
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=name,
                    line=dict(color=_rgba(colour[key], 0.45), width=9),
                )
            )
    fig.update_xaxes(title="time (s)", range=[first, last], zeroline=False, automargin=True)
    fig.update_yaxes(title="array mean (V)", zeroline=False, automargin=True)
    fig.update_layout(
        shapes=shapes,
        height=340,
        margin=dict(l=10, r=10, t=95, b=10),
        hovermode="x",
        legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"),
        title=dict(
            text="Where the gas puff is: the array-mean light over the whole record", font=dict(size=13), y=0.995
        ),
    )
    return fig


def hand_made(target):
    """The hand-made 1160616 mask as a ``(y, x)`` bool array, for that run day only; ``None`` for any other."""
    if target.shot // 1000 != 1160616:
        return None
    fusion_scripts.import_config()  # as in compute
    from density_scan.dead_pixel_mask import get_dead_pixel_mask

    return get_dead_pixel_mask().values


def explain(red=DeadPixelParams.red, gain=DeadPixelParams.gain):
    """The plain summary, always shown, and the technical method in an expander under it.

    Both dead-pixel views end their opening lines with this, so the two read alike and the words are in one place.
    ``red`` and ``gain`` are the thresholds METHOD quotes: the form's on a raw file, the defaults where a view has
    no form for them.
    """
    import streamlit as st

    try:
        st.markdown(plain_summary())
    except OSError as error:  # a broken deployment must not take the technical text down with it
        st.warning(f"The plain-language summary could not be read: {error}", icon="⚠️")
    with st.expander(METHOD_TITLE):
        st.markdown(METHOD.format(red=red, gain=gain))


def show_puff(result):
    """The figure of the cut with the window line under it, drawn above the grid it chose the window of.

    A result that predates the puff window has no search to draw: the note saying so stands in its place, and the
    grid is drawn all the same.
    """
    import streamlit as st

    found = puff_of(result)
    if found is None:
        st.info(PREDATES, icon="ℹ️")
        return
    st.plotly_chart(puff_figure(found), use_container_width=True)
    st.caption(window_line(seconds(found.analysis_window), seconds(found.discharge_window), found.rule))


def render(result, params, target):
    """Summary, the plain words and the method, the puff window, a view toggle and the grid; draws into Streamlit."""
    import streamlit as st

    st.caption(summary(result))
    by_hand = hand_made(target)
    if by_hand is not None and by_hand.shape == result.dead.shape:
        differ = by_hand != result.dead.values
        st.caption(
            "Agrees with the hand-made 1160616 mask."
            if not differ.any()
            else f"Differs from the hand-made 1160616 mask at {cells(differ)}."
        )
    explain(params.red, params.gain)
    show_puff(result)
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
        whole_record=True,
        description=(
            "The dead-pixel mask preprocessing uses, estimated from this raw file over the part of the record "
            "with the gas puff on (the cut is drawn first), with every pixel's PDF and spectrum laid out as "
            "the array is, to check it by eye. Reads the whole record: seconds per shot. Emits dead and "
            "psd_ratio at every pixel and number_dead."
        ),
    )
)
