"""The Mirror Langmuir Probe's profiles: its plunges, the paper's 1 ms means, ρ, and the far-SOL e-folding lengths.

The reduction behind the "MLP profiles" view (:mod:`fusion_ui.plots.mlp_profiles`), kept apart from the view so that
another can reuse it rather than copy it: the Thomson scattering view draws the same points on its own axes, from the
ASP file the single-shot page hands it as a companion, and judges whether the probe plunges in its window from the track
alone (:func:`strokes_in`). Pure: numpy, scipy and xarray, no Streamlit, no database, and no file but the dataset it is
handed.

The paper it reproduces is O E Garcia, J M Losada et al., "Velocity scaling regimes for blob-like filaments in the
Alcator C-Mod scrape-off layer", Plasma Phys. Control. Fusion 68 (2026) 055034, section 3: "The resulting measurement
time series are divided into windows of one millisecond duration, for which the mean value is calculated for the local
electron density and temperature." Its Fig. 2 draws those means against ρ, and its far-SOL e-folding lengths are
least-squares exponential fits to them (the caption of the same figure in its reference [63], Garcia et al., Nucl.
Fusion 66 (2026)).

What an ASP file holds
    ``cmod_functions.asp_functions.get_asp_dataset`` (uit-cosmo/cmod_functions) writes it from C-Mod's edge tree, and is
    the reference for the layout.

    ``<q>_<pin>`` on its own axis ``time_<q>_<pin>``
        The MLP's own values, one sample every 0.3 µs while it records: ``ne`` (the tree's DENSITY_FIT), ``Te``
        (TE_FIT), ``Js`` (JSAT_FIT), ``Is`` (ISAT_FIT), ``Vp`` (PHI_FIT, the plasma potential) and ``Vf`` (VF_FIT), at
        each of the four pins of the probe head: p0 to p3, the NE, SW, SE and NW pins by cmod_functions' own note. The
        MLP records around the plunges only, so an axis has gaps where the probe was parked.
    ``rho_<q>_<pin>`` on ``rho_time_<q>_<pin>``
        The tree's RHO of the pin, at 10 kHz over the whole shot, the same for every quantity of one pin: the pin's
        distance to the separatrix (:data:`UNITS` says how its unit was established). It reads exactly 0 where the
        tree has no mapping, before and after the plasma, and such samples are left out of a window's mean.
    the attributes ``plunge_time`` and ``plunge_depth``
        The probe's depth at 10 kHz, from the tree's PLUNGE node, "in metres and assumed purely horizontal"
        (cmod_functions). It grows as the probe goes in. Stored as arrays; a file that holds them as text is read too.

**No variable has a units attribute.** :data:`UNITS` gives each quantity's, with how it was established.

The reduction
    **Plunges** (:func:`plunges`) are found on the depth: a plunge is a maximum of depth at least :data:`PROMINENCE`
    deeper than where the probe started from and goes back to, numbered in the order of the shot, and it runs from the
    shallowest point before it to the shallowest point after it. The probe is moving while its depth changes faster
    than :data:`MOVING`: inward on the way **in**, outward on the way **out**. Where it holds still, as in a dwell, it
    is on no stroke, and its data make no profile.

    **1 ms means** (:func:`means`): the shot's clock is cut into windows of :data:`WINDOW`, on whole milliseconds, and
    every quantity of every pin is averaged over each window that lies wholly inside the page's window, with the pin's ρ
    at the samples that make the mean. A pin's mean counts only where it recorded at least :data:`MIN_FILL` of the
    window. Each window is labelled with the plunge it falls in and the stroke the probe was on.

    **The four pins** (:func:`combine`) are combined as the group's own code combines them
    (``cmod_functions.asp_functions.generate_average_mlp_data``, the mean of the four): one point per window, the mean
    of the pins' means, where every pin the file holds recorded. They sit on the same flux surface, their ρ agreeing to
    a tenth of a millimetre, but their n_e differ by about ±13% and their V_f by several volts, two by two, as the
    pins of a head facing both ways along the field do.

    **The far-SOL e-folding length** (:func:`efold`) of n_e or T_e: a least-squares fit of ``ln y = ln A − ρ/L`` to the
    combined means between two values of ρ, :data:`FAR_SOL` by default, on ρ as the file holds it.

    **The separatrix** (:func:`shift_to`): the paper moves its profiles so that T_e at ρ = 0 is the separatrix
    temperature from SOL power balance, which needs what the file does not hold (the power into the SOL, the heat-flux
    width). Given such a temperature, :func:`shift_to` finds where the combined T_e reaches it and says how far to move
    ρ; it never extrapolates.
"""

import re
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

#: The paper's window: "windows of one millisecond duration, for which the mean value is calculated".
WINDOW = 1e-3

#: The six quantities of the MLP, in the order the view draws them.
QUANTITIES = ("ne", "Te", "Js", "Is", "Vp", "Vf")

#: The unit of each quantity, in SI, with how it was established: the file holds none. ``rho`` and ``depth`` are the
#: two lengths. Checked on every sample of every pin of the eight files of 1160616 and of 1150618021 (2026-10-12).
UNITS = {
    "ne": (
        "m⁻³",
        "J_s = ½·e·n_e·c_s, with c_s = √(e·(T_e + T_i) / m_i), T_i = T_e and m_i = 2m_p, holds on every sample to five "
        "figures",
    ),
    "Te": ("eV", "the same Bohm relation holds only with T_e in eV; it reads no lower than 2 eV, the MLP's floor"),
    "Js": ("A m⁻²", "the same Bohm relation, with n_e in m⁻³"),
    "Is": ("A", "I_s / J_s is steady on each pin to within 3%, at 0.84 to 1.07 mm², the pin's collecting area"),
    "Vp": ("V", "(V_p − V_f) / T_e has a median of 2.7 to 2.8, the sheath drop in units of T_e"),
    "Vf": ("V", "the same"),
    "depth": (
        "m",
        "the PLUNGE node is in metres (cmod_functions); the probe's full stroke is 12 cm, and the scans of 1160616 run "
        "at about 0.5 m/s",
    ),
    "rho": (
        "m",
        "ρ moves 0.95 to 1.09 m per metre of plunge while the probe moves fast, is negative only at the tips of the "
        "deepest plunges, by up to 1.1 cm, and the profiles fall into the limiter shadow about 2 cm out",
    ),
}

#: The pins of the probe head, by the names cmod_functions gives them for the MLP.
PIN_NAMES = {0: "NE", 1: "SW", 2: "SE", 3: "NW"}

#: A plunge takes the probe at least this much deeper than where it starts from and goes back to, in metres. The
#: scans of 1160616 go 3.6 to 4.2 cm deeper than the depth they start from, and 11.8 cm deeper than the parking
#: position; 1150618021's plunges, 11 cm. The probe settles at its parking position with bumps of 1.1 to 1.2 cm,
#: which are not plunges.
PROMINENCE = 0.02

#: The probe is moving while its depth changes faster than this, in m/s: its strokes run at about 0.5 m/s, and it turns
#: at the tip within a millisecond or two.
MOVING = 0.05

#: A pin's mean over a window counts only where it recorded at least this fraction of the window's samples.
MIN_FILL = 0.5

#: The far SOL that the e-folding lengths are fitted over by default, on ρ as the file holds it, in metres: from 0.5 cm,
#: where the paper puts the break between the steep near SOL and the flatter far SOL, to 1.8 cm, where it says the
#: field lines meet the limiters (from its equilibrium, the file's axis). The paper's own axis is shifted from the
#: file's (see the Documentation page), so its far SOL on its own axis is not quite this one.
FAR_SOL = (0.005, 0.018)

#: A fit needs at least this many points.
MIN_FIT = 5

#: :func:`shift_to` fits ln T_e over the points within this factor of the separatrix temperature, and needs at least
#: :data:`MIN_SIDE` of them on each side of it.
BAND, MIN_SIDE = 2.0, 3


# ---------------------------------------------------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------------------------------------------------

_NAME = re.compile(r"^([A-Za-z]+)_(\d+)$")


def layout(ds):
    """``{quantity: [pin, …]}`` of the MLP quantities the file holds, in :data:`QUANTITIES` order. Names only."""
    found = {}
    for name in ds.data_vars:
        match = _NAME.match(str(name))
        if match and match.group(1) in QUANTITIES:
            found.setdefault(match.group(1), set()).add(int(match.group(2)))
    return {quantity: sorted(found[quantity]) for quantity in QUANTITIES if quantity in found}


def pins(ds):
    """The pins the file holds any quantity of, sorted."""
    return sorted({pin for found in layout(ds).values() for pin in found})


def _array(value):
    """An attribute as a float array: an array, a list, or the text of one (``"[0.1 0.2 …]"``). ``None`` when it is
    not one, or is text that was cut short (``…``) and so does not hold the whole array."""
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    if isinstance(value, str):
        if "..." in value or "…" in value:
            return None
        numbers = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf", value)
        if not numbers:
            return None
        value = [float(number) for number in numbers]
    try:
        array = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    return array if array.size else None


def track(ds):
    """``(time, depth)`` of the probe, in s and m, from the attributes ``plunge_time`` and ``plunge_depth``; ``None``
    when the file lacks them, or they do not pair up. Sorted by time, with the samples that are not finite left out."""
    time, depth = _array(ds.attrs.get("plunge_time")), _array(ds.attrs.get("plunge_depth"))
    if time is None or depth is None or time.size != depth.size or time.size < 3:
        return None
    keep = np.isfinite(time) & np.isfinite(depth)
    time, depth = time[keep], depth[keep]
    order = np.argsort(time, kind="stable")
    return (time[order], depth[order]) if time.size >= 3 else None


def origin(ds):
    """``(R, Z, φ)`` of the probe from ``probe_origin``, or ``None``."""
    found = _array(ds.attrs.get("probe_origin"))
    return tuple(float(v) for v in found[:3]) if found is not None and found.size >= 3 else None


# ---------------------------------------------------------------------------------------------------------------------
# The plunges
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Plunge:
    """One plunge of the probe, as its depth shows it: in from ``start`` to the tip at ``tip``, out until ``end``."""

    number: int  # 1 for the shot's first
    start: float  # s, the shallowest point before the tip (the start of the depth record for the first)
    tip: float  # s, the deepest point
    end: float  # s, the shallowest point after the tip (the end of the record for the last)
    depth: float  # m, the depth at the tip


def speed(time, depth):
    """d depth/dt in m/s, positive inward."""
    return np.gradient(depth, time)


def plunges(time, depth, prominence=PROMINENCE):
    """Every plunge of the record, in order: each maximum of ``depth`` at least ``prominence`` deeper than the probe's
    depth on both sides of it, between the plunges before and after it.

    A plateau, as in a dwell, is one maximum: the probe goes in, holds still, and goes out, and the windows at rest are on
    no stroke (:func:`means`).
    """
    from scipy.signal import find_peaks

    time, depth = np.asarray(time, dtype=float), np.asarray(depth, dtype=float)
    if time.size < 3:
        return []
    tips, _ = find_peaks(depth, prominence=prominence)
    found = []
    for k, tip in enumerate(tips):
        low = tips[k - 1] if k else 0
        high = tips[k + 1] if k + 1 < len(tips) else depth.size - 1
        start = low + int(np.argmin(depth[low : tip + 1])) if k else 0
        end = tip + int(np.argmin(depth[tip : high + 1])) if k + 1 < len(tips) else depth.size - 1
        found.append(Plunge(k + 1, float(time[start]), float(time[tip]), float(time[end]), float(depth[tip])))
    return found


# ---------------------------------------------------------------------------------------------------------------------
# The 1 ms means
# ---------------------------------------------------------------------------------------------------------------------


def edges(t_start, t_end, window=WINDOW):
    """The edges of the whole windows of ``window`` that lie within ``[t_start, t_end]``, on whole multiples of it on the
    shot's clock, so that a window is the same whatever the page's window."""
    first = int(np.ceil(t_start / window - 1e-9))
    last = int(np.floor(t_end / window + 1e-9))
    if last <= first:
        return np.empty(0)
    # Rounded to the nanosecond, so that an edge is the double nearest a whole millisecond (1251 × 0.001 is a hair above
    # 1.251, and would take a sample at 1.251 s into the window before), and a sample on an edge opens the next window.
    return np.round(np.arange(first, last + 1) * window, 9)


def _binned(time, values, edges):
    """``(sum, count)`` of the finite ``values`` in each window between ``edges``."""
    index = np.searchsorted(edges, time, side="right") - 1
    keep = (index >= 0) & (index < edges.size - 1) & np.isfinite(values)
    total = np.bincount(index[keep], weights=values[keep], minlength=edges.size - 1)
    count = np.bincount(index[keep], minlength=edges.size - 1)
    return total, count


def _series(ds, name, t_start, t_end):
    """``(time, values)`` of a 1-D variable on its own axis, within ``[t_start, t_end]``: only the window is read, through
    the axis' index where it is sorted, as the files' are.

    A file opened without indexing its axes (:func:`fusion_ui.core.loader.lazily`, as the Thomson scattering view opens
    one) has the axis read instead, and where it is sorted only the window of the values is read, the samples the index
    would have chosen: from the first at or after ``t_start`` to the last at or before ``t_end``, compared in the axis'
    own type, as the index looks a bound up. On an axis of float32, as the ρ companions' are, ``1.15`` is then the
    sample stored for 1.15, which a comparison in float64 would put a hair below it.
    """
    variable = ds[name]
    dim = variable.dims[0]
    index = ds.indexes.get(dim)
    if index is not None and index.is_monotonic_increasing:
        variable = variable.sel({dim: slice(t_start, t_end)})
        return np.asarray(variable[dim].values, dtype=float), np.asarray(variable.values, dtype=float)
    stored = np.asarray(variable[dim].values)
    time = stored.astype(float)
    if index is None and time.size and bool(np.all(np.diff(time) >= 0)):
        bounds = np.asarray([t_start, t_end], dtype=stored.dtype if stored.dtype.kind == "f" else float)
        first, last = int(np.searchsorted(stored, bounds[0], "left")), int(np.searchsorted(stored, bounds[1], "right"))
        return time[first:last], np.asarray(variable.isel({dim: slice(first, last)}).values, dtype=float)
    keep = np.flatnonzero((time >= t_start) & (time <= t_end))
    return time[keep], np.asarray(variable.isel({dim: keep}).values, dtype=float)


def _extent(ds, quantities):
    """The earliest and latest sample of any of the MLP's quantities: the record's span when no window is given."""
    starts, ends = [], []
    for quantity, found in quantities.items():
        for pin in found:
            axis = ds[f"{quantity}_{pin}"].dims[0]
            times = np.asarray(ds[axis].values, dtype=float)
            if times.size:
                starts.append(np.nanmin(times))
                ends.append(np.nanmax(times))
    return (min(starts), max(ends)) if starts else (np.nan, np.nan)


def _labels(found_track, grid, middle):
    """``(depth, speed, plunge, stroke, record)`` of each window between ``grid``'s edges, from the probe's track: what
    :func:`means` returns about the probe, and the plunges of the whole record."""
    n = middle.size
    depth, moving = np.full(n, np.nan), np.full(n, np.nan)
    plunge, stroke = np.zeros(n, dtype=int), np.full(n, "", dtype=object)
    if found_track is None:
        return depth, moving, plunge, stroke.astype(str), []
    time, depths = found_track
    velocity = speed(time, depths)
    record = plunges(time, depths)
    if n:
        total, count = _binned(time, depths, grid)
        depth = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        total, count = _binned(time, velocity, grid)
        moving = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        # A window with no sample of the track, which runs at 10 kHz, takes the track where it is.
        empty = ~np.isfinite(moving) & (middle >= time[0]) & (middle <= time[-1])
        if empty.any():
            moving[empty] = np.interp(middle[empty], time, velocity)
            depth[empty] = np.interp(middle[empty], time, depths)
    for found in record:
        inside = (middle >= found.start) & (middle <= found.end)
        plunge[inside & (plunge == 0)] = found.number
    with np.errstate(invalid="ignore"):
        stroke[(plunge > 0) & (moving > MOVING)] = "in"
        stroke[(plunge > 0) & (moving < -MOVING)] = "out"
    return depth, moving, plunge, stroke.astype(str), record


def means(ds, t_start=None, t_end=None, window=WINDOW, strokes_only=False):
    """The 1 ms means of every MLP quantity, and of ρ, pin by pin, over the whole windows within ``[t_start, t_end]``:
    the whole record where either end is ``None`` or not finite.

    An ``xr.Dataset`` on ``window`` and ``pin``:

    ``time`` (window)
        s, the middle of each window.
    ``<q>`` (window, pin), for each of :data:`QUANTITIES` the file holds
        The mean of the pin's samples in the window, NaN where it recorded less than :data:`MIN_FILL` of it, or does
        not hold that quantity.
    ``rho`` (window, pin)
        m, where the pin was: where it recorded, the mean of its ρ at the samples of its first quantity that make the
        window's mean, interpolated from the 10 kHz companion; elsewhere the mean of the companion's own samples in the
        window. ρ's exact zeros, where the tree has no mapping, are left out.
    ``depth``, ``speed`` (window)
        The probe's depth (m) and its speed (m/s, positive inward) averaged over the window; NaN without a track.
    ``plunge`` (window)
        The number of the plunge the window's middle falls in (:func:`plunges`), 0 for none.
    ``stroke`` (window)
        ``"in"`` or ``"out"`` while the probe moves faster than :data:`MOVING`, and ``""`` while it holds still or where
        the file has no track.

    With ``strokes_only``, the quantities are read only between the first and the last window on a stroke, and not at
    all where there is none: a dwell's file is 1.1 GB, and none of it makes a profile.

    The attributes say what was found: ``t_start`` and ``t_end`` of the windows, ``window``, ``layout`` (the pins of
    each quantity, :func:`layout`), ``plunges`` (the plunges of the whole record, as ``[number, start, tip, end,
    depth]`` rows) and ``track`` (whether the file has one).
    """
    quantities = layout(ds)
    all_pins = pins(ds)
    if t_start is None or t_end is None or not np.isfinite([t_start, t_end]).all():
        t_start, t_end = _extent(ds, quantities)
    grid = edges(t_start, t_end, window) if np.isfinite([t_start, t_end]).all() else np.empty(0)
    n = max(grid.size - 1, 0)
    middle = 0.5 * (grid[:-1] + grid[1:]) if n else np.empty(0)

    found_track = track(ds)
    depth, moving, plunge, stroke, record = _labels(found_track, grid, middle)

    # The span the quantities are read over: all the windows, or those from the first to the last on a stroke.
    span = (0, n)
    if strokes_only:
        on = np.flatnonzero(stroke != "")
        span = (int(on[0]), int(on[-1]) + 1) if on.size else (0, 0)
    part = grid[span[0] : span[1] + 1]
    values = {quantity: np.full((n, len(all_pins)), np.nan) for quantity in quantities}
    rho = np.full((n, len(all_pins)), np.nan)
    for column, pin in enumerate(all_pins):
        # The pin's ρ, with the tree's zeros (no mapping) left out, a little beyond the windows so that it interpolates
        # up to their edges. Where the pin did not record, a window's ρ is the mean of these samples: where it was.
        name = next((f"rho_{q}_{pin}" for q in quantities if f"rho_{q}_{pin}" in ds), None)
        if name is None or not n:
            continue
        rho_time, rho_values = _series(ds, name, grid[0] - window, grid[-1] + window)
        mapped = np.isfinite(rho_values) & (rho_values != 0.0)
        rho_time, rho_values = rho_time[mapped], rho_values[mapped]
        total, count = _binned(rho_time, rho_values, grid)
        rho[:, column] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        sampled = False
        for quantity, holding in quantities.items():
            if pin not in holding or span[1] <= span[0]:
                continue
            time, series = _series(ds, f"{quantity}_{pin}", part[0], part[-1])
            if time.size < 2:
                continue
            step = float(np.median(np.diff(time)))
            total, count = _binned(time, series, part)
            enough = count >= MIN_FILL * window / step if step > 0 else count > 0
            values[quantity][span[0] : span[1], column] = np.where(enough, total / np.maximum(count, 1), np.nan)
            if not sampled and rho_time.size >= 2:
                # Where the pin recorded, ρ is the mean of its ρ at the samples that make the window's mean, from the
                # 10 kHz companion interpolated: the companion's own samples sit elsewhere in the window, by tens of µs.
                at = np.interp(time, rho_time, rho_values, left=np.nan, right=np.nan)
                total, count = _binned(time, at, part)
                section = rho[span[0] : span[1], column]
                rho[span[0] : span[1], column] = np.where(enough & (count > 0), total / np.maximum(count, 1), section)
                sampled = True
    data = {quantity: (("window", "pin"), array) for quantity, array in values.items()}
    data["rho"] = (("window", "pin"), rho)

    data.update(
        depth=("window", depth),
        speed=("window", moving),
        plunge=("window", plunge),
        stroke=("window", stroke),
    )
    return xr.Dataset(
        data,
        coords={"time": ("window", middle), "pin": ("pin", np.asarray(all_pins, dtype=int))},
        attrs={
            "t_start": float(t_start) if np.isfinite(t_start) else np.nan,
            "t_end": float(t_end) if np.isfinite(t_end) else np.nan,
            "window": float(window),
            "track": int(found_track is not None),
            "layout": quantities,
            "plunges": [[p.number, p.start, p.tip, p.end, p.depth] for p in record],
        },
    )


def combine(found):
    """The pins combined as the group's code combines them: the mean of the pins' means, window by window.

    A quantity's mean counts only where every pin that holds it recorded, so that one pin's missing millisecond cannot
    move the point by the ±13% the pins differ by. ρ is the mean of the pins' ρ. The result is on ``window`` alone, with
    ``pins`` (window), how many pins recorded n_e there (or the first quantity the file holds).
    """
    holds = found.attrs.get("layout") or {}
    out = found.drop_vars([name for name in found.data_vars if "pin" in found[name].dims] + ["pin"])
    for quantity in QUANTITIES:
        if quantity not in found:
            continue
        columns = [k for k, pin in enumerate(found["pin"].values) if int(pin) in holds.get(quantity, [])]
        values = found[quantity].isel(pin=columns)
        every = np.isfinite(values).all("pin") if columns else xr.zeros_like(found["time"], dtype=bool)
        out[quantity] = values.mean("pin").where(every)
    out["rho"] = found["rho"].mean("pin")
    first = next((q for q in QUANTITIES if q in found), None)
    out["pins"] = np.isfinite(found[first]).sum("pin") if first else xr.zeros_like(found["time"], dtype=int)
    return out


def strokes(found):
    """``[(plunge, stroke), …]`` of the windows on a stroke, in the order the probe ran them."""
    seen = []
    for number, direction in zip(found["plunge"].values, found["stroke"].values):
        key = (int(number), str(direction))
        if number and direction and key not in seen:
            seen.append(key)
    return seen


def on_strokes(found):
    """``found`` cut to its points: the windows on a stroke, while the probe moved within a plunge, with a ρ and a
    measurement (every pin's, for the combined means)."""
    first = next((q for q in QUANTITIES if q in found), None)
    keep = found["stroke"].values != ""
    keep &= np.isfinite(found["rho"].values) if found["rho"].ndim == 1 else True
    if first is not None and found[first].ndim == 1:
        keep &= np.isfinite(found[first].values)
    return found.isel(window=np.flatnonzero(keep))


def strokes_in(ds, t_start=None, t_end=None, window=WINDOW):
    """``[(plunge, stroke), …]`` of the strokes the probe runs in the whole windows within ``[t_start, t_end]``, in the
    order it runs them, judged from its track alone (:func:`track`): **no variable is read**, so it costs nothing on a
    file opened lazily (:func:`fusion_ui.core.loader.lazily`).

    The windows are labelled as :func:`means` labels them, so a stroke found here is one whose windows the profile is
    read over. Empty where the probe holds still in the window, as in a dwell, or the file has no track. Over the whole
    track where either end is ``None`` or not finite.
    """
    found_track = track(ds)
    if found_track is None:
        return []
    if t_start is None or t_end is None or not np.isfinite([t_start, t_end]).all():
        t_start, t_end = float(found_track[0][0]), float(found_track[0][-1])
    grid = edges(t_start, t_end, window)
    if grid.size < 2:
        return []
    _, _, plunge, stroke, _ = _labels(found_track, grid, 0.5 * (grid[:-1] + grid[1:]))
    return strokes(xr.Dataset({"plunge": ("window", plunge), "stroke": ("window", stroke)}))


def recording(found):
    """``(on_strokes, partial)``: how many windows on a stroke any pin recorded in, and in how many of those not every
    pin did, which the combined means leave out. ``found`` is :func:`means`' pin-by-pin result."""
    first = next((q for q in QUANTITIES if q in found), None)
    if first is None:
        return 0, 0
    holding = len((found.attrs.get("layout") or {}).get(first, [])) or found.sizes["pin"]
    count = np.isfinite(found[first].values).sum(axis=1)
    moving = found["stroke"].values != ""
    return int((moving & (count > 0)).sum()), int((moving & (count > 0) & (count < holding)).sum())


# ---------------------------------------------------------------------------------------------------------------------
# The far-SOL e-folding lengths, and the separatrix
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EFold:
    """A least-squares exponential fit ``y = amplitude·exp(−ρ/length)`` over ``lo ≤ ρ ≤ hi`` (m, on the file's ρ).

    ``length`` and ``amplitude`` are NaN where there is no fit, and ``reason`` says why: too few points, or a profile
    that does not fall off. ``first`` and ``last`` are the smallest and the largest ρ of the points it rests on, which
    is where a drawing of it should stop.
    """

    length: float  # m
    amplitude: float  # the quantity's unit, at ρ = 0 of the file's axis
    points: int
    lo: float
    hi: float
    reason: str = ""
    first: float = np.nan  # m
    last: float = np.nan

    @property
    def ok(self):
        return not self.reason

    def at(self, rho):
        """The fitted curve at ``rho`` (m, on the file's ρ)."""
        return self.amplitude * np.exp(-np.asarray(rho, dtype=float) / self.length)


def efold(rho, values, lo=FAR_SOL[0], hi=FAR_SOL[1]):
    """The e-folding length of ``values`` against ``rho`` (m) over ``[lo, hi]``: a least-squares fit of ``ln y`` linear
    in ρ, of the points with a ρ and a positive value. At least :data:`MIN_FIT` points, spread over more than one value
    of ρ, or it says why not."""
    rho, values = np.asarray(rho, dtype=float), np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        keep = np.isfinite(rho) & np.isfinite(values) & (values > 0) & (rho >= lo) & (rho <= hi)
    points = int(keep.sum())
    if points < MIN_FIT or np.ptp(rho[keep]) <= 0:
        return EFold(np.nan, np.nan, points, lo, hi, f"{points} points between ρ {lo * 100:g} and {hi * 100:g} cm")
    slope, intercept = np.polyfit(rho[keep], np.log(values[keep]), 1)
    first, last = float(rho[keep].min()), float(rho[keep].max())
    if not slope < 0:
        return EFold(np.nan, np.nan, points, lo, hi, "the profile does not fall off there", first, last)
    return EFold(float(-1.0 / slope), float(np.exp(intercept)), points, lo, hi, "", first, last)


#: The quantities whose far-SOL e-folding lengths the paper gives: L_n and L_T.
FITTED = ("ne", "Te")


def far_sol(drawn, lo=FAR_SOL[0], hi=FAR_SOL[1]):
    """``{"ne": EFold, "Te": EFold}``: the far-SOL e-folding lengths of the drawn points (the combined means on a stroke,
    :attr:`Profiles.drawn`), over ``[lo, hi]`` of ρ as the file holds it (m)."""
    return {
        quantity: efold(drawn["rho"].values, drawn[quantity].values, lo, hi)
        if quantity in drawn
        else EFold(np.nan, np.nan, 0, lo, hi, f"the file has no {quantity}")
        for quantity in FITTED
    }


@dataclass(frozen=True)
class Shift:
    """How far to move ρ so that T_e is ``te_sep`` at ρ = 0: ``by`` (m), added to ρ, or 0 with ``reason`` saying why
    not."""

    te_sep: float  # eV
    by: float = 0.0
    points: int = 0
    reason: str = ""

    @property
    def ok(self):
        return not self.reason


def shift_to(rho, te, te_sep):
    """Where T_e reaches ``te_sep`` (eV), as a :class:`Shift` that puts it at ρ = 0.

    ``ln T_e`` is fitted linear in ρ over the points within a factor :data:`BAND` of ``te_sep``, and the shift is minus
    the ρ at which the fit reaches it. It needs :data:`MIN_SIDE` points on each side of ``te_sep`` within the band, so
    that it interpolates and never extrapolates: a profile that never reaches ``te_sep`` is not shifted, and the reason
    says so.
    """
    rho, te = np.asarray(rho, dtype=float), np.asarray(te, dtype=float)
    keep = np.isfinite(rho) & np.isfinite(te) & (te > 0)
    rho, te = rho[keep], te[keep]
    if not (np.isfinite(te_sep) and te_sep > 0):
        return Shift(te_sep, reason="no separatrix temperature")
    if not te.size:
        return Shift(te_sep, reason="no T_e to shift by")
    near = (te >= te_sep / BAND) & (te <= te_sep * BAND)
    above, below = int((near & (te >= te_sep)).sum()), int((near & (te < te_sep)).sum())
    if above < MIN_SIDE or below < MIN_SIDE:
        span = f"{np.min(te):.3g} to {np.max(te):.3g} eV"
        if te.max() < te_sep:
            why = f"T_e never reaches {te_sep:g} eV in the window (its 1 ms means run from {span})"
        elif te.min() >= te_sep:
            why = f"T_e is above {te_sep:g} eV everywhere in the window (its 1 ms means run from {span})"
        else:
            why = (
                f"too few 1 ms means of T_e near {te_sep:g} eV ({above} above and {below} below it within a factor "
                f"{BAND:g}, where {MIN_SIDE} on each side are needed)"
            )
        return Shift(te_sep, points=above + below, reason=why)
    slope, intercept = np.polyfit(rho[near], np.log(te[near]), 1)
    if not slope < 0:
        return Shift(te_sep, points=above + below, reason=f"T_e does not fall off with ρ near {te_sep:g} eV")
    crossing = (np.log(te_sep) - intercept) / slope
    return Shift(te_sep, by=float(-crossing), points=above + below)


@dataclass
class Profiles:
    """What an ASP file's MLP says over the page's window: the means, pin by pin and combined, and what was found."""

    means: xr.Dataset  # pin by pin (:func:`means`), every window in the page's window
    combined: xr.Dataset  # the pins combined (:func:`combine`), every window
    plunges: list  # [Plunge, …] of the whole record
    origin: tuple  # (R, Z, φ) or None
    probe_type: str
    t_start: float  # s, the page's window: NaN where it gave none, and the whole record was read
    t_end: float
    shift: Shift = field(default_factory=lambda: Shift(0.0, reason="no separatrix temperature"))

    @property
    def drawn(self):
        """The combined windows on a stroke: the points of the profile."""
        return on_strokes(self.combined)


def read(ds, t_start=None, t_end=None, te_sep=0.0):
    """The :class:`Profiles` of an open ASP file over ``[t_start, t_end]`` (the whole record when not finite), shifted to
    ``te_sep`` (eV) where it is given and reached. Only the windows on a stroke are read (:func:`means`)."""
    windowed = t_start is not None and t_end is not None and bool(np.isfinite([t_start, t_end]).all())
    found = means(ds, t_start, t_end, strokes_only=True)
    combined = combine(found)
    drawn = on_strokes(combined)
    if not te_sep:
        shift = Shift(0.0, reason="no separatrix temperature")
    elif "Te" not in drawn:
        shift = Shift(float(te_sep), reason="the file holds no T_e")
    else:
        shift = shift_to(drawn["rho"].values, drawn["Te"].values, float(te_sep))
    record = [Plunge(int(row[0]), *map(float, row[1:])) for row in found.attrs.get("plunges", [])]
    return Profiles(
        means=found,
        combined=combined,
        plunges=record,
        origin=origin(ds),
        probe_type=str(ds.attrs.get("probe_type") or ""),
        t_start=float(t_start) if windowed else np.nan,
        t_end=float(t_end) if windowed else np.nan,
        shift=shift,
    )
