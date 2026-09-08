"""One 1-D series from any diagnostic: naming it, and materialising it.

Today nothing in the app takes "a trace" as its input. Every analysis is a
``PlotSpec`` bound to one ``Target`` -- one machine, one shot, one diagnostic --
and the two ways a 1-D series is obtained are unrelated:
``loader.pixel_series`` for imaging, ``probes.load_trace`` for the ragged probe
files. Comparing an APD pixel with a probe channel is not expressible.

This module is the whole point of the statistics page: one type that names a
1-D series from any diagnostic (``TraceRef``), and one function that
materialises it over an arbitrary time window (``extract``). Pure -- numpy and
xarray only, no Streamlit, no database.
"""

from dataclasses import dataclass

import numpy as np

from fusion_ui.core import loader, probes

#: Diagnostics whose files carry one shared time axis and a 3-D image variable.
IMAGING = ("apd", "phantom")

#: Diagnostics whose files carry one ragged time axis per (quantity, position).
PROBE = ("asp", "fsp")


@dataclass(frozen=True)
class TraceRef:
    """Which 1-D series, on which file. Hashable, so it is a session-state
    value and a cache key."""

    machine: str
    shot: int
    diagnostic: str
    preprocessed: bool
    channel: tuple  # ("pixel", x, y) | ("probe", quantity, position)

    @property
    def target_key(self):
        """``cmod_1160616027_apd_p`` -- matches ``Target.key``."""
        suffix = "p" if self.preprocessed else "r"
        return f"{self.machine}_{self.shot}_{self.diagnostic}_{suffix}"

    @property
    def key(self):
        """``cmod_1160616027_apd_p_pixel_6_6`` -- unique per channel."""
        return f"{self.target_key}_{'_'.join(str(c) for c in self.channel)}"


@dataclass(frozen=True)
class Trace:
    ref: TraceRef
    time: np.ndarray
    value: np.ndarray
    dt: float
    coords: dict  # {"R", "Z", "rho", "dr_sep"} -- floats or None
    label: str


def channels(ds, diagnostic) -> list:
    """Every selectable channel in this file.

    All ``("pixel", x, y)`` for imaging, every ``("probe", quantity,
    position)`` from :func:`probes.quantities_and_positions` for a probe.
    Reads names and shapes only -- no data touched.
    """
    if diagnostic in IMAGING:
        return [
            ("pixel", int(x), int(y))
            for y in range(int(ds.sizes["y"]))
            for x in range(int(ds.sizes["x"]))
        ]
    found = probes.quantities_and_positions(ds)
    return [
        ("probe", quantity, position)
        for quantity, positions in sorted(found.items())
        for position in positions
    ]


def _imaging_coords(ds, x, y):
    """``{"R", "Z", ...}`` for one imaging pixel, ``None`` where unknown."""
    coords = {"R": None, "Z": None, "rho": None, "dr_sep": None}
    try:
        coords["R"] = float(ds["R"].isel(x=int(x), y=int(y)))
        coords["Z"] = float(ds["Z"].isel(x=int(x), y=int(y)))
    except (KeyError, IndexError, ValueError):
        pass
    return coords


def extract(ds, ref, t_start, t_end):
    """The series over ``[t_start, t_end]``, or ``None`` when the record
    misses it entirely.

    Imaging pixels are sliced with ``.sel(time=...)`` on the file's shared
    axis; probe channels are masked on their own ragged time array. ``.load()``
    happens here, so callers always get in-memory numpy -- and nothing outside
    the window is ever read.
    """
    import imaging_methods as im

    kind = ref.channel[0] if ref.channel else None
    if kind == "pixel":
        _, x, y = ref.channel
        variable = loader.image_variable(ds)
        da = (
            ds[variable]
            .isel(x=int(x), y=int(y))
            .sel({loader.TIME_DIM: slice(float(t_start), float(t_end))})
            .load()
        )
        time = np.asarray(da[loader.TIME_DIM].values, dtype=float)
        value = np.asarray(da.values, dtype=float)
        if time.size == 0:
            return None
        dt = float(im.get_dt(ds))
        coords = _imaging_coords(ds, x, y)
        try:
            from fusion_ui.core import geometry as _geometry

            dr = _geometry.pixel_dr_sep(
                ds, int(x), int(y), float((time[0] + time[-1]) / 2)
            )
            coords["dr_sep"] = dr
        except Exception:  # noqa: BLE001 - a label coordinate must never fail a trace
            pass
        return Trace(ref=ref, time=time, value=value, dt=dt, coords=coords, label="")
    if kind == "probe":
        _, quantity, position = ref.channel
        trace = probes.load_trace(ds, quantity, int(position))
        time = np.asarray(trace.time, dtype=float)
        value = np.asarray(trace.value, dtype=float)
        mask = (time >= float(t_start)) & (time <= float(t_end))
        time, value = time[mask], value[mask]
        if time.size == 0:
            return None
        dt = float(np.median(np.diff(time))) if time.size >= 2 else float("nan")
        coords = {"R": None, "Z": None, "rho": None, "dr_sep": None}
        try:
            from fusion_ui.core import geometry as _geometry

            coords["rho"] = _geometry.probe_rho(
                ds, quantity, int(position), float(t_start), float(t_end)
            )
        except Exception:  # noqa: BLE001 - a label coordinate must never fail a trace
            pass
        return Trace(ref=ref, time=time, value=value, dt=dt, coords=coords, label="")
    raise KeyError(f"unknown channel {ref.channel!r}")


def label(trace, style="magnetic") -> str:
    """A legend/row label for ``trace``.

    ``style`` is ``"channel"`` (``pixel (6, 6)`` / ``ne_0``), ``"rz"``
    (``R=89.1, Z=-2.3``), or ``"magnetic"`` (``R-R_sep=+1.2 cm`` for pixels,
    ``rho=0.42`` for probe channels). Falls back to ``"rz"`` -- then to
    ``"channel"`` -- when a coordinate is ``None``.
    """
    ref = trace.ref
    kind = ref.channel[0] if ref.channel else None
    coords = trace.coords or {}

    def channel_label():
        if kind == "pixel":
            _, x, y = ref.channel
            return f"{ref.shot} {ref.diagnostic} pixel ({x}, {y})"
        if kind == "probe":
            _, quantity, position = ref.channel
            return f"{ref.shot} {ref.diagnostic} {quantity}_{position}"
        return ref.key

    def rz_label():
        r, z = coords.get("R"), coords.get("Z")
        if r is not None and z is not None and kind == "pixel":
            return f"{ref.shot} {ref.diagnostic} R={r:.2f}, Z={z:.2f}"
        return channel_label()

    if style == "channel":
        return channel_label()
    if style == "rz":
        return rz_label()
    if style == "magnetic":
        if kind == "pixel":
            dr = coords.get("dr_sep")
            if dr is not None:
                return f"{ref.shot} {ref.diagnostic} R-R_sep={dr:+.2f} cm"
            return rz_label()
        if kind == "probe":
            rho = coords.get("rho")
            if rho is not None:
                return f"{ref.shot} {ref.diagnostic} rho={rho:.2f}"
            return channel_label()
        return channel_label()
    raise ValueError(f"unknown label style {style!r}")


def _nearest_within_half_step(base, point):
    """Boolean mask selecting the grid point nearest ``point``.

    Matches only when the sample rounds to that grid point, i.e. within half
    the reference spacing -- exact float equality would miss samples computed
    through different arithmetic on another file's clock. A degenerate
    single-point base falls back to ``isclose``; anything else (empty base,
    non-finite or zero spacing, sample too far away) selects nothing.
    """
    import math

    if base.size == 0 or not math.isfinite(float(point)):
        return np.zeros_like(base, dtype=bool)
    if base.size == 1:
        return np.isclose(base, point, rtol=1e-9, atol=1e-12)
    step = np.median(np.diff(base))
    if not math.isfinite(float(step)) or step <= 0:
        return base == point
    nearest = int(np.argmin(np.abs(base - point)))
    if abs(float(base[nearest]) - float(point)) <= float(step) / 2:
        mask = np.zeros_like(base, dtype=bool)
        mask[nearest] = True
        return mask
    return np.zeros_like(base, dtype=bool)


def common_grid(traces, reference) -> list:
    """Every trace linearly interpolated onto ``reference``'s time base.

    Only pairwise statistics need this -- a CCF between two different files is
    impossible without it, since probe channels live on independent ragged
    axes. When every trace already shares the reference's time base -- the
    common case, several pixels of one APD file -- this is a no-op, not a
    round-trip through ``np.interp``.

    Points of the reference base outside a trace's own range become NaN rather
    than ``np.interp``'s flat extrapolation: two non-overlapping windows must
    read as missing data, not as a bogus flat line with a spurious CCF peak.
    A trace with a single sample cannot be interpolated, so its value lands on
    the nearest grid point within half the reference spacing (nearest-neighbour
    binning) and everything else stays NaN.
    """
    base = np.asarray(reference.time, dtype=float)
    out = []
    for trace in traces:
        time = np.asarray(trace.time, dtype=float)
        if time.shape == base.shape and bool(np.array_equal(time, base)):
            out.append(trace)
            continue
        values = np.asarray(trace.value, dtype=float)
        if time.size < 2:
            resampled = np.full_like(base, float("nan"), dtype=float)
            if time.size == 1 and base.size:
                resampled[_nearest_within_half_step(base, time[0])] = values[0]
        else:
            resampled = np.interp(base, time, values).astype(float)
            # np.interp clamps outside xp; mask back to NaN.
            resampled[(base < time.min()) | (base > time.max())] = float("nan")
        out.append(
            Trace(
                ref=trace.ref,
                time=np.asarray(base),
                value=np.asarray(resampled),
                dt=trace.dt,
                coords=trace.coords,
                label=trace.label,
            )
        )
    return out
