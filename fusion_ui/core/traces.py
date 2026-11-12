"""One 1-D series from any diagnostic that has them: naming it and reading it.

``TraceRef`` names a series (a pixel of an imaging file, a channel of a probe
file) and ``extract`` reads it over a time window. That is what lets the
Statistics page put an APD pixel and a probe channel on one axis. Pure: no
Streamlit, no database.
"""

from dataclasses import dataclass

import numpy as np

from fusion_ui.core import geometry, loader, probes

#: Diagnostics whose files carry one shared time axis and a 3-D image variable.
IMAGING = ("apd", "phantom")

#: Diagnostics whose files carry one ragged time axis per (quantity, position).
PROBE = ("asp", "fsp")

#: Every diagnostic a trace can be taken from. A Thomson scattering file holds
#: profiles, one reading per laser pulse and channel, not traces.
SOURCES = IMAGING + PROBE


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
        """``cmod_1160616027_apd_p``, as ``Target.key``."""
        suffix = "p" if self.preprocessed else "r"
        return f"{self.machine}_{self.shot}_{self.diagnostic}_{suffix}"

    @property
    def key(self):
        """``cmod_1160616027_apd_p_pixel_6_6``, unique per channel."""
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
    """Every channel of this file: all ``("pixel", x, y)`` of an imaging file,
    every ``("probe", quantity, position)`` of a probe file.

    Reads names and shapes only. A diagnostic that is not in :data:`SOURCES`
    raises: an empty list would read as a file with no channels.
    """
    if diagnostic not in SOURCES:
        raise ValueError(
            f"{diagnostic!r} has no traces to take; they come from {', '.join(SOURCES)}"
        )
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


def extract(ds, ref, t_start, t_end):
    """The series over ``[t_start, t_end]`` in memory, or ``None`` when the
    record misses the window.

    A pixel is sliced on the file's shared time axis, so nothing outside the
    window is read. A probe channel is masked on its own ragged axis.
    """
    import imaging_methods as im

    t_start, t_end = float(t_start), float(t_end)
    coords = {"R": None, "Z": None, "rho": None, "dr_sep": None}
    kind = ref.channel[0]
    if kind == "pixel":
        x, y = int(ref.channel[1]), int(ref.channel[2])
        series = (
            ds[loader.image_variable(ds)]
            .isel(x=x, y=y)
            .sel({loader.TIME_DIM: slice(t_start, t_end)})
            .load()
        )
        time = np.asarray(series[loader.TIME_DIM].values, dtype=float)
        value = np.asarray(series.values, dtype=float)
        if time.size == 0:
            return None
        dt = float(im.get_dt(ds))
        try:
            coords["R"] = float(ds["R"].isel(x=x, y=y))
            coords["Z"] = float(ds["Z"].isel(x=x, y=y))
        except (KeyError, IndexError, ValueError):
            pass
        coords["dr_sep"] = geometry.pixel_dr_sep(ds, x, y, (time[0] + time[-1]) / 2)
    elif kind == "probe":
        _, quantity, position = ref.channel
        loaded = probes.load_trace(ds, quantity, int(position))
        time = np.asarray(loaded.time, dtype=float)
        value = np.asarray(loaded.value, dtype=float)
        mask = (time >= t_start) & (time <= t_end)
        time, value = time[mask], value[mask]
        if time.size == 0:
            return None
        dt = float(np.median(np.diff(time))) if time.size >= 2 else float("nan")
        coords["rho"] = geometry.probe_rho(ds, quantity, int(position), t_start, t_end)
    else:
        raise KeyError(f"unknown channel {ref.channel!r}")
    return Trace(ref=ref, time=time, value=value, dt=dt, coords=coords, label="")


def channel_text(channel) -> str:
    """``pixel (6, 6)`` or ``ne_0``."""
    if channel[0] == "pixel":
        return f"pixel ({channel[1]}, {channel[2]})"
    return f"{channel[1]}_{channel[2]}"


def magnetic_text(trace) -> str:
    """``R-R_sep=+1.20 cm`` for a pixel, ``rho=0.42`` for a probe channel,
    ``""`` when the coordinate is unknown."""
    kind = trace.ref.channel[0]
    coords = trace.coords or {}
    if kind == "pixel" and coords.get("dr_sep") is not None:
        return f"R-R_sep={coords['dr_sep']:+.2f} cm"
    if kind == "probe" and coords.get("rho") is not None:
        return f"rho={coords['rho']:.2f}"
    return ""


def label(trace, style="magnetic") -> str:
    """A legend label for ``trace``: its shot, its diagnostic and where it is.

    ``style`` is ``"channel"`` (:func:`channel_text`), ``"rz"``
    (``R=89.10, Z=-2.30``) or ``"magnetic"`` (:func:`magnetic_text`).
    ``"magnetic"`` falls back to ``"rz"`` and ``"rz"`` to ``"channel"`` when
    the coordinate is unknown.
    """
    if style not in ("channel", "rz", "magnetic"):
        raise ValueError(f"unknown label style {style!r}")
    ref = trace.ref
    coords = trace.coords or {}
    where = magnetic_text(trace) if style == "magnetic" else ""
    r, z = coords.get("R"), coords.get("Z")
    if not where and style != "channel" and ref.channel[0] == "pixel":
        if r is not None and z is not None:
            where = f"R={r:.2f}, Z={z:.2f}"
    return f"{ref.shot} {ref.diagnostic} {where or channel_text(ref.channel)}"


def _nearest_within_half_step(base, point):
    """A mask over ``base`` that selects the grid point nearest ``point``, when
    it is within half the grid spacing. Otherwise it selects nothing.

    Exact equality would miss a sample whose time came through other
    arithmetic on another file's clock.
    """
    nothing = np.zeros_like(base, dtype=bool)
    if base.size == 0 or not np.isfinite(point):
        return nothing
    if base.size == 1:
        return np.isclose(base, point, rtol=1e-9, atol=1e-12)
    step = np.median(np.diff(base))
    if not np.isfinite(step) or step <= 0:
        return base == point
    nearest = int(np.argmin(np.abs(base - point)))
    if abs(base[nearest] - point) <= step / 2:
        nothing[nearest] = True
    return nothing


def common_grid(traces, reference) -> list:
    """Every trace linearly interpolated onto ``reference``'s time base.

    Pairwise statistics need it: probe channels each have their own time axis.
    A trace that is already on the reference's base comes back as it is.

    Points of the base outside a trace's own range are NaN, not
    ``np.interp``'s flat extrapolation, which would give a spurious CCF peak.
    A trace of one sample cannot be interpolated: its value goes to the nearest
    grid point within half the spacing.
    """
    base = np.asarray(reference.time, dtype=float)
    out = []
    for trace in traces:
        time = np.asarray(trace.time, dtype=float)
        if np.array_equal(time, base):
            out.append(trace)
            continue
        values = np.asarray(trace.value, dtype=float)
        if time.size < 2:
            resampled = np.full_like(base, float("nan"), dtype=float)
            if time.size == 1 and base.size:
                resampled[_nearest_within_half_step(base, time[0])] = values[0]
        else:
            resampled = np.interp(base, time, values).astype(float)
            resampled[(base < time.min()) | (base > time.max())] = float("nan")
        out.append(
            Trace(
                ref=trace.ref,
                time=base,
                value=resampled,
                dt=trace.dt,
                coords=trace.coords,
                label=trace.label,
            )
        )
    return out
