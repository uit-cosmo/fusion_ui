"""Cached, lazy, time-window-sliced access to the imaging (APD/phantom) files.

APD files are ~500 MB and 583k samples; phantom is the same shape of problem.
**Never touch the full time axis** -- every consumer of this module gets a
dataset already restricted to the discharge DB's ``t_start..t_end`` (or a
centred 0.2 s window when there is no metadata yet), and even then only reads
one frame or one pixel at a time. ``frames`` itself is never ``.load()``-ed
whole.

APD's ``frames`` has dims ``(y, x, time)``; phantom's has ``(time, y, x)``.
Everything here indexes by dimension *name*, never position, so both work
unmodified -- ``isel(time=...)`` and ``isel(y=..., x=...)`` leave the
remaining dims in their original relative order either way.
"""

import math
import os

import numpy as np
import streamlit as st
import xarray as xr
from experimental_database.diagnostics import Diagnostic

from fusion_ui import config

# The default window used when a shot has no discharge-DB entry yet -- a
# centred slice of this width around the dataset's own time midpoint.
DEFAULT_WINDOW_SECONDS = 0.2

TIME_DIM = "time"
FRAMES_VAR = "frames"


def dataset_path(machine, shot, diagnostic, preprocessed):
    """Path to one diagnostic file. ``machine`` is accepted for symmetry with
    the rest of the app's API; the data tree is not yet partitioned by it."""
    return Diagnostic[diagnostic].get_dataset_path_for_shot(
        shot, config.DATA_FOLDER, preprocessed
    )


@st.cache_resource(show_spinner="Opening dataset…")
def _open_cached(path, mtime):
    # ``mtime`` is not used in the body -- it is in the signature so a
    # re-copied file (same path, new content) busts the cache, the same
    # pattern as ui.shot_table's fingerprint argument.
    return xr.open_dataset(path)


def open_dataset(path):
    """A lazy ``xr.Dataset`` for ``path``, opened once per server process.

    Safe to share across Streamlit's session threads: reads go through
    xarray's netCDF4 backend, which serializes access with its own lock
    rather than requiring a connection-per-thread the way sqlite3 does.
    """
    mtime = os.path.getmtime(path)
    return _open_cached(path, mtime)


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def time_window(ds, discharge=None, default_span=DEFAULT_WINDOW_SECONDS):
    """``(t_start, t_end, source)`` to slice ``ds`` to.

    The discharge DB's window when it has one (``source="metadata"``);
    otherwise a centred default around this dataset's own time midpoint
    (``source="default"``), clipped to the record.
    """
    if (
        discharge is not None
        and _finite(discharge.t_start)
        and _finite(discharge.t_end)
    ):
        return float(discharge.t_start), float(discharge.t_end), "metadata"

    t_min = float(ds[TIME_DIM].min())
    t_max = float(ds[TIME_DIM].max())
    center = (t_min + t_max) / 2
    half = default_span / 2
    return max(t_min, center - half), min(t_max, center + half), "default"


def sliced(ds, t_start, t_end):
    """A lazy view of ``ds`` restricted to ``[t_start, t_end]``."""
    return ds.sel({TIME_DIM: slice(t_start, t_end)})


def frame_times(ds):
    """The (small) time coordinate array -- one float per frame, not a frame."""
    return ds[TIME_DIM].values


@st.cache_data(show_spinner=False)
def _cached_times(path, mtime, t_start, t_end):
    return frame_times(sliced(open_dataset(path), t_start, t_end))


def cached_frame_times(path, t_start, t_end):
    """:func:`frame_times` for the sliced window at ``path``, memoized.

    A UI slider re-runs the whole page on every drag; without this, each of
    those reruns would re-read the time coordinate off disk.
    """
    return _cached_times(path, os.path.getmtime(path), t_start, t_end)


def nearest_index(times, t):
    """Index into ``times`` closest to ``t`` -- not just the next one after."""
    times = np.asarray(times)
    idx = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1))
    if abs(t - times[idx - 1]) <= abs(times[idx] - t):
        idx -= 1
    return idx


def frame(ds, index, variable=FRAMES_VAR):
    """One 2D ``(y, x)`` frame, loaded into memory."""
    return ds[variable].isel({TIME_DIM: index}).load()


def pixel_series(ds, iy, ix, variable=FRAMES_VAR):
    """One pixel's full time series over ``ds``'s (already-sliced) window."""
    da = ds[variable].isel(y=iy, x=ix).load()
    return da[TIME_DIM].values, da.values


def pixel_grid(ds):
    """``(R, Z)`` 2D coordinate grids in centimetres, or ``(None, None)``."""
    if "R" not in ds.coords or "Z" not in ds.coords:
        return None, None
    return ds["R"].values, ds["Z"].values
