"""Cached, lazy, time-window-sliced access to the imaging (APD/phantom) files.

APD files are ~500 MB and 583k samples; phantom is the same shape of problem.
**Never touch the full time axis** -- every consumer of this module gets a
dataset already restricted to the discharge DB's ``t_start..t_end`` (or a
centred 0.2 s window when there is no metadata yet), and even then only reads
one frame or one pixel at a time. ``frames`` itself is never ``.load()``-ed
whole.

The one exception is a spec that declares ``whole_record`` (see
:class:`fusion_ui.core.registry.PlotSpec`), which finds something against the
start of the record and cannot work from a cut of it. :func:`input_for` is the
one place that decides which of the two a spec computes on; the single-shot page
and ``fusion-ui precompute`` both call it, so they cannot disagree.

APD's ``frames`` has dims ``(y, x, time)``; phantom's has ``(time, y, x)``.
Everything here indexes by dimension *name*, never position, so both work
unmodified -- ``isel(time=...)`` and ``isel(y=..., x=...)`` leave the
remaining dims in their original relative order either way.
"""

import json
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


def image_variable(ds):
    """``"frames"`` for imaging data, or the only 3D variable if it is named
    something else -- phantom files have been seen both ways."""
    if "frames" in ds:
        return "frames"
    for name, variable in ds.data_vars.items():
        if variable.ndim == 3:
            return name
    raise KeyError("no 3D image variable in this dataset")


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


#: The attribute that carries the discharge window on a whole record
#: (:func:`whole_record`). Named for the app, so that nothing a data file stores
#: can collide with it: a preprocessed file keeps its own ``discharge_window``.
WINDOW_ATTR = "fusion_ui_discharge_window"


def as_window(value):
    """``(start, end)`` in seconds from a ``[start, end]`` attribute -- an array,
    a list or its JSON -- else ``None``."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    try:
        start, end = (float(v) for v in np.asarray(value, dtype=float).ravel())
    except (TypeError, ValueError):
        return None
    return (start, end) if np.isfinite([start, end]).all() else None


def stored_window(ds):
    """The window an opened file says it was cropped to, ``(start, end)``.

    A preprocessed file is cut to its analysis window (``density_scan.
    preprocess``) and stores it as the ``analysis_window`` attribute; ``None``
    for a file that stores none, or one that is not a window.
    """
    return as_window(ds.attrs.get("analysis_window"))


def whole_record(ds, t_start, t_end):
    """``ds`` uncut, carrying the discharge window ``t_start..t_end`` it would
    have been cut to.

    For a spec that declares ``whole_record``: it finds something against the
    start of the record, so it is handed all of it, and the window the others
    are cut to travels in an attribute (:func:`discharge_window` reads it). A
    shallow copy takes the attribute, so the shared, cached dataset is not
    touched, and nothing is loaded.
    """
    return ds.assign_attrs({WINDOW_ATTR: [float(t_start), float(t_end)]})


def discharge_window(ds):
    """The discharge window a whole record carries, ``(t_start, t_end)``.

    Raises ``ValueError`` for a dataset that carries none -- the one a caller
    cut itself, or built by hand. A spec that needs the whole record must not
    guess where its discharge window lies: the answer would pass for the
    record's own span, and the analysis window it finds would be wrong without
    a word.
    """
    window = as_window(ds.attrs.get(WINDOW_ATTR))
    if window is None:
        raise ValueError(
            "this record carries no discharge window: a spec that needs the "
            "whole record is handed it through loader.input_for, which attaches "
            "the window the other specs are cut to"
        )
    return window


def input_for(ds, t_start, t_end, whole=False):
    """What an analysis computes on: ``ds`` cut to ``[t_start, t_end]`` (a lazy
    view, as :func:`sliced`), or, when ``whole`` is set, all of it with the
    window attached (:func:`whole_record`).

    ``whole`` is the spec's ``PlotSpec.whole_record``. This is the one place
    that decides between the two, so that the single-shot page and ``fusion-ui
    precompute`` cannot. A dataset with no time axis -- a probe file, where
    every quantity has its own -- comes back as it is.
    """
    if TIME_DIM not in ds.dims:
        return ds
    if whole:
        return whole_record(ds, t_start, t_end)
    return sliced(ds, t_start, t_end)


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
