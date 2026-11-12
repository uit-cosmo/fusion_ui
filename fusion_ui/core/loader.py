"""Lazy, time-window-sliced access to the data files.

**Never load a full time axis.** An APD file is ~500 MB and 583k samples, so
every consumer gets a dataset already cut to the discharge DB's
``t_start..t_end`` (or a centred 0.2 s window when the shot has no entry), and
reads one frame or one pixel of it at a time.

The one exception is a spec that declares ``PlotSpec.whole_record``.
:func:`input_for` is the one place that decides which of the two a spec
computes on, so the single-shot page and ``fusion-ui precompute`` cannot
disagree.

APD's ``frames`` has dims ``(y, x, time)`` and phantom's ``(time, y, x)``, so
everything here indexes by dimension name, never by position.
"""

import json
import math
import os

import numpy as np
import streamlit as st
import xarray as xr
from experimental_database.diagnostics import Diagnostic

from fusion_ui import config

#: Width of the window for a shot with no discharge-DB entry, centred on the
#: middle of the record.
DEFAULT_WINDOW_SECONDS = 0.2

TIME_DIM = "time"


def image_variable(ds):
    """``"frames"``, or the first 3D variable when a file names it otherwise."""
    if "frames" in ds:
        return "frames"
    for name, variable in ds.data_vars.items():
        if variable.ndim == 3:
            return name
    raise KeyError("no 3D image variable in this dataset")


def dataset_path(machine, shot, diagnostic, preprocessed):
    """Path to one diagnostic file. The data tree is not partitioned by
    ``machine``, which is accepted and not used."""
    return Diagnostic[diagnostic].get_dataset_path_for_shot(
        shot, config.DATA_FOLDER, preprocessed
    )


@st.cache_resource(show_spinner="Opening dataset…")
def _open_cached(path, mtime):
    # ``mtime`` is in the signature so that a re-copied file busts the cache.
    return xr.open_dataset(path)


def open_dataset(path):
    """A lazy ``xr.Dataset`` for ``path``, opened once per server process.

    Safe to share across Streamlit's session threads: xarray's netCDF4 backend
    serializes access with its own lock.
    """
    return _open_cached(path, os.path.getmtime(path))


def lazily(path):
    """``path`` as an ``xr.Dataset`` that has read nothing but the file's
    attributes and its shared ``time`` axis.

    ``xr.open_dataset`` reads every dimension coordinate whole, to index it.
    In a probe file every quantity at every position has a time axis of its
    own, and they are most of the file (about 720 MB of a 1.1 GB ASP file).
    Here the variables and coordinates stay on disk until something reads
    them, and only a shared ``time`` axis is indexed, so that :func:`input_for`
    cuts the file as it cuts one opened whole. ``.sel`` on any other axis does
    not work. Nothing read is kept in memory.
    """
    store = xr.backends.NetCDF4DataStore.open(path, mode="r")
    try:
        variables, attrs = store.load()
        variables, attrs, coord_names = xr.conventions.decode_cf_variables(
            variables, attrs
        )
        coords = {
            name: variable
            for name, variable in variables.items()
            if name in coord_names or variable.dims == (name,)
        }
        data = {
            name: variable for name, variable in variables.items() if name not in coords
        }
        # An empty `indexes` is what keeps xarray from indexing, and so from
        # reading, every dimension coordinate.
        ds = xr.Dataset(data, coords=xr.Coordinates(coords, indexes={}), attrs=attrs)
        if TIME_DIM in ds.coords and ds[TIME_DIM].dims == (TIME_DIM,):
            ds = ds.set_xindex(TIME_DIM)
    except Exception:
        store.close()
        raise
    ds.set_close(store.close)
    return ds


@st.cache_resource(show_spinner=False)
def _lazily_cached(path, mtime):
    return lazily(path)


def open_lazily(path):
    """:func:`lazily`, opened once per server process: how the single-shot
    page opens a spec's companion."""
    return _lazily_cached(path, os.path.getmtime(path))


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def metadata_window(discharge):
    """``(t_start, t_end, "metadata")`` from the discharge DB's window, or
    ``(nan, nan, "none")`` when it has none.

    This is the whole answer for a probe file, which has no shared time axis to
    centre a default window on.
    """
    if (
        discharge is not None
        and _finite(discharge.t_start)
        and _finite(discharge.t_end)
    ):
        return float(discharge.t_start), float(discharge.t_end), "metadata"
    return float("nan"), float("nan"), "none"


def time_window(ds, discharge=None):
    """``(t_start, t_end, source)`` to slice ``ds`` to.

    The discharge DB's window when it has one (``source="metadata"``);
    otherwise :data:`DEFAULT_WINDOW_SECONDS` around the middle of the record,
    clipped to it (``source="default"``).
    """
    t_start, t_end, source = metadata_window(discharge)
    if source == "metadata":
        return t_start, t_end, source

    t_min = float(ds[TIME_DIM].min())
    t_max = float(ds[TIME_DIM].max())
    center = (t_min + t_max) / 2
    half = DEFAULT_WINDOW_SECONDS / 2
    return max(t_min, center - half), min(t_max, center + half), "default"


def sliced(ds, t_start, t_end):
    """A lazy view of ``ds`` restricted to ``[t_start, t_end]``."""
    return ds.sel({TIME_DIM: slice(t_start, t_end)})


#: The attribute that carries the discharge window on a whole record. Named
#: for the app, because a preprocessed file stores a ``discharge_window`` of
#: its own.
WINDOW_ATTR = "fusion_ui_discharge_window"


def as_window(value):
    """``(start, end)`` in seconds from a ``[start, end]`` attribute (an array,
    a list or its JSON), else ``None``."""
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

    A preprocessed file is cut to its analysis window and stores it as the
    ``analysis_window`` attribute. ``None`` for a file that stores none, or
    something that is not a window.
    """
    return as_window(ds.attrs.get("analysis_window"))


def whole_record(ds, t_start, t_end):
    """``ds`` uncut, carrying the discharge window ``t_start..t_end`` it would
    have been cut to in an attribute, which :func:`discharge_window` reads.

    The attribute goes on a shallow copy: the shared, cached dataset is not
    touched and nothing is loaded.
    """
    return ds.assign_attrs({WINDOW_ATTR: [float(t_start), float(t_end)]})


def discharge_window(ds):
    """The discharge window a whole record carries, ``(t_start, t_end)``.

    Raises ``ValueError`` for a dataset that carries none, rather than take the
    record's own span for the discharge window and be silently wrong.
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
    view), or, when ``whole`` is set, all of it with the window attached.

    ``whole`` is the spec's ``PlotSpec.whole_record``. A dataset with no shared
    time axis, a probe file, comes back as it is.
    """
    if TIME_DIM not in ds.dims:
        return ds
    if whole:
        return whole_record(ds, t_start, t_end)
    return sliced(ds, t_start, t_end)


@st.cache_data(show_spinner=False)
def _cached_times(path, mtime, t_start, t_end):
    return sliced(open_dataset(path), t_start, t_end)[TIME_DIM].values


def cached_frame_times(path, t_start, t_end):
    """The time of every frame of ``path`` in the window, memoized: a slider
    reruns the page on every drag."""
    return _cached_times(path, os.path.getmtime(path), t_start, t_end)


def nearest_index(times, t):
    """Index into ``times`` closest to ``t``."""
    times = np.asarray(times)
    idx = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1))
    if abs(t - times[idx - 1]) <= abs(times[idx] - t):
        idx -= 1
    return idx


def frame(ds, index):
    """One 2D ``(y, x)`` frame, loaded into memory."""
    return ds["frames"].isel({TIME_DIM: index}).load()


def pixel_series(ds, iy, ix):
    """``(times, values)`` of one pixel over ``ds``'s (already sliced) window."""
    da = ds["frames"].isel(y=iy, x=ix).load()
    return da[TIME_DIM].values, da.values


def pixel_grid(ds):
    """``(R, Z)`` 2D coordinate grids in centimetres, or ``(None, None)``."""
    if "R" not in ds.coords or "Z" not in ds.coords:
        return None, None
    return ds["R"].values, ds["Z"].values
