"""Read the stored decorrelation products from Python: blob parameters and velocity fields, by shot and pixel.

For scripts that plot and analyse what ``fusion-ui precompute`` stored, outside the app::

    from fusion_ui import api

    api.shots()                                   # the shots whose blob parameters are stored
    ds = api.blob_parameters(1140827019)          # every pixel of one shot, on (y, x)
    ds = api.blob_parameters(1140827019, pixels=[(4, 5), (5, 5)])   # those pixels, on "pixel"
    df = api.table()                              # every stored shot and pixel, one row each
    ds = api.velocity_fields(1140827019)          # the velocity fields of every method

**It only reads.** The app database is opened read-only (``db.open_readonly``) and nothing is computed:
a product that is not stored raises :class:`NotStored`, quoting the ``fusion-ui precompute`` command that
fills it. Run it with the app's own environment, so that it finds the same database and cache the app
does (``fusion_ui.config``; on the server, the checkout's ``.venv/bin/python``).

**Settings** pick which stored parameter set to read. ``None`` is the product's defaults, which is what
``precompute`` runs unless told otherwise. A dict is the fields to change, as ``--params-json`` takes
them, e.g. ``{"averages": {"window": 30}}``; or pass the product's own params instance. A blob parameter
set and the velocity fields' set are separate: ``blob_parameters`` reads ``neighbour_step`` and the
2DCA's ``averages``, ``velocity_fields`` its own groups (``tracking``, ``tde``, ...).

**Pixels** are ``(x, y)``, as the app names them (``refx``, ``refy``): column first, then row, on the
blob's ``(y, x)`` grid. ``dead`` marks the pixels the dead-pixel mask removed, which hold no result;
a NaN at a live pixel is a fit that was tried and failed.

The datasets are the stored blobs, unchanged but for the selection and, on the blob parameters, a
``long_name`` and ``units`` on each variable. The blob's attributes say what made it:
``fusion_ui_params_json``, ``fusion_ui_code_version`` and ``fusion_ui_created_at``.
"""

import logging

import pandas as pd
import xarray as xr

# The specs import Streamlit, which warns that it has no runtime when imported outside the app.
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

from fusion_ui import config  # noqa: E402
from fusion_ui.core import db, params_ui, precompute, registry, store  # noqa: E402
from fusion_ui.plots import blob_parameters as _blob_parameters  # noqa: E402
from fusion_ui.plots import method_fields as _method_fields  # noqa: E402
from fusion_ui.views.numbers import BLOB_PARAMETERS as _BLOB_TABLE  # noqa: E402

__all__ = ["NotStored", "shots", "blob_parameters", "velocity_fields", "table", "PRODUCTS"]

#: The products this reads, by the name a script uses, each with its plot key in the store.
PRODUCTS = {
    "blob_parameters": _blob_parameters.BlobParametersParams,
    "velocity_fields": _method_fields.MethodFieldsParams,
}
_PLOT = {"blob_parameters": "blob_parameters", "velocity_fields": "method_fields"}


class NotStored(LookupError):
    """A product that is not stored for a shot under the settings asked for. ``command`` fills it."""

    def __init__(self, message, command=None):
        super().__init__(message if command is None else f"{message}. To fill it: {command}")
        self.command = command


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _params(product, settings):
    if product not in PRODUCTS:
        raise ValueError(f"unknown product {product!r}; one of {', '.join(PRODUCTS)}")
    cls = PRODUCTS[product]
    if settings is None:
        return cls()
    if isinstance(settings, cls):
        return settings
    if isinstance(settings, dict):
        return params_ui.with_values(cls(), settings)
    raise TypeError(
        f"settings for {product} must be None, a dict of the fields to change,"
        f" or a {cls.__name__}; got {type(settings).__name__}"
    )


def _hash(product, settings):
    params = _params(product, settings)
    return params, params_ui.hash_params(_PLOT[product], params)[0]


def _target(shot, machine):
    return registry.Target(machine, int(shot), "apd", True, "", float("nan"), float("nan"), "none")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def shots(product="blob_parameters", settings=None, machine=None):
    """The shots with ``product`` stored under ``settings``, in order."""
    _, digest = _hash(product, settings)
    conn = db.open_readonly()
    try:
        rows = conn.execute(
            "SELECT DISTINCT shot FROM runs WHERE machine = ? AND diagnostic = 'apd'"
            " AND preprocessed = 1 AND plot = ? AND params_hash = ? AND status = 'ok'"
            " ORDER BY shot",
            (machine or config.MACHINE, _PLOT[product], digest),
        ).fetchall()
    finally:
        conn.close()
    return [row["shot"] for row in rows]


def _load(conn, product, shot, params, digest, machine):
    target = _target(shot, machine)
    run = store.find_run(conn, target, _PLOT[product], digest)
    command = precompute.command(registry.get(_PLOT[product]), target, params)
    if run is None:
        raise NotStored(f"no {product} for shot {shot} under these settings", command)
    if run["status"] != "ok":
        raise NotStored(
            f"{product} for shot {shot} failed when it was computed: {run['error']}",
            f"{command} --retry-failed",
        )
    result = store.load_result(conn, run)
    if result is None:
        raise NotStored(
            f"{product} for shot {shot} is in the ledger but its file cannot be read"
            f" ({run['blob_path']})",
            f"{command} --force",
        )
    if product == "blob_parameters":
        for name, meaning, unit in _BLOB_TABLE:
            if name in result:
                result[name].attrs.update(long_name=meaning, units=unit or "1")
    return result.assign_coords(shot=int(shot))


def _select(ds, pixels):
    if pixels is None:
        return ds
    pixels = [tuple(p) for p in pixels]
    if not pixels or any(len(p) != 2 for p in pixels):
        raise ValueError("pixels must be a non-empty list of (x, y) pairs")
    nx, ny = ds.sizes["x"], ds.sizes["y"]
    outside = [p for p in pixels if not (0 <= p[0] < nx and 0 <= p[1] < ny)]
    if outside:
        raise ValueError(f"pixels {outside} are outside the {nx} x {ny} (x, y) array")
    xs = xr.DataArray([int(p[0]) for p in pixels], dims="pixel")
    ys = xr.DataArray([int(p[1]) for p in pixels], dims="pixel")
    return ds.isel(x=xs, y=ys).assign_coords(x=xs, y=ys)


def _one(product, shot, pixels, settings, machine):
    params, digest = _hash(product, settings)
    conn = db.open_readonly()
    try:
        ds = _load(conn, product, shot, params, digest, machine or config.MACHINE)
    finally:
        conn.close()
    return _select(ds, pixels)


def blob_parameters(shot, pixels=None, settings=None, machine=None):
    """The thirteen blob parameters of ``shot``: on ``(y, x)``, or on ``pixel`` for the ``(x, y)`` given.

    ``nevents``, ``level``, ``area``, the contour ellipse (``lx_c``, ``ly_c``, ``theta_c``), the FWHM sizes
    (``lr``, ``lz``), the Gaussian fit (``lx_f``, ``ly_f``, ``theta_f``) and the PSD fit (``taud``,
    ``lam``), each with its ``long_name`` and ``units``; ``R`` and ``Z`` in metres; ``dead``.
    """
    return _one("blob_parameters", shot, pixels, settings, machine)


def velocity_fields(shot, pixels=None, settings=None, machine=None):
    """Every method's velocity field of ``shot`` (``method_fields``): on ``(y, x)``, or on ``pixel``.

    ``vr_<method>`` and ``vz_<method>`` [m/s] with each method's lags, levels and tracks, as the Fields
    page reads them; ``R`` and ``Z`` in metres; ``dead``.
    """
    return _one("velocity_fields", shot, pixels, settings, machine)


def table(product="blob_parameters", shot_numbers=None, settings=None, machine=None, dead=False):
    """One row per stored shot and pixel, one column per variable on the pixel grid.

    Columns ``shot``, ``x``, ``y``, ``R``, ``Z``, ``dead`` and the variables on ``(y, x)``; per-lag
    variables (the velocity fields' tracks) are left out. ``shot_numbers`` limits it to those shots, every one
    of which must be stored; by default it is every shot :func:`shots` lists. Dead pixels are left out
    unless ``dead`` is true.
    """
    params, digest = _hash(product, settings)
    machine = machine or config.MACHINE
    wanted = shots(product, params, machine) if shot_numbers is None else list(shot_numbers)
    frames = []
    conn = db.open_readonly()
    try:
        for shot in wanted:
            ds = _load(conn, product, shot, params, digest, machine)
            grid = [name for name, var in ds.data_vars.items() if var.dims == ("y", "x")]
            frame = ds[grid].drop_vars("shot").reset_coords(["R", "Z"]).to_dataframe().reset_index()
            frame.insert(0, "shot", int(shot))
            frames.append(frame)
    finally:
        conn.close()
    if not frames:
        return pd.DataFrame(columns=["shot", "x", "y", "R", "Z", "dead"])
    out = pd.concat(frames, ignore_index=True)
    first = ["shot", "x", "y", "R", "Z", "dead"]
    out = out[[c for c in first if c in out] + [c for c in out if c not in first]]
    if not dead and "dead" in out:
        out = out[~out["dead"].astype(bool)].reset_index(drop=True)
    return out
