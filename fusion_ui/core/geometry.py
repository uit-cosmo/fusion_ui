"""Magnetic-coordinate labels for the statistics page.

Imaging files carry no flux coordinate per pixel, but they carry the EFIT
boundary (``rlcfs``, ``zlcfs``, ``efit_time``). So a pixel is labelled with its
horizontal distance to the separatrix at its own height, ``R - R_sep(Z)`` in
centimetres, negative inside. **That is not a flux coordinate**, and the page
says so. ``calculate_splinted_LCFS`` holds numbers of C-Mod's outboard midplane
(``>= 86``, ``linspace(-8, 1)``) that will not carry to another machine.

Probe files carry a real rho, ``rho_<quantity>_<position>``, on its own time
base. The probe moves during its plunge, so the label is the mean over the
window.

Anything unexpected returns ``None``, and the label falls back to R, Z.
"""

import numpy as np
from imaging_methods.geometry import calculate_splinted_LCFS


def separatrix_radius(ds, t, z):
    """R [cm] of the outboard separatrix at height ``z``, at the EFIT slice
    nearest ``t``. ``None`` when the file carries no boundary, the spline
    cannot take its outboard leg, or ``z`` is outside it."""
    try:
        r_fine, z_fine = calculate_splinted_LCFS(
            float(t),
            np.asarray(ds["efit_time"].values, dtype=float),
            np.asarray(ds["rlcfs"].values),
            np.asarray(ds["zlcfs"].values),
        )
        r_fine = np.asarray(r_fine, dtype=float)
        z_fine = np.asarray(z_fine, dtype=float)
        if not z_fine.min() <= float(z) <= z_fine.max():
            return None
        return float(np.interp(float(z), z_fine, r_fine))
    except Exception:  # noqa: BLE001 - a label coordinate must never fail a trace
        return None


def pixel_dr_sep(ds, x, y, t):
    """``R(y, x) - separatrix_radius(ds, t, Z(y, x))``: centimetres outboard
    of the separatrix, negative inside. ``None`` when unknown."""
    try:
        r = float(ds["R"].isel(x=int(x), y=int(y)))
        z = float(ds["Z"].isel(x=int(x), y=int(y)))
    except (KeyError, IndexError, ValueError):
        return None
    r_sep = separatrix_radius(ds, t, z)
    return None if r_sep is None else r - r_sep


def probe_rho(ds, quantity, position, t_start, t_end):
    """Mean of ``rho_<quantity>_<position>`` over the window, or ``None``."""
    name = f"rho_{quantity}_{position}"
    if name not in ds.data_vars:
        return None
    try:
        rho = ds[name].load()
    except Exception:  # noqa: BLE001 - a label coordinate must never fail a trace
        return None
    time = np.asarray(rho[rho.dims[0]].values, dtype=float)
    values = np.asarray(rho.values, dtype=float)
    mask = (
        (time >= float(t_start))
        & (time <= float(t_end))
        & np.isfinite(time)
        & np.isfinite(values)
    )
    return float(np.mean(values[mask])) if mask.any() else None
