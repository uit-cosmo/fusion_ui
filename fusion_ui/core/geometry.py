"""Magnetic-coordinate labels for the statistics page.

Imaging files carry no per-pixel flux coordinate. They do carry the EFIT
boundary -- ``rlcfs(xlcfs, efit_time)``, ``zlcfs(ylcfs, efit_time)`` -- so a
pixel is labelled with its horizontal distance to the boundary at its own
height: ``R(y, x) - R_sep(Z(y, x))``, in centimetres, negative inside.

Be honest about what that is: it is *not* a flux coordinate, it is a
horizontal distance to the separatrix, and it is written ``R-R_sep`` with this
caveat in the docstring and in the page caption. The ``>= 86`` and
``linspace(-8, 1)`` inside ``calculate_splinted_LCFS`` are C-Mod
outboard-midplane numbers and will not carry to another machine.

Probe files carry a real rho -- ``rho_<quantity>_<position>`` on its own
coarser time base -- which is time-varying (the probe moves during its plunge),
so the label is the mean over the chosen window.

Pure numpy; ``fusion_scripts`` is imported **inside the function**, not at
module top -- it pulls in matplotlib, and every page import would pay for it.
Anything unexpected (no boundary in the file, a Z outside the spline range, a
non-monotonic outboard leg) returns ``None`` and the label falls back to R, Z.
"""

import numpy as np


def separatrix_radius(ds, t, z):
    """R [cm] of the outboard separatrix at height ``z``, at the EFIT slice
    nearest ``t``. ``None`` when the file carries no boundary or the leg is
    unusable."""
    try:
        rlcfs = np.asarray(ds["rlcfs"].values)
        zlcfs = np.asarray(ds["zlcfs"].values)
        efit_time = np.asarray(ds["efit_time"].values, dtype=float)
    except KeyError:
        return None
    try:
        from plotting_scripts.figure_plots import calculate_splinted_LCFS

        r_fine, z_fine = calculate_splinted_LCFS(
            float(t), np.asarray(efit_time), np.asarray(rlcfs), np.asarray(zlcfs)
        )
    except Exception:  # noqa: BLE001 - interpolate raises on a non-monotonic leg
        return None
    try:
        z_fine = np.asarray(z_fine, dtype=float)
        r_fine = np.asarray(r_fine, dtype=float)
        if float(z) < float(z_fine.min()) or float(z) > float(z_fine.max()):
            return None
        return float(np.interp(float(z), z_fine, r_fine))
    except (TypeError, ValueError):
        return None


def pixel_dr_sep(ds, x, y, t):
    """``R(y, x) - separatrix_radius(ds, t, Z(y, x))`` -- centimetres outboard
    of the separatrix, negative inside. ``None`` when unknown."""
    try:
        r = float(ds["R"].isel(x=int(x), y=int(y)))
        z = float(ds["Z"].isel(x=int(x), y=int(y)))
    except (KeyError, IndexError, ValueError):
        return None
    r_sep = separatrix_radius(ds, t, z)
    if r_sep is None:
        return None
    return r - r_sep


def probe_rho(ds, quantity, position, t_start, t_end):
    """Mean of the ``rho_<quantity>_<position>`` companion over the window."""
    rho_var = f"rho_{quantity}_{position}"
    if rho_var not in ds.data_vars:
        return None
    try:
        rho_da = ds[rho_var].load()
    except Exception:  # noqa: BLE001 - a label coordinate must never fail a page
        return None
    rho_time = np.asarray(rho_da[rho_da.dims[0]].values, dtype=float)
    rho = np.asarray(rho_da.values, dtype=float)
    mask = (
        (rho_time >= float(t_start))
        & (rho_time <= float(t_end))
        & np.isfinite(rho_time)
        & np.isfinite(rho)
    )
    if not bool(mask.any()):
        return None
    return float(np.mean(rho[mask]))
