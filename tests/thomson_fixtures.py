"""A synthetic Thomson scattering file in the layout of ``thomson_scattering_<shot>.nc``, as fusion_scripts writes it.

The reference for the layout is the module docstring of ``tools/mfe/get_thomson_scattering.py`` in fusion_scripts, which
``build_dataset`` there follows: the raw points of the edge and the core system by ``(time, <sys>_channel)`` over the
whole shot, with a 0 where a system measured nothing (and junk errors there), ``<sys>_rho_pol`` and ``<sys>_R_mid`` of
each point, the stored fit on a ``rho_pol`` grid, three shot-level scalars, and the attributes. It is written as
netCDF3 (64-bit offset) through scipy, as the real files are, so that the single-shot page reads it the way it reads them.

Nothing here is a measurement: the profiles are tanh pedestals, the points are those profiles plus the planted
irregularities a real file has (zeros with junk errors, a pulse only one system recorded, a pulse outside the EFIT time
base). Each option plants or leaves out one thing, so a test can say what it is about.
"""

import json
from pathlib import Path

import numpy as np
import xarray as xr

SHOT = 1160616027

#: The window the fit is made over, and the pulses are spaced the way the real ones are.
WINDOW = (1.0, 1.4)
PULSE = 0.02

#: The two ends of the ``fit_check`` attribute, as ``tools/mfe/get_thomson_scattering.py`` writes them: run_db.py's own
#: description of its test, then ``passed`` or ``failed: <reasons>``.
CHECK_TEXT = (
    "run_db.py's own test of a fit, which it flags and keeps: Te above 100 eV at the grid's outer end, or Te below "
    "0.5 keV or ne below 0.5e20 m^-3 on axis, means 'something must have silently failed'"
)
PASSED = f"{CHECK_TEXT}: passed"
FAILED = f"{CHECK_TEXT}: failed: Te on axis is 134 eV"  # the clause on axis
FAILED_OUTER = f"{CHECK_TEXT}: failed: Te at rho_pol 1.10 is 140 eV"  # the clause at the grid's outer end
FAILED_BOTH = f"{CHECK_TEXT}: failed: Te at rho_pol 1.10 is 140 eV; Te on axis is 134 eV"

CAVEAT = (
    "From the C-Mod_Analysis README: it is highly recommended to talk with the diagnostician who collected the data "
    "before using it in a public forum. The Thomson scattering data were collected by Jerry Hughes "
    "(jwhughes@psfc.mit.edu)."
)

#: ``fit_note`` as ``tools/mfe/get_thomson_scattering.py`` writes it: what the fit's pre-filter drops, what ``fit_reg``
#: names, and then a sentence on this file's window, which says the fit's core is an extrapolation when no core T_e
#: inside rho_pol 0.2 is 1 keV or more.
FIT_NOTE = (
    "The fit is C-Mod_Analysis' edge fit: edge and core TS together, after cmod_tools.prefit_filter, which among other "
    "cuts drops core Te below 1 keV and ne below 0.5e20 m^-3 inside rho_pol 0.2, and Te below 100 eV inside rho_pol "
    "0.95. fit_reg names the mtanh coefficients set to zero: [8] leaves the profile's outer side linear, [7, 8] makes "
    "it flat."
)
WINDOW_NOTE = "In the window, 12 of the 30 core Te points inside rho_pol 0.2 are 1 keV or more."
EXTRAPOLATION = (
    "In the window, 0 of the 30 core Te points inside rho_pol 0.2 are 1 keV or more: the fit has no Te inside rho_pol "
    "0.2, so its Te and pe there are an extrapolation."
)

FIT_SETTINGS = {
    "fit_type": "osborne",
    "apply_final_sep_stretch": False,
    "force_to_zero": False,
    "frac_err": True,
    "num_mc": 5,
}

#: Where the channels of each system look, in ρ_pol, at the first pulse; the plasma then drifts by up to ±0.01.
EDGE_RHO = np.linspace(0.90, 1.08, 8)
CORE_RHO = np.linspace(0.02, 0.85, 6)


def r_mid(rho):
    """The outboard midplane radius of a flux surface, in metres: the magnetic axis at 0.68 m, the separatrix at 0.89 m."""
    return 0.68 + 0.21 * np.asarray(rho)


def ne_profile(rho):
    """n_e in m^-3: a tanh pedestal with a scrape-off layer at 0.15e20."""
    rho = np.asarray(rho)
    return 0.15e20 + 1.05e20 * 0.5 * (1 - np.tanh((rho - 0.95) / 0.05))


def te_profile(rho):
    """T_e in eV."""
    rho = np.asarray(rho)
    return 20.0 + 1000.0 * 0.5 * (1 - np.tanh((rho - 0.9) / 0.15))


def thomson_dataset(
    shot=SHOT,
    *,
    window=WINDOW,
    edge=True,
    core=True,
    fit=True,
    pe=True,
    scalars=True,
    check=PASSED,
    fit_status="ok",
    caveat=CAVEAT,
    frac_err=True,
    zeros=True,
    unmapped=False,
    efit=True,
    fit_r_mid=True,
    extrapolation=False,
):
    """The dataset of one file, over a shot of 2 s.

    ``edge`` and ``core`` say which systems the file holds, ``fit`` whether the fit worked (``fit_status`` then reads
    ``"ok"``, else ``"failed: ..."``) and ``pe`` whether it stores a pressure fit. ``check`` is the ``fit_check``
    attribute. ``zeros`` plants readings of 0 with junk errors, which a real file has wherever a system measured nothing;
    ``unmapped`` plants pulses with no ρ_pol, outside the EFIT time base. ``efit=False`` is a shot whose EFIT tree could
    not be opened: no point has a ρ_pol or an R_mid, and there is no ``R_mid_sep``. ``fit_r_mid=False`` leaves the
    fit's ``fit_R_mid`` out. ``extrapolation`` makes ``fit_note`` call the fit's core an extrapolation, as the files
    do where no core T_e inside rho_pol 0.2 reaches 1 keV.
    """
    times = np.round(np.arange(0.0, 2.0, PULSE), 6)
    nt = times.size
    rng = np.random.default_rng(3)

    coords = {"time": ("time", times, {"units": "s"})}
    data_vars = {}
    for name, present, rho0 in (("edge", edge, EDGE_RHO), ("core", core, CORE_RHO)):
        if not present:
            continue
        chan = f"{name}_channel"
        nch = rho0.size
        coords[chan] = (chan, np.arange(nch, dtype="int32"), {"units": "1"})
        # The plasma drifts a little from pulse to pulse, so a window's points spread in rho_pol.
        rho = rho0[None, :] + 0.01 * np.sin(times / 0.3)[:, None]
        ne = ne_profile(rho) * (1 + 0.05 * rng.standard_normal(rho.shape))
        te = te_profile(rho) * (1 + 0.05 * rng.standard_normal(rho.shape))
        ne_err, te_err = 0.1 * ne, 0.1 * te
        if zeros:
            # No measurement: before the plasma, and now and then a channel in the middle of the shot. The errors there
            # are the junk the tree gives (often 2e20 m^-3 and 1000 eV on the edge).
            dark = (times < 0.2)[:, None] | (rng.random(rho.shape) < 0.1)
            ne, te = np.where(dark, 0.0, ne), np.where(dark, 0.0, te)
            ne_err, te_err = np.where(dark, 2e20, ne_err), np.where(dark, 1000.0, te_err)
        recorded = np.ones(nt, bool)
        if name == "core":
            recorded[(times > 1.1) & (times < 1.13)] = False  # a pulse only the edge system recorded
        for array in (ne, te, ne_err, te_err):
            array[~recorded] = np.nan
        mapped = np.full(nt, efit, bool)
        if unmapped:
            mapped[(times > 1.2) & (times < 1.25)] = False
        rho_mapped = np.where(mapped[:, None], rho, np.nan)
        rmid = np.where(mapped[:, None], r_mid(rho), np.nan)

        data_vars[f"{name}_time"] = ("time", np.where(recorded, times, np.nan), {"units": "s"})
        data_vars[f"{name}_R"] = (chan, np.full(nch, 0.9), {"units": "m"})
        data_vars[f"{name}_Z"] = (chan, np.linspace(-0.05, 0.02, nch), {"units": "m"})
        for key, value, units in (
            ("ne", ne, "m^-3"),
            ("ne_err", ne_err, "m^-3"),
            ("Te", te, "eV"),
            ("Te_err", te_err, "eV"),
            ("rho_pol", rho_mapped, "1"),
            ("R_mid", rmid, "m"),
        ):
            data_vars[f"{name}_{key}"] = (("time", chan), value, {"units": units})

    if fit:
        grid = np.linspace(0.0, 1.1, 200)
        coords["rho_pol"] = ("rho_pol", grid, {"units": "1"})
        ne_fit, te_fit = ne_profile(grid), te_profile(grid)
        for key, value, units in (("ne", ne_fit, "m^-3"), ("Te", te_fit, "eV")):
            data_vars[f"fit_{key}"] = ("rho_pol", value, {"units": units})
            data_vars[f"fit_{key}_err"] = ("rho_pol", 0.2 * value, {"units": units})
        if pe:
            pressure = 1.602176634e-19 * ne_fit * te_fit
            data_vars["fit_pe"] = ("rho_pol", pressure, {"units": "Pa"})
            data_vars["fit_pe_err"] = ("rho_pol", 0.28 * pressure, {"units": "Pa"})
        if fit_r_mid:
            data_vars["fit_R_mid"] = ("rho_pol", r_mid(grid), {"units": "m"})
    if scalars:
        data_vars["Te_sep_2pm"] = ((), 36.0, {"units": "eV"})
        data_vars["lambda_q_2pm"] = ((), 0.002, {"units": "m"})
        if efit:
            data_vars["R_mid_sep"] = ((), float(r_mid(1.0)), {"units": "m"})

    attrs = {
        "shot": int(shot),
        "title": f"Alcator C-Mod Thomson scattering, shot {shot}",
        "machine": "cmod",
        "window_t_start": float(window[0]),
        "window_t_end": float(window[1]),
        "window_source": "discharge DB t_start, t_end",
        "efit_tree": "EFIT20" if efit else "none",
        "efit": "EFIT20, 75 times from 0.0800 to 1.5598 s" if efit else "no EFIT tree could be opened (TreeNODATA)",
        "edge_ts": "found: 8 channels" if edge else "missing: TreeNODATA",
        "core_ts": "found: 6 channels" if core else "missing: TreeNODATA",
        "fit_status": fit_status,
        "fit_function": "cmod_tools.get_cmod_kin_profs",
        "fit_settings": json.dumps(dict(FIT_SETTINGS, frac_err=frac_err)),
        "fit_reg": "ne [7, 8], Te [7, 8], pe [7, 8]" if fit else "no fit",
        "fit_note": f"{FIT_NOTE} {EXTRAPOLATION if extrapolation else WINDOW_NOTE}",
        "fit_check": check if fit else "no fit",
        "two_point_model": "power_balance.Teu_2pt_model(shot, t_start, t_end, lambdaq_opt=1): ok",
        "cmod_analysis_commit": "d6a646625e92399dd4c8b534e9c93001754a7c91",
        "cmod_analysis_local_changes": "yes; modified: pyproject.toml",
        "script_commit": "fusion_scripts 8e3df50b63ac74bebd0a6490ba3df0ce02f49670 (the script as committed)",
        "created": "2026-10-09T08:37:26Z",
        "time_alignment": "edge recorded 100 pulses and core 98",
    }
    if caveat is not None:
        attrs["caveat"] = caveat
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def write_thomson(root, shot=SHOT, **options):
    """Write the file under ``root/thomson_scattering/`` the way the real ones are, and return its path."""
    folder = Path(root) / "thomson_scattering"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"thomson_scattering_{shot}.nc"
    thomson_dataset(shot, **options).to_netcdf(path, engine="scipy", format="NETCDF3_64BIT")
    return path
