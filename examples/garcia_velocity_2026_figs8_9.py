"""Figs 8 and 9 of Garcia, Losada et al., PPCF 68 (2026) 055034, from the results stored by Shot Explorer.

V_R is the paper's TDE velocity (``api.velocity_tde``) and ell_Z the vertical HWHM of the 2DCA (``api.blob_parameters``'
``lz`` / 2). Values over the blob detection area (y = 2..7, x = 4..7) are the mean of its column means; bars are its
smallest and largest pixel. C_s, ell* and V* come from the probe's T_e at each pixel's distance to the separatrix
(paper's eqs 5 and 6, c1 = c2 = 1, T_i = 3 T_e, deuterium). f_GW 0.80 has no probe data and uses 0.65's (hollow).

Run on the group server:
    /media/fusion/fusion-files/jlo044/fusion_ui/.venv/bin/python garcia_velocity_2026_figs8_9.py --out figs8_9.png
"""

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

import experimental_database  # noqa: E402
from fusion_ui import api, config  # noqa: E402
from fusion_ui.core import geometry, loader, mlp  # noqa: E402

E, M_D = 1.602176634e-19, 2.014101778 * 1.66053906660e-27  # C, kg (deuterium)
B0, R0, TAU = 5.4, 0.68, 3.0  # T, m, T_i / T_e
warnings.filterwarnings("ignore", "Mean of empty slice")  # columns with no velocity

PAPER_SHIFT = 0.65  # cm: the paper's rho sits this far outward of EFIT's (it only places L_par)
ROWS, COLUMNS = slice(2, 8), slice(4, 8)  # the blob detection area

#: shot, f_GW, marker, colour, the shot whose probe T_e normalises it
SHOTS = (
    (1160616009, 0.24, "v", "#ffb000", 1160616009),
    (1160616011, 0.30, "D", "#fe6100", 1160616011),
    (1160616016, 0.45, "o", "#dc267f", 1160616016),
    (1160616022, 0.65, "s", "#785ef0", 1160616022),
    (1160616027, 0.80, "^", "#376cf5", 1160616022),
)


def connection_length(rho):
    """L_par (m) at rho (cm): a fit to the paper's Fig. 2(c)."""
    return 4.524 / (rho + 0.2056) + 3.152


def normalisation(shot, major):
    """C_s, ell* and V* of ``shot`` averaged over the area; ``major`` is the pixels' R (m)."""
    entry = experimental_database.PlasmaDischargeManager(str(config.DISCHARGE_DB_PATH)).get_discharge_by_shot(shot)
    t0, t1 = float(entry.t_start), float(entry.t_end)
    with xr.open_dataset(loader.dataset_path("cmod", shot, "apd", True)) as ds:  # distance to the separatrix, cm
        times = ds["efit_time"].values[(ds["efit_time"].values >= t0) & (ds["efit_time"].values <= t1)]
        distance = np.full(major.shape, np.nan)
        for (y, x), _ in np.ndenumerate(distance):
            found = [d for d in (geometry.pixel_dr_sep(ds, x, y, t) for t in times) if d is not None]
            distance[y, x] = np.mean(found) if found else np.nan
    with xr.open_dataset(loader.dataset_path("cmod", shot, "asp", False)) as ds:  # the probe's 1 ms means
        drawn = mlp.read(ds, t0, t1).drawn
        rho, te = 100 * drawn["rho"].values, drawn["Te"].values
    local_te = np.full(distance.shape, np.nan)
    for (y, x), d in np.ndenumerate(distance):
        near = np.abs(rho - d) <= 0.1
        if np.isfinite(d) and near.sum() >= 3:
            local_te[y, x] = np.nanmean(te[near])
    cs = np.sqrt(E * local_te / M_D)
    rho_s = np.sqrt(M_D * E * local_te) / (E * B0 * R0 / major)
    parallel = connection_length(np.where(distance + PAPER_SHIFT > 0, distance + PAPER_SHIFT, np.nan))
    l_star = rho_s * (parallel**2 / (3 * rho_s * major)) ** 0.2
    v_star = np.sqrt(1 + TAU) * cs * (32 * rho_s**2 * parallel / (27 * major**3)) ** 0.2
    return {name: np.nanmean(v[ROWS, COLUMNS]) for name, v in (("cs", cs), ("l_star", l_star), ("v_star", v_star))}


def over_area(values):
    """(mean of the column means, smallest, largest) over the area."""
    area = values[ROWS, COLUMNS]
    return np.mean([np.nanmean(c) for c in area.T if np.isfinite(c).any()]), np.nanmin(area), np.nanmax(area)


def figs_8_and_9():
    fig, ((a8, b8), (a9, b9)) = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    a8.axvspan(89.4, 90.85, color="#eef0fb")  # the blob detection area
    constants = {}
    for shot, f_gw, marker, colour, probe in SHOTS:
        velocity = api.velocity_tde(shot)
        vr = velocity["vx"].values
        size = api.blob_parameters(shot)["lz"].values / 2
        if probe not in constants:
            constants[probe] = normalisation(probe, velocity["R"].values)
        norm = constants[probe]
        a8.plot(100 * np.nanmean(velocity["R"].values[ROWS], axis=0), np.nanmean(vr[ROWS], axis=0),
                marker, color=colour, ms=8, ls="none", label=f"$f_{{GW}}$ = {f_gw:.2f}")
        for ax, values, by in ((b8, vr, "cs"), (a9, size, "l_star"), (b9, vr, "v_star")):
            mean, low, high = (v / norm[by] for v in over_area(values))
            ax.errorbar(f_gw, mean, yerr=[[mean - low], [high - mean]], marker=marker, ms=8, color=colour,
                        mfc="white" if probe != shot else colour, capsize=3)
    a8.set(xlabel="$R$ / cm", ylabel="$V_R$ / m s$^{-1}$", ylim=(0, 1000))
    a8.legend(fontsize=8)
    b8.set(ylabel="$V_R / C_s$", ylim=(0, 0.06))
    a9.set(ylabel=r"$\ell_Z / \ell_*$", ylim=(0, 6))
    b9.set(ylabel="$V_R / V_*$", ylim=(0, 0.8))
    for ax, name in ((a8, "Fig. 8(a)"), (b8, "Fig. 8(b)"), (a9, "Fig. 9(a)"), (b9, "Fig. 9(b)")):
        ax.set_title(name, loc="left", fontsize=10)
        if ax is not a8:
            ax.set(xlim=(0, 1), xlabel="$f_{GW}$")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="garcia_velocity_2026_figs8_9.png")
    figs_8_and_9().savefig(parser.parse_args().out, dpi=200)
