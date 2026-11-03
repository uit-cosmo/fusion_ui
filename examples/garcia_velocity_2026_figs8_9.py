"""Figures 8 and 9 of Garcia, Losada et al., Plasma Phys. Control. Fusion 68 (2026) 055034, from Shot Explorer.

An example of reading the stored results with ``fusion_ui.api``. The paper's velocity is the three-point time delay
estimate with cross-conditional averages off each pixel's record (``api.velocity_tde``, which equals the paper's data
at every pixel), and its blob size the vertical half width at half maximum of the 2DCA average
(``api.blob_parameters``' ``lz`` / 2). The normalisations need the probe's T_e and each pixel's distance to the
separatrix, which are not stored results: they are read off the data files with the app's own functions
(``fusion_ui.core.mlp`` and ``fusion_ui.core.geometry``).

- Fig. 8(a): V_R at each column, averaged over the rows y = 2 to 7.
- Fig. 8(b): V_R / C_s over the blob detection area, y = 2 to 7 and x = 4 to 7 (R = 89.5 to 90.7 cm).
- Fig. 9(a): ell_Z / ell*, and Fig. 9(b): V_R / V*, over the same area.

A shot's value over the area is the mean of its four columns' means, which is how the paper's 508 and 748 m/s come
out of its data; its bar runs from the area's smallest to its largest pixel. C_s, rho_s, ell* and V* are evaluated at
each pixel of the area from the probe's T_e at the pixel's distance to the separatrix, with the paper's formulas
(eqs 5 and 6, c1 = c2 = 1, T_i = 3 T_e, deuterium, B = 5.4 T x 0.68 m / R), and averaged over the area; L_par is the
paper's Fig. 2(c), on its rho, which sits 0.65 cm outward of EFIT's. There is no probe data at f_GW = 0.80: as in the
paper, that shot takes the constants of f_GW = 0.65 and its normalised markers are hollow. The paper's own points,
read off its figures, are drawn over ours as black outlines.

What agrees and what does not. V_R over the area is the paper's to the m/s (508, 524, 672, 730, 748), and V_R / C_s
and V_R / V* are within 2%. Three differences, all known:

- Fig. 8(a) draws every column with a velocity. The paper leaves out the inner columns at low density, where the
  spectra are flat (its criterion, not applied here), so a few points inside 89.4 cm have no outline.
- The bars of Figs 8(b) and 9(b) reach lower than the paper's: their bottoms are the area's smallest pixel, and the
  paper's are its smallest only without pixels (5, 7) and (6, 3), the slowest of the area on every shot.
- ell_Z / ell* is about 5% above the paper's and its bars are wider: the stored sizes are at the 2DCA threshold 2.5,
  as the paper's text says, while its own sizes were made at 2.

Run it on the group server with the app's Python, which finds the app's database, cache and data files::

    /media/fusion/fusion-files/jlo044/fusion_ui/.venv/bin/python garcia_velocity_2026_figs8_9.py --out figs8_9.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

import experimental_database  # noqa: E402
from fusion_ui import api, config  # noqa: E402
from fusion_ui.core import geometry, loader, mlp  # noqa: E402


E = 1.602176634e-19  # C
M_D = 2.014101778 * 1.66053906660e-27  # kg, deuterium
B0, R0 = 5.4, 0.68  # T, m: the axial field and where it is
TAU = 3.0  # T_i / T_e, the paper's
#: How far the paper's rho sits outward of EFIT's, in cm. It only places L_par.
PAPER_SHIFT = 0.65
#: Half the width of the band of the probe's 1 ms means that T_e is averaged over at a pixel, in cm.
TE_BAND = 0.1
ROWS = slice(2, 8)  # y = 2..7
COLUMNS = slice(4, 8)  # x = 4..7

#: The paper's Table 1: shot, f_GW, marker, colour, and the shot whose probe T_e normalises it.
SHOTS = (
    (1160616009, 0.24, "v", "#ffb000", 1160616009),
    (1160616011, 0.30, "D", "#fe6100", 1160616011),
    (1160616016, 0.45, "o", "#dc267f", 1160616016),
    (1160616022, 0.65, "s", "#785ef0", 1160616022),
    (1160616027, 0.80, "^", "#376cf5", 1160616022),
)

#: The paper's points, read off its figures at 600 dpi by marker colour (about 1% of each axis). Fig. 8(a): V_R in
#: m/s at the columns R = 88.0 ... 90.7 cm (None where it draws none). Figs 8(b), 9(a), 9(b): (marker, bar's low end,
#: bar's high end).
PAPER = {
    "8a": {
        0.24: (None, None, None, None, 460, 434, 498, 638),
        0.30: (None, None, None, None, 437, 453, 530, 678),
        0.45: (None, None, None, 490, 664, 560, 635, 828),
        0.65: (734, 774, 647, 673, 854, 606, 663, 795),
        0.80: (764, 872, 751, 655, 911, 632, None, 774),
    },
    "8b": {
        0.24: (0.0227, 0.0184, 0.0313),
        0.30: (0.0214, 0.0163, 0.0292),
        0.45: (0.0295, 0.0242, 0.0398),
        0.65: (0.0413, 0.0342, 0.0514),
        0.80: (0.0422, 0.0321, 0.0542),
    },
    "9a": {
        0.24: (2.87, 2.45, 3.18),
        0.30: (2.65, 2.21, 3.00),
        0.45: (2.95, 2.50, 3.38),
        0.65: (4.10, 3.40, 5.06),
        0.80: (4.06, 2.90, 5.04),
    },
    "9b": {
        0.24: (0.273, 0.223, 0.378),
        0.30: (0.248, 0.190, 0.341),
        0.45: (0.354, 0.289, 0.477),
        0.65: (0.544, 0.451, 0.677),
        0.80: (0.556, 0.423, 0.713),
    },
}
PAPER_R = (88.0, 88.4, 88.8, 89.15, 89.5, 89.9, 90.3, 90.7)
#: The paper's V_R over the blob detection area, m/s: its text's 508 and 748, and the mean of its Fig. 8(a) over
#: x = 4..7 for the others.
PAPER_AREA = {0.24: 508, 0.30: 524, 0.45: 672, 0.65: 730, 0.80: 748}


# ---------------------------------------------------------------------------
# The normalisation: from the data files, not from stored results
# ---------------------------------------------------------------------------


def connection_length(rho):
    """L_par in m at rho in cm, on the paper's rho: a fit A/(rho + rho0) + L_inf to its Fig. 2(c), rho = 0 to 1.9 cm."""
    return 4.524 / (rho + 0.2056) + 3.152


def window(shot):
    """The discharge DB's analysis window of ``shot``, in s: the paper's Table 1."""
    entry = experimental_database.PlasmaDischargeManager(
        str(config.DISCHARGE_DB_PATH)
    ).get_discharge_by_shot(shot)
    return float(entry.t_start), float(entry.t_end)


def separatrix_distance(shot, t_start, t_end):
    """(y, x) distance of every pixel to the separatrix at its own height, in cm, averaged over the EFIT times in the
    window."""
    with xr.open_dataset(loader.dataset_path("cmod", shot, "apd", True)) as ds:
        times = ds["efit_time"].values
        times = times[(times >= t_start) & (times <= t_end)]
        distance = np.full((ds.sizes["y"], ds.sizes["x"]), np.nan)
        for (y, x), _ in np.ndenumerate(distance):
            found = [geometry.pixel_dr_sep(ds, x, y, t) for t in times]
            found = [d for d in found if d is not None]
            distance[y, x] = np.mean(found) if found else np.nan
    return distance


def probe_te(shot, t_start, t_end):
    """The Mirror Langmuir Probe's 1 ms means of T_e (eV) and their rho (cm) over the window."""
    with xr.open_dataset(loader.dataset_path("cmod", shot, "asp", False)) as ds:
        drawn = mlp.read(ds, t_start, t_end).drawn
        rho, te = 100 * drawn["rho"].values, drawn["Te"].values
    keep = np.isfinite(rho) & np.isfinite(te)
    return rho[keep], te[keep]


def normalisation(shot, major):
    """C_s (m/s), rho_s (m), ell* (m) and V* (m/s) of ``shot``, each averaged over the blob detection area.

    ``major`` is the (y, x) major radius of the pixels, in m.
    """
    t_start, t_end = window(shot)
    distance = separatrix_distance(shot, t_start, t_end)
    rho, te = probe_te(shot, t_start, t_end)
    local_te = np.full(distance.shape, np.nan)
    for (y, x), d in np.ndenumerate(distance):
        near = np.abs(rho - d) <= TE_BAND
        if np.isfinite(d) and near.sum() >= 3:
            local_te[y, x] = te[near].mean()
    cs = np.sqrt(E * local_te / M_D)
    rho_s = np.sqrt(M_D * E * local_te) / (E * B0 * R0 / major)
    on_paper = distance + PAPER_SHIFT
    parallel = connection_length(np.where(on_paper > 0, on_paper, np.nan))  # SOL only
    ell = rho_s * (parallel**2 / (3 * rho_s * major)) ** 0.2
    v_star = np.sqrt(1 + TAU) * cs * (32 * rho_s**2 * parallel / (27 * major**3)) ** 0.2
    local = {"cs": cs, "rho_s": rho_s, "l_star": ell, "v_star": v_star}
    return {name: float(np.nanmean(value[ROWS, COLUMNS])) for name, value in local.items()}


# ---------------------------------------------------------------------------
# The stored results, through fusion_ui.api
# ---------------------------------------------------------------------------


def over_area(values):
    """The paper's value over the area (the mean of its columns' means), and the area's smallest and largest pixel."""
    area = values[ROWS, COLUMNS]
    columns = [np.nanmean(c) for c in area.T if np.isfinite(c).any()]
    return float(np.mean(columns)), float(np.nanmin(area)), float(np.nanmax(area))


def shot_data():
    """Everything the four panels draw, one dict per shot."""
    rows, constants = [], {}
    for shot, f_gw, marker, colour, probe in SHOTS:
        velocity = api.velocity_tde(shot)  # (y, x); NaN at dead pixels and where no velocity was found
        blobs = api.blob_parameters(shot)  # (y, x)
        if probe not in constants:
            constants[probe] = normalisation(probe, api.blob_parameters(probe)["R"].values)
        vr = velocity["vx"].values
        size = blobs["lz"].values / 2  # the half width at half maximum, m: the paper's ell
        rows.append(
            {
                "shot": shot,
                "f_gw": f_gw,
                "marker": marker,
                "colour": colour,
                "hollow": probe != shot,
                "column_r": 100 * np.nanmean(velocity["R"].values[ROWS], axis=0),  # cm
                "column_v": np.array(
                    [np.nanmean(c) if np.isfinite(c).any() else np.nan for c in vr[ROWS].T]
                ),
                "v": over_area(vr),
                "l": over_area(size),
                "norm": constants[probe],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------


def _scaled(row, value, by):
    return tuple(v / row["norm"][by] for v in row[value])


def _point(ax, x, y, low, high, row):
    ax.errorbar(x, y, yerr=[[y - low], [high - y]], fmt="none", ecolor=row["colour"], capsize=3, zorder=2)
    face = "white" if row["hollow"] else row["colour"]
    ax.plot(x, y, row["marker"], ms=8, mfc=face, mec=row["colour"], mew=1.4, zorder=3)


def _paper_point(ax, x, value, marker):
    y, low, high = value
    x = x + 0.015
    ax.errorbar(x, y, yerr=[[y - low], [high - y]], fmt="none", ecolor="0.15", elinewidth=0.7, capsize=2, zorder=4)
    ax.plot(x, y, marker, ms=8, mfc="none", mec="0.15", mew=0.8, zorder=5)


def figs_8_and_9(rows):
    """Figs 8 and 9 of Garcia, Losada et al. PPCF 68 (2026) 055034 on one page, through fusion_ui.api.

    Coloured: Shot Explorer's stored velocity_tde (the paper's 3TDE with cross-conditional averages) and
    blob_parameters' lz / 2, at their default parameters, which are the paper's. (a) V_R averaged over rows y = 2..7 at
    each column; (b) over the blob detection area (y = 2..7, x = 4..7, the mean of its columns' means), normalised by
    C_s and by V*, and ell_Z over ell*; bars, the area's smallest and largest pixel. C_s, ell*, V* from the probe's T_e
    with the paper's formulas; f_GW 0.80 takes 0.65's, hollow, as in the paper. Black outlines: the paper's own
    points, read off its figures, so the two can be compared point by point.
    """
    fig, ((a8, b8), (a9, b9)) = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    a8.axvspan(89.4, 90.85, color="#eef0fb", zorder=0)  # the blob detection area
    for row in rows:
        r, v = row["column_r"], row["column_v"]
        a8.plot(r, v, row["marker"], ms=8, color=row["colour"], ls="none", zorder=3,
                label=f"$f_{{GW}}$ = {row['f_gw']:.2f}  ({row['shot']})")
        paper = PAPER["8a"][row["f_gw"]]
        a8.plot([x for x, p in zip(PAPER_R, paper) if p is not None], [p for p in paper if p is not None],
                row["marker"], ms=8, mfc="none", mec="0.15", mew=0.8, ls="none", zorder=5)
        for ax, key, value, by in ((b8, "8b", "v", "cs"), (a9, "9a", "l", "l_star"), (b9, "9b", "v", "v_star")):
            _point(ax, row["f_gw"], *_scaled(row, value, by), row)
            _paper_point(ax, row["f_gw"], PAPER[key][row["f_gw"]], row["marker"])
    a8.plot([], [], "o", ms=8, mfc="none", mec="0.15", mew=0.8, ls="none", label="the paper, read off its figures")
    a8.set(xlim=(87.8, 91.2), ylim=(0, 1000), xlabel="$R$ / cm", ylabel="$V_R$ / m s$^{-1}$")
    a9.legend(*a8.get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8, frameon=False)
    b8.set(ylim=(0, 0.06), ylabel="$V_R\\,/\\,C_s$")
    a9.set(ylim=(0, max(5.5, 1.05 * max(r["l"][2] / r["norm"]["l_star"] for r in rows))), ylabel="$\\ell_Z\\,/\\,\\ell_*$")
    b9.set(ylim=(0, 0.8), ylabel="$V_R\\,/\\,V_*$")
    for ax in (b8, a9, b9):
        ax.set(xlim=(0, 1), xlabel="$f_{GW}$")
    for ax, name in ((a8, "Fig. 8(a)"), (b8, "Fig. 8(b)"), (a9, "Fig. 9(a)"), (b9, "Fig. 9(b)")):
        ax.set_title(name, loc="left", fontsize=10)
        ax.tick_params(direction="in", top=True, right=True)
    fig.suptitle("Garcia, Losada et al., PPCF 68 (2026) 055034, Figs 8 and 9, from Shot Explorer", fontsize=11)
    return fig


def table(rows):
    """The numbers behind the panels, beside the paper's."""
    lines = ["f_GW   V_R m/s (paper)   V_R/C_s (paper)    l_Z/l* (paper)   V_R/V* (paper)"]
    for row in rows:
        f = row["f_gw"]
        lines.append(
            f"{f:.2f}   {row['v'][0]:5.0f} ({PAPER_AREA[f]:4.0f})      "
            f"{_scaled(row, 'v', 'cs')[0]:.4f} ({PAPER['8b'][f][0]:.4f})   "
            f"{_scaled(row, 'l', 'l_star')[0]:.2f} ({PAPER['9a'][f][0]:.2f})      "
            f"{_scaled(row, 'v', 'v_star')[0]:.3f} ({PAPER['9b'][f][0]:.3f})"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="garcia_velocity_2026_figs8_9.png", help="where to save the figure")
    args = parser.parse_args(argv)
    rows = shot_data()
    print(table(rows))
    fig = figs_8_and_9(rows)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
