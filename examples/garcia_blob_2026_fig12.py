"""Figure 12 of Garcia et al., Nucl. Fusion 66 (2026) 026043, redrawn from Shot Explorer's stored velocity fields.

An example of reading the stored products with ``fusion_ui.api``. Fig. 12 is the radial blob velocity averaged
over the far SOL against the Greenwald fraction: at each pixel the radial velocity, averaged over the rows (Z) at
each column, then over the columns R = 89.50 to 90.69 cm (x = 4 to 7), then over the shots of one f_GW group.
The bars run from the smallest to the largest pixel of the region in the group.

Run it on the group server with the app's Python, so that it finds the app's database and cache::

    /media/fusion/fusion-files/jlo044/fusion_ui/.venv/bin/python garcia_blob_2026_fig12.py --out fig12.png

**The velocity is not quite the paper's.** The paper uses the three-point time delay estimate with cross-conditional
averaging off each pixel's record, with its own selection criteria (a Lorentzian spectrum, live neighbours). The
stored velocity fields hold, among others, ``vr3_catde`` (the default here): the three-point time delay estimate
on the 2DCA conditional average, also at 2.5 standard deviations. The paper's published averages are drawn as open
black circles for comparison; ``vr3_catde`` agrees with them to within 7%, as does the app's own copy of the
paper's method (``velocity_tde``, which the API does not read). ``--method`` picks another radial velocity of the
stored fields: ``vr3_tde``, ``vr2_catde``, ``vr_max``, ``vr_com``, ``vr_2dcc``.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fusion_ui import api  # noqa: E402

try:  # the figure-provenance package, where it is installed
    import figure_provenance
    from figure_provenance import provenance
except ImportError:
    figure_provenance = None

    def provenance(function):
        return function


#: The paper's Table 1: the nine discharges of 16 June 2016, pooled by Greenwald fraction.
GROUPS = {
    0.24: [1160616009],
    0.30: [1160616011],
    0.45: [1160616016, 1160616017, 1160616018],
    0.65: [1160616022],
    0.80: [1160616025, 1160616026, 1160616027],
}
#: The far-SOL columns of the paper's average, R = 89.50 to 90.69 cm.
COLUMNS = range(4, 8)
#: The averages the paper's text quotes for Fig. 12, in m/s.
PAPER = {0.24: 502, 0.30: 509, 0.45: 677, 0.65: 767, 0.80: 819}
MARKERS = {0.24: "v", 0.30: "D", 0.45: "o", 0.65: "s", 0.80: "^"}


def far_sol_velocity(method="vr3_catde"):
    """One row per f_GW group: the mean far-SOL radial velocity and the smallest and largest pixel, in m/s."""
    shots = [shot for group in GROUPS.values() for shot in group]
    pixels = api.table("velocity_fields", shot_numbers=shots)  # live pixels only
    region = pixels[pixels.x.isin(COLUMNS)]
    rows = []
    for f_gw, group in GROUPS.items():
        values = region[region.shot.isin(group)]
        # Over the rows at each column, then over the columns of a shot, then over the shots of the group.
        by_shot = values.groupby(["shot", "x"])[method].mean().groupby("shot").mean()
        rows.append(
            {
                "f_gw": f_gw,
                "v_r": by_shot.mean(),
                "low": values[method].min(),
                "high": values[method].max(),
                "pixels": int(values[method].notna().sum()),
                "paper": PAPER[f_gw],
            }
        )
    return pd.DataFrame(rows)


@provenance
def figure_12(rows, method):
    """Fig. 12 of Garcia et al. NF 2026: far-SOL radial blob velocity against f_GW, one marker per density group.

    Filled markers are the stored velocity fields' ``method`` through fusion_ui.api, averaged over Z, then over
    x = 4 to 7, then over the group's shots; the bars span the region's smallest and largest pixel, as the paper's
    do. Open black circles are the averages the paper's text quotes, to show how far the stored method is from it.
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    for row in rows.itertuples():
        ax.errorbar(
            row.f_gw,
            row.v_r,
            yerr=[[row.v_r - row.low], [row.high - row.v_r]],
            marker=MARKERS[row.f_gw],
            linestyle="none",
            capsize=2,
            label=rf"$f_\mathrm{{GW}} = {row.f_gw:.2f}$",
        )
    ax.plot(rows.f_gw, rows.paper, "o", mfc="none", mec="k", ms=8, label="paper")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1500)
    ax.set_xlabel(r"$f_\mathrm{GW}$")
    ax.set_ylabel(r"$V_R$ / m s$^{-1}$")
    ax.set_title(f"far-SOL {method}", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def save(fig, path, function, arguments):
    """Through ``figure_provenance`` where it is installed; without it, the same record written by hand."""
    if figure_provenance is not None:
        figure_provenance.save(fig, path, stacklevel=2)
        return
    fig.savefig(path, dpi=200, bbox_inches="tight")

    def git(*args):
        here = Path(__file__).resolve().parent
        out = subprocess.run(["git", "-C", str(here), *args], capture_output=True, text=True)
        return out.stdout.strip()

    manifest = Path(path).parent / "provenance.json"
    entries = json.loads(manifest.read_text()) if manifest.exists() else {}
    entries[Path(path).name] = {
        "producer": f"{Path(__file__).resolve()}:{function.__name__}",
        "purpose": function.__doc__,
        "arguments": arguments,
        "command": " ".join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd()),
        "commit": git("rev-parse", "--short", "HEAD"),
        "uncommitted": git("status", "--porcelain", "--", str(Path(__file__).resolve())).splitlines(),
        "created": datetime.now(timezone.utc).isoformat(),
        "note": "written by the script's own fallback: figure_provenance is not installed",
    }
    manifest.write_text(json.dumps(entries, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="vr3_catde", help="a radial velocity of the stored fields")
    parser.add_argument("--out", default="garcia_blob_2026_fig12.png", help="where to save the figure")
    args = parser.parse_args(argv)

    rows = far_sol_velocity(args.method)
    with pd.option_context("display.float_format", "{:.0f}".format):
        print(rows.to_string(index=False))
    fig = figure_12(rows, args.method)
    save(fig, args.out, figure_12, {"method": args.method})
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
