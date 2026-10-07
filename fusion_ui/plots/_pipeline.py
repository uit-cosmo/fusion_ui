"""What the three product specs share: the physics API, and the record it takes.

``pixel_averages``, ``method_fields`` and ``blob_parameters`` are thin adapters over
``decorrelation.pipeline`` in ``fusion_scripts`` (the paper's code, which ``cmod_scan`` calls too), so the
paper and the app read the same numbers. This module is the one place fusion_ui imports it, and the one
place that prepares what it is called with.

**Import order.** ``decorrelation.pipeline`` imports ``fusion_scripts``' own ``config``, which fills every
``FUSION_*`` variable still unset from ``fusion_scripts/.env`` the moment it is imported -- and on the
group server that file names a folder that does not exist (``/hdd1/alcator``). ``fusion_ui.config`` fills
the same variables the same way, from its own ``.env``, and whichever is imported first wins. So
``fusion_ui.config`` is imported first, below, and every module that needs the API imports it from here.

Nothing in this module touches Streamlit, the database or the filesystem.
"""

# Before decorrelation.pipeline, always: see the module docstring.
from fusion_ui import config  # noqa: F401 - imported for what it does to os.environ

import numpy as np

from decorrelation import pipeline
from decorrelation.pipeline import Averages, Blobs, Tde, Tracking

__all__ = [
    "Averages",
    "Blobs",
    "Tde",
    "Tracking",
    "average_at",
    "bank_mask",
    "live_scalars",
    "pipeline",
    "record",
]

#: What R is, in the unit the imaging files store it in, for a machine this can run on: a major radius
#: between 0.1 m and 7 m (Alcator C-Mod 0.67 m, ASDEX Upgrade 1.65 m, JET 2.96 m, W7-X 5.5 m, ITER 6.2 m),
#: so 10 to 700 in centimetres. A value outside it is metres, or millimetres on a machine of C-Mod's size
#: or larger (C-Mod's R reads 880 to 910), not centimetres. A heuristic, and one that cannot tell 300 cm
#: from 300 mm: it catches the mistakes a C-Mod-sized machine can make, which is the machine there is.
CENTIMETRES = (10.0, 700.0)


def record(ds):
    """The record as ``decorrelation.pipeline`` takes it: R and Z in metres, loaded into memory.

    The APD files hold R and Z in centimetres and the frames' time axis in seconds, so a velocity
    computed off them as they are would come out in cm/s. The API's outputs are all in SI units, which
    makes the conversion the adapter's job and keeps ``pipeline`` free of any one machine's convention.

    The magnitude of R is checked rather than the convention trusted: a record that already holds metres
    (a file written by something else, or another machine's) would be divided by 100 again and every
    velocity would come out a hundred times too small, with nothing to say so. It raises instead.

    ``ds`` is the record sliced to the discharge window, lazily opened or already in memory (a pool
    worker loads it). The 2DCA reads every pixel's series once per reference, so a lazy record would
    re-read the file from disk every time: this loads it. ``ds`` itself is left as it was.
    """
    for name in ("R", "Z"):
        if name not in ds.coords:
            raise ValueError(
                f"the record has no {name} coordinate, so its pixel positions are unknown"
            )
    reach = float(np.nanmax(np.abs(np.asarray(ds["R"].values, dtype=float))))
    low, high = CENTIMETRES
    if not low <= reach <= high:
        raise ValueError(
            f"R reaches {reach:g} on this record, which is not a major radius in centimetres"
            f" ({low:g} to {high:g}, that is 0.1 m to 7 m): it may already be in metres, or in"
            " millimetres. Nothing was converted."
        )
    # The same expression as the paper's own loader (``apd_check.fields.load``), so that the two prepare
    # a record bit for bit alike: R and Z keep the dtype they have on disk (float32 for C-Mod).
    return ds.assign_coords(R=ds.R / 100, Z=ds.Z / 100).load()


def bank_mask(bank):
    """The dead-pixel mask ``bank`` was computed with, as ``fields`` and ``blobs`` take it.

    The bank's own ``dead`` with its ``dead_mask_source``, which the API reads off the DataArray's
    attributes. A product built on a bank never computes the mask again: the file may have changed since
    (a corrected mask), and the bank, the fields and the blobs of one shot must agree on which pixels
    were never computed.
    """
    dead = bank["dead"]
    source = bank.attrs.get("dead_mask_source")
    return dead if source is None else dead.assign_attrs(dead_mask_source=source)


def average_at(bank):
    """``average_at(x, y)`` as the API takes it: reference (x, y)'s average, read back out of ``bank``."""
    return lambda x, y: pipeline.at(bank, x, y)


def live_scalars(result, scalars):
    """``{(x, y, name): value}`` at every live pixel of ``result``, one scalar per entry of ``scalars``.

    ``scalars`` maps a scalar's name, as the store records it, to the variable of ``result`` it is read
    from. Each spec writes the two side by side in its ``SCALARS``, because they are not always the same
    word: the blob keeps the API's variable names, which the paper's code reads it by, while a scalar takes
    the name the store already uses for that quantity. The value is the variable's, untouched.

    A dead pixel gets no row: it was never computed. A live pixel whose estimate failed gets a NaN, which
    the store writes as NULL, so "tried and failed" stays different from "never tried".
    """
    live_y, live_x = np.nonzero(~np.asarray(result["dead"].values, dtype=bool))
    out = {}
    for name, variable in scalars.items():
        values = np.asarray(result[variable].values, dtype=float)
        for x, y in zip(live_x.tolist(), live_y.tolist()):
            out[(x, y, name)] = float(values[y, x])
    return out
