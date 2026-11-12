"""What the three product specs share: the physics API, and what it is called with.

``pixel_averages``, ``method_fields`` and ``blob_parameters`` are thin adapters over
``imaging_methods.decorrelation``. This is the one module of fusion_ui that imports it.

Nothing here touches Streamlit, the database or the filesystem.
"""

import numpy as np

from imaging_methods import decorrelation as pipeline
from imaging_methods.decorrelation import Averages, Blobs, Tde, Tracking

__all__ = [
    "Averages",
    "Blobs",
    "Tde",
    "Tracking",
    "average_at",
    "bank_mask",
    "hand_made",
    "live_scalars",
    "pipeline",
    "record",
]

#: A major radius in centimetres: 0.1 m to 7 m. C-Mod's R reads 88 to 91; in metres or millimetres it
#: falls outside. A heuristic that cannot tell 300 cm from 300 mm.
CENTIMETRES = (10.0, 700.0)


def record(ds):
    """The record as the API takes it: R and Z in metres, loaded into memory. ``ds`` is left as it was.

    The APD files hold R and Z in centimetres, and the API works in SI units. The magnitude of R is
    checked, since a record already in metres would be divided again and every velocity would come out
    a hundred times too small. Loaded because the 2DCA reads every pixel's series once per reference.
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
    # R and Z keep the dtype they have on disk (float32 for C-Mod).
    return ds.assign_coords(R=ds.R / 100, Z=ds.Z / 100).load()


def hand_made():
    """The hand-made dead-pixel masks by run day, for the files that store none (``dead_mask``)."""
    from experimental_database import hand_made_dead_pixel_masks

    return hand_made_dead_pixel_masks()


def bank_mask(bank):
    """The dead-pixel mask ``bank`` was computed with, with its source, as ``fields`` and ``blobs`` take it.

    A product built on a bank never computes the mask again: the file may have changed since, and the
    bank, the fields and the blobs of one shot must agree on which pixels were never computed.
    """
    dead = bank["dead"]
    source = bank.attrs.get("dead_mask_source")
    return dead if source is None else dead.assign_attrs(dead_mask_source=source)


def average_at(bank):
    """``average_at(x, y)`` as the API takes it: reference (x, y)'s average, read back out of ``bank``."""
    return lambda x, y: pipeline.at(bank, x, y)


def live_scalars(result, scalars):
    """``{(x, y, name): value}`` at every live pixel of ``result``.

    ``scalars`` maps a scalar's name, as the store records it, to the variable of ``result`` it is read
    from. A dead pixel gets no row: it was never computed. A live pixel whose estimate failed gets a
    NaN, which the store writes as NULL.
    """
    live_y, live_x = np.nonzero(~np.asarray(result["dead"].values, dtype=bool))
    out = {}
    for name, variable in scalars.items():
        values = np.asarray(result[variable].values, dtype=float)
        for x, y in zip(live_x.tolist(), live_y.tolist()):
            out[(x, y, name)] = float(values[y, x])
    return out
