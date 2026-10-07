"""What a builder is handed: one shot's products as loaded, the pixel in view, and the view cuts.

The page loads the three products (``pixel_averages``, ``method_fields``, ``blob_parameters``) and
puts them in a :class:`Bundle`; a builder reads whatever it needs off it and never touches the
store. Any product may be ``None``: the page only calls a builder whose ``reads`` are present, but a
builder that can do without one (the lag strip without ``method_fields``) is told so by ``optional``.

All three products carry ``R`` and ``Z`` as 2-D ``(y, x)`` coordinates in metres, and velocities are
in m/s. Nothing here converts units.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class Cuts:
    """View cuts: which pixels the panels leave out of the arrows. View state, never parameters.

    Moving one never recomputes anything, so none of them is part of a product's cache key.

    The cuts apply per method (``views.methods.METHODS``), to what each estimate rests on, unless
    ``paper`` is set.

    min_lags: fewest lags a 2DCA track's slope may rest on, for the three tracks (the 2DCC keeps it).
        A slope through two or three lags is a secant, not a fit, and the wildest velocities in a
        field come from such pixels.
    min_events: fewest events the conditional average at a pixel may rest on, for the methods read
        off that average: the 2DCA maximum and centroid, and the TDEs applied to it. The 2DCC is
        read off the cross-correlation of the whole record, and a TDE off the record: neither
        depends on the events, and this never cuts them.
    interior_only: leave out the pixels on the array's border, where the 2DCA track leaves the view
        within a few lags and a TDE loses half its neighbour pairs. Every method.
    paper: the paper's own cut instead of those three, one set of pixels for every method
        (``views.reliable``, which is ``reliable()`` of ``apd_check/figures.py``): the 2DCA
        centroid on at least ``min_lags`` lags and the average on at least ``min_events`` events, a
        number from the centroid and from both TDEs, and not on the border. The two minimums are the
        ones above; the border is always out, whatever ``interior_only`` says.
    """

    min_lags: int = 8
    min_events: int = 200
    interior_only: bool = False
    paper: bool = False


@dataclass(frozen=True)
class Bundle:
    """The products of one shot, the pixel in view and the cuts, as a builder sees them."""

    shot: int = 0
    bank: Optional[xr.Dataset] = None  # pixel_averages
    fields: Optional[xr.Dataset] = None  # method_fields
    blobs: Optional[xr.Dataset] = None  # blob_parameters
    pixel: Optional[tuple] = None  # (x, y) in view, or None
    cuts: Cuts = field(default_factory=Cuts)
    #: Pixels between the peak and the neighbours the contour level is read off
    #: (``Tracking.neighbour_step``): the page reads it out of the settings in use.
    neighbour_step: int = 1
    #: Names the 2DCA bank these products rest on, so a view's own state (a lag slider) is shared
    #: by shots with the same lags and apart from those with another window.
    tag: str = ""

    def reference(self):
        """The first product present, whichever it is: the array's shape and grid are the same in all."""
        for dataset in (self.fields, self.bank, self.blobs):
            if dataset is not None:
                return dataset
        return None

    @property
    def shape(self):
        """``(ny, nx)``."""
        dataset = self.reference()
        if dataset is None:
            return (0, 0)
        return int(dataset.sizes["y"]), int(dataset.sizes["x"])

    @property
    def grid(self):
        """``(R, Z)`` as float ``(y, x)`` arrays in metres, or the pixel indices when a product has none."""
        dataset = self.reference()
        ny, nx = self.shape
        if dataset is not None and "R" in dataset.coords and "Z" in dataset.coords:
            return (
                np.asarray(dataset["R"].values, dtype=float),
                np.asarray(dataset["Z"].values, dtype=float),
            )
        x, y = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
        return x, y

    @property
    def has_coordinates(self):
        dataset = self.reference()
        return dataset is not None and "R" in dataset.coords and "Z" in dataset.coords

    @property
    def dead(self):
        """The mask the products were computed with, ``(y, x)`` bool; nothing dead when none is stored."""
        for dataset in (self.fields, self.bank):
            if dataset is not None and "dead" in dataset:
                return np.asarray(dataset["dead"].values, dtype=bool)
        return np.zeros(self.shape, dtype=bool)

    @property
    def mask_source(self):
        """Where the mask came from, as the products record it (``dead_mask_source``), or ``None``."""
        for dataset in (self.fields, self.bank):
            if dataset is not None and dataset.attrs.get("dead_mask_source"):
                return str(dataset.attrs["dead_mask_source"])
        return None

    @property
    def min_cc(self):
        """The TDE's minimum correlation, as ``method_fields`` records it."""
        if self.fields is not None and "min_cc" in self.fields.attrs:
            try:
                return float(self.fields.attrs["min_cc"])
            except (TypeError, ValueError):
                pass
        return 0.5

    @property
    def labels(self):
        """Axis titles for the grid: metres when the products carry coordinates, indices otherwise."""
        return ("R [m]", "Z [m]") if self.has_coordinates else ("x", "y")

    def field_values(self, name):
        """``(y, x)`` floats of a ``method_fields`` variable, or ``None`` when it is not there."""
        if self.fields is None or name not in self.fields:
            return None
        return np.asarray(self.fields[name].values, dtype=float)
