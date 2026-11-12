"""What a builder is handed: one shot's products as loaded, the pixel in view, and the view cuts.

Any product may be ``None``. All three carry ``R`` and ``Z`` as 2-D ``(y, x)`` coordinates in
metres, and velocities are in m/s.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class Cuts:
    """View cuts: which pixels the panels leave out. View state, never parameters.

    They apply per method (``views.methods.METHODS``), to what each estimate rests on, unless
    ``paper`` is set.

    min_lags: fewest lags a 2DCA track's slope may rest on (the three tracks). A slope through two
        or three lags is a secant, not a fit.
    min_events: fewest events the conditional average may rest on, for the methods read off it:
        the 2DCA maximum and centroid, and the TDEs applied to the average. The 2DCC and the TDEs
        off the record do not depend on the events.
    interior_only: leave out the pixels on the array's border. Every method.
    paper: the paper's cut instead, one set of pixels for every method (``views.reliable``): the
        2DCA centroid on at least ``min_lags`` lags and ``min_events`` events, a number from the
        centroid and from both TDEs, and not on the border, whatever ``interior_only`` says.
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
    #: Pixels between the peak and the neighbours the contour level is read off (``Tracking.neighbour_step``).
    neighbour_step: int = 1
    #: Names the 2DCA bank the products rest on: a view's widget state (a lag slider) is keyed on it,
    #: so banks with another lag window get their own.
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
        if self.has_coordinates:
            dataset = self.reference()
            return (
                np.asarray(dataset["R"].values, dtype=float),
                np.asarray(dataset["Z"].values, dtype=float),
            )
        ny, nx = self.shape
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
