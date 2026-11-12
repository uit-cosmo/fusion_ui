"""Adapter over the ragged ASP/FSP probe datasets.

Every quantity at every probe position has its own time dimension,
``time_<quantity>_<position>``, with no axis shared across positions or
quantities, so the imaging code paths, which assume one time axis, do not
apply. Each also has a ``rho_<quantity>_<position>`` companion on its own,
coarser time base ``rho_time_<quantity>_<position>``; it is read as its own
pair, not interpolated.
"""

import re
from dataclasses import dataclass

import numpy as np

_VAR_RE = re.compile(r"^([A-Za-z]+)_(\d+)$")


@dataclass(frozen=True)
class ProbeTrace:
    quantity: str
    position: int
    time: np.ndarray
    value: np.ndarray
    rho_time: "np.ndarray | None" = None
    rho: "np.ndarray | None" = None


def quantities_and_positions(ds):
    """``{quantity: [position, ...]}`` present in ``ds``, sorted, from the
    variable names alone."""
    found = {}
    for name in ds.data_vars:
        match = _VAR_RE.match(name)  # a rho_<quantity>_<position> does not match
        if match is None:
            continue
        quantity, position = match.group(1), int(match.group(2))
        found.setdefault(quantity, set()).add(position)
    return {
        quantity: sorted(positions) for quantity, positions in sorted(found.items())
    }


def load_trace(ds, quantity, position):
    """One (quantity, position) trace with its ρ, loaded into memory.

    Raises ``KeyError`` if that combination is not in ``ds``.
    """
    var = f"{quantity}_{position}"
    if var not in ds.data_vars:
        raise KeyError(f"{var!r} not in dataset")

    da = ds[var].load()
    time = da[da.dims[0]].values

    rho_time = rho = None
    rho_var = f"rho_{var}"
    if rho_var in ds.data_vars:
        rho_da = ds[rho_var].load()
        rho_time = rho_da[rho_da.dims[0]].values
        rho = rho_da.values

    return ProbeTrace(
        quantity=quantity,
        position=position,
        time=time,
        value=da.values,
        rho_time=rho_time,
        rho=rho,
    )


def probe_geometry(ds):
    """The file's attributes that describe the probe head."""
    origin = ds.attrs.get("probe_origin")
    return {
        "probe_type": ds.attrs.get("probe_type", ""),
        "probe_origin": np.asarray(origin).tolist() if origin is not None else None,
    }
