"""Adapter over the ragged ASP/FSP probe datasets.

Every quantity x probe position carries its own time dimension --
``time_<quantity>_<position>``, 107k-207k samples, with no shared axis across
positions or quantities. A shot's ASP file typically holds six quantities
(``ne``, ``Is``, ``Js``, ``Te``, ``Vf``, ``Vp``) at four probe positions --
twenty-four independent traces -- so this cannot reuse the imaging (APD/
phantom) code paths, which assume one shared time axis.

Each quantity also has a ``rho_<quantity>_<position>`` companion: the same
quantity mapped onto normalised flux coordinate, but computed on its own,
coarser time base (``rho_time_<quantity>_<position>``) -- not a resampling of
the raw trace, so it is read as its own pair rather than interpolated here.

Unlike APD/phantom, these files are tens of MB, not hundreds: a full trace is
loaded outright, no time-window slicing needed.
"""

import re
from dataclasses import dataclass

import numpy as np

_VAR_RE = re.compile(r"^([A-Za-z]+)_(\d+)$")
_RHO_PREFIX = "rho_"


@dataclass(frozen=True)
class ProbeTrace:
    quantity: str
    position: int
    time: np.ndarray
    value: np.ndarray
    rho_time: "np.ndarray | None" = None
    rho: "np.ndarray | None" = None


def quantities_and_positions(ds):
    """``{quantity: [position, ...]}`` present in ``ds``, sorted.

    Reads variable names only -- no data touched.
    """
    found = {}
    for name in ds.data_vars:
        if name.startswith(_RHO_PREFIX):
            continue
        match = _VAR_RE.match(name)
        if match is None:
            continue
        quantity, position = match.group(1), int(match.group(2))
        found.setdefault(quantity, set()).add(position)
    return {
        quantity: sorted(positions) for quantity, positions in sorted(found.items())
    }


def load_trace(ds, quantity, position, with_rho=True):
    """One (quantity, position) trace, loaded into memory.

    Raises ``KeyError`` if that combination is not in ``ds`` -- callers should
    build their selector from :func:`quantities_and_positions` rather than
    guessing a (quantity, position) pair.
    """
    var = f"{quantity}_{position}"
    if var not in ds.data_vars:
        raise KeyError(f"{var!r} not in dataset")

    da = ds[var].load()
    time = da[da.dims[0]].values

    rho_time = rho = None
    if with_rho:
        rho_var = f"{_RHO_PREFIX}{var}"
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
    """The per-shot attrs describing the probe head -- not per quantity."""
    origin = ds.attrs.get("probe_origin")
    return {
        "probe_type": ds.attrs.get("probe_type", ""),
        "probe_origin": np.asarray(origin).tolist() if origin is not None else None,
    }
