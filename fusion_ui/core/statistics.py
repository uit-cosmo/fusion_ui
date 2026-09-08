"""``StatSpec`` -- the sibling of ``PlotSpec`` for statistics over traces.

``PlotSpec.render`` is ``(result, params, target) -> figure``: one target, one
result. A statistics view is many traces from many targets on one axis, and
``Target`` cannot express that. Rather than bend the contract every spec
depends on, this module adds a small sibling with the same purity rule --
``compute`` and ``render`` never touch Streamlit, the database or the
filesystem -- and the same one-file-plus-one-import registration.

Statistics are live: a Welch PSD or a histogram of a 583k-sample trace is tens
of milliseconds, these views produce no scalar anyone wants on a multi-shot
axis (that is what the cached ``taud_psd`` spec is *for*), so there are no
``runs`` rows, no blobs, no schema change.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class StatSpec:
    """One statistic: its parameters, how to compute it, and how to draw it."""

    #: Stable identifier. Keys the session state and the params prefix.
    key: str
    #: What the statistic selectbox shows.
    label: str
    #: A dataclass. Walked by :mod:`fusion_ui.core.params_ui` exactly like a
    #: ``PlotSpec``'s -- only ``int``/``float``/``str``/``bool`` leaves.
    params: type
    #: ``(trace, params) -> xr.Dataset``, or
    #: ``(trace, reference, params) -> xr.Dataset`` when ``pairwise``.
    compute: Callable
    #: ``(items, params) -> go.Figure | None``. ``items`` is
    #: ``[(Trace, xr.Dataset), ...]`` in basket order.
    render: Callable
    #: Needs a reference trace, and a common time base (see
    #: :func:`fusion_ui.core.traces.common_grid`).
    pairwise: bool = False
    #: One line under the statistic picker.
    description: str = ""


REGISTRY = {}


def register(spec):
    """Add ``spec`` to the registry and return it.

    A duplicate key is a programming error, not something to resolve silently.
    """
    if spec.key in REGISTRY:
        raise ValueError(f"a StatSpec is already registered under {spec.key!r}")
    REGISTRY[spec.key] = spec
    return spec


def get(key):
    return REGISTRY[key]


def all_specs():
    """Every registered spec, in registration order."""
    return list(REGISTRY.values())
