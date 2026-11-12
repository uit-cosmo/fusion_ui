"""``PlotSpec``: the one shape every analysis in this app takes.

An analysis registers once, in :mod:`fusion_ui.plots`, and yields a figure for
the single-shot view and scalars for the multi-shot view. A page picks a spec
out of the registry and knows nothing else about it.

A spec is *cached* when it has a ``compute``: the result is written to netCDF
under ``CACHE_DIR`` and recorded in ``runs``, and ``render`` gets it back. It is
*live* when ``compute`` is ``None``: the time-sliced dataset is the result and
``render`` draws it directly.

``compute``, ``render``, ``scalars`` and ``overlay`` never touch the database or
the filesystem; :mod:`fusion_ui.core.store` does that. View state (a frame
index, a selected pixel, a zoom) is never a parameter: it lives in
``st.session_state`` keyed off :attr:`Target.key`.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from fusion_ui.core import catalog


@dataclass(frozen=True)
class Target:
    """The file a plot is computed on, and the window it is restricted to."""

    machine: str
    shot: int
    diagnostic: str
    preprocessed: bool
    path: str
    t_start: float
    t_end: float
    window_source: str = "metadata"  # "metadata" | "default" | "none"
    #: The shot's discharge-DB entry, or ``None``, for a view to quote. It is
    #: not part of what the plot is computed on, so it is left out of equality,
    #: hashing and :attr:`key`.
    discharge: Optional[object] = field(default=None, compare=False, repr=False)

    @property
    def key(self):
        """``cmod_1160616027_apd_p``: safe in widget keys, session state and
        blob filenames, and different for the raw and preprocessed file."""
        suffix = "p" if self.preprocessed else "r"
        return f"{self.machine}_{self.shot}_{self.diagnostic}_{suffix}"

    @property
    def label(self):
        kind = "preprocessed" if self.preprocessed else "raw"
        return f"{self.machine} {self.shot} · {self.diagnostic} ({kind})"


@dataclass(frozen=True)
class PlotSpec:
    """One analysis: its parameters, how to compute it, and how to draw it."""

    #: Permanent: part of the cache key and of ``runs.plot``.
    key: str
    #: What the plot selectbox shows.
    label: str
    #: Diagnostic names this accepts, as strings (``"apd"``, ``"asp"``).
    diagnostics: tuple
    #: A dataclass. :mod:`fusion_ui.core.params_ui` walks it for both the
    #: widget panel and the hash, so the form and the cache key cannot disagree.
    params: type
    #: ``(result, params, target) -> go.Figure | None``, with a fourth argument
    #: ``companions`` for a spec that names some. ``None`` means it drew into
    #: Streamlit itself (a slider, a click target, several figures).
    render: Callable
    #: ``(ds, params) -> xr.Dataset``, or ``(ds, params, upstream)`` with
    #: :attr:`requires`. ``None`` for a live spec.
    compute: Optional[Callable] = None
    #: ``(result) -> dict``. A ``str`` key is a shot-level scalar, an
    #: ``(x, y, name)`` key a value that belongs to one pixel.
    scalars: Optional[Callable] = None
    #: ``(ds, field_path, chosen) -> tuple | None``: options only the opened
    #: file knows. ``chosen`` holds the values picked so far, so selectboxes
    #: can chain.
    choices: Optional[Callable] = None
    #: ``(items, params, target) -> go.Figure`` over several pixels. ``items``
    #: is ``[((x, y), result), …]`` in selection order; ``result`` is that
    #: pixel's stored result for a cached spec, the open dataset for a live one.
    overlay: Optional[Callable] = None
    #: Plot key of a spec whose cached result this one's ``compute`` is handed.
    requires: Optional[str] = None
    #: ``(params) -> upstream params``, required with :attr:`requires`. It must
    #: read out of this spec's own parameters, so the two cache keys stay in step.
    upstream_params: Optional[Callable] = None
    #: One line under the plot picker.
    description: str = ""
    #: ``False`` the raw file only, ``True`` the preprocessed one only, ``None``
    #: either.
    preprocessed: Optional[bool] = None
    #: Too slow for a page: computed only by ``fusion-ui precompute``, and read
    #: back from cache everywhere else.
    batch_only: bool = False
    #: ``compute`` gets the whole record, not the cut to the discharge window,
    #: with that window in its attributes (``loader.discharge_window``). For a
    #: spec that must see the record from its start.
    whole_record: bool = False
    #: Other diagnostics of the same shot that a live spec draws beside its own
    #: file, such as ``("asp",)``. The single-shot page opens each lazily and
    #: passes ``render`` a dict ``{diagnostic: dataset or None}``.
    companions: tuple = ()

    @property
    def cached(self):
        return self.compute is not None

    def accepts(self, preprocessed):
        """Whether this spec runs on the raw (``False``) or preprocessed file."""
        return self.preprocessed is None or self.preprocessed == bool(preprocessed)


REGISTRY = {}


def register(spec):
    """Add ``spec`` to the registry and return it.

    Every rule is checked here, at import, so a mistake does not surface as a
    failed run on someone's shot.
    """
    if spec.key in REGISTRY:
        raise ValueError(f"a PlotSpec is already registered under {spec.key!r}")
    if not spec.diagnostics:
        raise ValueError(f"{spec.key!r} accepts no diagnostics")
    if spec.batch_only and spec.compute is None:
        raise ValueError(
            f"{spec.key!r} is batch only but a live spec: there is nothing to "
            "compute in batch, so nothing a page could read back"
        )
    if spec.whole_record and spec.compute is None:
        raise ValueError(
            f"{spec.key!r} needs the whole record but is a live spec: a live "
            "spec has no compute to hand it to, and its view is the cut dataset"
        )
    _check_companions(spec)
    if spec.requires is not None:
        if spec.requires not in REGISTRY:
            raise ValueError(
                f"{spec.key!r} requires {spec.requires!r}, which is not "
                "registered -- import it first in fusion_ui/plots/__init__.py"
            )
        if spec.upstream_params is None:
            raise ValueError(
                f"{spec.key!r} sets requires but no upstream_params, so there "
                "is no way to say which upstream result it wants"
            )
        if spec.compute is None:
            raise ValueError(f"{spec.key!r} sets requires but is a live spec")
        upstream = REGISTRY[spec.requires]
        missing = set(spec.diagnostics) - set(upstream.diagnostics)
        if missing:
            raise ValueError(
                f"{spec.key!r} accepts {sorted(missing)} but its upstream "
                f"{spec.requires!r} does not"
            )
        if spec.whole_record != upstream.whole_record:
            raise ValueError(
                f"{spec.key!r} and its upstream {spec.requires!r} disagree on "
                "whole_record, but the store hands both the same dataset"
            )
    elif spec.upstream_params is not None:
        raise ValueError(f"{spec.key!r} sets upstream_params but no requires")
    REGISTRY[spec.key] = spec
    return spec


def _check_companions(spec):
    """Refuse a ``companions`` declaration the single-shot page could not honour."""
    companions = spec.companions
    if not isinstance(companions, tuple):
        # A bare string would iterate as its letters: ("asp") is "asp".
        raise ValueError(
            f"{spec.key!r} names its companions as {companions!r}: give a tuple "
            "of diagnostic names, such as ('asp',)"
        )
    if not companions:
        return
    if spec.compute is not None:
        raise ValueError(
            f"{spec.key!r} names companions but is a cached spec: a result "
            "computed from other files would need them in its cache key"
        )
    if spec.overlay is not None:
        raise ValueError(
            f"{spec.key!r} names companions and has an overlay: many-pixel mode "
            "opens no companion, so its overlay would draw without them"
        )
    for name in companions:
        if not isinstance(name, str) or name not in catalog.DIAGNOSTICS:
            raise ValueError(
                f"{spec.key!r} names {name!r} as a companion, which is no "
                f"diagnostic: they are {', '.join(catalog.DIAGNOSTICS)}"
            )
        if name in spec.diagnostics:
            raise ValueError(
                f"{spec.key!r} names {name!r} as a companion, but it is one of "
                "its own diagnostics: a companion is another diagnostic's file"
            )
    if len(set(companions)) != len(companions):
        raise ValueError(f"{spec.key!r} names a companion twice: {companions!r}")


def get(key):
    return REGISTRY[key]


def chain(spec):
    """``[spec, its upstream, …]`` down the ``requires`` links."""
    links = [spec]
    while links[-1].requires is not None:
        links.append(get(links[-1].requires))
    return links


def for_diagnostic(diagnostic, preprocessed=None):
    """Every spec that accepts ``diagnostic``, in registration order; only those
    that run on that version of the file when ``preprocessed`` is given."""
    return [
        s
        for s in REGISTRY.values()
        if diagnostic in s.diagnostics
        and (preprocessed is None or s.accepts(preprocessed))
    ]
