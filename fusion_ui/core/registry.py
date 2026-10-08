"""``PlotSpec`` -- the one shape every analysis in this app takes.

An analysis registers once and yields two things: a figure for the single-shot
view and a dict of scalars for the multi-shot view. There is no separate
multi-shot pipeline to keep in sync -- plotting a quantity for one shot is the
same call that deposits its scalars into the store the scatter reads. Adding a
plot is one file in :mod:`fusion_ui.plots` and one :func:`register` call; no UI
code, no new storage, and it appears in both views at once.

Two kinds of spec live in the same registry, distinguished only by whether
``compute`` is set:

*cached* (``compute`` is a callable)
    ``compute(ds, params) -> xr.Dataset`` runs once per (target, params), its
    result is written to netCDF under ``CACHE_DIR`` and recorded in ``runs``,
    and ``scalars(result)`` is written to ``scalars``. ``render`` gets the
    stored result back and returns a figure.

*live* (``compute`` is ``None``)
    There is nothing to cache: the time-sliced dataset *is* the result, and
    ``render`` draws it directly. The frame viewer and the probe trace are
    this -- they are interactive, cheap, and produce no derived quantity.

``render`` may therefore either return a ``go.Figure`` (the page draws it, with
the shared chrome) or draw into Streamlit itself and return ``None`` (for a
view that needs a slider, a click target, or several figures). Anything more
rigid would have forced the frame viewer to stay outside the registry, which
is the thing this design is for.

A cached spec may also be built *on another one*. Set ``requires`` to the
upstream plot key and ``upstream_params`` to the function that pulls the
upstream's parameters out of this spec's own, and ``compute`` is called as
``compute(ds, params, upstream)`` with the upstream's cached result already in
hand. That exists because 2DCA costs about half a minute on a real shot and
almost every blob quantity is derived from the same conditional average:
without chaining, each derived plot would pay for its own copy of it and store
a duplicate blob. Because ``upstream_params`` reads out of the downstream
parameters, the two hashes stay in step -- change the 2DCA threshold and both
the average and everything derived from it get a new cache entry.

A cached spec may also be *batch only* (``batch_only=True``): too expensive
to start from a page, so it is computed by ``fusion-ui precompute`` and pages
only ever read it back from cache. The 2DCA at every pixel of a shot is this
-- 35 to 70 minutes on one core, inside a Streamlit process that serves the
whole group. A spec built on a batch-only one is still computed on demand,
but only once every batch-only link beneath it is cached; see
:func:`fusion_ui.core.store.missing_batch_upstreams`.

A cached spec may also need *the whole record* (``whole_record=True``). Every
other spec computes on the file cut to the discharge window, which is what
keeps a 500 MB record out of memory. A spec that has to find something
*against the start of the record* cannot work from that cut: the gas puff is
found against the dark level at the start of the record, and a discharge window
that begins after the puff has risen would read "already on". Such a spec is
handed the whole record, with the discharge window it would have been cut to
carried in the dataset's attributes
(:func:`fusion_ui.core.loader.discharge_window`), and cuts the record itself.
Every place that cuts a dataset before ``compute`` honours the flag, through
:func:`fusion_ui.core.loader.input_for`: the single-shot page and
``fusion-ui precompute``.

What must never enter a spec's ``params``: view state. A frame index, a
selected pixel, a zoom -- those live in ``st.session_state`` keyed off
:attr:`Target.key`. A slider drag must not mint a new ``param_sets`` row.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class Target:
    """What a plot is being computed *on*, plus the window it is restricted to.

    Built once per page render and passed to every spec, so nothing downstream
    has to re-derive a path, re-open a file to find the time window, or invent
    its own session-state key convention.
    """

    machine: str
    shot: int
    diagnostic: str
    preprocessed: bool
    path: str
    t_start: float
    t_end: float
    window_source: str = "metadata"  # "metadata" | "default" | "none"

    @property
    def key(self):
        """A stable, filesystem- and session-state-safe identifier.

        ``cmod_1160616027_apd_p``. Used for widget keys, session-state keys and
        blob filenames, so a second shot or the preprocessed variant of the
        same one can never share state with the first.
        """
        suffix = "p" if self.preprocessed else "r"
        return f"{self.machine}_{self.shot}_{self.diagnostic}_{suffix}"

    @property
    def label(self):
        kind = "preprocessed" if self.preprocessed else "raw"
        return f"{self.machine} {self.shot} · {self.diagnostic} ({kind})"


@dataclass(frozen=True)
class PlotSpec:
    """One analysis: its parameters, how to compute it, and how to draw it."""

    #: Stable identifier. Part of the cache key and of ``runs.plot``, so
    #: renaming one orphans its cached results -- treat it as permanent.
    key: str
    #: What the plot selectbox shows.
    label: str
    #: Diagnostic names this accepts, as the strings used everywhere else in
    #: the app (``"apd"``, ``"asp"``) -- never ``Diagnostic`` members. The enum
    #: stays an implementation detail inside :mod:`fusion_ui.core.loader`.
    diagnostics: tuple
    #: A dataclass. Walked by :mod:`fusion_ui.core.params_ui` to build both the
    #: widget panel and the canonical dict that is hashed, so the form and the
    #: cache key cannot disagree.
    params: type
    #: ``(result, params, target) -> go.Figure | None``. Returning ``None``
    #: means the callable drew into Streamlit itself.
    render: Callable
    #: ``(ds, params) -> xr.Dataset``, or ``None`` for a live view.
    compute: Optional[Callable] = None
    #: ``(result) -> dict``. Keys are either a ``str`` (a shot-level scalar,
    #: stored at the ``x = y = -1`` sentinel) or an ``(x, y, name)`` tuple for
    #: a value that belongs to one pixel.
    scalars: Optional[Callable] = None
    #: ``(ds, field_path, chosen) -> tuple | None`` for options that only the
    #: opened file knows -- the probe view's quantity and position lists.
    #: ``chosen`` holds the values picked so far, so selectboxes can chain.
    choices: Optional[Callable] = None
    #: ``(items, params, target) -> go.Figure``. Draws one figure over several
    #: pixels: ``items`` is ``[((x, y), result), …]`` in selection order, and
    #: the spec owns the axes, scales and legend -- the same purity contract
    #: ``render`` has. For a cached spec ``result`` is the stored result of
    #: that pixel's run; for a live spec it is the time-sliced dataset itself
    #: (the same object ``render`` gets), so the overlay reads each pixel off
    #: the already-open file -- exactly what ``render`` already does. A cached
    #: spec without one falls back to the scalar view in
    #: :mod:`fusion_ui.core.multipixel`.
    overlay: Optional[Callable] = None
    #: Plot key of a spec whose result this one consumes. When set, ``compute``
    #: is called as ``compute(ds, params, upstream)`` and the store resolves --
    #: from cache where it can -- the upstream result first.
    requires: Optional[str] = None
    #: ``(params) -> upstream params``. Required with :attr:`requires`, and it
    #: must read out of this spec's own parameters so the two cache keys cannot
    #: drift apart.
    upstream_params: Optional[Callable] = None
    #: One line under the plot picker.
    description: str = ""
    #: Which version of a file this accepts: ``False`` the raw file only, ``True``
    #: the preprocessed one only, ``None`` either. The dead-pixel check needs
    #: the raw file -- preprocessing interpolates dead pixels away.
    preprocessed: Optional[bool] = None
    #: Computed only by ``fusion-ui precompute``, never from a page: the store
    #: refuses to compute it unless the caller says it is a batch job, the
    #: single-shot page shows it from cache or names the command that fills
    #: it, and many-pixel mode offers neither it nor anything built on it.
    #: Only a cached spec can be batch only.
    batch_only: bool = False
    #: ``compute`` is handed the whole record, not the file cut to the
    #: discharge window, with the discharge window in its attributes
    #: (``loader.discharge_window(ds)``). The one exception to "never load a
    #: full time axis", for a spec that must see the record from its start: the
    #: dead-pixel view finds the gas puff against the dark level there. Only a
    #: cached spec can need it, and a chain agrees on it, since the store hands
    #: every link the same dataset.
    whole_record: bool = False

    @property
    def cached(self):
        return self.compute is not None

    def accepts(self, preprocessed):
        """Whether this spec runs on the raw (``False``) or preprocessed file."""
        return self.preprocessed is None or self.preprocessed == bool(preprocessed)


REGISTRY = {}


def register(spec):
    """Add ``spec`` to the registry and return it, so it can decorate a module.

    A duplicate key is a programming error, not something to resolve silently:
    it would mean two analyses sharing a cache namespace.
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
    if spec.requires is not None:
        # Checked at registration, not at compute time: a typo here would
        # otherwise surface as a failed run on someone's shot.
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
        missing = set(spec.diagnostics) - set(REGISTRY[spec.requires].diagnostics)
        if missing:
            raise ValueError(
                f"{spec.key!r} accepts {sorted(missing)} but its upstream "
                f"{spec.requires!r} does not"
            )
        if spec.whole_record != REGISTRY[spec.requires].whole_record:
            # The store hands the same dataset down the chain: one link given
            # the whole record would silently compute on a time axis it was
            # never written for, or one given the cut would look for the start
            # of a record that is not there.
            raise ValueError(
                f"{spec.key!r} and its upstream {spec.requires!r} disagree on "
                "whole_record, but the store hands both the same dataset"
            )
    elif spec.upstream_params is not None:
        raise ValueError(f"{spec.key!r} sets upstream_params but no requires")
    REGISTRY[spec.key] = spec
    return spec


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
