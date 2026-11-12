"""What an entry of ``SHOT_VIEWS`` or ``PIXEL_VIEWS`` is: a pure builder, what it reads, its controls.

A builder is ``build(bundle, **controls)`` and returns a ``go.Figure``, a ``pandas.DataFrame``, a
``str`` (a sentence the page shows as information) or a list of these.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Control:
    """One widget of a view: view state, never a parameter, so moving it recomputes nothing.

    ``kind`` is ``"radio"`` (``options``, with ``labels`` to show), ``"slider"`` or ``"number"``
    (``bounds`` = ``(low, high, step)``; a number starts empty and the builder reads ``None`` as
    "work it out", the ``placeholder`` saying what that comes to) or ``"text"``.
    """

    key: str
    label: str
    kind: str
    default: Any = None
    options: tuple = ()
    labels: dict = field(default_factory=dict)
    bounds: tuple = ()
    placeholder: str = ""
    help: str = ""


@dataclass(frozen=True)
class View:
    """One figure or table of the Fields page."""

    key: str
    title: str
    #: Product keys the builder cannot do without. The page names the command that fills a missing one.
    reads: tuple
    build: Callable
    #: ``controls(bundle) -> tuple of Control``; bounds and placeholders may depend on the data.
    controls: Optional[Callable] = None
    #: A shot-level figure whose markers carry ``[x, y]``: clicking one opens the pixel level.
    selectable: bool = False
    caption: str = ""
