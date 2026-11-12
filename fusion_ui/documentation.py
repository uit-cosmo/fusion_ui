"""The Documentation page's content: the hand-written text, and what it quotes from the code.

The text is ``fusion_ui/data/documentation.md``. :func:`parse` reads it into the blocks the page draws:

- a ``## `` heading starts a section, and a ``### `` heading made only of scalar names in backticks is the
  entry of those names, which the scalar table links to;
- a line holding only ``{{table:settings}}`` or ``{{table:scalars}}`` is a table generated from the code
  (:func:`table`), and a fenced ``dot`` block is the dependency diagram;
- ``{{value:NAME}}`` in a sentence is a number read from the code (:func:`facts`) and ``{{default:PATH}}``
  the default of a product's setting (:func:`defaults`), so that the text cannot drift from what runs.
  :func:`fill` replaces them;
- ``<!-- ... -->`` comments are dropped.

Nothing here touches Streamlit or the database.
"""

import dataclasses
import os
import re
from dataclasses import dataclass
from fractions import Fraction

from fusion_ui.core import params_ui, registry, scalar_labels
from fusion_ui.core.scalar_labels import NO_UNIT
from fusion_ui.plots import blob_parameters, method_fields
from fusion_ui.views.bundle import Cuts
from fusion_ui.views.products import KEYS as PRODUCTS

DOC_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "documentation.md"
)

#: The scalar names each spec writes, by plot key. The two products' are their ``SCALARS``; the others are
#: written out, and a test holds them to what each spec's ``scalars`` returns.
SOURCES = {
    "method_fields": tuple(method_fields.SCALARS),
    "blob_parameters": tuple(blob_parameters.SCALARS),
    "taud_psd": ("taud_psd", "lambda_psd"),
    "velocity_tde": ("vx_tde", "vy_tde"),
    "dead_pixels": ("dead", "psd_ratio", "number_dead"),
}

#: The unit of each name that has no label in ``core/scalar_labels.py``.
UNITS = {
    "vx_tde": "m/s",
    "vy_tde": "m/s",
    "dead": NO_UNIT,
    "psd_ratio": NO_UNIT,
    "number_dead": NO_UNIT,
}


# -- the text --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """One piece of a section, in the order the page draws them.

    ``kind`` is ``markdown`` (``text``), ``entry`` (a scalar entry: its ``heading``, the ``names`` in it and
    the ``text`` under it), ``table`` (``name``) or ``diagram`` (``text`` is DOT).
    """

    kind: str
    text: str = ""
    name: str = ""
    names: tuple = ()
    heading: str = ""

    @property
    def anchor(self):
        """The anchor of an entry: its first name's."""
        return "name-" + self.names[0].replace("_", "-") if self.names else ""


@dataclass(frozen=True)
class Section:
    title: str
    anchor: str
    blocks: tuple


@dataclass(frozen=True)
class Document:
    intro: tuple  # the blocks before the first section
    sections: tuple


_COMMENT = re.compile(r"<!--.*?-->", re.S)
_NAME = r"`([A-Za-z_][A-Za-z0-9_]*)`"
_ENTRY = re.compile(rf"^{_NAME}(?:\s*,\s*{_NAME})*$")
_TABLE = re.compile(r"^\{\{\s*table:([a-z_]+)\s*\}\}$")
_PLACEHOLDER = re.compile(r"\{\{\s*(value|default)\s*:\s*([A-Za-z0-9_.]+)\s*\}\}")


def entry_names(heading):
    """The scalar names a ``### `` heading lists, or ``()`` when it is an ordinary heading."""
    heading = heading.strip()
    return tuple(re.findall(_NAME, heading)) if _ENTRY.match(heading) else ()


def parse(text):
    """The :class:`Document` the text describes. See the module docstring for the format."""
    found = [("", [])]  # (title, blocks): the introduction, then each section
    lines, entry, in_diagram = [], "", False

    def flush(kind="markdown"):
        nonlocal entry
        body = "\n".join(lines).strip()
        if entry:
            names = entry_names(entry)
            found[-1][1].append(Block("entry", body, heading=entry, names=names))
        elif body:
            found[-1][1].append(Block(kind, body))
        lines.clear()
        entry = ""

    for line in _COMMENT.sub("", text).splitlines():
        stripped = line.strip()
        table = _TABLE.match(stripped)
        if in_diagram and stripped == "```":
            flush("diagram")
            in_diagram = False
        elif in_diagram:
            lines.append(line)
        elif stripped == "```dot":
            flush()
            in_diagram = True
        elif line.startswith("## "):
            flush()
            found.append((line[3:].strip(), []))
        elif line.startswith("### ") and entry_names(line[4:]):
            flush()
            entry = line[4:].strip()
        elif table:
            flush()
            found[-1][1].append(Block("table", name=table.group(1)))
        else:
            if line.startswith("### "):
                flush()
            lines.append(line)
    flush()
    sections = tuple(
        Section(title, re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-"), tuple(b))
        for title, b in found[1:]
    )
    return Document(intro=tuple(found[0][1]), sections=sections)


def load(path=None):
    """The text file, parsed. Raises ``OSError`` when it cannot be read: the page says so."""
    with open(path or DOC_PATH, encoding="utf-8") as handle:
        return parse(handle.read())


def fill(text, values=None, settings=None):
    """``(text, unknown)``: ``text`` with its placeholders replaced, and the ones that name nothing.

    An unknown placeholder is left in the text as it was written, so a typo shows on the page.
    """
    values = facts() if values is None else values
    settings = defaults() if settings is None else settings
    unknown = []

    def replace(match):
        kind, name = match.group(1), match.group(2)
        known = values if kind == "value" else settings
        if name not in known:
            unknown.append(f"{kind}:{name}")
            return match.group(0)
        return known[name]

    return _PLACEHOLDER.sub(replace, text), unknown


# -- what the text quotes from the code ------------------------------------------------------------------


def _number(value):
    """A value as the text shows it: ``2.5``, ``60``, ``1e+06``, ``true``, ``none``, ``lsq``."""
    if value is None or isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)


def _ms(seconds):
    return f"{seconds * 1e3:g} ms"


def _band(band):
    return f"{band[0] / 1e3:g}–{band[1] / 1e3:g} kHz"


def _ordinal(n):
    suffix = (
        "th"
        if n % 100 in (11, 12, 13)
        else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )
    return f"{n}{suffix}"


def facts():
    """``{name: text}``: the numbers the text quotes with ``{{value:NAME}}``, read from the code that uses them."""
    import imaging_methods as im
    from imaging_methods import dead_pixels, puff

    from fusion_ui.core import loader, mlp
    from fusion_ui.plots import thomson_profiles, velocity_tde

    radius = im.get_default_apd_method_params().preprocessing.radius
    tde = velocity_tde.TdeVelocityParams()
    cuts = Cuts()
    return {
        "preprocess.window": _number(2 * radius + 1),
        "preprocess.trimmed": _number(2 * radius),
        "puff.smooth": _ms(puff.SMOOTH),
        "puff.baseline": _ms(puff.BASELINE_SPAN),
        "puff.min_on": _ms(puff.MIN_ON),
        "puff.min_span": _ms(puff.MIN_SPAN),
        "puff.flat_start": f"{puff.FLAT_START * 100:g}%",
        "puff.clear_rise": _number(puff.CLEAR_RISE),
        "puff.fallback": f"{_ordinal(round(puff.FALLBACK_QUANTILE * 100))} percentile",
        "dead.red_band": _band(dead_pixels.RED_BAND),
        "dead.noise_band": _band(dead_pixels.NOISE_BAND),
        "dead.blob_band": _band(dead_pixels.BLOB_BAND),
        "dead.red": _number(dead_pixels.RED),
        "dead.gain": _number(dead_pixels.GAIN),
        "dead.day_fraction": str(
            Fraction(dead_pixels.DAY_FRACTION).limit_denominator(12)
        ),
        "loader.default_window": f"{loader.DEFAULT_WINDOW_SECONDS:g} s",
        "method_fields.scalars": _number(len(method_fields.SCALARS)),
        "blob_parameters.scalars": _number(len(blob_parameters.SCALARS)),
        "cuts.min_lags": _number(cuts.min_lags),
        "cuts.min_events": _number(cuts.min_events),
        "cuts.interior_only": "on" if cuts.interior_only else "off",
        "cuts.paper": "on" if cuts.paper else "off",
        "velocity_tde.min_threshold": _number(tde.min_threshold),
        "velocity_tde.ccf_min_lag": _number(tde.ccf_min_lag),
        "thomson.margin": _number(thomson_profiles.MARGIN),
        "thomson.percentile": f"{_ordinal(thomson_profiles.WHOLE_PERCENTILE)} percentile",
        "thomson.sigmas": _number(1 / thomson_profiles.WELL_MEASURED),
        "thomson.min_well": _number(thomson_profiles.MIN_WELL),
        "thomson.fit_factor": _number(thomson_profiles.FIT_FACTOR),
        "thomson.nearest": _number(thomson_profiles.NEAREST),
        "mlp.window": _ms(mlp.WINDOW),
        "mlp.min_fill": f"{mlp.MIN_FILL * 100:g}%",
        "mlp.prominence": f"{mlp.PROMINENCE * 100:g} cm",
        "mlp.moving": f"{mlp.MOVING * 100:g} cm/s",
        "mlp.far_sol": f"{mlp.FAR_SOL[0] * 100:g} to {mlp.FAR_SOL[1] * 100:g} cm",
        "mlp.min_fit": _number(mlp.MIN_FIT),
        "mlp.band": _number(mlp.BAND),
        "mlp.min_side": _number(mlp.MIN_SIDE),
    }


def _leaves(instance, prefix=""):
    """``(path, owner class, field name, value)`` of every leaf of a params dataclass, in field order."""
    for field in dataclasses.fields(instance):
        value = getattr(instance, field.name)
        if dataclasses.is_dataclass(value):
            yield from _leaves(value, f"{prefix}{field.name}.")
        else:
            yield prefix + field.name, type(instance), field.name, value


def settings():
    """``{path: (default, products, help)}``: every setting of the three products.

    In the order ``method_fields`` and then ``blob_parameters`` declare them. A setting's products are
    those whose parameters hold it, which are those whose cache key a change to it moves.
    """
    rows = {}
    for key in ("method_fields", "blob_parameters", "pixel_averages"):
        for path, owner, name, value in _leaves(registry.get(key).params()):
            help_text = params_ui.help_for(owner, name, path) or ""
            rows.setdefault(path, (value, [], help_text))[1].append(key)
    return {
        path: (value, sorted(products, key=PRODUCTS.index), help_text)
        for path, (value, products, help_text) in rows.items()
    }


def defaults():
    """``{path: text}``: every setting's default, as the text quotes it with ``{{default:PATH}}``."""
    return {path: _number(value) for path, (value, *_) in settings().items()}


def scalars():
    """``{name: (label or None, unit, sources)}``: every scalar name a registered spec writes."""
    rows = {}
    for source, names in SOURCES.items():
        for name in names:
            label = scalar_labels.LABELS.get(name)
            unit = label.unit if label else UNITS[name]
            rows.setdefault(name, (label, unit, []))[2].append(source)
    return rows


# -- the tables, as Markdown, which the browser's find can search --------------------------------------


def _markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = (str(cell).replace("|", "\\|").replace("\n", " ") for cell in row)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _code(items):
    return ", ".join(f"`{item}`" for item in items)


def table(name, document=None):
    """The Markdown of the table a ``{{table:NAME}}`` line names: ``settings`` or ``scalars``.

    ``document`` gives the scalar table its links to the entries. Raises ``KeyError`` for any other name.
    """
    if name == "settings":
        rows = [
            (f"`{path}`", f"`{_number(value)}`", _code(products), help_text)
            for path, (value, products, help_text) in settings().items()
        ]
        return _markdown_table(
            ("setting", "default", "a change gives new", "what it is"), rows
        )
    if name == "scalars":
        sections = document.sections if document else ()
        links = {
            scalar: block.anchor
            for section in sections
            for block in section.blocks
            for scalar in block.names
        }
        rows = [
            (
                f"[`{scalar}`](#{links[scalar]})" if scalar in links else f"`{scalar}`",
                label.quantity if label else "*none: the axis shows the name*",
                unit,
                _code(sources),
            )
            for scalar, (label, unit, sources) in scalars().items()
        ]
        return _markdown_table(
            ("name", "label on the Multi shot axis", "unit", "written by"), rows
        )
    raise KeyError(f"no table called {name!r}; there are settings and scalars")
