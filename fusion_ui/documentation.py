"""The Documentation page's content: the prose, and everything in it that is read off the code.

``fusion_ui/pages/6_documentation.py`` draws the page and decides nothing. What it draws is built here, as
plain data, so that each part can be tested without a Streamlit runtime: this module never calls Streamlit
or the database, and reads no file but the prose. It sits outside ``core/`` because it reads the specs and
the Fields page's views, as ``jump.py`` does.

**The prose** is ``fusion_ui/data/documentation.md``, written by hand so that it is easy to reword, as
``run_days.md`` is. :func:`parse` reads it:

- a ``## `` heading starts a section of the page;
- a line holding only ``{{table:NAME}}`` (:data:`TABLES`), ``{{diagram}}`` or ``{{checked}}`` is replaced
  by a table, the dependency diagram, or the commits the text was checked against;
- ``{{value:NAME}}`` in a sentence is a number read from the code (:func:`facts`), and
  ``{{default:PATH}}`` a setting's default read from the products' parameter classes
  (:func:`defaults`), so that the text cannot drift from what runs;
- a ``### `` heading made only of scalar names in backticks is the **entry** of those names: the page
  puts each name's label, unit and sources under it (:func:`entry_lines`);
- ``<!-- ... -->`` comments are dropped.

**The tables are generated** and cannot drift: the scalar names with their labels
(``core/scalar_labels.py``), units and sources (:func:`scalar_rows`); the settings, with their defaults and
the products whose results a change replaces (:func:`setting_rows`); the products; the Fields page's cuts.
Two things are written out here by hand: what the single-shot specs and the seed write
(:data:`OLDER_SOURCES`), which a test holds to the code by running every registered spec's ``scalars``, and
the units of the names that have no label (:data:`UNITS`).
"""

import dataclasses
import os
import re
from dataclasses import dataclass
from fractions import Fraction

from fusion_ui.core import params_ui, registry, scalar_labels, seed
from fusion_ui.core.fusion_scripts import import_config
from fusion_ui.core.scalar_labels import NO_UNIT
from fusion_ui.plots import blob_parameters, method_fields
from fusion_ui.views import products as product_views
from fusion_ui.views.bundle import Cuts
from fusion_ui.views.methods import METHODS

DOC_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "documentation.md"
)

#: The three products, in the order they are built: the bank first.
PRODUCTS = product_views.KEYS

#: The scalars each product writes per live pixel, by name: ``{scalar name: blob variable}``.
PRODUCT_SCALARS = {
    "method_fields": method_fields.SCALARS,
    "blob_parameters": blob_parameters.SCALARS,
}

#: The seed: the scalars imported from ``density_scan/results.json``.
SEED = seed.IMPORT_PLOT

#: What every source but the products writes: each single-shot spec that writes scalars, by plot key, and
#: the seed. A test runs every registered spec's ``scalars`` and holds this to it; the seed's names are the
#: fields of ``density_scan.discharge.BlobParameters``.
OLDER_SOURCES = {
    "two_dca": ("number_events",),
    "taud_psd": ("taud_psd", "lambda_psd"),
    "velocity_contour": ("vx_c", "vy_c", "area_c"),
    "fwhm_sizes": ("lr", "lz"),
    "gaussian_sizes": ("lx_f", "ly_f", "theta_f"),
    "velocity_2dca_tde": ("vx_2dca_tde", "vy_2dca_tde"),
    "velocity_tde": ("vx_tde", "vy_tde"),
    "trajectories": ("vx_2dca_lsq", "vy_2dca_lsq", "vx_ccf_lsq", "vy_ccf_lsq"),
    "two_sided_exp": ("tau_prime", "sigma_t", "l_prime", "sigma_sp"),
    "dead_pixels": ("dead", "psd_ratio", "number_dead"),
    SEED: (
        "vx_c",
        "vy_c",
        "area_c",
        "vx_2dca_tde",
        "vy_2dca_tde",
        "vx_tde",
        "vy_tde",
        "lx_f",
        "ly_f",
        "lr",
        "lz",
        "theta_f",
        "taud_psd",
        "lambda_psd",
        "number_events",
    ),
}

#: The unit of each name that has no label (a labelled name's unit is the last thing in its label), as
#: each module computes it: velocities and sizes divided by 100 from centimetres.
UNITS = {
    "vx_c": "m/s",
    "vy_c": "m/s",
    "area_c": "m²",
    "vx_2dca_tde": "m/s",
    "vy_2dca_tde": "m/s",
    "vx_tde": "m/s",
    "vy_tde": "m/s",
    "vx_2dca_lsq": "m/s",
    "vy_2dca_lsq": "m/s",
    "vx_ccf_lsq": "m/s",
    "vy_ccf_lsq": "m/s",
    "tau_prime": "s",
    "sigma_t": NO_UNIT,
    "l_prime": "m",
    "sigma_sp": NO_UNIT,
    "dead": NO_UNIT,
    "psd_ratio": NO_UNIT,
    "number_dead": NO_UNIT,
}

#: The tables a ``{{table:NAME}}`` line can name.
TABLES = ("products", "settings", "scalars", "cuts")

#: The code this text was last read against, by repository: the commit at which every account of a
#: computation was checked. Update it whenever the text is read against the code again.
CHECKED_AGAINST = {
    "fusion_scripts": "9e14f01",
    "imaging-methods": "8a9bc07",
    "velocity-estimation": "b3b6945",
    "experimental_database": "47b2d6b",
    "fpp-analysis-tools": "6f98750",
}


# ---------------------------------------------------------------------------------------------------------
# The prose
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """One piece of a section, in the order the page draws them.

    ``kind`` is ``"markdown"`` (``text``), ``"table"`` (``name``), ``"diagram"``, ``"checked"`` or
    ``"entry"``: a scalar entry, whose ``names`` the page lists with their labels under ``heading``, then
    ``text``.
    """

    kind: str
    text: str = ""
    name: str = ""
    names: tuple = ()
    heading: str = ""

    @property
    def anchor(self):
        """The anchor of an entry: its first name's, which the scalar table links to."""
        return entry_anchor(self.names[0]) if self.names else ""


@dataclass(frozen=True)
class Section:
    title: str
    anchor: str
    blocks: tuple


@dataclass(frozen=True)
class Document:
    intro: tuple  # blocks before the first section
    sections: tuple


_COMMENT = re.compile(r"<!--.*?-->", re.S)
_SECTION = re.compile(r"^##\s+(.+?)\s*$")
_SUBSECTION = re.compile(r"^###\s+(.+?)\s*$")
_NAME = r"`([A-Za-z_][A-Za-z0-9_]*)`"
_ENTRY = re.compile(rf"^{_NAME}(?:\s*,\s*{_NAME})*$")
_ENTRY_NAMES = re.compile(_NAME)
_BLOCK = re.compile(r"^\{\{\s*(table:[a-z_]+|diagram|checked)\s*\}\}$")
_PLACEHOLDER = re.compile(r"\{\{\s*(value|default)\s*:\s*([A-Za-z0-9_.]+)\s*\}\}")


def slug(text):
    """An anchor for a heading: lower case, words joined by hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def entry_anchor(name):
    """The anchor of the entry that holds ``name``."""
    return f"name-{name.replace('_', '-')}"


def entry_names(heading):
    """The scalar names a ``### `` heading lists, or ``()`` when it is an ordinary heading."""
    if not _ENTRY.match(heading.strip()):
        return ()
    return tuple(_ENTRY_NAMES.findall(heading))


def parse(text):
    """The :class:`Document` the prose describes. See the module docstring for the format."""
    text = _COMMENT.sub("", text)
    intro, sections = [], []
    title, blocks = None, intro
    lines = []
    entry = None  # (heading, names) of the entry being read

    def flush():
        nonlocal lines, entry
        body = "\n".join(lines).strip("\n")
        if entry is not None:
            blocks.append(
                Block("entry", text=body.strip(), heading=entry[0], names=entry[1])
            )
        elif body.strip():
            blocks.append(Block("markdown", text=body.strip()))
        lines, entry = [], None

    for line in text.splitlines():
        section = _SECTION.match(line)
        if section and not line.startswith("###"):
            flush()
            if title is not None:
                sections.append(Section(title, slug(title), tuple(blocks)))
            title, blocks = section.group(1), []
            continue
        sub = _SUBSECTION.match(line)
        if sub and entry_names(sub.group(1)):
            flush()
            entry = (sub.group(1), entry_names(sub.group(1)))
            continue
        if sub:
            flush()
        block = _BLOCK.match(line.strip())
        if block:
            flush()
            what = block.group(1)
            if what.startswith("table:"):
                blocks.append(Block("table", name=what.split(":", 1)[1]))
            else:
                blocks.append(Block(what))
            continue
        lines.append(line)
    flush()
    if title is not None:
        sections.append(Section(title, slug(title), tuple(blocks)))
    return Document(intro=tuple(intro), sections=tuple(sections))


def load(path=None):
    """The prose file, parsed. Raises ``OSError`` when it cannot be read: the page says so."""
    with open(path or DOC_PATH, encoding="utf-8") as handle:
        return parse(handle.read())


def fill(text, values=None, settings=None):
    """``(text, unknown)``: ``text`` with its placeholders replaced, and the ones that name nothing.

    An unknown placeholder is left in the text as it was written, so a typo shows on the page, and is
    returned so that the page can say so; a test holds the prose to having none.
    """
    values = facts() if values is None else values
    settings = defaults() if settings is None else settings
    unknown = []

    def replace(match):
        kind, name = match.group(1), match.group(2)
        table = values if kind == "value" else settings
        if name not in table:
            unknown.append(f"{kind}:{name}")
            return match.group(0)
        return table[name]

    return _PLACEHOLDER.sub(replace, text), unknown


# ---------------------------------------------------------------------------------------------------------
# Numbers read off the code
# ---------------------------------------------------------------------------------------------------------


def _number(value):
    """A number as the text shows it: ``2.5``, ``60``, ``1e+06``; ``true``/``false``; ``none``."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)


def _ms(seconds):
    return f"{seconds * 1e3:g} ms"


def _band(band):
    low, high = band
    return f"{low / 1e3:g}–{high / 1e3:g} kHz"


def _ordinal(n):
    suffix = (
        "th"
        if n % 100 in (11, 12, 13)
        else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )
    return f"{n}{suffix}"


def facts():
    """``{name: text}``: the numbers the prose quotes, read from the code that uses them.

    The fusion_scripts constants of preprocessing, the puff window and the dead-pixel estimate; the
    Fields page's default cuts; and the defaults of the older single-shot specs.
    """
    import_config()  # density_scan reads its settings by the bare name `config`
    from density_scan import dead_pixels, preprocess, puff

    from fusion_ui.core import loader
    from fusion_ui.plots import two_dca, velocity_contour, velocity_tde

    contour = velocity_contour.ContourVelocityParams()
    tde = velocity_tde.TdeVelocityParams()
    cuts = Cuts()
    return {
        "preprocess.radius": _number(preprocess.RADIUS),
        "preprocess.window": _number(2 * preprocess.RADIUS + 1),
        "preprocess.trimmed": _number(2 * preprocess.RADIUS),
        "puff.smooth": _ms(puff.SMOOTH),
        "puff.baseline": _ms(puff.BASELINE_SPAN),
        "puff.min_on": _ms(puff.MIN_ON),
        "puff.min_span": _ms(puff.MIN_SPAN),
        "puff.min_dip": _ms(puff.MIN_DIP),
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
        "dead.nperseg": _number(dead_pixels.NPERSEG),
        "loader.default_window": f"{loader.DEFAULT_WINDOW_SECONDS:g} s",
        "cuts.min_lags": _number(cuts.min_lags),
        "cuts.min_events": _number(cuts.min_events),
        "older.two_dca_threshold": _number(
            two_dca.TwoDcaSpecParams().two_dca.threshold
        ),
        "older.contour_level": _number(contour.contouring.threshold_factor),
        "older.window_size": _number(contour.position_filter.window_size),
        "older.mask_signal_factor": _number(contour.position_filter.mask_signal_factor),
        "older.estimator": _number(contour.velocity.estimator),
        "older.tde_threshold": _number(tde.min_threshold),
        "older.tde_ccf_min_lag": _number(tde.ccf_min_lag),
    }


# ---------------------------------------------------------------------------------------------------------
# The settings
# ---------------------------------------------------------------------------------------------------------


def _leaves(instance, prefix=""):
    """``(path, owner class, field name, value)`` of every leaf of a params dataclass, in field order."""
    for field in dataclasses.fields(instance):
        value = getattr(instance, field.name)
        path = f"{prefix}.{field.name}" if prefix else field.name
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            yield from _leaves(value, path)
        else:
            yield path, type(instance), field.name, value


@dataclass(frozen=True)
class Setting:
    """One leaf of the products' parameters: what the settings table shows of it."""

    path: str
    default: object
    products: tuple  # the products whose cache key holds it, so whose results a change replaces
    help: str


def setting_rows():
    """Every setting of the three products, in the order their parameter classes declare them.

    The defaults are the classes' own, and a setting's products are those whose parameters hold it: a
    change to it gives those products a new cache key, and new results beside the old ones.
    """
    rows = {}
    for key in PRODUCTS:
        for path, owner, name, value in _leaves(registry.get(key).params()):
            if path not in rows:
                rows[path] = dict(
                    default=value,
                    products=[],
                    help=params_ui.help_for(owner, name, path) or "",
                )
            rows[path]["products"].append(key)
    # In the order a person reads the settings: as method_fields and then blob_parameters declare them.
    reading = []
    for key in ("method_fields", "blob_parameters", "pixel_averages"):
        for path, *_ in _leaves(registry.get(key).params()):
            if path not in reading:
                reading.append(path)
    return [
        Setting(
            path,
            rows[path]["default"],
            tuple(rows[path]["products"]),
            rows[path]["help"],
        )
        for path in reading
    ]


def defaults():
    """``{path: text}``: every setting's default as the prose quotes it with ``{{default:PATH}}``."""
    return {row.path: _number(row.default) for row in setting_rows()}


# ---------------------------------------------------------------------------------------------------------
# The scalar names
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scalar:
    """One scalar name: what the axis calls it, its unit, and who writes it."""

    name: str
    label: object  # scalar_labels.Label, or None for a name without one
    unit: str
    sources: tuple  # plot keys, and the seed's

    @property
    def quantity(self):
        return self.label.quantity if self.label is not None else None


def scalar_rows():
    """Every scalar name a result can hold, the products' first (in the order they write them).

    A product's name takes its label and unit from ``core/scalar_labels.py``; an older one has no label,
    and its unit is in :data:`UNITS`. The sources are the product that writes it and every older source
    in :data:`OLDER_SOURCES` that writes the same name.
    """
    names = []
    owner = {}
    for key, scalars in PRODUCT_SCALARS.items():
        for name in scalars:
            names.append(name)
            owner[name] = key
    for written in OLDER_SOURCES.values():
        for name in written:
            if name not in owner and name not in names:
                names.append(name)
    rows = []
    for name in names:
        label = scalar_labels.LABELS.get(name)
        older = tuple(
            source for source, written in OLDER_SOURCES.items() if name in written
        )
        sources = ((owner[name],) if name in owner else ()) + older
        rows.append(
            Scalar(
                name, label, label.unit if label is not None else UNITS[name], sources
            )
        )
    return rows


def documented_names():
    """The set of scalar names the page describes."""
    return {row.name for row in scalar_rows()}


def undocumented(names):
    """The names in ``names`` (say, every name in a ledger) that the page does not describe, sorted."""
    return sorted(set(names) - documented_names())


def entry_lines(names):
    """The markdown the page puts under an entry's heading: each name's label, unit and sources."""
    by_name = {row.name: row for row in scalar_rows()}
    lines = []
    for name in names:
        row = by_name.get(name)
        if row is None:
            lines.append(f"- `{name}` · *not a name any source writes*")
            continue
        what = (
            f"**{row.label}**"
            if row.label is not None
            else f"no label, the Multi shot axis shows the name · [{row.unit}]"
        )
        lines.append(f"- `{name}` · {what} · written by {_code_list(row.sources)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------
# The tables, as markdown: a table drawn as text can be searched with the browser's find, which a
# st.dataframe cannot.
# ---------------------------------------------------------------------------------------------------------


def _cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def markdown_table(header, rows):
    """A GitHub-flavoured markdown table."""
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _code_list(items):
    return ", ".join(f"`{item}`" for item in items)


def anchors(document):
    """``{scalar name: the anchor of the entry that holds it}`` in ``document``."""
    out = {}
    for section in document.sections:
        for block in section.blocks:
            if block.kind == "entry":
                for name in block.names:
                    out.setdefault(name, block.anchor)
    return out


def scalars_table(links=None):
    """Every scalar name, with its label, unit and sources; each name links to its entry in ``links``."""
    links = links or {}

    def name_cell(name):
        return f"[`{name}`](#{links[name]})" if name in links else f"`{name}`"

    rows = [
        (
            name_cell(row.name),
            row.quantity if row.label is not None else "*no label*",
            row.unit,
            _code_list(row.sources),
        )
        for row in scalar_rows()
    ]
    return markdown_table(("name", "label", "unit", "written by"), rows)


def settings_table():
    rows = [
        (
            f"`{row.path}`",
            f"`{_number(row.default)}`",
            _code_list(row.products),
            row.help or "",
        )
        for row in setting_rows()
    ]
    return markdown_table(
        ("setting", "default", "a change gives new", "what it is"), rows
    )


def products_table():
    rows = []
    for key in PRODUCTS:
        spec = registry.get(key)
        built_on = (
            "the record"
            if spec.requires is None
            else f"`{spec.requires}` and the record"
        )
        computed = (
            "in batch only, by `fusion-ui precompute`"
            if spec.batch_only
            else "by `fusion-ui precompute`, or on the Single shot page once its bank is stored"
        )
        scalars = PRODUCT_SCALARS.get(key)
        rows.append(
            (
                f"`{key}`",
                spec.label,
                built_on,
                computed,
                product_views.COST.get(key, ""),
                f"{len(scalars)} per live pixel" if scalars else "none",
            )
        )
    return markdown_table(
        ("product", "what it is", "read from", "computed", "cost a shot", "scalars"),
        rows,
    )


#: The Fields page's cuts, as its sidebar names them, with the field of ``Cuts`` each sets.
CUT_LABELS = (
    ("Minimum lags", "min_lags"),
    ("Minimum events", "min_events"),
    ("Interior pixels only", "interior_only"),
    ("The paper's cut, reliable()", "paper"),
)


def cut_rows():
    """``[(cut, default, the panels it applies to)]``, from ``views.methods.METHODS`` and ``Cuts``."""
    default = Cuts()
    applies = {
        "min_lags": ", ".join(m.label for m in METHODS if m.nlags),
        "min_events": ", ".join(m.label for m in METHODS if m.uses_events),
        "interior_only": "every panel",
        "paper": "every panel, the same pixels in each, in place of the three above",
    }
    rows = []
    for label, field in CUT_LABELS:
        value = getattr(default, field)
        shown = (
            ("on" if value else "off") if isinstance(value, bool) else _number(value)
        )
        rows.append((label, shown, applies[field]))
    return rows


def cuts_table():
    return markdown_table(("cut", "default", "applies to"), cut_rows())


def table(name, document=None):
    """The markdown of the table a ``{{table:NAME}}`` line names; ``document`` gives the scalar table its
    links to the entries."""
    builders = {
        "products": products_table,
        "settings": settings_table,
        "scalars": lambda: scalars_table(anchors(document) if document else None),
        "cuts": cuts_table,
    }
    if name not in builders:
        raise KeyError(f"no table called {name!r}; there are {', '.join(TABLES)}")
    return builders[name]()


# ---------------------------------------------------------------------------------------------------------
# The commits the text was checked against
# ---------------------------------------------------------------------------------------------------------


def checked_text():
    """The line a ``{{checked}}`` block shows: the commits the text was read against.

    Stated, not compared with what runs: most commits to these repositories leave the code described here
    alone, and a notice raised by every one of them would teach the reader to skip it.
    """
    return (
        "Checked against the code at "
        + ", ".join(f"{name} `{commit}`" for name, commit in CHECKED_AGAINST.items())
        + "."
    )


# ---------------------------------------------------------------------------------------------------------
# The diagram
# ---------------------------------------------------------------------------------------------------------


def _fields_of(instance):
    return ", ".join(f.name for f in dataclasses.fields(instance))


def diagram():
    """How the quantities depend on one another, as DOT text for ``st.graphviz_chart``.

    Blue boxes are data, white boxes computations, yellow notes settings and green boxes the places a
    result is read. The settings groups are read off the products' parameter classes.
    """
    method = method_fields.MethodFieldsParams()
    averages = _fields_of(method.averages).replace(", ", ",\\n", 1)
    tde = _fields_of(method.tde)
    data = 'style="filled", fillcolor="#dbe9f6"'
    setting = 'shape=note, style="filled", fillcolor="#fff3c4"'
    read = 'style="rounded,filled", fillcolor="#e3f1df"'
    return f"""digraph dependencies {{
  graph [rankdir=TB, fontname="Helvetica", fontsize=11, nodesep=0.3, ranksep=0.32, newrank=true, bgcolor="white"];
  node [fontname="Helvetica", fontsize=10, shape=box, style="rounded,filled", fillcolor="white", color="#555555", margin="0.08,0.04"];
  edge [color="#666666", arrowsize=0.6, fontname="Helvetica", fontsize=9, fontcolor="#555555"];

  subgraph cluster_files {{
    label="preprocessing (fusion_scripts, density_scan)"; style="rounded,dashed"; color="#999999"; fontcolor="#444444";
    raw [label="raw APD file", {data}];
    window [label="discharge window\\n(discharge database)", {data}];
    puff [label="analysis window\\npuff.find_puff"];
    dead [label="dead-pixel mask\\ndead_pixels.estimate_shot,\\nconsolidate"];
    pre [label="preprocessed file\\npreprocess.preprocess_shot", {data}];
  }}

  subgraph cluster_record {{
    label=""; style="invis";
    record [label="the record\\npreprocessed file over the discharge window,\\nR and Z in metres (_pipeline.record)", {data}];
  }}

  subgraph cluster_bank {{
    label="pixel_averages (batch only)"; style="rounded"; color="#3b6ea5"; fontcolor="#3b6ea5";
    s_averages [label="averages\\n{averages}", {setting}];
    twodca [label="2DCA at every live reference\\nfind_events_and_2dca"];
    bank [label="cond_av, cond_repr, cross_corr,\\nnevents, dead", {data}];
  }}

  subgraph cluster_fields {{
    label="method_fields"; style="rounded"; color="#3b6ea5"; fontcolor="#3b6ea5";
    s_step [label="tracking.neighbour_step", {setting}];
    s_filter [label="tracking.position_filter,\\ntracking.cross_corr_mask_signal_factor,\\ntracking.velocity", {setting}];
    s_tde [label="tde\\n{tde}", {setting}];
    level_com [label="level_com\\nneighbour_level"];
    t_max [label="maximum track"];
    t_com [label="centroid track"];
    t_2dcc [label="2DCC track"];
    fit [label="fit lags and slope\\npipeline.track"];
    tde [label="TDE off the record\\npipeline.tde_fields"];
    catde [label="TDE on the average\\npipeline.ca_tde"];
    fields_blob [label="per track: vr, vz, nlags, level, pos, fit;\\nnevents; the TDE velocities, cc_tde", {data}];
  }}

  subgraph cluster_blobs {{
    label="blob_parameters"; style="rounded"; color="#3b6ea5"; fontcolor="#3b6ea5";
    s_bstep [label="neighbour_step", {setting}];
    s_gauss [label="blobs.gauss_fit", {setting}];
    s_taud [label="blobs.taud_estimation", {setting}];
    level [label="level\\nneighbour_level"];
    contour [label="zero-lag contour\\nget_contour_evolution"];
    fwhm [label="FWHM\\nestimate_fwhm_sizes"];
    gauss [label="Gaussian fit\\nfit_ellipse_to_event"];
    psd [label="PSD fit\\nDurationTimeEstimator"];
    blobs_blob [label="nevents, level, area, lx_c, ly_c, theta_c,\\nlr, lz, lx_f, ly_f, theta_f, taud, lam", {data}];
  }}

  cuts [label="view cuts: minimum lags,\\nminimum events, interior only,\\nthe paper's reliable()", shape=note, style="filled", fillcolor="#f2e3f5"];
  ledger [label="scalars in the ledger", shape=cylinder, style="filled", fillcolor="#e3f1df"];
  multishot [label="Multi shot page\\n(every pixel with a value)", {read}];
  fieldspage [label="Fields page", {read}];

  raw -> puff; window -> puff;
  raw -> dead; puff -> dead;
  raw -> pre; dead -> pre; puff -> pre;
  pre -> record; window -> record [label=" slice"];
  record -> twodca; s_averages -> twodca; twodca -> bank;

  bank -> t_max [label=" cond_av"];
  bank -> level_com [label=" cond_av"];
  bank -> t_2dcc [label=" cross_corr"];
  bank -> catde [label=" cond_av"];
  s_step -> level_com; level_com -> t_com;
  t_max -> fit; t_com -> fit; t_2dcc -> fit; s_filter -> fit;
  record -> tde; s_tde -> tde;
  fit -> fields_blob; tde -> fields_blob; catde -> fields_blob;

  bank -> level [label=" cond_av"]; s_bstep -> level; level -> contour;
  bank -> fwhm; bank -> gauss; s_gauss -> gauss;
  record -> psd; s_taud -> psd;
  contour -> blobs_blob; fwhm -> blobs_blob; gauss -> blobs_blob; psd -> blobs_blob;

  fields_blob -> ledger [label=" {len(method_fields.SCALARS)} names"];
  blobs_blob -> ledger [label=" {len(blob_parameters.SCALARS)} names"];
  ledger -> multishot;
  fields_blob -> fieldspage; bank -> fieldspage [style=dashed]; blobs_blob -> fieldspage [style=dashed];
  cuts -> fieldspage;
}}
"""
