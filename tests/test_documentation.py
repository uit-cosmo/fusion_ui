"""The Documentation page: its text, the tables generated from the code, and the page itself.

What these tests hold the page to:

- every label of ``core/scalar_labels.py`` has exactly one entry on the page, and every entry shows the
  labels of its names, as the code has them;
- every scalar name a registered spec or the seed can write has an entry, and no entry names anything else;
- every number the text quotes from the code resolves, and so does every table it places;
- the settings table says which products a change gives new results for, as the cache keys do;
- the diagram is valid DOT, and names every product and every group of settings;
- the page renders, draws the diagram without the Python ``graphviz`` package, and names any scalar in the
  ledger it does not describe.
"""

import ast
import dataclasses
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from streamlit.testing.v1 import AppTest

import fusion_ui.plots  # noqa: F401 - registers every spec the page describes
from fusion_ui import documentation
from fusion_ui.core import db, params_ui, registry, scalar_labels, seed, store
from fusion_ui.plots import blob_parameters, method_fields
from fusion_ui.views.bundle import Cuts
from fusion_ui.views.methods import METHODS

REPO = Path(__file__).resolve().parent.parent
PAGE = str(REPO / "fusion_ui" / "pages" / "6_documentation.py")


@pytest.fixture(scope="module")
def doc():
    return documentation.load()


def entries(document):
    return [
        block
        for section in document.sections
        for block in section.blocks
        if block.kind == "entry"
    ]


# -- the labels and the entries, both ways ---------------------------------------------------------------


def test_every_label_has_exactly_one_entry(doc):
    named = [name for entry in entries(doc) for name in entry.names]
    assert len(named) == len(set(named)), "a name has two entries"
    missing = set(scalar_labels.LABELS) - set(named)
    assert not missing, f"labelled names without an entry: {sorted(missing)}"


def test_every_entry_names_a_scalar_some_source_writes_and_every_such_name_has_one(doc):
    named = {name for entry in entries(doc) for name in entry.names}
    assert named == documentation.documented_names()


def test_the_entry_of_the_gaussian_fit_says_what_the_seed_held_and_that_it_was_removed(
    doc,
):
    """The seed's values under ``lx_f``, ``ly_f`` and ``theta_f`` were a contour ellipse at 0.3 of the
    maximum, and are gone: the text says both, with the date, where a reader of the three names looks.
    """
    (entry,) = [e for e in entries(doc) if "lx_f" in e.names]
    assert set(entry.names) == set(seed.NOT_IMPORTED)
    text = " ".join(entry.text.split())
    assert "seed" in text and "contour at 0.3 of the maximum" in text
    assert "removed on 2026-10-08" in text


def test_each_entry_shows_its_names_labels_as_the_code_has_them(doc):
    for entry in entries(doc):
        lines = documentation.entry_lines(entry.names).splitlines()
        assert len(lines) == len(entry.names)
        for name, line in zip(entry.names, lines):
            label = scalar_labels.LABELS.get(name)
            assert line.startswith(f"- `{name}` · ")
            if label is not None:
                assert f"**{label}**" in line
            else:
                assert "no label" in line and f"[{documentation.UNITS[name]}]" in line


def test_the_scalar_table_has_a_row_and_a_link_for_every_name(doc):
    text = documentation.table("scalars", doc)
    links = documentation.anchors(doc)
    for row in documentation.scalar_rows():
        cell = f"[`{row.name}`](#{links[row.name]})"
        assert cell in text, row.name
        if row.label is not None:
            assert f"| {row.label.quantity} | {row.label.unit} |" in text
    anchors = {entry.anchor for entry in entries(doc)}
    assert set(links.values()) <= anchors


def test_a_labelled_names_unit_is_its_labels_and_an_unlabelled_one_has_its_own():
    for row in documentation.scalar_rows():
        if row.label is not None:
            assert row.unit == row.label.unit
            assert row.name not in documentation.UNITS
        else:
            assert row.unit in scalar_labels.UNITS, row.name


# -- what each source writes, held to the code -------------------------------------------------------------


def older_result():
    """A result with every variable an older spec's ``scalars`` reads, on a 2 x 3 array."""
    return xr.Dataset(
        {
            "refx": 1,
            "refy": 0,
            "number_events": 5,
            "taud": 2e-5,
            "lam": 0.4,
            "vx": 400.0,
            "vy": -50.0,
            "area": 1e-4,
            "lr": 0.01,
            "lz": 0.02,
            "lx": 0.01,
            "ly": 0.02,
            "theta": 0.1,
            "slope_r_cond_av": 400.0,
            "slope_z_cond_av": 10.0,
            "slope_r_cross_corr": 380.0,
            "slope_z_cross_corr": 12.0,
            "tau_prime": 1e-5,
            "sigma_t": 0.4,
            "l_prime": 0.005,
            "sigma_sp": 0.5,
            "dead": (
                ("y", "x"),
                np.array([[True, False, False], [False, False, True]]),
            ),
            "red_ratio": (("y", "x"), np.full((2, 3), 200.0)),
        }
    )


def written_by(spec):
    return {
        key if isinstance(key, str) else key[2] for key in spec.scalars(older_result())
    }


def test_every_registered_spec_that_writes_scalars_is_described():
    for spec in registry.REGISTRY.values():
        if spec.scalars is None:
            continue
        if spec.key in documentation.PRODUCT_SCALARS:
            continue
        assert (
            spec.key in documentation.OLDER_SOURCES
        ), f"{spec.key} writes scalars the page lists nowhere"
        assert written_by(spec) == set(documentation.OLDER_SOURCES[spec.key]), spec.key


def test_every_older_source_listed_is_a_registered_spec_or_the_seed():
    for source in documentation.OLDER_SOURCES:
        assert (
            source == seed.IMPORT_PLOT or registry.get(source).scalars is not None
        ), source


def test_the_seeds_names_are_the_fields_of_its_records_but_the_three_it_leaves_out():
    discharge = pytest.importorskip("density_scan.discharge")
    fields = {field.name for field in dataclasses.fields(discharge.BlobParameters)}
    assert len(fields) == 15 and set(seed.NOT_IMPORTED) <= fields
    assert set(documentation.OLDER_SOURCES[seed.IMPORT_PLOT]) == fields - set(
        seed.NOT_IMPORTED
    )


def test_the_seed_is_no_source_of_the_names_it_no_longer_writes():
    by_name = {row.name: row for row in documentation.scalar_rows()}
    for name in seed.NOT_IMPORTED:
        assert seed.IMPORT_PLOT not in by_name[name].sources, name
        assert by_name[name].sources == ("blob_parameters", "gaussian_sizes"), name
    # Every other name of the seed is still its.
    for name in documentation.OLDER_SOURCES[seed.IMPORT_PLOT]:
        assert seed.IMPORT_PLOT in by_name[name].sources, name


def test_the_products_names_are_their_scalars():
    assert documentation.PRODUCT_SCALARS == {
        "method_fields": method_fields.SCALARS,
        "blob_parameters": blob_parameters.SCALARS,
    }
    product_names = set(method_fields.SCALARS) | set(blob_parameters.SCALARS)
    assert product_names == set(scalar_labels.LABELS)


def test_the_shared_names_have_the_sources_the_labels_module_lists():
    by_name = {row.name: row for row in documentation.scalar_rows()}
    for name, sources in scalar_labels.SHARED.items():
        assert set(by_name[name].sources[1:]) == set(sources), name


def test_every_unlabelled_name_has_a_unit_and_every_unit_a_name():
    unlabelled = {row.name for row in documentation.scalar_rows() if row.label is None}
    assert unlabelled == set(documentation.UNITS)


# -- what the text quotes from the code ------------------------------------------------------------------


def test_every_placeholder_and_table_in_the_text_resolves(doc):
    values, settings = documentation.facts(), documentation.defaults()
    unknown = []
    for block in [*doc.intro, *(b for section in doc.sections for b in section.blocks)]:
        if block.kind == "table":
            assert block.name in documentation.TABLES
            assert documentation.table(block.name, doc).startswith("| ")
        _, missing = documentation.fill(block.text, values, settings)
        unknown += missing
    assert not unknown


def test_the_numbers_quoted_are_the_codes():
    import_config = documentation.import_config
    import_config()
    from density_scan import dead_pixels, preprocess, puff

    values = documentation.facts()
    assert values["preprocess.window"] == str(2 * preprocess.RADIUS + 1)
    assert values["dead.red"] == f"{dead_pixels.RED:g}"
    assert values["dead.gain"] == f"{dead_pixels.GAIN:g}"
    assert values["puff.min_on"] == f"{puff.MIN_ON * 1e3:g} ms"
    assert values["cuts.min_lags"] == str(Cuts().min_lags)
    defaults = documentation.defaults()
    assert (
        defaults["averages.threshold"]
        == f"{method_fields.MethodFieldsParams().averages.threshold:g}"
    )
    assert defaults["neighbour_step"] == str(
        blob_parameters.BlobParametersParams().neighbour_step
    )


def test_a_placeholder_that_names_nothing_is_reported_and_left_as_written():
    text, unknown = documentation.fill(
        "a {{value:dead.red}} b {{value:no.such}} c {{default:averages.window}} d {{default:nope}}"
    )
    assert unknown == ["value:no.such", "default:nope"]
    assert "{{value:no.such}}" in text and "{{default:nope}}" in text
    assert (
        "{{value:dead.red}}" not in text and "{{default:averages.window}}" not in text
    )


def test_every_section_of_the_text_names_code_as_file_and_function(doc):
    """Each account of a computation names the code it describes: every section but the conventions and
    the diagram's names at least one ``file.py: function``."""
    for section in doc.sections:
        text = " ".join(block.text for block in section.blocks)
        if section.title in ("Conventions", "How the quantities depend on one another"):
            continue
        assert ".py: " in text, section.title


# -- the parser ------------------------------------------------------------------------------------------


TEXT = """<!-- a comment
that spans lines, with a ## heading in it -->
Intro.

{{checked}}

## First section

Some text {{value:dead.red}}.

### An ordinary heading

More text.

{{table:cuts}}

### `vr_com`, `vz_com`

Entry text.

### `level`
Level text.

## Second section

{{diagram}}
"""


def test_the_parser_splits_sections_blocks_and_entries():
    parsed = documentation.parse(TEXT)
    assert [b.kind for b in parsed.intro] == ["markdown", "checked"]
    assert parsed.intro[0].text == "Intro."
    first, second = parsed.sections
    assert (first.title, first.anchor) == ("First section", "first-section")
    kinds = [(b.kind, b.name or b.names) for b in first.blocks]
    assert kinds == [
        ("markdown", ()),
        ("markdown", ()),
        ("table", "cuts"),
        ("entry", ("vr_com", "vz_com")),
        ("entry", ("level",)),
    ]
    assert first.blocks[1].text.startswith("### An ordinary heading")
    assert first.blocks[3].text == "Entry text."
    assert (
        first.blocks[3].anchor == documentation.entry_anchor("vr_com") == "name-vr-com"
    )
    assert first.blocks[4].text == "Level text."
    assert [b.kind for b in second.blocks] == ["diagram"]
    assert "comment" not in " ".join(b.text for b in parsed.intro)


def test_only_a_heading_made_of_names_is_an_entry():
    assert documentation.entry_names("`vr_com`, `vz_com`") == ("vr_com", "vz_com")
    assert documentation.entry_names("`level`") == ("level",)
    assert documentation.entry_names("The `level` and more") == ()
    assert documentation.entry_names("pixel_averages") == ()


# -- the generated tables --------------------------------------------------------------------------------


def _changed(value):
    """Another valid value for a leaf of the products' parameters."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 1.5 + 0.1
    if value is None:
        return 0.01
    return {"hann": "hamming", "lsq": "central_diff"}[value]


def _nested(path, value):
    head, *rest = path.split(".")
    return {head: _nested(".".join(rest), value) if rest else value}


def test_a_settings_products_are_those_whose_cache_key_a_change_moves():
    """What the table says a change replaces is what the hashes do: a change to a setting moves the cache
    key of every product whose parameters hold it, the bank's through ``upstream_params``.
    """
    for row in documentation.setting_rows():
        moved = set()
        for key in documentation.PRODUCTS:
            spec = registry.get(key)
            params = spec.params()
            leaves = {
                path: value for path, _, _, value in documentation._leaves(params)
            }
            if row.path not in leaves:
                continue
            assert leaves[row.path] == row.default, row.path
            other = params_ui.with_values(
                params, _nested(row.path, _changed(row.default))
            )
            if (
                params_ui.hash_params(key, other)[0]
                != params_ui.hash_params(key, params)[0]
            ):
                moved.add(key)
            if spec.upstream_params is not None:
                bank = spec.requires
                before = params_ui.hash_params(bank, spec.upstream_params(params))[0]
                after = params_ui.hash_params(bank, spec.upstream_params(other))[0]
                if before != after:
                    moved.add(bank)
        assert moved == set(row.products), row.path


def test_the_settings_table_lists_every_setting_once_with_its_help():
    rows = documentation.setting_rows()
    paths = [row.path for row in rows]
    assert len(paths) == len(set(paths))
    every = set()
    for key in documentation.PRODUCTS:
        every |= {
            path for path, *_ in documentation._leaves(registry.get(key).params())
        }
    assert set(paths) == every
    assert all(row.help for row in rows), [row.path for row in rows if not row.help]
    text = documentation.settings_table()
    for row in rows:
        assert f"`{row.path}`" in text


def test_the_cuts_table_is_the_fields_pages():
    rows = {
        label: (default, applies)
        for label, default, applies in documentation.cut_rows()
    }
    cuts = Cuts()
    assert rows["Minimum lags"][0] == str(cuts.min_lags)
    assert rows["Minimum events"][0] == str(cuts.min_events)
    assert rows["Minimum lags"][1] == ", ".join(m.label for m in METHODS if m.nlags)
    assert rows["Minimum events"][1] == ", ".join(
        m.label for m in METHODS if m.uses_events
    )
    labels = [label for label, _ in documentation.CUT_LABELS]
    page = (REPO / "fusion_ui" / "pages" / "5_fields.py").read_text()
    for label in labels:
        assert f'"{label}"' in page, f"the Fields page has no cut called {label!r}"


def test_the_products_table_has_every_product():
    text = documentation.products_table()
    for key in documentation.PRODUCTS:
        assert f"`{key}`" in text
    assert f"{len(method_fields.SCALARS)} per live pixel" in text
    assert f"{len(blob_parameters.SCALARS)} per live pixel" in text


def test_an_unknown_table_is_refused():
    with pytest.raises(KeyError):
        documentation.table("nothing")


# -- the diagram -----------------------------------------------------------------------------------------


def test_the_diagram_names_every_product_and_every_group_of_settings():
    dot = documentation.diagram()
    for key in documentation.PRODUCTS:
        assert key in dot
    groups = set()
    for row in documentation.setting_rows():
        parts = row.path.split(".")
        groups.add(
            ".".join(parts[:2]) if parts[0] in ("tracking", "blobs") else parts[0]
        )
    for group in groups:
        assert group in dot, group


@pytest.mark.skipif(
    shutil.which("dot") is None, reason="Graphviz's dot is not installed here"
)
def test_the_diagram_is_valid_dot():
    done = subprocess.run(
        ["dot", "-Tsvg"],
        input=documentation.diagram(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert "<svg" in done.stdout
    assert "Warning" not in done.stderr and "Error" not in done.stderr, done.stderr


# -- purity ----------------------------------------------------------------------------------------------


def _imports(path):
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""


def test_the_module_never_imports_streamlit():
    assert not [
        m for m in _imports(documentation.__file__) if m.split(".")[0] == "streamlit"
    ]


# -- the page --------------------------------------------------------------------------------------------


@pytest.fixture
def deployment(monkeypatch, tmp_path):
    """A fresh ledger holding one product's name and one no source writes; every cache cleared."""
    import streamlit as st

    database = tmp_path / "state" / "shot_explorer.sqlite"
    monkeypatch.setenv("FUSION_UI_DB", str(database))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(tmp_path / "alcator"))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(tmp_path / "plasma_discharges.json"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")
    conn = db.open_db(database)
    target = registry.Target(
        machine="cmod",
        shot=1160616027,
        diagnostic="apd",
        preprocessed=True,
        path="",
        t_start=float("nan"),
        t_end=float("nan"),
        window_source="none",
    )
    digest, _ = store.record_params(
        conn, "method_fields", method_fields.MethodFieldsParams()
    )
    run = store.record_run(
        conn,
        target,
        "method_fields",
        digest,
        blob_path=None,
        status="ok",
        error=None,
        seconds=None,
        code_version="test",
    )
    store.write_scalars(
        conn, run["id"], {(5, 4, "vr_com"): 600.0, (5, 4, "vx_field"): 1.0}
    )
    conn.close()
    st.cache_data.clear()
    st.cache_resource.clear()
    yield database
    st.cache_data.clear()
    st.cache_resource.clear()


def run_page():
    app = AppTest.from_file(PAGE, default_timeout=120).run()
    assert not app.exception, app.exception
    return app


def test_the_page_renders_every_section_entry_and_label(deployment, doc):
    app = run_page()
    assert [t.value for t in app.title] == ["Documentation"]
    assert [h.value for h in app.header] == [section.title for section in doc.sections]
    assert [h.proto.anchor for h in app.header] == [
        section.anchor for section in doc.sections
    ]
    assert [s.value for s in app.subheader] == [entry.heading for entry in entries(doc)]
    prose = "\n".join(m.value for m in app.markdown)
    for name, label in scalar_labels.LABELS.items():
        assert f"`{name}` · **{label}**" in prose, name
    assert "{{" not in prose, "a placeholder was left in the page"


def test_each_entry_is_followed_by_its_labels(deployment, doc):
    """The label lines come right under their own entry's heading, not merely somewhere on the page."""
    app = run_page()
    nodes = [
        node
        for node in app.main
        if getattr(node, "type", None) in ("subheader", "markdown")
    ]
    for index, node in enumerate(nodes):
        if node.type != "subheader":
            continue
        names = documentation.entry_names(node.value)
        assert names, node.value
        lines = nodes[index + 1]
        assert lines.type == "markdown"
        for name in names:
            label = scalar_labels.LABELS.get(name)
            assert f"`{name}` · " in lines.value
            if label is not None:
                assert f"**{label}**" in lines.value


def test_the_page_draws_the_diagram_from_dot_text(deployment):
    app = run_page()
    charts = app.get("graphviz_chart")
    assert len(charts) == 1
    assert charts[0].proto.spec == documentation.diagram()


def test_the_page_names_a_ledger_scalar_it_does_not_describe(deployment):
    app = run_page()
    warnings = [w.value for w in app.warning]
    assert any("`vx_field`" in w for w in warnings), warnings
    assert not any("`vr_com`" in w for w in warnings)


def test_the_page_says_when_its_text_names_what_the_code_lacks(
    deployment, monkeypatch, tmp_path
):
    text = tmp_path / "documentation.md"
    text.write_text(
        "## Only\n\nA value {{value:no.such.thing}}.\n\n{{table:nothing}}\n"
    )
    monkeypatch.setattr(documentation, "DOC_PATH", str(text))
    app = run_page()
    warnings = " ".join(w.value for w in app.warning)
    assert "value:no.such.thing" in warnings and "table:nothing" in warnings


def test_the_page_survives_a_missing_text(deployment, monkeypatch, tmp_path):
    monkeypatch.setattr(documentation, "DOC_PATH", str(tmp_path / "gone.md"))
    app = run_page()
    assert [e.value for e in app.error][0].startswith(
        "The text of this page could not be read"
    )
