"""The Documentation page: its text, what the text quotes from the code, and the page itself."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from streamlit.testing.v1 import AppTest

import fusion_ui.plots  # noqa: F401 - registers every spec the page describes
from fusion_ui import documentation
from fusion_ui.core import registry
from fusion_ui.plots import method_fields

PAGE = str(
    Path(__file__).resolve().parent.parent
    / "fusion_ui"
    / "pages"
    / "6_documentation.py"
)


@pytest.fixture(scope="module")
def doc():
    return documentation.load()


def blocks(document, kind):
    every = [*document.intro, *(b for s in document.sections for b in s.blocks)]
    return [block for block in every if block.kind == kind]


# -- the parser ------------------------------------------------------------------------------------------

TEXT = """<!-- a comment
with a ## heading in it -->
Intro.

## First section

Text {{value:dead.red}}.

### An ordinary heading

More.

{{table:scalars}}

### `vr_com`, `vz_com`

Entry text.

## Second section

```dot
digraph { a -> b }
```
"""


def test_the_parser_splits_sections_blocks_entries_and_the_diagram():
    parsed = documentation.parse(TEXT)
    assert [(b.kind, b.text) for b in parsed.intro] == [("markdown", "Intro.")]
    first, second = parsed.sections
    assert (first.title, first.anchor) == ("First section", "first-section")
    assert [(b.kind, b.name or b.names or b.text) for b in first.blocks] == [
        ("markdown", "Text {{value:dead.red}}."),
        ("markdown", "### An ordinary heading\n\nMore."),
        ("table", "scalars"),
        ("entry", ("vr_com", "vz_com")),
    ]
    entry = first.blocks[3]
    assert (entry.heading, entry.text) == ("`vr_com`, `vz_com`", "Entry text.")
    assert entry.anchor == "name-vr-com"
    assert [(b.kind, b.text) for b in second.blocks] == [
        ("diagram", "digraph { a -> b }")
    ]
    # The scalar table links each name to the entry that holds it.
    table = documentation.table("scalars", parsed)
    assert "[`vr_com`](#name-vr-com)" in table and "[`vz_com`](#name-vr-com)" in table
    assert "| `vr_max` |" in table


def test_only_a_heading_made_of_names_is_an_entry():
    assert documentation.entry_names("`vr_com`, `vz_com`") == ("vr_com", "vz_com")
    assert documentation.entry_names("The `level` and more") == ()
    assert documentation.entry_names("pixel_averages") == ()


def test_a_placeholder_is_filled_from_the_code_and_one_that_names_nothing_is_reported():
    from imaging_methods import dead_pixels

    window = method_fields.MethodFieldsParams().averages.window
    text, unknown = documentation.fill(
        "a {{value:dead.red}} b {{value:no.such}} c {{default:averages.window}} d {{default:nope}}"
    )
    assert text == (
        f"a {dead_pixels.RED:g} b {{{{value:no.such}}}} c {window} d {{{{default:nope}}}}"
    )
    assert unknown == ["value:no.such", "default:nope"]


def test_an_unknown_table_is_refused():
    with pytest.raises(KeyError):
        documentation.table("nothing")


# -- the text --------------------------------------------------------------------------------------------


def test_every_placeholder_and_table_in_the_text_resolves(doc):
    for kind in ("markdown", "entry", "diagram"):
        for block in blocks(doc, kind):
            text, unknown = documentation.fill(block.text)
            assert not unknown and "{{" not in text, unknown or text
    assert [block.name for block in blocks(doc, "table")] == ["settings", "scalars"]
    for block in blocks(doc, "table"):
        assert documentation.table(block.name, doc).startswith("| ")


def test_every_scalar_a_spec_writes_has_one_entry_and_every_entry_names_one(doc):
    named = [name for entry in blocks(doc, "entry") for name in entry.names]
    assert len(named) == len(set(named)), "a name has two entries"
    assert set(named) == set(documentation.scalars())


def test_the_sources_are_the_specs_that_write_scalars_with_the_names_they_write():
    """``documentation.SOURCES`` is held to the registry: a spec that gains a scalar fails here."""
    result = xr.Dataset(
        {
            "refx": 1,
            "refy": 0,
            "taud": 2e-5,
            "lam": 0.4,
            "vx": 400.0,
            "vy": -50.0,
            "dead": (("y", "x"), np.array([[True, False], [False, False]])),
            "red_ratio": (("y", "x"), np.full((2, 2), 200.0)),
        }
    )
    writers = {
        key for key, spec in registry.REGISTRY.items() if spec.scalars is not None
    }
    assert writers == set(documentation.SOURCES)
    # The two products' names are read off their SCALARS, which their own tests hold to what they write.
    for key in writers - {"method_fields", "blob_parameters"}:
        written = registry.get(key).scalars(result)
        names = {name if isinstance(name, str) else name[2] for name in written}
        assert names == set(documentation.SOURCES[key]), key


def test_every_setting_says_what_it_is_and_whose_results_a_change_replaces():
    settings = documentation.settings()
    for path, (_, products, help_text) in settings.items():
        assert help_text and products, path
        assert f"| `{path}` |" in documentation.table("settings")
    # A change to the 2DCA reaches all three products, and a product's own setting only itself.
    assert settings["averages.threshold"][1] == list(documentation.PRODUCTS)
    assert settings["tde.min_cc"][1] == ["method_fields"]
    assert settings["neighbour_step"][1] == ["blob_parameters"]


@pytest.mark.skipif(
    shutil.which("dot") is None, reason="Graphviz's dot is not installed here"
)
def test_the_diagram_is_valid_dot(doc):
    (diagram,) = blocks(doc, "diagram")
    done = subprocess.run(
        ["dot", "-Tsvg"],
        input=documentation.fill(diagram.text)[0],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert "<svg" in done.stdout
    assert "Warning" not in done.stderr and "Error" not in done.stderr, done.stderr


# -- the page --------------------------------------------------------------------------------------------


def run_page():
    app = AppTest.from_file(PAGE, default_timeout=120).run()
    assert not app.exception, app.exception
    return app


def test_the_page_draws_every_section_entry_and_the_diagram(doc):
    app = run_page()
    assert [t.value for t in app.title] == ["Documentation"]
    assert [h.value for h in app.header] == [s.title for s in doc.sections]
    assert [h.proto.anchor for h in app.header] == [s.anchor for s in doc.sections]
    entries = blocks(doc, "entry")
    assert [s.value for s in app.subheader] == [entry.heading for entry in entries]
    assert [s.proto.anchor for s in app.subheader] == [e.anchor for e in entries]
    prose = "\n".join(m.value for m in app.markdown)
    assert "{{" not in prose, "a placeholder was left in the page"
    assert not app.warning
    # Every section is in the contents, and every entry is linked from the scalar table.
    for anchor in [*(s.anchor for s in doc.sections), *(e.anchor for e in entries)]:
        assert f"](#{anchor})" in prose, anchor
    (chart,) = app.get("graphviz_chart")
    assert chart.proto.spec == documentation.fill(blocks(doc, "diagram")[0].text)[0]


def test_the_page_says_when_its_text_names_what_the_code_lacks(monkeypatch, tmp_path):
    text = tmp_path / "documentation.md"
    text.write_text(
        "## Only\n\nA value {{value:no.such.thing}}.\n\n{{table:nothing}}\n"
    )
    monkeypatch.setattr(documentation, "DOC_PATH", str(text))
    warnings = " ".join(w.value for w in run_page().warning)
    assert "value:no.such.thing" in warnings and "table:nothing" in warnings


def test_the_page_survives_a_missing_text(monkeypatch, tmp_path):
    monkeypatch.setattr(documentation, "DOC_PATH", str(tmp_path / "gone.md"))
    app = run_page()
    assert [e.value for e in app.error][0].startswith(
        "The text of this page could not be read"
    )


def test_the_python_api_tab_shows_the_example_and_its_output():
    app = run_page()
    assert [t.label for t in app.tabs] == ["Quantities", "Python API"]
    tab = app.tabs[1]
    code = [c.value for c in tab.code]
    assert any(
        "from fusion_ui import api" in c and "def figs_8_and_9" in c for c in code
    )
    assert any(type(node).__name__ == "Image" for node in tab.children.values())
