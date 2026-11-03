"""Documentation: what every quantity means, how it is computed, and how the quantities depend on one another.

The text is ``fusion_ui/data/documentation.md``, read at every rerun, so a reworded file shows on the next
one with no restart. The tables, the diagram and every number the text quotes are generated from the code by
:mod:`fusion_ui.documentation`; this page draws and decides nothing.

It reads one thing that changes under it, at most once a minute: the scalar names in the ledger, so that a
name the text does not describe is named on the page rather than missed.

A second tab, **Python API**, is how to read the stored results from a script (``fusion_ui.api``): the
hand-written ``fusion_ui/data/python_api.md``, then the example ``examples/garcia_velocity_2026_figs8_9.py``
and its output as it was last run on the group server (``fusion_ui/data/examples/``), not computed here.
"""

from pathlib import Path

import streamlit as st

import fusion_ui.plots  # noqa: F401 - importing the package registers every spec the text describes
from fusion_ui import config, documentation, ui

st.set_page_config(page_title="Documentation · Shot Explorer", layout="wide")


@st.cache_data(ttl=60, show_spinner=False)
def _ledger_names(database):
    """The scalar names in the ledger at ``database``; ``None`` when it cannot be read."""
    try:
        rows = (
            ui.get_connection().execute("SELECT DISTINCT name FROM scalars").fetchall()
        )
    except Exception:  # noqa: BLE001 - a page of text must not fail on the ledger
        return None
    return sorted(row[0] for row in rows)


def ledger_names():
    try:
        database = str(config.UI_DB_PATH)
    except RuntimeError:  # an unconfigured machine has no ledger to read
        return None
    return _ledger_names(database)


def show_ledger():
    """Whether every scalar name the ledger holds has an entry on this page."""
    names = ledger_names()
    if names is None:
        return
    if not names:
        st.caption("The ledger of this deployment holds no scalars yet.")
        return
    missing = documentation.undocumented(names)
    if missing:
        st.warning(
            f"The ledger holds {len(missing)} scalar name{'' if len(missing) == 1 else 's'} this page "
            "does not describe: " + ", ".join(f"`{name}`" for name in missing) + ".",
            icon="⚠️",
        )
    else:
        st.caption(
            f"Each of the {len(names)} scalar names in this deployment's ledger has an entry below."
        )


def show_text(text, values, settings, unknown):
    filled, missing = documentation.fill(text, values, settings)
    unknown.extend(missing)
    st.markdown(filled)


def show_block(block, doc, values, settings, unknown):
    if block.kind == "markdown":
        show_text(block.text, values, settings, unknown)
    elif block.kind == "entry":
        st.subheader(block.heading, anchor=block.anchor)
        st.markdown(documentation.entry_lines(block.names))
        if block.text:
            show_text(block.text, values, settings, unknown)
    elif block.kind == "table":
        try:
            st.markdown(documentation.table(block.name, doc))
        except KeyError as error:
            unknown.append(f"table:{block.name}")
            st.warning(str(error), icon="⚠️")
            return
        if block.name == "scalars":
            show_ledger()
    elif block.kind == "diagram":
        st.graphviz_chart(documentation.diagram(), width="stretch")
    elif block.kind == "checked":
        st.caption(documentation.checked_text())


DATA = Path(__file__).resolve().parent.parent / "data"
API_TEXT = DATA / "python_api.md"
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "garcia_velocity_2026_figs8_9.py"
#: The example's output, as it was last run on the group server: what it printed, and the figure it saved.
EXAMPLE_TABLE = DATA / "examples" / "garcia_velocity_2026_figs8_9.txt"
EXAMPLE_FIGURE = DATA / "examples" / "garcia_velocity_2026_figs8_9.png"


def show_api():
    try:
        st.markdown(API_TEXT.read_text(encoding="utf-8"))
    except OSError as error:
        st.error(f"The text of this tab could not be read: {error}", icon="⚠️")
    st.markdown("### Example: Figs 8 and 9 of Garcia, Losada et al., PPCF 68 (2026) 055034")
    st.markdown(
        "The script reads the paper's velocities (`api.velocity_tde`) and blob sizes (`api.blob_parameters`)"
        " of its five shots and redraws both figures, with the paper's own points over them as black outlines."
        f" It is `examples/{EXAMPLE.name}` in the checkout; copy it and change what you need."
    )
    try:
        st.code(EXAMPLE.read_text(encoding="utf-8"), language="python")
    except OSError:
        st.info(f"The script is not in this checkout (examples/{EXAMPLE.name}).")
    st.markdown("### Its output")
    st.caption(
        "As the script printed and saved it when last run on the group server, not computed here."
        " The paper's values are in parentheses."
    )
    if EXAMPLE_TABLE.exists():
        st.code(EXAMPLE_TABLE.read_text(encoding="utf-8"), language=None)
    if EXAMPLE_FIGURE.exists():
        st.image(str(EXAMPLE_FIGURE))


def show_documentation():
    try:
        doc = documentation.load()
    except OSError as error:
        st.error(f"The text of this page could not be read: {error}", icon="⚠️")
        return

    values, settings, unknown = documentation.facts(), documentation.defaults(), []
    for block in doc.intro:
        show_block(block, doc, values, settings, unknown)
    st.markdown(
        "**Contents** · "
        + " · ".join(
            f"[{section.title}](#{section.anchor})" for section in doc.sections
        )
    )
    for section in doc.sections:
        st.header(section.title, anchor=section.anchor)
        for block in section.blocks:
            show_block(block, doc, values, settings, unknown)
    if unknown:
        st.warning(
            "The text names what the code does not have: "
            + ", ".join(f"`{name}`" for name in dict.fromkeys(unknown))
            + ". Correct it in fusion_ui/data/documentation.md.",
            icon="⚠️",
        )


def main():
    st.title("Documentation")
    quantities, python_api = st.tabs(["Quantities", "Python API"])
    with quantities:
        show_documentation()
    with python_api:
        show_api()


main()
