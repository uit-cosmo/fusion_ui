"""Documentation: what every quantity means, how it is computed, and how the quantities depend on one another.

The text is ``fusion_ui/data/documentation.md``, read at every rerun, so an edit shows on the next one with
no restart. :mod:`fusion_ui.documentation` parses it and supplies what it quotes from the code; this page
only draws.

A second tab, **Python API**, is how to read the stored results from a script (``fusion_ui.api``): the
hand-written ``fusion_ui/data/python_api.md``, then the example ``examples/garcia_velocity_2026_figs8_9.py``
and the figure it saved when last run on the group server (``fusion_ui/data/examples/``), not computed here.
"""

from pathlib import Path

import streamlit as st

import fusion_ui.plots  # noqa: F401 - importing the package registers every spec the text describes
from fusion_ui import documentation

st.set_page_config(page_title="Documentation · Shot Explorer", layout="wide")

DATA = Path(__file__).resolve().parent.parent / "data"
API_TEXT = DATA / "python_api.md"
EXAMPLE = (
    Path(__file__).resolve().parents[2] / "examples" / "garcia_velocity_2026_figs8_9.py"
)
EXAMPLE_FIGURE = DATA / "examples" / "garcia_velocity_2026_figs8_9.png"


def show_documentation():
    try:
        doc = documentation.load()
    except OSError as error:
        st.error(f"The text of this page could not be read: {error}", icon="⚠️")
        return
    values, settings = documentation.facts(), documentation.defaults()
    unknown = []

    def filled(text):
        text, missing = documentation.fill(text, values, settings)
        unknown.extend(missing)
        return text

    def show(block):
        if block.kind == "entry":
            st.subheader(block.heading, anchor=block.anchor)
        if block.kind == "diagram":
            st.graphviz_chart(filled(block.text), width="stretch")
        elif block.kind == "table":
            try:
                st.markdown(documentation.table(block.name, doc))
            except KeyError:
                unknown.append(f"table:{block.name}")
        elif block.text:
            st.markdown(filled(block.text))

    for block in doc.intro:
        show(block)
    st.markdown(
        "**Contents** · "
        + " · ".join(f"[{s.title}](#{s.anchor})" for s in doc.sections)
    )
    for section in doc.sections:
        st.header(section.title, anchor=section.anchor)
        for block in section.blocks:
            show(block)
    if unknown:
        st.warning(
            "The text names what the code does not have: "
            + ", ".join(f"`{name}`" for name in dict.fromkeys(unknown))
            + ". Correct it in fusion_ui/data/documentation.md.",
            icon="⚠️",
        )


def show_api():
    try:
        st.markdown(API_TEXT.read_text(encoding="utf-8"))
    except OSError as error:
        st.error(f"The text of this tab could not be read: {error}", icon="⚠️")
    st.markdown(
        "### Example: Figs 8 and 9 of Garcia, Losada et al., PPCF 68 (2026) 055034"
    )
    st.markdown(
        "The script reads the paper's velocities (`api.velocity_tde`) and blob sizes (`api.blob_parameters`)"
        " of its five shots and redraws both figures."
        f" It is `examples/{EXAMPLE.name}` in the checkout; copy it and change what you need."
    )
    try:
        st.code(EXAMPLE.read_text(encoding="utf-8"), language="python")
    except OSError:
        st.info(f"The script is not in this checkout (examples/{EXAMPLE.name}).")
    st.markdown("### Its output")
    st.caption(
        "As the script saved it when last run on the group server, not computed here."
    )
    if EXAMPLE_FIGURE.exists():
        st.image(str(EXAMPLE_FIGURE))


st.title("Documentation")
quantities, python_api = st.tabs(["Quantities", "Python API"])
with quantities:
    show_documentation()
with python_api:
    show_api()
