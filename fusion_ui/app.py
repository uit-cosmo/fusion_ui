"""Shot Explorer -- entry point.

Run with ``streamlit run fusion_ui/app.py``. Streamlit discovers the numbered
files in ``fusion_ui/pages/`` and builds the navigation from them.
"""

import os

import streamlit as st

from fusion_ui import config, ui
from fusion_ui.core import db

st.set_page_config(page_title="Shot Explorer", page_icon="🔥", layout="wide")

_PATHS = [
    ("Discharge DB (read-only)", "DISCHARGE_DB_PATH", "FUSION_DISCHARGE_DB", True),
    ("Data folder", "DATA_FOLDER", "FUSION_DATA_FOLDER", True),
    ("App database", "UI_DB_PATH", "FUSION_UI_DB", False),
    ("Result cache", "CACHE_DIR", "FUSION_UI_CACHE", False),
]


def health_section(conn):
    """The strip that makes a broken deployment diagnosable from the browser."""
    st.subheader("Configuration")

    rows, problems = [], []
    for label, attribute, variable, must_exist in _PATHS:
        value, error = ui.resolve(attribute)
        if error:
            problems.append(f"`{variable}` is not set. Copy `.env.example` to `.env`.")
            rows.append({"": label, "path": f"({variable} unset)", "on disk": "✗"})
            continue
        exists = os.path.exists(value)
        if must_exist and not exists:
            problems.append(f"`{variable}` points at `{value}`, which does not exist.")
        rows.append({"": label, "path": value, "on disk": "✓" if exists else "✗"})

    st.dataframe(rows, hide_index=True, use_container_width=True)
    for problem in problems:
        st.error(problem, icon="⚠️")

    counts = conn.execute(
        "SELECT diagnostic, COUNT(*) AS n FROM shots GROUP BY diagnostic"
    ).fetchall()
    files, shots = st.columns(2)
    files.metric("Indexed files", sum(row["n"] for row in counts))
    shots.metric(
        "Shots", conn.execute("SELECT COUNT(DISTINCT shot) FROM shots").fetchone()[0]
    )

    if counts:
        st.caption(
            "Index: " + ", ".join(f"{row['diagnostic']} {row['n']}" for row in counts)
        )
    else:
        st.info(
            "The shot index is empty. Run `fusion-ui rescan`, or use the Rescan "
            "button on the Shot browser page.",
            icon="ℹ️",
        )

    versions = " · ".join(f"{n} `{v}`" for n, v in ui.code_version().items())
    st.caption(
        f"machine `{config.MACHINE}` · schema v{db.schema_version(conn)} · {versions}"
    )


def main():
    # The selection contract every later page reads. Set once, here, so phases
    # 01-04 do not each invent their own shape for it.
    st.session_state.setdefault("selection", None)
    st.session_state.setdefault("shot_selection", [])

    st.title("🔥 Shot Explorer")
    st.markdown("""
Point-and-click access to the group's fusion diagnostic data.

- **Shot browser** — every shot on disk, joined to the discharge metadata.
- **Single shot** — pick a plot and its parameters; frames and pixel traces,
  probe traces, duration times, movies.
- **Multi shot** — a scalar against shot number, f_GW or line-averaged density. *(phase 04)*

The discharge database is read-only here and stays hand-curated; shots with
files but no entry in it are listed anyway, flagged as missing metadata.
""")

    health_section(ui.get_connection())


main()
