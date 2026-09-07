"""Shot Explorer -- entry point.

Run with ``streamlit run fusion_ui/app.py``. Streamlit discovers the numbered
files in ``fusion_ui/pages/`` and builds the navigation from them.
"""

import os

import streamlit as st

from fusion_ui import config, ui
from fusion_ui.core import db, rundays

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


def human_bytes(value):
    """Byte count as something a person reads. Duplicated from the browser page
    on purpose -- a page cannot import another page, and one shared formatter
    is not worth a module."""
    if not value:
        return ""
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024


def _entry(row, run_days):
    """One run day as markdown: the heading line, then its purpose."""
    held = f"{row.on_disk} on disk" if row.on_disk else "none on disk"
    counts = f"{row.shots} curated shot{'' if row.shots == 1 else 's'}, {held}"
    if row.diagnostics:
        counts += f" ({row.diagnostics})"

    entry = run_days.get(row.day)
    if entry is None:
        return (
            f"**{row.date} · {row.day}** — {counts}\n\n"
            "*No run-day entry yet.* Add a `## " + row.day + "` section to "
            "`fusion_ui/data/run_days.md`.\n"
        )
    return (
        f"**{row.date} · {row.day}** — {counts}\n\n"
        f"{entry.mp} · *{entry.title}*\n\n{entry.summary}\n"
    )


def overview_section():
    """What the collection is: every run day, and what it was run for."""
    st.subheader("Run days")

    table = ui.cached_run_day_table()
    if table.empty:
        st.info(
            "No run days to show: neither the discharge database nor the shot "
            "index has anything in it yet.",
            icon="ℹ️",
        )
        return

    days, curated, held, size = st.columns(4)
    days.metric("Run days", len(table))
    curated.metric("Curated shots", int(table["shots"].sum()))
    held.metric("Shots on disk", int(table["on_disk"].sum()))
    size.metric("Size on disk", human_bytes(int(table["bytes"].sum())) or "0 B")

    st.caption(
        "A shot number is `1YYMMDDnnn`, so the first seven digits are the run "
        "day. Miniproposal numbers and titles are quoted from the C-Mod run "
        "pages; the summaries under them are written by hand in "
        "`fusion_ui/data/run_days.md`."
    )

    st.dataframe(
        table.assign(size=table["bytes"].map(human_bytes)).drop(columns="bytes"),
        hide_index=True,
        use_container_width=True,
        height=min(36 * len(table) + 38, 420),
        column_config={
            "date": st.column_config.TextColumn("date"),
            "day": st.column_config.TextColumn("run day"),
            "shots": st.column_config.NumberColumn(
                "shots", format="%d", help="shots on this day in the discharge DB"
            ),
            "on_disk": st.column_config.NumberColumn(
                "on disk", format="%d", help="shots on this day with files indexed here"
            ),
            "diagnostics": st.column_config.TextColumn("diagnostics"),
            "mp": st.column_config.TextColumn("MP"),
            "title": st.column_config.TextColumn(
                "miniproposal", help="quoted verbatim from the C-Mod run page"
            ),
            "size": st.column_config.TextColumn("size"),
        },
    )

    run_days = rundays.load()
    with st.container(height=460, border=True):
        st.markdown("\n\n".join(_entry(row, run_days) for row in table.itertuples()))


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
- **Multi shot** — a scalar against shot number, f_GW, line-averaged density
  or I_p, coloured by confinement mode.

The discharge database is read-only here and stays hand-curated; shots with
files but no entry in it are listed anyway, flagged as missing metadata.
""")

    overview_section()
    health_section(ui.get_connection())


main()
