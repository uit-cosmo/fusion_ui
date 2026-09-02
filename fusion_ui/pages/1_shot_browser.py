"""Shot browser -- every shot on disk, joined to the discharge metadata.

Reads the ``shots`` table only; it never walks the data tree. Shots whose files
are on disk but which nobody has curated yet are listed like any other, flagged
in the ``meta`` column, so a newly copied shot is visible the moment it lands.
"""

import math

import pandas as pd
import streamlit as st

from fusion_ui import config, ui
from fusion_ui.core import catalog

st.set_page_config(page_title="Shot browser · Shot Explorer", layout="wide")

_RANGE_FILTERS = [
    ("f_GW", "Greenwald fraction"),
    ("I_p", "Plasma current [MA]"),
    ("n_e_bar", "Line-averaged density [10²⁰ m⁻³]"),
]

# Only a quarter of the curated shots carry every quantity, so "no value" has to
# be a first-class option on each filter rather than a silent exclusion.
_MISSING_LABEL = {
    "f_GW": "…keep shots with no f_GW",
    "I_p": "…keep shots with no I_p",
    "n_e_bar": "…keep shots with no density",
}


def human_bytes(value):
    if not value:
        return ""
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024


def rescan_now():
    """Run a rescan from the UI and drop the cached table so it shows up."""
    data_folder, error = ui.resolve("DATA_FOLDER")
    if error:
        st.error(error, icon="⚠️")
        return
    discharge_db, db_error = ui.resolve("DISCHARGE_DB_PATH")
    stats = catalog.rescan(
        ui.get_connection(),
        data_folder,
        config.MACHINE,
        discharge_db_path=None if db_error else discharge_db,
    )
    st.cache_data.clear()
    st.toast(stats.summary(), icon="✅")


def range_filter(table, column, label):
    """A slider over the finite values, plus what to do with the missing ones.

    Most metadata columns are only partly filled in, so silently dropping the
    NaNs would hide most of the collection behind a filter nobody touched.
    """
    values = table[column].dropna()
    values = values[values.apply(math.isfinite)]
    if values.empty:
        return pd.Series(True, index=table.index)

    low, high = float(values.min()), float(values.max())
    if math.isclose(low, high):
        return pd.Series(True, index=table.index)

    chosen = st.sidebar.slider(label, low, high, (low, high))
    keep_missing = st.sidebar.checkbox(
        _MISSING_LABEL[column], value=True, key=f"keep_missing_{column}"
    )
    in_range = table[column].between(*chosen)
    return in_range | (table[column].isna() & keep_missing)


def sidebar_filters(table):
    st.sidebar.header("Filters")

    machines = sorted(table["machine"].unique())
    if len(machines) > 1:
        chosen = st.sidebar.multiselect("Machine", machines, default=machines)
        table = table[table["machine"].isin(chosen)]

    diagnostics = st.sidebar.multiselect(
        "Has diagnostic", list(catalog.DIAGNOSTICS), default=[]
    )
    if diagnostics:
        available = pd.concat([table[d] != "" for d in diagnostics], axis=1).any(axis=1)
        table = table[available]
        if st.sidebar.checkbox("…preprocessed only", value=False):
            preprocessed = pd.concat(
                [table[d].str.contains("P") for d in diagnostics], axis=1
            ).any(axis=1)
            table = table[preprocessed]

    metadata = st.sidebar.radio(
        "Discharge metadata", ["Any", "Curated only", "Missing only"], horizontal=True
    )
    if metadata == "Curated only":
        table = table[table["has_metadata"]]
    elif metadata == "Missing only":
        table = table[~table["has_metadata"]]

    modes = sorted(m for m in table["mode"].unique() if m)
    if modes:
        chosen = st.sidebar.multiselect("Confinement mode", modes, default=[])
        if chosen:
            table = table[table["mode"].isin(chosen)]

    for column, label in _RANGE_FILTERS:
        table = table[range_filter(table, column, label)]

    search = st.sidebar.text_input("Shot number contains", "")
    if search.strip():
        table = table[table["shot"].astype(str).str.contains(search.strip())]

    return table


def default_target(row):
    """The (diagnostic, preprocessed) a click on this shot should land on.

    Preprocessed wins where it exists -- it is what every analysis actually
    consumes -- and diagnostics are tried in the enum's order.
    """
    for diagnostic in catalog.DIAGNOSTICS:
        available = row[diagnostic]
        if available:
            return diagnostic, "P" in available
    return None, False


def main():
    st.session_state.setdefault("selection", None)
    st.session_state.setdefault("shot_selection", [])

    st.title("Shot browser")

    header, button = st.columns([5, 1])
    header.caption(
        "The index is filled by `fusion-ui rescan` (on cron on the server). "
        "Rescan here after copying new files in."
    )
    if button.button("Rescan", use_container_width=True):
        rescan_now()

    table = ui.cached_shot_table()
    if table.empty:
        st.warning(
            "The shot index is empty — run a rescan, or check that "
            "`FUSION_DATA_FOLDER` points at the data tree.",
            icon="⚠️",
        )
        return

    filtered = sidebar_filters(table)
    st.caption(
        f"{len(filtered)} of {len(table)} shots · "
        f"{int((~filtered['has_metadata']).sum())} without discharge metadata"
    )

    shown = filtered.assign(
        meta=filtered["has_metadata"].map({True: "✓", False: "⚠"}),
        size=filtered["bytes"].map(human_bytes),
        window=[
            "" if not math.isfinite(a) or not math.isfinite(b) else f"{a:.2f}–{b:.2f}"
            for a, b in zip(filtered["t_start"], filtered["t_end"])
        ],
    )
    columns = [
        "shot",
        *catalog.DIAGNOSTICS,
        "I_p",
        "n_e_bar",
        "f_GW",
        "f_GW_source",
        "mode",
        "window",
        "mlp_mode",
        "meta",
        "size",
    ]
    if filtered["machine"].nunique() > 1:
        columns.insert(0, "machine")

    # Missing measurements show as a greyed "None": st.dataframe renders NaN
    # that way and neither a Styler's na_rep nor column_config overrides it.
    # Left as numeric columns anyway -- sorting on f_GW and I_p is the point of
    # this table, and a formatted string column sorts lexicographically.
    event = st.dataframe(
        shown[columns],
        hide_index=True,
        use_container_width=True,
        height=min(38 * len(shown) + 38, 640),
        on_select="rerun",
        selection_mode="multi-row",
        key="shot_rows",
        column_config={
            "shot": st.column_config.NumberColumn("shot", format="%d"),
            "I_p": st.column_config.NumberColumn("I_p [MA]", format="%.2f"),
            "n_e_bar": st.column_config.NumberColumn(
                "n_e [10²⁰ m⁻³]", format="%.2f", help="line-averaged density"
            ),
            "f_GW": st.column_config.NumberColumn("f_GW", format="%.2f"),
            "f_GW_source": st.column_config.TextColumn(
                "f_GW from",
                help="db = curated value; derived = n_e · π · 0.22² / I_p",
            ),
            "mode": st.column_config.TextColumn("mode", help="discharge DB comment"),
            "window": st.column_config.TextColumn("window [s]", help="t_start–t_end"),
            "meta": st.column_config.TextColumn(
                "meta", help="⚠ = files on disk, no discharge DB entry"
            ),
        },
    )

    picked = filtered.iloc[event.selection["rows"]] if event.selection["rows"] else None
    selection_row, use_column = st.columns([3, 2])

    if picked is not None and len(picked) == 1:
        row = picked.iloc[0]
        diagnostic, preprocessed = default_target(row)
        st.session_state["selection"] = {
            "machine": row["machine"],
            "shot": int(row["shot"]),
            "diagnostic": diagnostic,
            "preprocessed": preprocessed,
        }
        selection_row.success(
            f"Selected shot **{int(row['shot'])}** "
            f"({diagnostic}{', preprocessed' if preprocessed else ''}) — "
            "the single-shot view lands in phase 01.",
            icon="🎯",
        )
    elif picked is not None:
        selection_row.info(f"{len(picked)} shots selected.", icon="🎯")

    cohort = (picked if picked is not None else filtered)["shot"].astype(int).tolist()
    if use_column.button(
        f"Use these {len(cohort)} shots for multi-shot", use_container_width=True
    ):
        st.session_state["shot_selection"] = cohort
        st.toast(f"{len(cohort)} shots carried over.", icon="📈")

    if st.session_state["shot_selection"]:
        st.caption(
            f"Carried over for the multi-shot view: "
            f"{len(st.session_state['shot_selection'])} shots."
        )


main()
