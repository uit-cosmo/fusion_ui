# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

**Shot Explorer** — a Streamlit web app that gives the research group
point-and-click access to fusion diagnostic data (Alcator C-Mod GPI imaging and
Langmuir probes) and to the analyses in `imaging_methods`, `fusion_scripts` and
`velocity_estimation`. It runs as a hosted service on the group server, next to
the data.

**Read `docs/PLAN.md` before starting work.** It is the source of truth for the
architecture, the database schema, the phase order, and what has been decided.
This file holds only the conventions that apply while writing code.

## Current state

Phase 00 (skeleton, deployed) and Phase 01 (raw data browser) — **done**.
Config, schema, `rescan`, the shot browser, the deploy files, and
`core/loader.py` / `core/decimate.py` / `core/probes.py` /
`pages/2_single_shot.py` are in: frame view with a time slider and
jump-to-time, click-a-pixel decimated time series, mp4 movie export, and the
ASP/FSP (quantity, position) probe trace view. Written directly against
Streamlit and Plotly, ahead of the registry on purpose. Phase 02 (registry,
parameter forms, result store) is next. Nothing in `fusion_ui/plots/` exists
yet.

## Setup

```bash
python -m venv .venv
# not on PyPI — install each editable from its local checkout first
.venv/bin/pip install -e ../imaging-methods -e ../experimental_database \
  -e ../fusion_scripts -e ../velocity-estimation -e ../fpp-analysis-tools
.venv/bin/pip install -e ".[dev]"
cp .env.example .env        # then edit for this machine
.venv/bin/fusion-ui init-db && .venv/bin/fusion-ui rescan
.venv/bin/streamlit run fusion_ui/app.py
```

On the dev machine the checkouts are under `/home/sosno/Git` and the data is
`~/Data/alcator` (45 APD shots, raw + preprocessed, and one ASP shot).

## Configuration

**No hardcoded paths.** Everything machine-specific goes through
`fusion_ui/config.py`, which reads `.env` with shell environment variables
taking precedence — the same pattern, and the same first two variables, as
`fusion_scripts/config.py`. Attributes resolve lazily, so importing the module
on an unconfigured machine works and only *using* a path raises.

| `config.` | env var | |
|---|---|---|
| `DISCHARGE_DB_PATH` | `FUSION_DISCHARGE_DB` | `plasma_discharges.json`, **read-only** |
| `DATA_FOLDER` | `FUSION_DATA_FOLDER` | one subfolder per diagnostic |
| `UI_DB_PATH` | `FUSION_UI_DB` | this app's SQLite file |
| `CACHE_DIR` | `FUSION_UI_CACHE` | netCDF result blobs (phase 02 on) |
| `MACHINE` | `FUSION_MACHINE` | defaults to `cmod` |

## The two databases

**The discharge DB is read-only.** `experimental_database` and its
`plasma_discharges.json` are consumed, never written. It is small (about 100
curated shots, with a few exact duplicate entries) and is read straight from
JSON through `PlasmaDischargeManager` — never mirrored into SQLite, where it
would go stale. Confinement mode is the free-text `comment` field; do not
normalise it. f_GW is missing for most shots and is derived from I_p and n̄_e
where possible — `catalog.greenwald_fraction` returns `(value, source)` and the
source is always shown next to the value.

**Everything this app generates lives in one SQLite file**, schema in
`fusion_ui/core/db.py` (`param_sets`, `runs`, `scalars`, `presets`, `shots`) and
documented in `docs/PLAN.md`. Open it with `db.open_db()`; never with bare
`sqlite3.connect`. Schema changes append a function to `db.MIGRATIONS` — the
list index is the version it produces — and never edit `db.SCHEMA` in place.

Only `shots` is written today. `runs.code_version` is stored but deliberately
**not** part of any cache key.

## The catalog

`core/catalog.rescan()` is the only thing that walks the data tree, and it
**stats files, it never opens them** — a single APD file is ~425 MB and the tree
is tens of terabytes. Pages read the `shots` table, never the filesystem. A file
that disappears loses its row; a shot with files but no discharge-DB entry is
indexed with `has_metadata = 0` and shown flagged, not hidden.

`catalog.shot_table(conn, discharge_db_path)` is the join: one row per
`(machine, shot)`, diagnostics collapsed into a `"R+P" | "R" | "P" | ""` column
each. `catalog.index_fingerprint(conn)` is the `st.cache_data` key that makes
the Rescan button take effect.

## Streamlit conventions

- Shared helpers live in `fusion_ui/ui.py`. A page cannot import `app.py` —
  Streamlit executes it as a script, so importing it would run the landing page.
- **Get connections from `ui.get_connection()`**, which is thread-local. A
  connection cached with `st.cache_resource` breaks as soon as a second browser
  tab connects: Streamlit runs each session on a pooled thread and a sqlite3
  connection may only be used on the thread that created it.
- Selection contract, set in `app.py` and read by later pages:
  `st.session_state["selection"]` is
  `{"machine", "shot", "diagnostic", "preprocessed"}` or `None`;
  `st.session_state["shot_selection"]` is a `list[int]` of shot numbers.
- `st.dataframe` renders a missing numeric as a greyed `"None"`; neither a
  Styler's `na_rep` nor `column_config` overrides it. Keep numeric columns
  numeric anyway — sorting is what the table is for.

## Analysis conventions

- **Every analysis is a `PlotSpec`.** Once the registry lands in phase 02, no
  plot gets wired into a page directly. The contract will be documented here.
- **Never load a full time axis.** APD files are ~500 MB and 583k samples;
  always slice to the discharge DB's `t_start..t_end` (or a centred 0.2 s window
  when there is no metadata).
- **Decimate before handing a 1D trace to Plotly.** Use the shared min/max
  envelope helper in `core/decimate.py` — striding drops spikes.
- **Interactive plots are Plotly.** Movies stay matplotlib, rendered to mp4 and
  served with `st.video`.
- **Development uses one small shot.** Do not loop over the data tree to "check
  something" — it is tens of terabytes and growing.

## Testing

```bash
pytest                 # all tests
pytest tests/ -v
```

Tests are hermetic: an autouse fixture clears the `FUSION_*` variables so a
developer's `.env` cannot point the suite at the real data, and the fixtures
build a tmp tree of empty `.nc` files. Streamlit pages are smoke-tested with
`streamlit.testing.v1.AppTest`.

## Deployment

systemd unit, nginx site and the exact sudo sequence are in `deploy/`. The
shared password lives in nginx, not in the app. The one check that matters is
the websocket handshake returning `101` — a proxy that drops the `Upgrade`
header serves a page that loads and then hangs forever with no error.
