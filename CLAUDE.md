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

Phases 00 (skeleton, deployed), 01 (raw data browser) and 02 (registry,
parameter forms, result store) — **done**. Phase 03 (velocity and conditional
averaging) is **in progress**: the 2DCA chain has landed, along with
contour-tracking velocity and FWHM sizes. The remaining ports (Gaussian fit
sizes, 2DCA-TDE and TDE velocities, quiver and trajectory plots) are
`PlotSpec`s built the same way — see the contract below.

`core/registry.py`, `core/params_ui.py`, `core/store.py` and `core/seed.py` are
in, `pages/2_single_shot.py` is a thin dispatcher over the registry, and
`fusion_ui/plots/` holds six specs:

| module | spec | |
|---|---|---|
| `raw.py` | `raw_frames` | live: frames, click-a-pixel trace, mp4 export |
| `probe.py` | `probe_trace` | live: the ragged ASP/FSP trace |
| `spectra.py` | `taud_psd` | cached: the PSD duration-time fit |
| `two_dca.py` | `two_dca` | cached: the conditional average — **the base of the phase-03 chain** |
| `velocity_contour.py` | `velocity_contour` | cached, `requires="two_dca"`: contour-tracking velocity |
| `fwhm_sizes.py` | `fwhm_sizes` | cached, `requires="two_dca"`: FWHM blob size, `lr`/`lz` |

**Copy `velocity_contour.py` for a new phase-03 analysis.** Almost every blob
quantity the group reports is derived from the conditional average, not from
the raw frames, and 2DCA costs ~21 s on a real shot — so a derived spec
declares `requires="two_dca"` rather than running its own. Copying `spectra.py`
instead is right only for something computed straight off the raw frames.

The `density_scan/results.json` seed is imported with `fusion-ui
import-results`: 50 shots, 3880 pixels, 58 200 scalars under the plot key
`density_scan_import`. Values computed today agree with the seed on
`taud_psd`/`lambda_psd` exactly and on `number_events` exactly, and to 1–5% on
the contour quantities — the seed is from 1 June 2026 and predates upstream's
non-uniform-grid fix in `contours.py`. `theta` changed convention outright.
A **large** disagreement is a finding; these are drift.

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
Currently at **v2**, which added `runs.preprocessed`: the raw and preprocessed
files are different data and give different answers, so they must not share a
ledger row.

`presets` is the only table nothing writes yet. Never write `runs`, `scalars`
or `param_sets` with raw SQL — go through `core/store.py`, which is what keeps
the hash, the blob path and the ledger consistent.

`runs.code_version` is stored but deliberately **not** part of any cache key:
hashing it would invalidate every result on every commit. It is shown under the
figure with a Recompute button, and the person looking at the plot decides
whether the version matters.

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

## The `PlotSpec` contract

**Every analysis is a `PlotSpec`, registered in `fusion_ui/plots/`.** No plot is
ever wired into a page directly; a page picks a spec out of the registry and
knows nothing else about it. Adding one is a new module plus its import in
`fusion_ui/plots/__init__.py` — see the recipe in `README.md`.

```python
@dataclass(frozen=True)
class PlotSpec:
    key:         str        # permanent: it is part of the cache key
    label:       str
    diagnostics: tuple      # ("apd", "phantom") -- strings, see below
    params:      type       # a dataclass -> the widget panel and the hash
    render:      Callable   # (result, params, target) -> go.Figure | None
    compute:     Callable | None = None   # (ds, params) -> xr.Dataset
    scalars:     Callable | None = None   # (result) -> dict
    choices:     Callable | None = None   # (ds, field_path, chosen) -> tuple | None
    requires:    str | None = None        # plot key of an upstream spec
    upstream_params: Callable | None = None   # (params) -> the upstream's params
    description: str = ""
```

- **`compute is None` means live.** The time-sliced dataset *is* the result:
  no `runs` row, no blob, widgets take effect on the next rerun. That is what
  the frame viewer and the probe trace are. With a `compute`, the result is
  written to netCDF under `CACHE_DIR`, recorded in `runs`, and the parameter
  form goes behind a **Compute** button so a nudged slider cannot start a
  four-minute analysis.
- **`render` may draw.** Return a `go.Figure` and the page draws it with the
  shared chrome; draw into Streamlit yourself and return `None` when the view
  needs a slider, a click target or several figures. Prefer returning a figure.
- **`scalars` keys are `str` (shot-level, stored at the `x = y = -1` sentinel)
  or `(x, y, name)` tuples** for a value belonging to one pixel. Write at the
  pixel whenever the parameters name one — it is how the seeded `density_scan`
  rows are laid out, so the two line up on one axis.
- **`compute`, `render` and `scalars` never touch Streamlit, the database or
  the filesystem.** `core/store.py` does all of that. Keeping them pure is what
  makes them testable and what will let phase 05 move compute into a process
  pool without touching a single spec.
- **Diagnostics are strings** — `"apd"`, `"asp"` — matching `catalog.DIAGNOSTICS`,
  `shots.diagnostic` and `loader.dataset_path`. `experimental_database`'s
  `Diagnostic` enum stays an implementation detail inside `core/loader.py`.
  (`docs/PLAN.md` sketched `tuple[Diagnostic, ...]`; this supersedes it.)
- **`choices` is for options only the opened file knows** — a probe's quantity
  list, a valid pixel index on this array. It is consulted before the static
  `params_ui.CHOICES` table and receives the values chosen so far, so
  selectboxes chain.
- **`requires` chains one spec onto another.** Set it to the upstream plot key
  and set `upstream_params` to the function that lifts the upstream's
  parameters out of this spec's own; `compute` is then called as
  `compute(ds, params, upstream)` with the upstream's cached result in hand.
  The store resolves the chain depth-first and **only on a miss**, so a cache
  hit on the derived quantity never pays for its upstream. Each link keeps its
  own `runs` row and its own blob.

  Three rules follow, and `register()` enforces the first two:

  - the upstream must already be registered — mind the import order in
    `plots/__init__.py` — and must accept every diagnostic the downstream does;
  - `upstream_params` must **read out of the downstream parameters** (as
    `velocity_contour` does with its `two_dca` field), never construct fresh
    defaults. That is what keeps the two cache keys in step: changing the 2DCA
    threshold has to give both a new entry;
  - an upstream that fails is reported on the downstream run, quoting the
    upstream's own error. The person is looking at the derived plot.

### Parameters, and what must never be one

`params` is a plain dataclass, ideally composing `imaging_methods` classes
directly (`TaudEstimationParams`, `TwoDcaParams`) so the form and the analysis
cannot disagree about a knob's name. One walk in `core/params_ui.py` produces
the widget panel, the canonical dict and the sha1 that keys `param_sets`.

- **View state is not a parameter.** A frame index, a selected pixel, a zoom
  live in `st.session_state` keyed off `Target.key`. A slider drag must never
  mint a `param_sets` row.
- **The hash includes the plot key.** `param_sets.hash` is the primary key while
  `plot` is an ordinary column, so hashing parameters alone would let two plots
  sharing a default parameter set collide on it.
- **Every leaf is coerced by its annotation, and anything unrecognised raises.**
  The upstream defaults do not match their own types — `size_penalty: float = 5`
  is an int, `size_max: float = None` is really optional, `radius: int` is
  checked with a strict `isinstance` downstream — and a cache key that quietly
  disagrees with itself is the failure mode that makes the tool untrustworthy.
  New leaf types get a rule in `params_ui._leaf`, never a silent pass-through.
- **A dataclass with no fields raises.** `velocity_estimation`'s
  `EstimationOptions` and friends carry `@dataclass` but declare their own
  `__init__`, so `fields()` is empty and `asdict()` is `{}` — an empty,
  colliding hash with no error at all.
- `params_ui.CHOICES`, `OPTIONAL` and `HELP` hold what upstream does not carry:
  the allowed values for its string fields and help text for the seven leaves
  whose docstrings say nothing. Add there rather than patching a dependency.

## Analysis conventions

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
