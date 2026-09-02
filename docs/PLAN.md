# Shot Explorer — build plan

A browser UI over the group's diagnostic data: pick a shot and a plot, get the
plot; pick a scalar and an x-axis, get it across every shot. Built on the
analysis code that already exists (`imaging_methods`, `fusion_scripts`,
`velocity_estimation`), not beside it.

This file is the source of truth for the build. A session implementing a phase
should read it first and need nothing else.

Shareable version of this document (for the group):
<https://claude.ai/code/artifact/2f20887b-57f8-433c-a72a-75e0cbc5021e>

## Decisions locked

| | |
|---|---|
| Deployment | Hosted service on the group server |
| Framework | Streamlit |
| Compute | On demand, disk-cached |
| Plots | Interactive (Plotly) for new plots; matplotlib for movies |
| Server access | sudo — a real systemd service behind nginx |
| Auth | One shared password, enforced in nginx |
| Data location | On the server; the collection grows continuously |
| Disk | ~30 TB free — not a constraint |
| Machine column | In the schema from day one |
| Discharge DB | Read-only to the UI; stays hand-curated |
| App database | One SQLite file: params, runs, scalars, presets, shot index |
| v1 scope | Raw browser, velocity, conditional average, multi-shot scatter |

## The one idea the design rests on

Every analysis registers once and yields two things: **a figure** for the
single-shot view and **a dict of scalars** for the multi-shot view. There is no
separate multi-shot pipeline to keep in sync — plotting a velocity field for
shot 1160616027 is the same call that deposits its `vx_c` into the store that
the f_GW scatter reads.

```
catalog ──▶ loader ──▶ compute()  ──┬──▶ render() ──▶ figure     (single-shot view)
 (index)    (lazy,      (cached by  │
            sliced)     params_hash)└──▶ scalars() ──▶ SQLite    (multi-shot view)
```

Adding a new plot is one file and one registration — no UI code, no new
storage, and it appears in both views at once.

```python
@dataclass(frozen=True)
class PlotSpec:
    key:         str                     # "velocity_2dca"
    label:       str                     # "Velocity field (2DCA)"
    diagnostics: tuple[Diagnostic, ...]  # which data it accepts
    params:      type                    # a dataclass -> the widget panel
    compute:     Callable[[xr.Dataset, P], xr.Dataset]
    render:      Callable[[xr.Dataset, P], go.Figure]
    scalars:     Callable[[xr.Dataset], dict[str, float]] | None

REGISTRY: dict[str, PlotSpec] = {}
def register(spec): REGISTRY[spec.key] = spec
```

The `params` field is why this is cheap: `imaging_methods.MethodParameters` is
already a nested tree of dataclasses with defaults and docstrings. One
introspection helper walks it and emits a Streamlit sidebar — `int` becomes a
number input, `bool` a checkbox, a `str` field with known choices a selectbox.
The same walk produces the canonical dict that gets hashed into the cache key,
so the form and the key can never disagree.

## The app's own database

The discharge DB stays hand-curated and read-only. Everything the UI generates
lives in one SQLite file it owns outright — parameter sets, run provenance,
scalars, saved presets, and the shot index. One file to back up, one file to
delete for a clean slate.

```sql
-- what a params_hash actually means. Written once per distinct parameter set.
CREATE TABLE param_sets (
    hash         TEXT PRIMARY KEY,   -- sha1 of canonical JSON
    plot         TEXT NOT NULL,
    params_json  TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

-- one row per (what was computed, on what). The provenance record.
CREATE TABLE runs (
    id           INTEGER PRIMARY KEY,
    machine      TEXT    NOT NULL,    -- 'cmod', 'w7x', ...
    shot         INTEGER NOT NULL,
    diagnostic   TEXT    NOT NULL,
    plot         TEXT    NOT NULL,
    params_hash  TEXT    NOT NULL REFERENCES param_sets(hash),
    blob_path    TEXT,                -- netCDF on disk; NULL if scalars-only
    status       TEXT    NOT NULL,    -- ok | failed
    error        TEXT,
    seconds      REAL,
    code_version TEXT,                -- git describe: fusion_ui + imaging_methods
    created_at   TEXT    NOT NULL,
    UNIQUE (machine, shot, diagnostic, plot, params_hash)
);

-- x = y = -1 means a shot-level scalar. Sentinel, not NULL: SQLite allows
-- NULLs in a non-INTEGER primary key, which would silently break uniqueness.
CREATE TABLE scalars (
    run_id  INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    x       INTEGER NOT NULL DEFAULT -1,
    y       INTEGER NOT NULL DEFAULT -1,
    name    TEXT    NOT NULL,         -- 'vx_c', 'taud_psd', 'number_events'
    value   REAL,
    PRIMARY KEY (run_id, x, y, name)
);

-- named parameter sets, saved from the UI and shared across the group
CREATE TABLE presets (
    name         TEXT PRIMARY KEY,
    plot         TEXT NOT NULL,
    params_hash  TEXT NOT NULL REFERENCES param_sets(hash),
    note         TEXT,
    created_at   TEXT NOT NULL
);

-- the catalog: refreshed by a scan job, never walked per request
CREATE TABLE shots (
    machine      TEXT    NOT NULL,
    shot         INTEGER NOT NULL,
    diagnostic   TEXT    NOT NULL,
    preprocessed INTEGER NOT NULL DEFAULT 0,
    path         TEXT    NOT NULL,
    bytes        INTEGER,
    mtime        TEXT,
    has_metadata INTEGER NOT NULL DEFAULT 0,  -- present in the discharge DB?
    PRIMARY KEY (machine, shot, diagnostic, preprocessed)
);
```

### Two consequences of the data growing

- **The catalog is an index, not a directory walk.** With shots arriving
  continuously, `os.listdir` per page load stops being acceptable and the
  discharge DB stops being the authoritative shot list. A `fusion-ui rescan`
  command — on cron plus a button in the UI — fills the `shots` table; the
  browser reads only from it. Files with no discharge-DB entry still appear,
  flagged `has_metadata = 0`, so a newly copied shot is visible the moment it
  lands rather than waiting for someone to curate it.
- **With 30 TB free, the cache never evicts.** No LRU eviction — an unevicted
  cache is strictly better, and eviction would solve a problem we don't have.
  The `runs` table is the ledger; `fusion-ui prune` handles deliberate cleanup
  (drop failed runs, drop everything for one plot after fixing a bug in it).
  Deletion is an operator decision, not a background process that silently
  throws away a four-minute computation.

### Why `code_version` is stored but not hashed

A cached result computed by last month's `imaging_methods` is the trap that
makes people distrust the tool. Putting the git hash in the cache key would
invalidate everything on every commit, which is worse. Store it, show it under
the figure, and put a **Recompute** button next to it — the person looking at
the plot decides whether the version matters for what they're doing.

## Repository layout

```
fusion_ui/
  fusion_ui/
    app.py               # entrypoint, nav, session state
    config.py            # paths from .env, same conventions as fusion_scripts
    cli.py               # rescan | precompute | prune
    pages/
      1_shot_browser.py  # filterable shot index -> feeds selections
      2_single_shot.py   # shot + diagnostic + plot + params -> figure
      3_multi_shot.py    # scalar vs. shot / f_GW / n_e over a selection
    core/
      db.py              # schema, migrations, connection (WAL)
      catalog.py         # rescan job; joins shots table to discharge metadata
      loader.py          # cached, lazy, time-window-sliced dataset access
      registry.py        # PlotSpec + register()
      params_ui.py       # dataclass -> widgets -> canonical dict -> hash
      store.py           # run ledger, netCDF blobs, scalar read/write
      decimate.py        # min/max envelope downsampling for Plotly traces
      probes.py          # adapter over the ragged ASP/FSP time dims
    plots/
      raw.py  velocity.py  condav.py  probe.py  spectra.py
  deploy/
    fusion-ui.service  nginx.conf
  docs/PLAN.md
  tests/
```

Separate from `fusion_scripts` on purpose: the deployment's dependencies can be
pinned without freezing exploratory work, and the UI is forced to consume stable
APIs rather than reaching into scripts.

## Four facts from the data that shape the build

- **APD files are ~500 MB each** — 583,279 time samples × 9×10 pixels, and the
  collection grows. Never touch the full time axis: default every view to the
  discharge DB's `t_start..t_end`, typically a 0.3 s slice of a 3 s record. For
  a shot with no metadata yet, default to a centred 0.2 s window rather than the
  whole record.
- **Plotly dies above roughly 50k points.** Interactive traces were the choice,
  so decimation is not optional — one shared min/max-envelope helper on the way
  to every 1D trace, preserving spikes that naive striding would drop. This is
  the single most important performance detail in the project.
- **ASP probe files are ragged.** Every quantity × probe position carries its
  own time dimension (`time_ne_0` through `time_Vf_3`, 107k–207k samples each)
  with no shared axis. The probe view needs a (quantity, position) selector and
  its own adapter; it cannot reuse the imaging code paths.
- **Movies stay matplotlib.** `plotting_scripts/plot_movies.py` already produces
  what we want; render to mp4 and serve through `st.video`. Interactive is for
  scatter, traces, and frame views — not for animation.

## Phased build

Ordered so something usable is deployed at the end of the first phase, and each
later phase adds a view without reworking the last.

### Phase 00 — Skeleton, deployed (1–2 days) — **done**

*Schema: Opus 5, in plan mode. Scaffolding: Sonnet 5. Deployment steps: you.*

Config module following the `.env` conventions from `fusion_scripts`; the SQLite
schema above; the `rescan` command that indexes the data tree; and the shot
browser — a filterable table joining the shot index to discharge metadata, with
sortable f_GW, I_p, n̄_e, confinement mode, and a visible flag on shots that have
files but no metadata. Then the full deployment path — systemd, nginx, password
— before any analysis exists.

**Ships: a URL the group can already open.**

### Phase 01 — Raw data browser (2–3 days) — **done**

*Sonnet 5, except `core/probes.py` — Opus 5.*

APD/phantom frame view with a time slider, click-a-pixel to get its decimated
time series, movie export, and the probe adapter with a (quantity, position)
trace view. Written directly against Streamlit and Plotly — deliberately before
the registry exists, so the registry gets designed against two real, different
consumers rather than one imagined one.

**Ships: the view people will use daily.**

### Phase 02 — Registry, parameter forms, result store (2–3 days) — **done**

*Opus 5, in plan mode.*

`PlotSpec`, the dataclass→widget walk over `MethodParameters`, the canonical
params hash, the netCDF blob store and the scalar writer. Retrofit phase 01's
views onto it and delete the bespoke wiring. Import
`density_scan/results.json` through `ResultManager` as seed rows so the
multi-shot view has data before any heavy compute runs.

**Ships: adding a plot becomes a one-file job.**
**On landing: write the `PlotSpec` contract into `CLAUDE.md`.** — done; it is
the section of that file phase 03 should read first.

Four things came out differently from the sketch above, all deliberate:

- **`render` may draw, and `compute` is optional.** `render(result, params,
  target) -> go.Figure | None`; returning `None` means it drew into Streamlit
  itself. `compute is None` marks a *live* spec whose result is the time-sliced
  dataset — no run row, no blob. Without this the frame viewer, with its
  slider, click target, movie expander and two figures, could not have been a
  spec at all, and the registry would have been designed against one imagined
  consumer instead of the two real ones phase 01 was written to provide.
- **Diagnostics are strings, not `Diagnostic` members.** Every other part of the
  app already used the string (`catalog.DIAGNOSTICS`, `shots.diagnostic`,
  `loader.dataset_path`); the enum stays inside `loader`.
- **The params hash includes the plot key.** `param_sets.hash` is the primary
  key while `plot` is an ordinary column, so two plots sharing a default
  parameter set — easy, since several will use bare `TwoDcaParams()` — would
  have collided on it.
- **Schema v2 adds `runs.preprocessed`.** The original UNIQUE key could not tell
  a result computed from the raw file from one computed from the preprocessed
  one, so the second silently returned the first's cached result under its own
  label. Caught by a test before any real data was written.

Phase 02 also shipped one real cached analysis rather than infrastructure alone:
`plots/spectra.py`, the PSD duration-time fit ported from
`density_scan/utils.py:get_taud_from_psd`. It reproduces the stored value for
shot 1160616027 at pixel (6, 6) exactly — `1.9861501630052958e-05` from both —
which is the end-to-end proof that params → compute → blob → scalars works, and
it is the file phase 03's ports should be copied from.

### Phase 03 — Velocity and conditional averaging (1 week)

*First port: Opus 5. Remaining ports: Sonnet 5. Physics validation: you.*

The physics payload, ported as `PlotSpec`s:

- quiver and trajectory plots from `plotting_scripts/twodca_plots.py`
- contour, TDE and 2DCA-TDE velocities from `density_scan/utils.py`
- `imaging_methods.find_events_and_2dca` with event counts, and two-sided
  exponential fits from `waveform_analysis/fitting.py`
- duration times via `density_scan/utils.py:get_taud_from_psd`

Each implements `scalars()`, which is what makes phase 04 nearly free.

**Ships: the reason the tool exists.**

### Phase 04 — Multi-shot view and precompute CLI (2–3 days)

*Sonnet 5.*

Pick a scalar, an x-axis (shot number, f_GW, n̄_e, I_p), a shot selection carried
over from the browser, and a pixel or pixel-aggregate; get the scatter, coloured
by confinement mode. Clicking a point jumps to that shot's single-shot view with
the same parameters — the move that makes an outlier immediately explainable.
Plus `fusion-ui precompute` for overnight fills.

**Ships: the trend plots, and warm caches.**

### Phase 05 — Hardening (ongoing)

*Sonnet 5; docs and test scaffolding: Haiku 4.5.*

Compute moved into a `ProcessPoolExecutor` so one person's 2DCA doesn't freeze
the page for everyone; `prune`; error surfaces that name the missing file rather
than dumping a traceback; smoke tests via `streamlit.testing.v1.AppTest`; and a
short recipe in the README for adding a plot.

## Who builds what

The split is about how expensive a wrong decision is to undo, not difficulty in
the abstract. Work whose mistakes are caught by the next test run is cheap to
delegate widely. Work that silently poisons a cache, or that only a physicist
can tell is wrong, is not.

| Work | Who | Why |
|---|---|---|
| SQLite schema, catalog design, the machine dimension | **Opus 5** | Everything downstream depends on it and a migration later is the expensive kind. Plan mode before any code. |
| Registry, dataclass→widget walk, canonical params hash | **Opus 5** | The hash is subtle in a way tests don't catch: unsorted keys, float repr drift, or an unhandled nested type produce a cache that quietly misses or collides. Highest-consequence code in the project. |
| The ragged ASP probe adapter | **Opus 5** | Twenty-four independent time axes with no shared grid. Fiddly enough that a plausible-looking wrong answer is easy to produce. |
| The first velocity port — setting the PlotSpec pattern | **Opus 5** | Every later port copies it. |
| Repo scaffolding, systemd unit, nginx config, pyproject | **Sonnet 5** | Well-trodden and immediately verifiable. |
| Frame viewer, time series, movie export, multi-shot page | **Sonnet 5** | Ordinary Streamlit/Plotly work against an interface that already exists. |
| Velocity/CA ports 2..N, following the pattern | **Sonnet 5** | Mechanical translation once the first PlotSpec exists. The bulk of phase 03 by volume. |
| Docstrings, README, test scaffolding, CLI help text | **Haiku 4.5** | Fast and cheap; errors visible on reading. |
| Anything needing sudo, SSH, or an IT ticket | **You** | Firewall rules, `htpasswd`, systemd install, the reachability test. Have the model write the exact commands; run them yourself. |
| Physics validation of every ported analysis | **You** | No model can tell you a velocity field is wrong. Pick two shots with figures you already trust from `density_scan` and require the UI to reproduce them before phase 03 is done. |
| Which scalars matter, and what the sane defaults are | **You** | Default reference pixel, threshold, window, which quantities belong on the multi-shot axis list. Research judgments dressed as configuration. |
| Curating the discharge DB as new shots land | **You** | Stays hand-maintained outside the UI. The `has_metadata` flag shows what's waiting. |

Fable 5 is available and more capable, but nothing here is reasoning-hard enough
to earn its cost. Hold it for a genuinely stuck debugging session.

### Two practices that matter more than the model choice

- **Keep `CLAUDE.md` current.** After phase 00 it holds the schema and path
  conventions; after phase 02 it holds the `PlotSpec` contract. That file is what
  stops six sessions in phase 03 from inventing six different shapes.
- **Keep the growing data tree out of exploratory loops.** Point development
  sessions at one small shot. A session that loads six shots to "check
  something" costs minutes and memory for nothing.

## Deployment

Streamlit bound to loopback, nginx in front on 443 doing TLS and basic auth,
ufw restricted to the campus subnet.

```
# /etc/systemd/system/fusion-ui.service
ExecStart=/opt/fusion-ui/.venv/bin/streamlit run fusion_ui/app.py \
  --server.port 8501 --server.address 127.0.0.1 \
  --server.headless true --browser.gatherUsageStats false

# nginx — the shared password lives here, not in the app
location / {
    auth_basic           "Shot Explorer";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass           http://127.0.0.1:8501;
    proxy_http_version   1.1;
    proxy_set_header     Upgrade $http_upgrade;   # Streamlit needs
    proxy_set_header     Connection "upgrade";    # websockets
    proxy_read_timeout   3600s;                   # long analyses
}

sudo ufw allow from <campus-subnet> to any port 443
```

**Two things that bite here.** Streamlit runs over websockets: a reverse proxy
or campus proxy that drops the `Upgrade` header produces a page that loads and
then hangs forever with no error — test the websocket path explicitly, not just
that the HTML arrives. And put the password in nginx rather than in the app, so
an unauthenticated request never reaches Python at all.

If the campus firewall blocks inbound 443, nothing about the build changes:
anyone who can already SSH to the box gets the identical app through
`ssh -N -L 8501:localhost:8501 user@server`. Confirm which world you're in
during phase 00, before anything is built on the assumption.

## Still open

1. **Are presets shared or per-person?** With one shared password there are no
   user identities, so a saved parameter set is global by construction.
   *Recommendation:* shared and append-only — saving under an existing name
   creates a new entry rather than replacing, and the `note` field carries who
   and why.
2. **What does a scalar mean when it is per-pixel?** Most scalars live on a
   pixel grid, but a multi-shot scatter needs one number per shot. Mean over the
   array, a fixed reference pixel, a median over a masked region, and the pixel
   of maximum signal are all defensible and give different answers.
   *Recommendation:* an explicit control in the multi-shot view, with the choice
   shown in the axis label so nobody misreads a plot later.
3. **How much concurrency is actually needed?** Streamlit is one process; a
   four-minute 2DCA blocks everyone unless compute is pushed off the main
   thread. *Recommendation:* build for the process pool in phase 05, revisit
   only if people are waiting.

## Before phase 00

Settled:

- **The data is on the server**, in the same layout as the dev laptop:
  `FUSION_DATA_FOLDER` with one subfolder per diagnostic (`apd/`, `phantom/`,
  `asp/`, `fsp/`) and files named `<diagnostic>_<shot>.nc`, plus
  `_preprocessed` variants. That is what `rescan` walks.
- **Everyone in the group has SSH to the server; only the maintainer has sudo.**
  So the SSH-tunnel fallback is available to every user with no help needed, and
  all deployment steps are done by one person.
- Paths are set in `.env.example` (`/hdd1/...`).

- **Inbound 443 is reachable.** Verified with `python -m http.server 443` on the
  server and `curl http://10.228.13.5:443` from a laptop on the network → `200`.
  The initial failure was the *host* firewall rejecting with ICMP
  host-unreachable (immediate "No route to host", not a timeout — the campus
  firewall was never the problem); opening 443 on the server fixed it. So
  **phase 00 ends in a URL, not an SSH tunnel**, and the nginx + systemd path in
  the Deployment section applies as written.

- **`machine` is `cmod`** for everything currently on disk. Other machines are
  expected later — W7-X most likely — which is why the column exists from the
  start. Nothing else about the schema should assume a single machine: shot
  numbers are unique only within a machine, and diagnostics differ between them.

Nothing outstanding — phase 00 can start.
