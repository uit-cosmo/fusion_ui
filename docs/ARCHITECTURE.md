# Shot Explorer: architecture

How the app is built and why, for someone about to change it. Using it is in
[`../README.md`](../README.md), the rules to keep while coding in
[`../CLAUDE.md`](../CLAUDE.md), and the server in [`../deploy/README.md`](../deploy/README.md).

## The one idea

Every analysis is a `PlotSpec` (`core/registry.py`), registered once in
`fusion_ui/plots/`. It yields a figure for the Single shot page and a dict of
scalars for the Multi shot page, from the same stored result. There is no second
pipeline for the multi-shot view to drift from, and a page never names a plot:
it asks the registry what can be drawn for a diagnostic.

```
catalog ──▶ loader ──▶ compute() ──┬──▶ render()  ──▶ figure   (Single shot)
(index)    (lazy,      (stored by   │
            cut to a    plot, target└──▶ scalars() ──▶ SQLite   (Multi shot)
            window)     and params)
```

A spec is one of two kinds, told apart by whether it has a `compute`:

- **Live** (`compute is None`): the dataset cut to the window is the result.
  Nothing is stored, and widgets take effect on the next rerun. The frame
  viewer, the probe trace and the two profile views are live.
- **Cached**: `compute` runs once per target and parameter set. Its result is
  a netCDF blob, its run a row in the ledger, its scalars rows the Multi shot
  page reads. The form sits behind a **Compute** button, so that a nudged
  slider cannot start an analysis.

`render` may return a figure or draw into Streamlit itself. The second exists
so that a view with a slider, a click target and several figures (the frame
viewer) can be a spec like any other.

A few fields cover what the plain contract cannot, each for one reason.
`register()` refuses the combinations that would break silently (`CLAUDE.md`
lists them).

| Field | Why it exists |
|---|---|
| `requires`, `upstream_params` | Build a result on another spec's stored result. The 2DCA is nearly all the cost of the decorrelation quantities, so it is stored once and read back |
| `batch_only` | The 2DCA at every pixel takes 35 to 70 minutes a shot, and a page computes inside the process that serves the group. Only `fusion-ui precompute` may compute such a spec |
| `preprocessed` | Raw and preprocessed files are different data. `dead_pixels` needs the raw file (preprocessing interpolates dead pixels away); the three products need the preprocessed one |
| `whole_record` | `dead_pixels` finds the gas puff against the dark level at the start of the record, which a file cut to the discharge window no longer has |
| `companions` | A live view may draw another diagnostic of the same shot: the Thomson profiles offer the probe's on the same axes |
| `overlay` | Draw several pixels on one axis in many-pixel mode |
| `choices` | Options only the opened file knows, such as a probe's quantities |

## The two databases

**The discharge DB** is `experimental_database`'s `plasma_discharges.json`: about
a hundred hand-curated shots with a few exact duplicate entries. The app only
reads it, straight from JSON, and never mirrors it into SQLite, where it would
go stale. Confinement mode is the free-text `comment`. f_GW is missing for most
shots and derived from I_p and n̄_e where possible: `catalog.greenwald_fraction`
returns the value with its source, which is shown beside it.

**The app database** is one SQLite file (`core/db.py`) holding everything the
app generates: one file to back up, one to delete for a clean slate. It is in
WAL mode, so that the service, `rescan` on cron and a batch run do not block
each other's reads. `PRAGMA user_version` holds a file's schema version (4
now), and `db.MIGRATIONS` the steps, whose index is the version each produces.

| Table | Holds |
|---|---|
| `param_sets` | What a parameter hash means: `hash` (sha1 of the canonical JSON, the primary key), `plot`, `params_json` |
| `runs` | The ledger: one row per plot, parameter set and file. `status` is `ok` or `failed`, with `error`, `seconds`, `blob_path`, `code_version` and `created_at` |
| `scalars` | `(run_id, x, y, name, value)`. `x = y = -1` marks a shot-level value: a sentinel, because SQLite allows NULLs in a non-integer primary key, which would break its uniqueness. A NULL `value` is an estimate that was tried and failed |
| `shots` | The index of files on disk: `(machine, shot, diagnostic, preprocessed)` with `path`, `bytes`, `mtime`, `has_metadata` |
| `presets` | Named parameter sets. Nothing writes it yet |

Columns worth knowing:

- `machine`: shot numbers are unique only within a machine. Everything on disk
  is C-Mod; W7-X is expected, so the column was there from the start.
- `runs.preprocessed` is part of the unique key. Without it a result from the
  raw file was returned for the preprocessed one under its label.
- `runs.code_version`: the commit of `fusion_ui`, `imaging_methods`,
  `velocity_estimation` and `experimental_database` (`core/versions.py`).
- `runs.input_mtime`, `runs.upstream_run_id`: what a result was computed from,
  for staleness. Not in the cache key.
- `shots.dt`: a phantom file's frame interval, which varies from shot to shot.
  `fusion-ui backfill-dt` measures it; a rescan neither fills nor clears it.

## The result store

`core/store.py` is the only writer of `runs`, `scalars` and `param_sets`, which
is what keeps the hash, the blob path and the ledger consistent.

**The cache key** is the file (`machine`, `shot`, `diagnostic`, `preprocessed`),
the plot key and the parameter hash. The blob is at
`CACHE_DIR/runs/<plot>/<hash>/<target key>.nc` and carries its parameters and
code version as attributes, so a blob found without the database describes
itself.

**The hash** comes from one walk over the params dataclass (`core/params_ui.py`),
which also builds the widget form, so the form and the key cannot disagree. It
includes the plot key, since two plots may share a default parameter set and
`param_sets.hash` is a primary key. Every leaf is coerced by its annotation and
an unknown type raises: the upstream defaults do not match their own types (an
int under a `float` annotation, a `None` under `float`), and a key that
disagrees with itself between the form and a JSON round trip is the failure
that would make the tool untrustworthy.

**Decisions.**

- *The cache never evicts.* Disk is not the constraint, and deleting a long
  computation is an operator's decision: `fusion-ui prune`, or **Recompute**.
- *`code_version` is stored, not hashed.* Hashing it would invalidate every
  result on every commit. It is shown under the figure with **Recompute**, and
  the person looking decides whether the version matters.
- *A failure is a row.* A compute that raises is stored as `failed` with its
  message and returned on later calls, so a page does not retry a long failure
  on every rerun. `--retry-failed` or **Retry** clears it.

**Chains.** A spec with `requires` is called as `compute(ds, params, upstream)`.
The store resolves the upstream depth first and only on a miss, so a hit on the
derived result never pays for its upstream. Each link has its own row and blob.
`upstream_params` reads the upstream's parameters out of the downstream's own,
which keeps the two keys in step. An upstream that fails is reported on the
downstream run, quoting its error.

**Batch-only specs.** `store.result` raises `BatchOnlyError` for one unless
called with `batch=True`, which only `precompute` passes. A page reads such a
spec with `store.lookup`, which never computes and writes nothing, and shows
the command that fills what is missing.

**Staleness.** `store.stale_runs` lists a run when its input file's `mtime` in
the index differs from the one it recorded, when its upstream row was deleted or
recomputed in place, or when its upstream is stale. A run with no `input_mtime`
is unknown and never listed. Nothing recomputes by itself: the Fields page
badges a stale result, and `precompute --stale` recomputes them in place,
upstream first, each with its own parameters.

**Many-pixel mode** (`core/multipixel.py`) is N ordinary runs, one per pixel of
the selected rectangle, with `refx`/`refy` stamped into each parameter set. It
needed no schema change and shares the cache with single-pixel use. The price:
the Multi shot page lists those runs as N sources, one per pixel.

**The Multi shot page** reads `store.scalar_frame`. Several plots and parameter
sets can write one name, so a scalar is picked by name and source. Per-pixel
values need one number per shot, and the mean, the median, the maximum and a
fixed pixel give different answers, so the choice is a control shown in the axis
label. A click opens the run that made the point (`fusion_ui/jump.py`).

## Catalog and loader

`catalog.rescan` is the only thing that walks the data tree
(`<diagnostic>/<diagnostic>_<shot>[_preprocessed].nc`), and it stats files
without opening them. Pages read the `shots` table. A shot with files but no
discharge-DB entry is indexed with `has_metadata = 0` and shown flagged, so a
newly copied shot is visible before anyone curates it. `catalog.shot_table` is
the join to the discharge DB, cached on `catalog.index_fingerprint`.

`core/loader.py` opens a file lazily and cuts it to the discharge DB's
`t_start..t_end`, or to a centred 0.2 s window for a shot without metadata.

A shot number is `1YYMMDDnnn`: its first seven digits are the run day. What each
day was run for is hand-written in `fusion_ui/data/run_days.md`. There is no
scraper: the run pages are behind an internal server and written once, and a
start page that reaches the network hangs when the network is not there.

## The batch products and the Fields page

`pixel_averages` is the 2DCA at every live reference pixel of a preprocessed
shot, batch only. `method_fields` (seven methods' velocities and tracks) and
`blob_parameters` (size, area, duration) declare `requires="pixel_averages"` and
read it back, so a change of tracking or TDE settings costs minutes, not the
2DCA again. The three are thin adapters over `imaging_methods.decorrelation`;
`plots/_pipeline.py` is the only module that imports it. The averages are one
blob per shot, not one run per pixel, because a field needs every pixel's
average and `requires` names one upstream. The dead-pixel mask is the one the
preprocessed file stores (a hand-made one from `experimental_database` for the
1160616 files, which store none), and the derived products take it from the
averages so that all three agree on which pixels were never computed.

Three scalars are not named after their blob variable: `taud_psd`, `lambda_psd`
and `number_events` (`taud`, `lam`, `nevents` in the blobs) take the names the
`taud_psd` plot and older rows in a ledger use for the same quantities, so that
the sources line up on one Multi shot axis.

The Fields page (`pages/5_fields.py`) reads the ledger and loads each blob
through a cache keyed on path and mtime. Its *settings* are a `method_fields`
parameter set; `views.products.related_params` derives the other two products'
parameters from it. Its figures are pure builders in `fusion_ui/views/`, listed
in `SHOT_VIEWS` and `PIXEL_VIEWS`; the cuts, lags and pixel are view state.
`fusion_ui/api.py` reads the same blobs from scripts, read-only
(`db.open_readonly`), and its names are what scripts call.

## Traces and statistics

The Statistics page compares 1-D series from any diagnostic, which a `PlotSpec`
bound to one target cannot express. `core/traces.py` names a series (`TraceRef`:
a file and a pixel, or a probe quantity at a position) and extracts it over an
absolute window. `core/statistics.py` holds `StatSpec`, a small sibling of
`PlotSpec` with the same registration and purity rule; the statistics are in
`fusion_ui/stats/`. They are live: each takes tens of milliseconds and gives no
scalar for the Multi shot axis, which is what the cached `taud_psd` plot is for.

Pixels are labelled with `R - R_sep`, the horizontal distance to the separatrix
at the pixel's own height (`core/geometry.py`): not a flux coordinate, and the
spline behind it holds C-Mod numbers. A probe channel's ρ is its mean over the
window, while the probe moves during a plunge.

## What runs where

- **The Streamlit process** serves every user and computes whatever a page
  computes. Each session runs on a pooled thread, so database connections are
  per thread (`ui.get_connection`).
- **`fusion-ui precompute`** is the only process pool: `--workers N` spawns
  processes over the shots, largest file first. It skips cache hits from the
  ledger without opening a file, and Ctrl-C stops between writes.
- **Two accounts write the state directory**: the service, and whoever runs
  `precompute` or `rescan` by hand. So `core/shared.py` creates everything
  group-writable, blobs are written aside and renamed into place, and
  `precompute` checks that it can write before paying for an analysis.
- **nginx** terminates TLS and checks the one shared password, so the app has
  no user identities.

## Facts about the data that shaped this

- An APD file is about 500 MB: 583k samples of a 9×10 array. The tree is tens of
  terabytes and grows. Hence the index, the lazy loader, and the rule never to
  load a full time axis.
- Plotly fails above roughly 50k points, so every 1-D trace goes through the
  min/max envelope in `core/decimate.py` (striding would drop spikes), and a box
  selection re-decimates the full data to that window.
- ASP and FSP probe files are ragged: every quantity at every position has its
  own time axis, and those axes are most of the file's gigabyte. The probe views
  have their own adapters (`core/probes.py`, `core/mlp.py`), and a companion
  file is opened without indexing them (`loader.open_lazily`).
- APD frames are `(y, x, time)` and phantom's `(time, y, x)`, so the loader
  indexes by dimension name.
- The discharge windows of several shots include the gas-puff ramp, so live
  pixels start dark. Preprocessing crops to the puff, and `dead_pixels` judges
  its mask over the same window.

## Known limits and open questions

- **`velocity_tde` gives NaN on purely radial motion** at the default
  `ccf_min_lag = 1`, which rejects a neighbour whose cross-correlation peaks at
  zero lag. The render names the missing axis.
- **Contour tracking misreads the velocity at the array's radial edges**, where
  the structure is cut off. The Fields page's "Interior pixels only" and the
  paper's cut leave those pixels out.
- **The FWHM sizes** (`lr`, `lz`) come from an interpolation that leaves out the
  last point of the falling stretch; the Documentation page describes it. A fix
  belongs in `imaging_methods` and changes published numbers, so it waits for a
  decision.
- **A run does not record its discharge window.** If `t_start` or `t_end` is
  edited in the discharge DB after a result is stored, nothing marks it stale.
- **Under `streamlit run fusion_ui/app.py`, `fusion_ui/` is on `sys.path`**, so
  a module there shadows a top-level module of the same name.
- **`method_fields` has no group for the paper's TDE with cross-conditional
  averages.** That estimator is the per-pixel `velocity_tde` plot, whose runs
  `api.velocity_tde` gathers onto the grid.
- **Open: presets.** With one shared password a saved parameter set is global.
  Whether they are append-only is undecided, and nothing writes the table.
- **Open: should the 2DCA window adapt to each shot**, or are per-shot parameter
  sets (`--params-json`) enough? Only the second exists.

## Planned, not built

- **Page computes off the serving process**: a process pool, or a job queue with
  a worker service and queue buttons on the Fields page. Until then nothing
  slow may start from a page, which is what `batch_only` enforces.
- **The API over HTTP**, so that scripts on laptops can read the products.
- **Radial profiles of the velocity** across shots, and Thomson scattering fit
  values as scalars on the Multi shot axis.
- **Saving a preset from the UI**, and running `precompute` with one.
- **W7-X and phantom data through the batch products.**
