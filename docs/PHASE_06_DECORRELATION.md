# Phase 06 — Decorrelation fields, served

Every method's velocity field for every shot, and every pixel's conditional
average at several lags, computed in batch on the group server and browsed in
Shot Explorer. The physics stays in `fusion_scripts` (the paper's code); Shot
Explorer owns the parameters, the cache, the batch and the views.

This file is written for an **orchestrating session** that launches one agent
per job (see [Orchestration](#orchestration)). A job's agent should need only
this file and the files its job lists. `docs/PLAN.md` and `CLAUDE.md` still
hold the app's architecture and conventions; this file adds to them and does
not repeat them.

Status, 2026-10-07: JD done. The dead-pixel view is deployed and cached for all
111 raw shots ([Dead pixels](#dead-pixels-done-2026-10-07)), and G1, the user's
check of the masks, is open. J0 done at 828 files: the user left the `_ca`
group out of phase 06 (Decisions). J2a merged (e8d87f8): schema v4, `lookup`,
`stale_runs`, `batch_only`, and `code_version` with all four repositories.
J2b merged (342d57f): `precompute` takes several plots, `--workers`,
`--run-day`, `--stale` and `--params-json`. J4 merged (99f11aa): the Fields page
and its builders. J1 merged into fusion_scripts (7e0d38f): `decorrelation.pipeline`
is bit-equal to J0's snapshot on the server, on all nine shots. G2 settled: the
three keys as proposed, three scalars renamed to the store's existing names,
and the paper's `reliable()` as a Fields-page toggle
([Products](#products-three-plotspecs-and-their-blob-schemas)). J3 merged
(48a3e33): the three product specs, with G2's names. J10 merged (bdf4806):
`velocity_field` is gone, and `fusion-ui prune` clears its runs in J7. J4b (J4's
integration round) and J5 are running. Everything else is planned and not
started.

## Decisions (the user, 2026-10-07)

| | |
|---|---|
| Shots analysed | The 17 preprocessed APD shots: 1160616 ×9, 1140827 ×8 |
| Shots preprocessed | All 111 APD shots on the server (94 are still raw only), so later batches need only a command |
| Where the physics lives | A pure API in `fusion_scripts` (`decorrelation/pipeline.py`). The fusion_ui specs are thin adapters, and `cmod_scan.py` becomes a second client of the same functions, so the paper and the app read the same numbers |
| Recompute | From the CLI first (`fusion-ui precompute`, run in parallel). A job queue and a worker service come later (L1) |
| Per-pixel view | Interactive Plotly in the app |
| Dead pixels | The hand-made mask applies to 1160616 only. Every other shot gets a mask estimated in the preprocessing step, stored in the preprocessed file and shown in the UI. The estimate is automated, and every pixel's PDF is drawn against it so the mask can be checked |
| The old `velocity_field` | Remove the spec and its stored results |
| Lags | Flexible: chosen in the view, with a shorter 2DCA window where a shot's dynamics are faster |
| W7-X | Comes later. Nothing in the API may assume C-Mod beyond its defaults |
| The `_ca` group | Left out of phase 06 (2026-10-07). The paper's `ca_tde_field` needs velocity-estimation fc5e59a, which the server does not have. Ported to the server's b3b6945, its numbers move (median 0.2–0.7%, up to 12% in v_R). It comes back as its own product once its algorithm is chosen (L9) |

## Facts every agent needs

**The analysis today** (`fusion_scripts`, all read-only for the jobs that do
not say otherwise):

- `decorrelation/apd_check/fields.py:velocity_fields` builds every method's field
  on one record:
  - three 2DCA tracks through `twodca_manuscript/velocity_field.py:compute_many`:
    `cond_av` tracked by its subpixel maximum (suffix `_max`), `cond_av` by the
    centroid of its contour (`_com`), and `cross_corr` by its maximum (`_2dcc`);
  - three- and two-point TDE off the record's cross-correlation (`tde_fields`,
    `_tde`);
  - three- and two-point TDE on the conditional average itself (`ca_tde`,
    `_catde`).

  `ca_tde_field` adds velocity_estimation's cross-conditional-average TDE
  (`TDEMethod.CA`). `ca_field` caches it as `apd<shot>_tde_ca.nc`, and the
  radial profiles read that file. It is written against velocity-estimation
  fc5e59a (`CAOptions(delta, window)`) and raises `TypeError` under b3b6945,
  whose CA method selects events with PlasmaPy. It is **not part of phase 06**
  and stays as it is (Decisions).
- `decorrelation/apd_check/cmod_scan.py` is the multi-shot entry point. It fills
  the per-pixel averages in parallel over (shot, row) with `fields.fill_row`.
  Then, per shot, it runs `fields.compute` and `blob_parameters`, which gives 13
  numbers per pixel: `nevents level area lx_c ly_c theta_c lr lz lx_f ly_f
  theta_f taud lam`.
- Caching: `twodca_manuscript/average_cache.py` keeps one netCDF per (record,
  reference pixel), about 146 kB, stamped with the 2DCA settings. Per shot,
  `apd<shot>_velocities.nc` and `apd<shot>_blobs.nc` sit beside them, all under
  `config.DECORRELATION_CACHE / "apd"`.
- The settings are `twodca_manuscript/datasets/cmod.py`'s `SPEC` pointed at
  another shot (`fields.spec_for`): the deck's C-Mod settings, **unchanged from
  shot to shot**. A shot they do not suit shows up as failed pixels; it is not
  retuned.

  | knob | value | from |
  |---|---|---|
  | 2DCA threshold, window, check_max, single_counting | 2.5, 60 (61 lags, ±15 µs), 1, True | `cmod.params()` |
  | `cond_av` mask_signal_factor, mask_distance, window_size | 0.65, 2, 1 | `cmod.params()` |
  | `cross_corr` mask_signal_factor | 0.75 | `cmod.SPEC.params_by_variable` |
  | velocity estimator | `lsq` | `cmod.params()` |
  | contour level | read off each average, `contour_level.neighbour_level`, `neighbour_step` 1 | `DatasetSpec.level_for` |
  | TDE | CC, minimum CC 0.5, running mean, interpolated peak | `fields.tde_fields`, `tde_field.MIN_CC` |
  | dead pixels | one hardcoded 10×9 mask, 22 dead | `density_scan/dead_pixel_mask.py` |
  | units | R and Z converted from cm to m on load, so velocities come out in m/s | `fields.load` |

- The cost is all in the 2DCA: about 30 s per reference pixel on one core,
  because `imaging_methods.find_events_and_2dca` recomputes the reference's
  cross-correlation with every pixel over the whole record on every call. That
  is 35–70 min per shot. Everything after it takes seconds per shot.
- `decorrelation/apd_check/figures.py` holds the reference drawings:
  - `fig_lags`: the average at a few lags, with contour and track, through
    `twodca_manuscript.presentation_figures.frames` and then
    `plotting_scripts.plot_frames_with_contour`;
  - `fig_pixel`: the tracks with the TDE lines, through
    `plotting_scripts.plot_trajectories`;
  - `fig_slices`, the field grids, and `radial_profiles`.

  Its view cuts are `MIN_LAGS = 8`, `MIN_EVENTS = 200` and interior pixels only.
  The 2DCA track it compares against is the centroid (`TRACK = "_com"`): the
  maximum sticks to the reference pixel where a pulse passes near its centre
  (1160616027 pixel (5, 7): 166 m/s by the maximum against 471 m/s by the
  centroid).

**Shot Explorer today**:

- The store, the `requires` chain, failures recorded as rows, and
  `code_version` stored but not hashed are all described in `CLAUDE.md`.
  `code_version` covers only `fusion_ui` and `imaging_methods` today.
- `fusion-ui precompute` computes one plot at a time, sequentially over the
  targets.
- `plots/velocity_field.py` is **a different estimator** from the paper's:
  contour centroid at a fixed contour level, in cm, with its own loop that
  re-runs the 2DCA and keeps no averages. The server holds 115 good runs of it.
  J10 removes it, as the user asked.
- Phase 05's process pool never landed. Anything a page computes runs inside
  the Streamlit process, so a 40-minute compute must never start from a page.

**The server**:

- `ssh fusion` reaches it with a key login (the alias is in `~/.ssh/config`).
  8 cores, 125 GB RAM, 6.7 TB free on `/hdd1`.
- The checkouts live in the home directory, `~`:
  `fusion_ui`, `fusion_scripts`, `imaging-methods`, `velocity-estimation`,
  `experimental_database`. **One venv**, `~/fusion_ui/.venv`, imports all of
  them editable. The service and every batch run therefore use the same code: a
  `git pull` in `~/fusion_scripts` changes the next batch run at once, and the
  service at its next restart.
- The service `fusion-ui` runs as user `fusionui` from `~/fusion_ui`. Restarting
  it (`sudo systemctl restart fusion-ui`) is the user's job.
- `~/fusion_ui/.env` points at the data (`/hdd1/fusion_data/`), the database
  (`/hdd1/fusion_ui/shot_explorer.sqlite`) and the cache
  (`/hdd1/fusion_ui/cache`).
- **`~/fusion_scripts/.env` points at `/hdd1/alcator`, which lacks these
  shots.** Anything run through fusion_scripts' `config` needs
  `FUSION_DATA_FOLDER=/hdd1/fusion_data`.
- Data: 111 raw APD shots over 20 run days, every one in the discharge DB with
  `t_start` and `t_end`. 17 are preprocessed. The 94 raw-only files are
  0.16–0.6 GB each, 27 GB in total. **Server data is canonical**: the laptop's 1160616
  preprocessed files were replaced by the server's on 2026-10-05.
- The server's decorrelation cache for the nine 1160616 shots is in
  `~/fusion_scripts/decorrelation/cache/apd`: 828 files, 98 MB, written by
  `cmod_scan` at fusion_scripts `8c59f96`. Only `figures.py` has changed since
  (`a3699f8`). The averages were computed at all 90 pixels, dead ones included.
- A tmux session `precompute` from 16 September is still open. Long runs go in
  tmux, at `nice -n 10`, with at most 7 workers: the service shares the
  machine.
- **The discharge windows of several shots include the gas-puff ramp-up.** Live
  pixels start dark and then light up (1120814026–032 and 1160929, for
  example). Expect bimodal PDFs there, and a non-stationary start to the
  record.

**Dead pixels** (worked out on 2026-10-07; see
[Dead pixels](#dead-pixels-done-2026-10-07) for the method):

- **The hardware set is stable within a campaign.** It is the same 18 pixels on
  every run day from 2012-02 to 2015-09. In 2016 it is those 18 plus (4, 8),
  (8, 7), (8, 8) and (9, 7), which is exactly the hand-made 1160616 mask. 2009,
  2010 and 2011 give 6, 22 and 8 dead pixels; there is no reference for them.
- **The eight 1140827 preprocessed files were made with the 2016 mask.**
  Pixels (4, 8), (8, 7), (8, 8) and (9, 7), live in 2014, were overwritten by
  interpolation from their neighbours. Those files must be preprocessed again
  (J6) before any 1140827 result is computed.
- The server's raw files are float32 with no NaN at dead pixels, so dead pixels
  have to be found from the signal. The laptop's older raw 1160616 files carry
  NaN there, from an older mask that also marked the live pixel (4, 7).

**Environments** (checked 2026-10-07):

- **Use `~/Git/fusion_ui/.venv` for every phase-06 test, regression and run,
  in both repos.** It matches the server's `~/fusion_ui/.venv` package for
  package: numpy 1.26.4, scipy 1.15.3, xarray 2025.6.1, netCDF4 1.7.4,
  plasmapy 2024.10.0, and the same commits, all editable, of imaging-methods
  (8a9bc07), velocity-estimation (b3b6945), fpp-analysis-tools (6f98750) and
  experimental_database (47b2d6b).
- The laptop's `~/Git/fusion_scripts/.venv`, where the paper's scripts run,
  differs: velocity-estimation fc5e59a in site-packages, fppanalysis 0.2.0 from
  PyPI, xarray 2025.1.2, netCDF4 1.7.2, h5py 3.15.1. It belongs to the user,
  and no job changes it.
- fusion_scripts' tests on main after J1 (7e0d38f), run as `python -m pytest
  --continue-on-collection-errors` from the root:
  - paper venv: 136 pass and 8 skip (the fusion_ui checks);
  - app venv: 134 pass and 1 skips. Three modules need `figure_provenance` or
    `seaborn`, which the app venv lacks (1 failure, 2 collection errors).
  
  That is the baseline to keep.
- **Floating point differs between the machines.** The server's Xeon W-2125 has
  AVX-512 and the laptop's i7-8565U only AVX2. With the same packages, numpy's
  BLAS rounds differently: a 300×300 matrix product already differs in the last
  bit. Fields and blobs computed on the laptop differ from the server's in the
  last bits, and by more at a few ill-conditioned pixels: max-track velocities
  that are numerically zero, and edge-pixel ellipse fits by up to 4e-3.
  - **Exact regressions run on the server.**
  - On the laptop, compare against a laptop-made reference.
- The two machines' 1160616 preprocessed files are byte-identical, and so are
  their discharge DBs.

**Pitfalls that have bitten before**:

- Import `netCDF4` before `h5py` in any process that can load both, or every
  netCDF read fails with `Errno -101`.
- Editable installs are shadowed by the current directory. Run Python from a
  worktree's root (`python -m pytest`, `python -m ...`) and the worktree's code
  is what gets imported; this was checked for `fusion_ui` and `decorrelation`.
  Run from anywhere else and the main checkout is imported, so you test the
  wrong code without any error.
- Development uses synthetic fixtures or one small shot. Never loop over the
  data tree.
- A fusion_scripts worktree has no `.env` (gitignored) and its own, empty
  `decorrelation/cache/`, since `config.DECORRELATION_CACHE` is fixed to the
  checkout `config` is imported from. Copy `~/Git/fusion_scripts/.env` in (or
  export `FUSION_DISCHARGE_DB` and `FUSION_DATA_FOLDER`). When a check needs
  cached averages, seed that cache with a writable copy of J0's snapshot. Never
  run anything from `~/Git/fusion_scripts` itself: its cache is the paper's.
- `*.nc` is gitignored in fusion_scripts. A committed netCDF fixture needs a
  narrow `!` exception in `.gitignore`, not `git add -f`.

## Design

### One expensive product, cheap ones on top

```
apd_<shot>_preprocessed.nc, sliced to the discharge window
   │
   ▼
pixel_averages     batch only · 35–70 min/shot/core · one ~12 MB blob per shot
(the 2DCA at every live reference: cond_av, cond_repr, cross_corr)
   │ requires                              │ requires
   ▼                                        ▼
method_fields                             blob_parameters
every method's v_R, v_Z, nlags,           the 13 blob parameters
track positions per lag, TDE off          at every pixel
the record · 1–3 min/shot                 ~1 min/shot
   │                                        │
   └─────────────────┬──────────────────────┘
                     ▼
   Fields page (looks up results, never computes) · multi-shot scatter (scalars)
```

- The 2DCA is over 95% of the cost and depends on four settings only. Every
  method, the blob parameters and any future per-pixel quantity read it.
  Stored once per (shot, 2DCA settings), it turns a tracking change from a
  two-hour recompute into minutes. This is the argument `average_cache.py`
  makes for the paper and `requires` makes for `two_dca`.
- **One blob per shot, not one `two_dca` run per pixel.** A field-level product
  would otherwise need 68 upstream runs, which `requires` cannot express. The
  multi-shot page would also list 68 sources per shot (the multipixel
  limitation in `PLAN.md`).
- The page never computes. A cache miss shows the command that would fill it.

### The physics API: `fusion_scripts/decorrelation/pipeline.py`

This is the only module fusion_ui imports from the decorrelation code. It is
pure: arguments in, `xr.Dataset` out. No caching, no paths, no `config` at call
time, no printing, no matplotlib at import.

A proposed surface follows. J1 may rename things; the contract is the
invariants after it.

```python
@dataclass
class Averages:            # what one reference's 2DCA depends on, bar the pixel
    threshold: float
    window: int
    check_max: int
    single_counting: bool

@dataclass
class Tracking:            # the three 2DCA tracks
    position_filter: PositionFilterParams      # cond_av's
    cross_corr_mask_signal_factor: float       # was cmod.SPEC.params_by_variable
    velocity: VelocityParams
    neighbour_step: int                         # contour_level.neighbour_level

@dataclass
class Tde:
    min_cc: float
    running_mean: bool
    interpolate: bool

@dataclass
class Blobs:
    gauss_fit: GaussFitParams
    taud_estimation: TaudEstimationParams

def dead_mask(ds) -> xr.DataArray                 # bool (y, x)
def average(ds, averages, x, y) -> xr.Dataset     # one reference's 2DCA
def stack(averages_by_pixel, ds) -> xr.Dataset    # into the pixel_averages layout
def at(bank, x, y) -> xr.Dataset | None           # one reference back out, as average() made it
def fields(ds, average_at, averages, tracking, tde, dead) -> xr.Dataset
def blobs(ds, average_at, averages, neighbour_step, blob_settings, dead) -> xr.Dataset
```

Invariants:

1. **The defaults are the deck's C-Mod settings**, read from
   `twodca_manuscript.datasets.cmod` and `tde_field.MIN_CC`, never restated. A
   test rebuilds them from `cmod.SPEC` and compares. If the deck's settings
   change, the default params hash in fusion_ui changes too and a new cache
   entry appears beside the old one. That is the intended behaviour.
2. **Every leaf is one `fusion_ui.core.params_ui` can hash**: `int`, `float`,
   `str`, `bool`, `Enum`, a nested dataclass or an Optional of these.
   - No `refx`/`refy`: they would mint one cache key per pixel for one result.
   - No tuples, lists or dicts: `params_by_variable` is flattened to the one
     knob it holds.
   - No view cuts such as `MIN_LAGS` or `MIN_EVENTS`.
   - A spec's params hold only what changes its result. `blob_parameters`
     takes `neighbour_step` alone from the tracking settings, so that moving
     the `cond_av` mask does not mint a new blob-parameter key for the same
     numbers.
3. **The input is the preprocessed record with R and Z in metres, loaded into
   memory.** The 2DCA reads every pixel series once per reference, and a lazy
   record re-reads them from disk every time. All outputs are in SI units.
4. **There is one implementation.** `fields.py`, `cmod_scan.py` and
   `twodca_manuscript.velocity_field.compute_many` all call the API.
   `compute_many` gets an optional `average_at` that bypasses `average_cache`;
   its default path is unchanged, so the W7-X deck is untouched.
5. **J1 is a refactor, not a change.** The API reproduces the frozen 1160616
   cache (J0) exactly; that is J1's acceptance test.
6. **The dead-pixel mask** comes from `dead_mask(ds)`:
   - `ds["dead"]` when the file carries one (J6 writes it);
   - otherwise, for a 1160616 shot only, the hand-made `get_dead_pixel_mask()`
     (the detector agrees with it exactly). The shot number is in
     `ds.attrs["shot_number"]`;
   - otherwise it **raises**: such a file was preprocessed before masks were
     stored, and must be preprocessed again (the 1140827 files were wrong this
     way).

   `pixel_averages` skips dead references (22 of 90). The old cache computed
   them and set the fields to NaN there afterwards, so the fields still agree.
   Products record the mask they used (`dead`, `dead_mask_source`), so the views
   can show it.
7. **No casting.** `average()` keeps the dtypes `find_events_and_2dca` returns
   (float64). A derived product computed straight after the 2DCA must be
   bit-identical to one computed later from the stored blob. Rounding to
   float32 is enough to move a mask edge or a contour closure by one lag, and a
   velocity by a few percent.
8. **Never read `config` paths at call time.** In the service process,
   fusion_scripts' `config` can resolve the server's `/hdd1/alcator`.
9. **No machine in the code paths.** W7-X data comes later. Everything
   C-Mod-specific lives in the defaults and in the fusion_ui adapter: the
   settings, cm to m, and the hand-made 1160616 mask. Array shapes, pixel
   pitches and sampling intervals are read off the data.

### Products: three PlotSpecs and their blob schemas

The keys below were confirmed at G2. **They become permanent**, since they are
part of the cache key, once the first batch writes them.
`time` is the lag in seconds; R and Z are 2-D coordinates in metres. Each blob
also carries the store's own `fusion_ui_*` attrs.

**`pixel_averages`** — batch only. Params: `PixelAveragesParams(averages:
Averages)`.

```
dims    ref_y 10, ref_x 9, y 10, x 9, time 61
cond_av, cond_repr, cross_corr   (ref_y, ref_x, y, x, time)  float64, NaN at skipped refs
nevents                          (ref_y, ref_x)              int64, 0 where none or skipped
computed                         (ref_y, ref_x)              bool, False at dead references
dead                             (y, x)                      bool, the mask used
attrs   averages = average_cache.stamp(...) string, dead_mask_source
```

**`method_fields`** — requires `pixel_averages`, and its `upstream_params`
reads `p.averages`. Params: `MethodFieldsParams(averages, tracking, tde)`.

```
dims  y 10, x 9, time 61
for each track T in max, com, 2dcc:
  vr_T, vz_T, nlags_T      (y, x)          as velocities.nc
  level_T                  (y, x)          contour level (NaN for maximum tracks)
  pos_r_T, pos_z_T         (y, x, time)    tracked position [m], NaN where untracked
  fit_T                    (y, x, time)    bool: the lags the slope rests on
nevents                                    (y, x)
vr3_tde vz3_tde vr2_tde vz2_tde cc_tde     (y, x)   TDE off the record (CC)
vr3_catde vz3_catde vr2_catde vz2_catde    (y, x)   TDE on the conditional average
dead                                       (y, x)
attrs min_cc, settings stamp, dead_mask_source
```

- The variable names are those of `apd<shot>_velocities.nc`, so the regression
  compares them by name and the paper's figure code could read a blob as it is.
- `level_*`, `pos_*` and `fit_*` are new. They are what the pixel view draws.
- There is no `_ca` group (velocity_estimation's `TDEMethod.CA`): it is left
  out of phase 06 (Decisions, L9).
- dtypes are those of `fields.velocity_fields` today: `where(~dead)` makes the
  `(y, x)` variables float64, with NaN at dead pixels.

**`blob_parameters`** — requires `pixel_averages`. Params:
`BlobParametersParams(averages, neighbour_step, blobs)`. The variables are the
13 of `apd<shot>_blobs.nc`, `(y, x)`, units `"m, m^2, s, rad"`.

**Scalars** are written per live pixel as `(x, y, name)`:

- `method_fields`, 20 names: `vr_max vz_max nlags_max vr_com vz_com nlags_com
  level_com vr_2dcc vz_2dcc nlags_2dcc number_events vr3_tde vz3_tde vr2_tde
  vz2_tde cc_tde vr3_catde vz3_catde vr2_catde vz2_catde`;
- `blob_parameters`, 12 names: `level area lx_c ly_c theta_c lr lz lx_f ly_f
  theta_f taud_psd lambda_psd`, its variables except `nevents`, which
  `method_fields` already writes.

Dead pixels get no rows, since they were never computed. A NaN at a live pixel
is written as NULL, meaning "tried and failed", as `velocity_field` does. That
makes 1360 + 816 rows per shot on a 68-pixel mask. `PLAN.md`'s phase-03 rule
applies: the user confirms new scalar names before they go on an axis.

**G2 (the user, 2026-10-07)** confirmed the three keys and these names. Each
scalar is named after its blob variable except three. Those take the name the
store already gives the same quantity, so that the seed (`density_scan_import`),
`two_dca` and the `taud_psd` spec show up as other sources on the same
multi-shot axis:

- `number_events` is read from `nevents`;
- `taud_psd` from `taud`;
- `lambda_psd` from `lam`.

The blobs keep the API's variable names, which the regression and the paper's
code read.

- `lr lz lx_f ly_f theta_f` already exist, from the seed, `fwhm_sizes` and
  `gaussian_sizes`. They are the same estimators, in the same units.
- `vr_com vz_com area` read a contour at a level set per pixel, where
  `velocity_contour`'s `vx_c vy_c area_c` read one at a fixed fraction of the
  maximum.
- `vr3_tde` and `vr3_catde` use other settings than `vx_tde` and `vx_2dca_tde`.

  So these keep names of their own.
- `level` and `level_com` agree while the two products' neighbour steps do,
  which they do at the defaults. Both stay, since each product sets its own.

### The Fields page

`fusion_ui/pages/5_fields.py`, with pure builders in a new package
`fusion_ui/views/` (no Streamlit inside; add it to `pyproject.toml`'s
packages).

**Sidebar**

- Run day, then shot. Shots with a good `method_fields` run under the chosen
  settings come first; the rest are marked *not computed*.
- Settings: the default parameter set, or any other `method_fields` params hash
  in the ledger, shown as its difference from the default.
- View cuts, which are view state and never params: minimum lags (8), minimum
  events (200), interior pixels only (off).

**Shot level**

- One panel per method: 2DCA max, 2DCA centroid, 2DCC, 3TDE (CC), 2TDE (CC),
  3TDE on the CA, 2TDE on the CA. (3TDE and 2TDE by velocity_estimation's CA
  method wait for L9.)
- **One arrow scale and one key for all panels**, so the methods compare by eye.
  Start from `plots/velocity_field.py:figure` and replace its per-figure scale
  with a shared one.
- A toggle between the quivers and v_R/v_Z maps on one diverging scale.
- Equal aspect, or the arrow directions lie.
- Dead pixels and failed fits are marked differently: a failed fit is not a
  dead pixel. Hovering shows the pixel's numbers.
- **The mask is shown with its source**, from the products' `dead` and
  `dead_mask_source`: "estimated at preprocessing, run day 1140827" or "hand-made,
  1160616". A link opens the `dead_pixels` view of the raw file, which holds the
  evidence.
- Clicking a pixel in any panel opens the pixel level. Each marker carries
  `[x, y]` as customdata, read back through `on_select="rerun"`, the way
  `core/multipixel.py:selector` does it.
- A download button for the shot's blobs as netCDF, for work on the laptop.

**Pixel level** (`fields.pixel` in session state; previous/next buttons in
reading order, skipping dead pixels)

- **Lag strip.** Rows for `cond_av` and `cross_corr` at a few lags.
  - **The lags are flexible, and they are view state**: a span (± the largest
    lag) and a number of panels, or lags typed in. Some shots have much faster
    dynamics, and ±10 µs then shows a structure long gone.
  - The default span comes from the pixel's own data: the lags the centroid
    track's slope was fitted on (`fit_com`), widened by a quarter on each
    side. It falls back to the deck's lags (`cmod.SPEC.lags`: −20, −10, 0, 10,
    20 frames, i.e. ±10 µs) when the track did not fit.
  - The lags offered stop at the bank's window. A shot that needs a shorter or
    longer window has its own parameter set (see [Recompute](#recompute)), and
    the sidebar's settings picker lists every set computed for the shot.
  - Each row keeps one colour scale across its lags, because a per-frame
    rescale makes a decaying average look like it never decays.
  - The reference pixel is marked, the contour is drawn at the level the
    average implies, and each track's position is marked at each lag.
  - The reference for what to draw, including the contour-level semantics, is
    `plot_frames_with_contour` as `figures.fig_lags` calls it.
- One large frame with a lag slider, for any of `cond_av`, `cond_repr` and
  `cross_corr`, as `plots/two_dca.py` draws it.
- **Tracks.** R(τ) and Z(τ) for the three tracks, with the fitted lags
  highlighted and the fitted lines drawn, plus the 3TDE and 2TDE lines. The
  reference is `figures.fig_pixel`.
- **Numbers.** Every method's v_R, v_Z and |v|, with nlags, nevents and level,
  plus the blob parameters when they have been computed.

**Around both levels**

- The products missing for this shot and these settings, each with the exact
  command that fills it.
- Each product's created_at, code_version and params hash, with a badge when it
  is stale (see [Staleness](#staleness)).

The page reads only through `store.lookup`. Loaded results are cached with
`st.cache_data` on (blob path, mtime), so flipping through 68 pixels never
reloads the 12 MB bank.

**Room to grow.** The shot-level panels and the pixel-level sections are
entries in two lists in `fusion_ui/views/__init__.py`, `SHOT_VIEWS` and
`PIXEL_VIEWS`. Each entry names the products it reads and points to a pure
builder, so a new figure is one builder plus one entry.

### Recompute

The first fill (J7):

```bash
cd ~/fusion_ui && nice -n 10 .venv/bin/fusion-ui precompute method_fields blob_parameters \
    --run-day 1160616 --workers 7 2>&1 | tee -a ~/phase06.log
```

- `precompute` takes several plots and runs them in dependency order within a
  target, so one worker handles a shot end to end and computes its bank once.
- `--workers N` runs a process pool over the targets, largest file first, so
  the last shots do not start alone.
- 9 shots take 2 waves on 7 workers, about 1.5 h. All 17 take about 3 h, and
  all 111 later about 13 h.

| what changed | run | cost |
|---|---|---|
| A shot was preprocessed | `rescan`, then `precompute method_fields blob_parameters --shot N` | ~1 h |
| Tracking, TDE or blob code (`pipeline.py`, velocity_estimation) | `precompute method_fields blob_parameters --force --workers 7`; the banks are reused | minutes per shot |
| The 2DCA itself (`imaging_methods.find_events_and_2dca`) | `precompute pixel_averages --force --workers 7`, then `precompute method_fields blob_parameters --stale --workers 7` | as a first fill |
| A setting | Not a recompute: a new parameter set (`--params-json`, or the single-shot form for the derived products) gives new cache entries beside the old ones | — |
| A shot with faster dynamics needs a shorter 2DCA window | A parameter set with a smaller `averages.window`, for that shot only: `precompute method_fields blob_parameters --shot N --params-json short.json`. A shorter window keeps more events, since single counting drops events closer than one window apart, and it narrows the lags a view can show | as a first fill, for that shot |
| A preprocessed file was rewritten (e.g. a corrected mask) | `precompute pixel_averages method_fields blob_parameters --stale` | as a first fill |
| One result looks wrong | Recompute on the single-shot page (derived products), or `--force --shot N` | — |

The derived products stay computable from the single-shot page whenever their
bank is cached, in 1–3 min. That is how someone tries a tracking setting on one
shot without the CLI. `pixel_averages` itself is batch only.

### Staleness

Two things can make a stored result disagree with what a recompute would give.
Neither is in the cache key, on purpose.

- **Code.** `code_version` gains `fusion_scripts` and `velocity_estimation`.
  fusion_scripts' modules are top-level, so its root is the directory holding
  `decorrelation/`, not a package's parent. The page marks a product computed
  under a different fusion_scripts commit from the current checkout. The mark
  is information only and never triggers a recompute.
- **Inputs and upstreams.** Schema v4 adds `runs.input_mtime`, the input file's
  mtime as `shots` records it, and `runs.upstream_run_id`. A result is stale
  when:
  - its input file changed;
  - its upstream link is NULL (the upstream row was deleted);
  - or its upstream row is newer than itself (the upstream was recomputed in
    place).

  `precompute --stale` recomputes exactly those. Rows written before v4 have
  NULLs and count as *unknown*, not stale, so legacy runs are not mass
  recomputed. The existing `two_dca` → `velocity_contour` chain has this gap
  today, and J2a closes it for every chain.

### Adding a product or a view later

1. Write a pure function in `decorrelation/pipeline.py`, `(ds, average_at,
   settings…, dead) -> xr.Dataset`, with a test.
2. Add a PlotSpec in `fusion_ui/plots/` with `requires="pixel_averages"`, whose
   compute calls that function and which defines `scalars`. Copy
   `plots/method_fields.py`.
3. Add a builder in `fusion_ui/views/` and an entry in `SHOT_VIEWS` or
   `PIXEL_VIEWS`.
4. Run `fusion-ui precompute <key> --workers 7`. The banks are reused.

### Dead pixels (done, 2026-10-07)

Built, tested and pushed to main on 2026-10-07, and deployed the same day (JD).

- **fusion_scripts, `density_scan/dead_pixels.py`.**
  - `estimate(ds)` returns `dead`, `evidence` (−1 no data, 0 dead, 1 follows a
    live neighbour, 2 red spectrum), `red_ratio` and `gain`, plus `psd` on
    request.
  - `consolidate(masks)` gives the run day's mask: dead in at least a third of
    the day's shots.
  - `python -m density_scan.dead_pixels --run-day D` prints the per-shot and
    per-day masks.
  - Tests: `tests/test_dead_pixels.py`.
- **fusion_ui, `plots/dead_pixels.py`.** A cached spec `dead_pixels`, raw files
  only.
  - The new `PlotSpec.preprocessed` field makes it raw-only; the single-shot
    page and `precompute` respect the field.
  - One panel per pixel, laid out as the array is (Z up, R to the right): the
    PDF, standardised or in volts and binned on the digitizer's levels, or the
    spectrum with the two bands shaded. Each panel is coloured by its verdict
    and shows the PSD ratio.
  - An expander documents the method. On 1160616 the page states whether the
    estimate agrees with the hand-made mask.
  - Scalars: `dead` and `psd_ratio` per pixel, and `number_dead`. About 7 s per
    shot.
- **The method.** A pixel is live when its spectrum is red: the median PSD over
  1–20 kHz divided by the median over 300–900 kHz exceeds 150. It is also live
  when it follows an already-live neighbour: the regression gain of its 2–20 kHz
  signal on that neighbour exceeds 0.1. This second step repeats until no pixel
  joins. Every other pixel is dead. Per run day, a pixel dead in at least a
  third of the shots is dead all day. The full text is in the module docstring
  and on the page.
- **Why not the PDF alone, as the masks used to be made.**
  - On some shots every standardised PDF is near-Gaussian (1150916025).
  - Dim live pixels inside the separatrix have PDFs that look dead.
  - Dead channels can be skewed by spikes and by telegraph noise.

  The view still draws every PDF, as the user asked, next to the spectra the
  decision used.
- **Validation on all 111 raw shots.**
  - The 2012–2016 run days were checked against the stable hardware sets
    (Facts): no errors after consolidation.
  - Per shot, the only misses were 12: the (6, 3)/(9, 3) pair on 1120814026–032,
    which consolidation catches.
  - The result is insensitive to the 150 threshold anywhere from 100 to 300.
  - A per-pixel band-passed correlation does **not** work: dead channels share
    pickup and correlate strongly while carrying nothing, which is why the gain
    is used.
- **How preprocessing uses it (J6) and where it shows.** Each preprocessed file
  stores the mask it was made with, and the shot's own verdict next to it. The
  stored-mask view (J6d) draws them on the preprocessed file. The Fields page
  marks dead pixels with the mask's source. `dead_pixels` on the raw file holds
  the evidence.

## Jobs

| ID | Job | Who | Repo | After | Runs alongside |
|---|---|---|---|---|---|
| JD | Deploy the dead-pixel view, which is already on main, and precompute it on every raw shot | orchestrator + **user** (restart) | server | — | everything |
| G1 | Check the estimated masks in the UI, run day by run day | **user** | — | JD | — |
| J0 | Freeze the regression baseline | orchestrator | server | — | — |
| J1 | Physics API | **Opus 5.5** | fusion_scripts | J0 | J2a |
| J2a | Store: batch-only specs, lookup, staleness, code_version | **Opus 5.5** | fusion_ui | — | J1 |
| J2b | Parallel, multi-plot precompute | **Opus 5.5** | fusion_ui | J2a | J1, J3, J4 |
| J3 | The three product specs | **Sonnet 5.5** | fusion_ui | J1, J2a | J2b, J4 |
| G2 | Confirm plot keys and scalar names | **user** | — | J3 | — |
| J4 | Fields page and builders | **Sonnet 5.5** | fusion_ui | J2a; integrate after J3 | J1, J2b, J3 |
| J5 | Multi-shot jump and labels | **Sonnet 5.5** | fusion_ui | J4 | — |
| J6 | Preprocess with estimated masks: the 94 raw shots, and the eight 1140827 files again | **Sonnet 5.5** | fusion_scripts + server | G1 | J1–J5 |
| J6d | Stored-mask view for preprocessed files | **Sonnet 5.5** | fusion_ui | J6's file format | J6 |
| J10 | Remove the old `velocity_field`; `fusion-ui prune` | **Sonnet 5.5** | fusion_ui | J4, whose quiver starts from its drawing | J5, J6 |
| G3 | Approve push and deploy; restart the service | **user** | — | J2b, J3, J4, J10, G2 | — |
| J7 | Deploy, first batch, regression on the server | orchestrator | server | G3; 1140827 after J6 | — |
| J8 | Physics validation | **user** | — | J7 | — |
| J9 | Docs | **Haiku 4.5** | both | J8 | — |

**Why these models.** `PLAN.md` splits work by how expensive a wrong decision
is to undo, not by how hard it is, and so does this table.

- **Opus 5.5** takes J1, J2a and J2b. Each can go wrong silently:
  - a refactor that moves a velocity by 2% reads as physics;
  - a ledger race or a missed cascade leaves stale results looking valid.

  The dead-pixel method needed the same care, and was done by Opus 5.5 on
  2026-10-07.
- **Sonnet 5.5** takes J3, J4, J5, J6, J6d and J10. These follow a pattern
  against a fixed contract (the API, the frozen schemas, `dead_pixels.estimate`),
  and identity tests or the screen catch their mistakes. J6 writes the input
  every result is computed from, so the orchestrator checks three of its files
  independently before the batch reads any of them.
- **Haiku 4.5** takes J9, where errors are visible on reading.
- **Fable 5.1** is not used by default. Escalate to it when J1's regression or
  L3's equivalence fails and an Opus session could not explain why.
- **The user** owns sudo, the masks, the names, the physics check, and any push
  or deploy.

---

### J0 — Freeze the regression baseline · orchestrator

**Done 2026-10-07, at 828 files.** The `ca_field` step below raised `TypeError`
under the server's velocity-estimation, before writing anything, and the user
left the `_ca` group out (Decisions). The snapshot is
`/hdd1/fusion_ui/reference/decorrelation_8c59f96` and
`~/Data/reference/decorrelation_8c59f96`: 810 averages, 9 velocities, 9 blobs
and a README, read-only. Checksums are in `decorrelation_8c59f96.sha256` beside
it and verified on both machines. The server's fusion_scripts was at 8c59f96
when JD's pull moved it to 7b761fd, which changes no code the snapshot came
from.

`cmod_scan` rewrites the decorrelation cache whenever it runs, so copy it
before anything changes. Add the `_ca` group, which the server lacks; it takes
minutes.

```bash
ssh fusion 'git -C ~/fusion_scripts rev-parse --short HEAD'      # expect 8c59f96
ssh fusion 'cd ~/fusion_scripts && FUSION_DATA_FOLDER=/hdd1/fusion_data PYTHONPATH=. \
  ~/fusion_ui/.venv/bin/python -c "from decorrelation.apd_check import fields as fl
for s in fl.SHOTS: fl.ca_field(s)"'
ssh fusion 'R=/hdd1/fusion_ui/reference/decorrelation_8c59f96 && mkdir -p $R \
  && cp -a ~/fusion_scripts/decorrelation/cache/apd/. $R/ \
  && echo "1160616 decorrelation cache, fusion_scripts 8c59f96 (cmod_scan + fields.ca_field), frozen $(date -I)" > $R/README.txt \
  && chmod -R a-w $R'
rsync -a fusion:/hdd1/fusion_ui/reference/decorrelation_8c59f96/ ~/Data/reference/decorrelation_8c59f96/
```

**Accept when** the snapshot holds 837 files (810 averages, plus 9 each of
velocities, blobs and tde_ca) and its README, on both machines. The laptop's
1160616 preprocessed files are the server's (2026-10-05), so J1 can run its
regression locally.

### J1 — Physics API · Opus 5.5 · fusion_scripts

**Read first:**

- [Facts](#facts-every-agent-needs) and [the API](#the-physics-api-fusion_scriptsdecorrelationpipelinepy) above;
- `decorrelation/apd_check/fields.py` and `cmod_scan.py`;
- in `twodca_manuscript/`: `velocity_field.py`, `average_cache.py`,
  `datasets/base.py`, `datasets/cmod.py`, `contour_level.py`, `tde_field.py`;
- `plotting_scripts.get_positions_and_mask`;
- fusion_ui `CLAUDE.md` on parameters (the leaf rules).

**Deliver:**

- `decorrelation/pipeline.py` meeting every invariant.
- `fields.py`, `cmod_scan.py` and `compute_many` rewired onto it.
- Hermetic tests in `decorrelation/tests/`:
  - the defaults equal what `cmod.SPEC` gives;
  - `average()` equals `find_events_and_2dca`'s average;
  - `stack`/`at` round-trip;
  - `fields()` and `blobs()` on a small synthetic record equal the
    pre-refactor `velocity_fields`/`blob_parameters`. Generate the fixture
    before refactoring and commit it.
- `decorrelation/apd_check/regress_pipeline.py --reference DIR [--shot N]`.
  It recomputes fields and blobs from the reference averages (no 2DCA, so
  minutes) and compares them variable by variable. There is no `_ca` group.
  It also recomputes one average with `average()` and compares it with the
  cached file. A `--fusion-ui` mode, used by J7, compares the products in the
  store instead.

**Accept when:**

- `regress_pipeline` passes on the laptop against J0's snapshot for all nine
  1160616 shots: the same NaN pattern, and values equal to ≤1e-12 relative,
  which in practice should mean bit-equal;
- the new tests pass in the app venv, and the existing tests keep their
  baseline in both venvs ([Environments](#facts-every-agent-needs));
- `~/Git/fusion_ui/.venv/bin/python -c "import decorrelation.pipeline"` runs
  from the worktree root;
- `python -m decorrelation.apd_check.cmod_scan 1160616027 -j 2 --force` still
  runs, in the app venv from the worktree root. Seed the worktree's cache with a
  writable copy of the snapshot first: the averages are then cached, so it
  recomputes that shot's velocities and blobs only, rewriting them in the
  worktree's cache. The snapshot stays the reference.

**Pitfalls:**

- `average_cache.get` mutates `two_dca.refx/refy`.
- The contour level is per pixel, not a setting.
- `params_for("cross_corr")` applies the 0.75 mask factor.
- `out.where(~dead)` turns ints into floats.
- `warnings` are silenced in the loops.
- `experimental_database` is imported lazily.
- Units are metres.
- Leave `ca_tde_field` and `ca_field` as they are. They are the paper's, need
  velocity-estimation fc5e59a, and are not part of the API.
- **Never relax the tolerance to pass.** A mismatch is a finding: stop and
  report it. Escalate to Fable 5.1 if it cannot be explained.

**Landed 2026-10-07 (fusion_scripts 7e0d38f).** What later jobs build on:

- **Acceptance.** On the server, `regress_pipeline` against J0's snapshot passes
  9/9 shots, every variable bit-equal and the recomputed average (5, 4)
  bit-equal. It ran from a scratch clone in `~/phase06_j1`; the log is
  `~/phase06_j1/regress.log`.
  - On the laptop, the same run differs from the snapshot in the last bits on
    all nine shots, from the CPU (Environments).
  - There J1 is bit-equal to the pre-refactor code on all nine.
  - The user accepted J1 on that evidence, with the server check run
    alongside.
- **The API differs from the sketch in these places:**
  - `stack(averages_by_pixel, ds, averages, dead)`;
  - `fields(…, pixels=None)`;
  - `dead_mask` returns its source in `attrs["dead_mask_source"]`;
  - more is public: `references`, `track`, `tde_fields`, `ca_tde`,
    `pixel_blob`, `nevents`, `TRACKS` and `BLOB_PARAMETERS`;
  - the hand-made 1160616 mask is the one-entry table `HAND_MADE_MASKS`.
    Invariant 6 was followed over invariant 9.
- **The pixel order is part of the TDE's result**, because it caches delays by
  pixel pair. Use the default order.
- **Positions and fits.** `pos_*` is NaN where the tracker found nothing, yet
  `fit_*` can include such a lag, where the slope used `smooth_da`'s
  interpolated value. 1160616027's centroid track has 18 such lags, at 4
  pixels.
- **Import order.** Importing the API imports fusion_scripts' `config`, which
  fills unset `FUSION_*` variables from its own `.env`. fusion_ui's `config`
  does the same, so whichever is imported first wins. In fusion_ui, its own
  must come first.
- **`regress_pipeline` flags:** `-j`, `--average X Y`, `--no-average`, `-v`,
  `--db`, `--machine` and `--hash PLOT=HASH`.
  - `--fusion-ui` refuses a database whose schema is not current.
  - Run standalone on the server, it needs
    `FUSION_DATA_FOLDER=/hdd1/fusion_data/` (The server).

### J2a — Store: batch-only specs, lookup, staleness · Opus 5.5 · fusion_ui

**Read first:** `CLAUDE.md`, `docs/PLAN.md` (schema, store), `core/store.py`,
`core/registry.py`, `core/db.py`, `core/precompute.py`, `ui.py`,
`pages/2_single_shot.py`, `core/multipixel.py`.

**Deliver:**

1. `PlotSpec.batch_only: bool = False`. `register()` refuses a batch-only live
   spec.
2. `store.lookup(conn, spec, target, params) -> (result | None, run | None)`.
   It never computes and never resolves an upstream; for a failed run it
   returns `(None, run)`.
3. `store.missing_batch_upstreams(conn, spec, target, params)`. The single-shot
   page then shows a batch-only spec from cache, or the CLI command if it is
   missing. It computes a derived spec inline only when no batch-only link in
   its chain is missing. Many-pixel mode skips both kinds.
4. Schema v4 (append to `db.MIGRATIONS`): `runs.input_mtime TEXT` and
   `runs.upstream_run_id INTEGER REFERENCES runs(id) ON DELETE SET NULL`.
   `compute_and_store` fills both; `input_mtime` must compare with
   `shots.mtime`.
5. `store.stale_runs(conn, plot=None)`, following the rules under
   [Staleness](#staleness). Pre-v4 rows count as unknown.
6. `code_version` gains `fusion_scripts` and `velocity_estimation`. Check on the
   server that it is not `unknown` under either account, since git's
   `safe.directory` can refuse the service user. If it is, read `.git/HEAD` and
   the ref directly.

**Accept when** these hermetic tests pass:

- `lookup` on a spec whose compute raises when called;
- the v3 → v4 migration on a copy of a v3 file;
- `upstream_run_id` is set, and becomes stale when the upstream is recomputed
  in place and when it is deleted;
- the single-shot `AppTest` shows a batch-only spec without computing it;
- `code_version` includes `fusion_scripts`.

The full suite must stay green (`python -m pytest`, run from the worktree
root).

**Landed 2026-10-07 (e8d87f8, 412 tests).** What later jobs build on:

- `store.result` and `store.compute_and_store` take `batch=False`. A batch-only
  spec, or a missing batch-only upstream, raises `store.BatchOnlyError` unless
  `batch=True`, and no row is written. `precompute` passes `batch=True`; so must
  J2b's workers and any test that computes `pixel_averages` through the store.
- `store.lookup` loads the blob on every call. A page that caches on (blob path,
  mtime) takes the row from `store.find_run` and caches `store.load_result`.
- `store.stale_runs(conn, plot=None)` lists runs of any status, each with a
  `stale` reason (`input changed`, `upstream deleted`, `upstream recomputed`,
  `upstream stale`). It judges only runs whose input is recorded and still
  indexed, and reads the registry, so import `fusion_ui.plots` first.
  `store.result` hands a stale cached row straight back: recompute one with
  `compute_and_store` (in place), upstream before downstream.
- Also new: `store.missing_batch_upstreams`, `store.input_mtime`,
  `registry.chain`, and `precompute.command`, whose command names
  `--params-json params.json` for non-default parameters (J2b adds the flag).
- `code_version` reads `name=<7 hex>[-dirty]` for fusion_ui, imaging_methods,
  fusion_scripts and velocity_estimation (`core/versions.py`), through git with
  optional locks off or straight from `.git`. Compare commits with
  `versions.commit`. `runs.created_at` now has microseconds.

### J2b — Parallel, multi-plot precompute · Opus 5.5 · fusion_ui

**Deliver:**

- `precompute PLOT [PLOT ...]`, run in dependency order within each target.
- `--workers N`. The default of 1 behaves exactly as today.
  - The pool uses the `spawn` context: no inherited SQLite handles or
    threads.
  - Each worker opens its own connection with `busy_timeout` ≥ 30 s and loads
    the windowed record into memory once per target.
  - The parent skips cache hits from the ledger without opening any file, and
    sorts the remaining targets by file size, largest first.
  - Each worker prints its own timestamped lines, flushed.
- `--run-day D` (repeatable, 7 digits), `--stale`, and `--params-json PATH`
  (canonical JSON as in `param_sets.params_json`; add
  `params_ui.from_canonical` if missing).
- `--nice` (default 10 when workers > 1).
- Ctrl-C cancels the pending targets and leaves no `failed` rows and no partial
  blobs.
- The existing rules still hold: infrastructure errors leave no row,
  `--retry-failed` works, and the writability check runs before compute.

**Accept when** hermetic tests pass with toy cached specs (instant compute,
plus a two-link chain):

- 2 workers over 4 targets write 4 good rows and their blobs;
- a second run skips them all as cached;
- `--stale` picks only the targets whose input mtime changed;
- an infrastructure error stays unrecorded;
- the single-worker output is unchanged.

**Landed 2026-10-07 (342d57f, 465 tests).** What later jobs build on:

- One worker, the default, runs in the CLI's own process and reads the record
  lazily, as before. Its output is line for line what it was: the orchestrator
  checked 16 commands against faeaede.
- Pool workers are spawned and import `precompute.WORKER_MODULES`
  (`fusion_ui.plots`). They receive each parameter set pickled, so J3's params
  classes must be importable at module level.
- `--params-json` takes one of two things, and checks either before the
  database is opened:
  - a complete parameter set as `param_sets.params_json` stores it, accepted
    only when its plot is the only one named;
  - a partial values tree such as `{"averages": {"window": 30}}`, applied to
    every plot named and refused unless every path fits every one.
  
  `params_ui.from_canonical` and `with_values` are the strict inverses.
- `--stale` recomputes in place every stale run of the plots named, each with
  its stored parameters, upstream first. A run whose upstream is stale and not
  named is skipped, and the line names the command that fixes it. `--stale`
  does not combine with `--force` or `--retry-failed`. The fix for any stale
  product of a shot is `precompute pixel_averages method_fields blob_parameters
  --stale --shot N`.
- `--nice N` sets the niceness rather than adding to it, so `nice -n 10 …
  --workers 7` computes at 10. It defaults to 10 when there are several workers.
- Ctrl-C cancels the targets not yet started and stops running ones outside
  the store's writes. A second Ctrl-C kills the workers.
- Exit status is 0 for done, 1 when a worker died or a parameter file did not
  fit, and 130 when interrupted.
- Unchanged: a `PermissionError` reading the input file gets the
  result-cache hint.

### J3 — The three product specs · Sonnet 5.5 · fusion_ui

**Read first:** [Products](#products-three-plotspecs-and-their-blob-schemas),
`CLAUDE.md` (the PlotSpec contract), `plots/velocity_contour.py` (the chained
pattern), `plots/velocity_field.py`, and `decorrelation/pipeline.py` (J1,
merged).

**Deliver:**

- `plots/pixel_averages.py`, `method_fields.py` and `blob_parameters.py`,
  imported in `plots/__init__.py` in that order. Only `pixel_averages` is
  batch only.
- Each compute converts R and Z from cm to m (`assign_coords(R=ds.R / 100, …)`),
  checks the magnitude rather than trusting the convention, calls `.load()`,
  calls the API and returns the schema exactly.
- Minimal pure renders, upgraded by J4's builders when they land:
  - `pixel_averages`: the lag-slider frame for a pixel picked as view state;
  - `method_fields`: one v_R map per method;
  - `blob_parameters`: one map per parameter.
- Scalars as specified, with no rows for dead pixels.

**Accept when** `python -m pytest -m "not slow"` passes from the worktree root,
plus:

- the adapter's output equals a direct API call on the synthetic fixture;
- a schema test checks names, dims and dtypes against
  [Products](#products-three-plotspecs-and-their-blob-schemas);
- the scalar count equals live pixels × names;
- the hash test passes: `averages.threshold` moves all three keys,
  `tracking.*` only `method_fields`', and `blobs.*` only `blob_parameters`'.

Then **stop for G2**: list the keys and the 32 names for the user before
merging.

**Landed 2026-10-07 (48a3e33, 638 tests).** What later jobs build on:

- **Where the API comes in.** `plots/_pipeline.py` is the only module in
  `fusion_ui` that imports `decorrelation`, and it imports `fusion_ui.config`
  above it. A test fails if any other module imports the API, and another
  checks, in a fresh interpreter, that the UI's data folder survives the import.
  The module also holds the shared helpers:
  - `record`, the record in metres;
  - `bank_mask`;
  - `average_at`;
  - `live_scalars`.
- **Preprocessed files only.** All three specs set `preprocessed=True`, so a
  `--run-day` fill never selects a raw file.
- **The record.** `record()` converts R and Z with the paper's own expression
  (`fields.load`). It keeps R and Z in their on-disk dtype and loads the record
  into memory. It refuses an R outside 10–700, which would not be centimetres.
  `frames` is not widened, so J6's float32 files go in as float32, as in the
  paper's loader. On the synthetic record, float32 frames gave products within
  5e-7 relative of float64 ones.
- **The derived products.** They take the bank's own `dead` and
  `dead_mask_source` and never call `dead_mask` again. They use the API's
  default pixel order.
- **Parameters.** The classes are module-level:
  - `PixelAveragesParams(averages)`;
  - `MethodFieldsParams(averages, tracking, tde)`;
  - `BlobParametersParams(averages, neighbour_step, blobs)`.

  `upstream_params` deep-copies `averages`. Each class's module and name are
  part of the hash, so they are as permanent as the keys.
  `test_the_default_keys_are_stable` pins the three default hashes:
  `302e4217…`, `d190bcb9…` and `42728c93…`.
- **Scalars.** `SCALARS` is a dict from scalar name to blob variable, which
  holds G2's three renames.
- **Renders.**
  - `pixel_averages` draws J4's lag strip at the reference with the most
    events. A render has no state, so there is no slider; the Fields page has
    the picker.
  - `method_fields` draws J4's v_R panels at the default cuts.
  - `blob_parameters` draws its own grid of 13 maps.
- **Tests.** `tests/test_products.py` on `tests/product_fixtures.py`, a 3×3
  synthetic record in centimetres with two masks. Its `direct_*` functions call
  the API without the adapters, and J4b and J5 reuse them.
  `fields_fixtures.World.install` now remembers only the first original.
- **On the laptop, on 1160616027,** with J0's 68 frozen averages stacked as
  the bank (no 2DCA was run):
  - `method_fields` took 205 s and `blob_parameters` 43 s; the bank blob is
    11.9 MB;
  - all three are bit-equal to the paper's own path on the same machine;
  - against J0's server snapshot they agree within 1e-12, except one pixel of
    `vr_2dcc` at 2.2e-11 relative, which is the machines' floating point. J7's
    exact check runs on the server.

### J4 — Fields page and builders · Sonnet 5.5 · fusion_ui

Build what [The Fields page](#the-fields-page) describes. The schema is frozen,
so start against synthetic blobs from a fixture factory
(`tests/fields_fixtures.py`) while J3 is in flight, and wire up the real
`store.lookup` once J3 is merged.

**Accept when:**

- the builders are unit-tested: each returns a `go.Figure`; all panels share
  one arrow scale; dead and failed markers differ; NaNs are handled; a pixel
  without events shows a message, not an exception;
- the `AppTest` smoke test passes:
  - with no products, the page lists the missing products and their commands;
  - with seeded products, the grid draws, and after setting
    `st.session_state["fields.pixel"]` the pixel sections draw too;
- building the pixel level from a loaded bank takes well under a second
  (measure and report it).

**Landed 2026-10-07 (99f11aa, 585 tests; the pixel level takes 117 ms of
CPU).** What later jobs build on:

- **Reading.** The page reads through `store.find_run` and a `st.cache_data` on
  (path, mtime), and never computes. A test replaces every store writer with
  one that raises.
- **`fields.open` is J5's way in.** Set it to `{"shot", "settings": <a
  method_fields params hash>, "pixel": (x, y)}`, then call
  `st.switch_page("pages/5_fields.py")`. The page reads it once and removes it.
- **The single-shot page opens the raw file** when `selection["preprocessed"]
  is False` names this shot. The mask link and multi-shot points from raw-file
  specs depend on that.
- **The arrow scale is shared by all seven panels.** The three 2DCA tracks set
  it, and an "Arrow length ×" slider moves it. The other panels set it only
  when no 2DCA track drew anything.
- **Cuts apply per method.**
  - The minimum of lags applies to the three tracks.
  - The minimum of events applies to the 2DCA max and centroid and to the
    TDEs on the CA, not to the 2DCC or the TDEs off the record.
  - Interior-only applies to every panel.
  
  The paper's `reliable()` cuts per pixel, across all methods instead. At G2
  the user kept the per-method cuts as the default and asked for a checkbox
  that applies `reliable()`'s pixel set to every panel, at the page's lags and
  events thresholds (J4b).
- **Stale products get one command for the shot:** `precompute pixel_averages
  method_fields blob_parameters --stale --shot N`.
- **J3 must keep three field names.** `views/products.related_params` derives
  the `blob_parameters` params from the `method_fields` settings through
  `averages`, `tracking.neighbour_step` and the top-level `neighbour_step`.
- **Contours** use skimage's `find_contours` with imaging_methods' primitives,
  equal to `get_contour_evolution` to 1e-12. skimage is not listed, since
  imaging_methods pins it. `streamlit>=1.63`, for `download_button` with a
  callable.
- **For J8:** the 2DCC arrows are coloured by events, while the paper's
  `fig_2dcc` colours them by lags.

### J5 — Multi-shot jump and labels · Sonnet 5.5 · fusion_ui

- Clicking a multi-shot point sourced from `method_fields` or `blob_parameters`
  opens the Fields page on that shot and settings, and on that pixel when the
  aggregate is a fixed pixel.
- Add a label table for the 32 names, e.g. `vr_com` → "v_R, 2DCA centroid
  [m/s]".

**Accept when** a multi-shot `AppTest` covers the jump and a test covers the
labels.

### JD — Deploy the dead-pixel view · orchestrator + user

The code is on main in both repos: fusion_scripts' `density_scan/dead_pixels.py`
and fusion_ui's `plots/dead_pixels.py`, with the `PlotSpec.preprocessed` field.

1. On the server, run `git -C ~/fusion_scripts pull --ff-only` and
   `git -C ~/fusion_ui pull --ff-only`. There is no schema change and no new
   dependency.
2. Ask the user to run `sudo systemctl restart fusion-ui`, and wait for it.
3. Run `cd ~/fusion_ui && nice -n 10 .venv/bin/fusion-ui precompute
   dead_pixels` in a tmux session: 111 raw shots at about 10 s each. Every
   shot's view then opens at once.
4. Tell the user where it is: the single-shot page, any APD shot, version Raw,
   "Dead pixels (PDF and spectrum of every pixel)". That starts G1, which runs
   while the other jobs go on.

### G1 — the user checks the masks in the UI

Worth looking at first:

- **The days without a reference:** 1091216 (2009, 6 dead), 1100803 (2010, one
  shot, 22 dead, among them channels sitting at zero) and 1110201 (2011, 8
  dead).
- **Pixels whose per-shot verdict changes within a day:**
  - 1091216: (6, 3) and (7, 4);
  - 1110201: (1, 3), (2, 5), (3, 8), (4, 5), (7, 0) and (9, 0);
  - 1120814: (6, 3) and (9, 3) on shots 026–032. These are dead; consolidation
    catches them.
- **The orange pixels**, live only through a neighbour, mostly in the innermost
  columns.

If a pixel is wrong, say which one. J6 then adds an override file
(`density_scan/dead_pixel_overrides.json`, per run day, recorded in each file's
attrs) rather than retuning the thresholds.

### J6 — Preprocess with estimated masks · Sonnet 5.5 · fusion_scripts + server

**Deliver** `density_scan/preprocess.py`:
`python -m density_scan.preprocess [--shot N]... [--run-day D]... [--workers 4]
[--force]`.

- **Per run day:** run `dead_pixels.estimate_shot` on every raw shot of the
  day, not only the ones being preprocessed, then `dead_pixels.consolidate`.
  1160616 uses the hand-made mask instead, by the user's rule; the detector
  agrees with it.
- **Per shot:** wrap
  `PlasmaDischargeManager.preprocess_dataset(ds, shot, radius=1000, mask=day_mask)`
  and do not reimplement it. Use a local manager, not `data_processing`'s
  module global.
- **What it stores:**
  - `dead (y, x)`: the mask used;
  - `dead_shot`: the shot's own verdict;
  - `dead_evidence` and `dead_psd_ratio`;
  - attrs `dead_mask_source` ("estimated, run day D, N shots" or "hand-made,
    1160616"), `dead_thresholds`, `preprocess_radius`, `fusion_scripts_commit`
    and `created`. Keep `shot_number`, which the API's `dead_mask` reads.
- **Writing:** write to `<file>.tmp` and rename into place. Never overwrite
  without `--force`, and leave files group-readable.
- **Order:**
  1. The eight 1140827 files first (`--run-day 1140827 --force`), since they
     were made with the wrong mask. **Move the old files to
     `/hdd1/fusion_data/apd/superseded/`; never delete them.** The user removes
     them once J7's batch 2 is validated.
  2. Then the 94 raw-only shots.
  3. Then `~/fusion_ui/.venv/bin/fusion-ui rescan`.

  The 1160616 files are not touched.
- Measure one shot first (time, peak RSS) and choose the worker count from it.

**Accept when:**

- on one shot, the frames are identical to `preprocess_dataset`'s with the same
  mask;
- on three files, `dead` equals a `consolidate` of fresh `estimate_shot` calls,
  recomputed by the orchestrator itself;
- `rescan` reports 111 preprocessed files;
- the 1160616 files are unchanged (same mtime);
- the replaced 1140827 files are in `superseded/`.

### J6d — Stored-mask view · Sonnet 5.5 · fusion_ui

A live spec `stored_mask`, "Dead-pixel mask stored at preprocessing", with
`preprocessed=True`.

- It reads `dead`, `dead_shot`, `dead_evidence`, `dead_psd_ratio` and
  `dead_mask_source` off the opened file.
- It draws the array map: dead, live, live by a neighbour, and the pixels where
  the day's mask overrides the shot's own verdict. A line under the map gives
  the mask's source.
- A file without a stored mask reads "hand-made mask" on 1160616. Anywhere else
  it gets a warning that the file predates stored masks and must be
  preprocessed again.

**Accept when** an `AppTest` passes on a fixture preprocessed file both with and
without the variables.

### J10 — Remove the old `velocity_field` · Sonnet 5.5 · fusion_ui

- Delete `plots/velocity_field.py`, its import and
  `tests/test_plots_velocity_field.py`. Update the tests that name it:
  `test_multipixel.py`, `test_plots_roundtrip.py` and `test_app.py`'s plot
  list. `test_shared.py` uses the name only as an arbitrary path string.
- Add `fusion-ui prune --plot KEY [--yes]`. It deletes every run of a plot
  (blobs, ledger rows, and scalars by cascade) and the `param_sets` rows nothing
  references any more. It prints the counts first and refuses to delete
  without `--yes`.
- The server's 115 runs (41,400 scalars) are pruned in J7, after G3. The user
  asked for the removal on 2026-10-07, and G3 restates it.

**Accept when** the suite passes, plus a hermetic test of `prune`.

**Landed 2026-10-07 (bdf4806, 656 tests).** What later jobs build on:

- **`velocity_field` is gone.** The module, its tests, its import and every
  mention in the code are removed. CLAUDE.md's spec table loses its row and
  gains a note on pruning. The stale physics statements, "twelve new scalar
  names" and the R-edge caveat, are left for J9. Four of those twelve
  (`vx_field`, `vy_field`, `number_events_field` and `nlags_field`) go with the
  prune.
- **`fusion-ui prune --plot KEY [--yes]`** is `store.plan_prune` plus
  `store.prune`. It counts:
  - runs, by status;
  - scalar rows;
  - blobs on disk, and blobs listed but already missing;
  - param sets left unreferenced (a preset keeps its own);
  - runs of other plots built on these: their link is nulled and they read as
    stale;
  - blobs in directories this user cannot write.

  Without `--yes` it deletes nothing and exits 1. The key need not be
  registered.
- **Deletion order.** Blobs go first. Then one transaction deletes the rows of
  the runs whose blob is gone.
  - A blob that cannot be removed keeps its run and is listed, and the command
    exits 1. Rerun it once the permissions are fixed.
  - A blob outside `CACHE_DIR` refuses the whole plan. The check resolves `..`
    and symlinked directories.
  - A connection with `foreign_keys` off is refused.
- **For J7.** Dry-run first, with the service's `FUSION_UI_CACHE`. J10 expects
  115 runs and 41,400 scalar rows.
  - An `outside cache` line means the ledger's blob paths are not under that
    cache.
  - An `unwritable` line needs the user to fix permissions (sudo).

### G3 — the user approves deployment

The orchestrator summarises what was merged and the test results, then asks to:

- push `fusion_scripts` main and `fusion_ui` main to GitHub;
- deploy to the server;
- prune the old `velocity_field` results there (115 runs, 41,400 scalars),
  which the user asked for on 2026-10-07.

The user runs `sudo systemctl restart fusion-ui` and `deploy/verify.sh` (the
websocket check must return `101`).

### J7 — Deploy, first batch, regression · orchestrator · server

1. On the server, run `git -C ~/fusion_scripts pull --ff-only` and
   `git -C ~/fusion_ui pull --ff-only`.
2. Run `~/fusion_ui/.venv/bin/pip install -e ~/fusion_ui --no-deps`, but only
   if `pyproject.toml` changed.
3. Run `fusion-ui init-db` (v4), then `fusion-ui status`.
4. Ask the user to restart the service (G3). Then run
   `fusion-ui prune --plot velocity_field --yes` (J10).
5. In a tmux session `phase06`, run batch 1: the 1160616 command under
   [Recompute](#recompute). Watch the log.
6. Run `regress_pipeline --fusion-ui` against J0's snapshot on all nine shots.
   It must pass exactly.
   - If every variable of one shot is off, suspect the time window: fusion_ui
     slices to the discharge DB's current window, which may have been edited
     since the file was preprocessed.
   - **Any mismatch stops the phase** and goes to the user.
7. Run batch 2 (`--run-day 1140827`) only after J6 has preprocessed those
   files again with their own mask. The old files carry the 2016 mask, and
   `dead_mask` refuses them.
8. Report per shot: status, seconds, failures with their errors, disk used, and
   the page URL for J8.

### J8 — the user checks the physics

- **1160616027 fields.** The 2DCA-centroid field matches the deck's field grid,
  and pixel (5, 7) reads about 471 m/s by the centroid against 166 m/s by the
  maximum.
- **Pixel (5, 4) lag strip.** It looks like `fig_lags`.
- **A few edge pixels.** These are where the estimators are known to misbehave.

### J9 — Docs · Haiku 4.5 · both repos

- `README.md`: the Fields page, the new `precompute` flags, and the recompute
  table.
- `CLAUDE.md`: current state; the batch-only, lookup, `views/` and product
  conventions.
- `PLAN.md`: phase 06 marked done, with the decisions that came out
  differently.
- In fusion_scripts: `decorrelation/README.md` (the `pipeline.py` API, with
  `cmod_scan` as a client) and notes on `density_scan/preprocess.py` and
  `density_scan/dead_pixels.py`.

The orchestrator reviews the result before merging.

## Orchestration

**The session.** Run the orchestrator as Opus 5.5 in `~/Git/fusion_ui`. It does
JD, J0 and J7 itself, because they are short operations on the shared server
and need judgment. It does JD first, since the user wants to check the masks
while the rest goes on. Every other job goes to an agent. The kickoff prompt:

```
You are the orchestrator of phase 06 of Shot Explorer. Read
docs/PHASE_06_DECORRELATION.md in full, then CLAUDE.md and docs/PLAN.md. The
plan is the source of truth; follow its "Orchestration" section.

- Coordinate; do not implement. Do JD, J0 and J7 yourself, over `ssh fusion`.
  Launch every other job as a background agent with the Agent tool:
  subagent_type "general-purpose", the model the Jobs table names ("opus" for
  Opus 5.5, "sonnet" for Sonnet 5.5, "haiku" for Haiku 4.5), and
  isolation "worktree" for fusion_ui jobs. fusion_scripts jobs make their own
  worktree, as the plan says. Brief each agent with the plan's template.
- Order: JD first; I restart the service when you ask. G1 is my check of the
  masks in the UI, and it runs while you go on with J0, then J1 and J2a
  together, then J2b, J4, J3, G2, J5 and J10. J6 and J6d wait for G1, J7 for
  G3, then J8 and J9. No more than three agents at once.
- After each job, read its diff and rerun its acceptance checks yourself;
  only then merge its branch into main locally. Never relax a tolerance or a
  test to make a job pass.
- Stop and ask me at JD's restart, G1, G2, G3 and J8, at any regression
  mismatch, and before any push, deploy or deletion on the server. Never use
  sudo: give me the command.
- A job that fails acceptance twice goes one model up (Sonnet, then Opus,
  then Fable 5.1), with both failure reports in its brief.
- Keep the plan's Status line current as jobs land. Report one line per
  finished job.

Start with JD.
```

**Launching a job.**

```
Agent(subagent_type="general-purpose", model="opus" | "sonnet" | "haiku",
      isolation="worktree",            # fusion_ui jobs
      description="J3 product specs", prompt=<brief>)
```

- **fusion_scripts jobs.** Worktree isolation would give a fusion_ui tree, so
  the brief tells the agent to make its own:
  `git -C ~/Git/fusion_scripts worktree add ~/Git/fusion_scripts-<job> -b phase06/<job>`.
- Agents run Python from the worktree root, with
  `~/Git/fusion_ui/.venv/bin/python` in both repos: it matches the server
  ([Environments](#facts-every-agent-needs)). fusion_scripts jobs also run the
  existing tests with `~/Git/fusion_scripts/.venv/bin/python`, so the paper's
  environment keeps working.
- Merge J1 before starting J3, so that no job needs an unmerged branch of the
  other repo on its path.

**The brief.**

```
You are implementing job <ID> of docs/PHASE_06_DECORRELATION.md
(~/Git/fusion_ui). Read its sections "Facts every agent needs", "Design" and
"<ID>", then the files <ID> lists, before writing code.
Repo: <repo>; branch phase06/<id>; work in <worktree>, running Python from its root.
Deliver exactly what <ID> lists; meet its "Accept when" and run those checks yourself.
Do not push, do not write to the server unless <ID> says so, never use sudo,
never edit the discharge DB, and do not add dependencies (the server's venv is
shared with the live service) — stop and ask instead.
Report: files changed, the acceptance output verbatim, anything you did
differently from the plan and why, and open questions.
```

**Review gate.** After each job, the orchestrator reads the diff and **reruns
the acceptance checks itself**: an agent's report is not evidence. Only then
does it merge `phase06/<id>` into main locally. Commit messages end with the
session's attribution line.

**Escalation.** A job that fails acceptance twice is relaunched one model up
(Sonnet → Opus → Fable 5.1), with both failure reports in the brief.

**Parallelism.**

1. JD, then G1 while the rest goes on.
2. J0.
3. J1 and J2a together.
4. J2b and J4 once J2a is merged.
5. J3 once J1 and J2a are merged.
6. G2, then J5 and J10.
7. J6 and J6d after G1.
8. J7 after G3.
9. J8, then J9.

At most three agents run at once. They share one laptop and, for J6, one
server.

**Gates.** Stop and ask the user at JD's restart, G1, G2, G3 and J8, at any
regression mismatch, and before any push, deploy, or deletion on the server.

**Reporting.** Send one line per finished job, and the gate's question when one
is reached.

## Later

| ID | Job | Who |
|---|---|---|
| L1 | **Queue and worker.** Schema v5 `jobs` table (plots, target, params hash, status queued/running/done/failed/cancelled, timestamps, worker, message). Atomic claim with `UPDATE … RETURNING`. `fusion-ui worker --processes 4` reuses J2b's per-target function. A systemd unit `fusion-ui-worker` (`User=fusionui`, `Nice=10`). Queue buttons on the Fields page for missing or stale products, and a Jobs page | Opus 5.5 (schema, claim, worker), Sonnet 5.5 (UI), user (systemd) |
| L2 | **Radial profiles** as a multi-shot view: v_R(R) per method, averaged over rows and grouped by f_GW, a port of `figures.radial_profiles` over `method_fields` blobs | Sonnet 5.5 |
| L3 | **A faster bank.** Profile first. If the per-reference full-record cross-correlation dominates, compute every pair once per shot, or only at the window's lags, in imaging_methods, equal to today's output within 1e-12 | Opus 5.5; Fable 5.1 if stuck |
| L4 | **Presets.** Save a parameter set from the UI (the `presets` table, unused so far) and run `precompute --preset NAME` | Sonnet 5.5 |
| L5 | **The 94 newly preprocessed shots** through the batch (~13 h on 7 workers today), and a nightly cron fill | orchestrator |
| L7 | **An adaptive 2DCA window.** Read the window off each shot (e.g. a multiple of its median duration time) instead of fixing 60 samples, the way the contour level is read off each average. Only if per-shot parameter sets prove too manual | Opus 5.5 |
| L8 | **W7-X and phantom data** through the same products. The API is kept machine-agnostic for this (invariant 9) | Opus 5.5 for the first, Sonnet 5.5 after |
| L6 | **One cache.** `figures.py`/`cmod_scan` read fusion_ui's blobs instead of their own cache | Opus 5.5 |
| L10 | **Record the discharge window on each run.** The products are computed over the discharge DB's `t_start..t_end`, which no run records: if the window is edited after a bank is computed, the products on it, or a bank computed afterwards, disagree silently, and `stale_runs` cannot see it (J3) | Opus 5.5 |
| L9 | **The CA TDE** (velocity_estimation's `TDEMethod.CA`, the paper's `_ca` group) as its own product, once the user chooses its event selector: fc5e59a's in-package `cond_av`, which the paper used, or PlasmaPy's `ConditionalEvents` in b3b6945. On 1160616027 they differ by a median of 0.2–0.7%, at most 12% in v_R, and about 2% in events (2026-10-07). The paper's nine files are kept in `~/Data/reference/tde_ca_fc5e59a_laptop/` | Opus 5.5 |

## Open questions for the user

1. ~~Plot keys and scalar names (G2).~~ Settled on 2026-10-07: see
   [Products](#products-three-plotspecs-and-their-blob-schemas).
2. Whether the masks look right in the UI, above all for 2009–2011, which have
   no reference (G1).
3. When J6 has run, can the superseded 1140827 files in
   `/hdd1/fusion_data/apd/superseded/` be deleted?
4. Should the 2DCA window adapt to each shot by itself (L7), or are per-shot
   parameter sets enough?
5. Which CA-TDE event selector is the right one (L9)? And should the paper's
   venv be aligned with the app's ([Environments](#facts-every-agent-needs))?
