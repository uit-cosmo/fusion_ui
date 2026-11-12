# CLAUDE.md

Rules for working in this repository.

**Shot Explorer** is a Streamlit app that gives the research group
point-and-click access to fusion diagnostic data (Alcator C-Mod GPI imaging,
Langmuir probes, Thomson scattering) and to the analyses in `imaging_methods`
and `velocity_estimation`. It runs as a hosted service on the group server, next
to the data.

- `README.md`: the pages, the plots, setup, the commands, how to add a plot.
- `docs/ARCHITECTURE.md`: the design and its reasons. Read it before changing
  `core/`, the schema, the products or the Fields page.
- `deploy/README.md`: the server.

## Configuration

**No hardcoded paths.** Everything machine-specific goes through
`fusion_ui/config.py`, which reads `.env`; shell variables take precedence.
Attributes resolve lazily: importing the module on an unconfigured machine
works, and only using a path raises.

| `config.` | env var | |
|---|---|---|
| `DISCHARGE_DB_PATH` | `FUSION_DISCHARGE_DB` | `plasma_discharges.json`, **read-only** |
| `DATA_FOLDER` | `FUSION_DATA_FOLDER` | one subfolder per diagnostic |
| `UI_DB_PATH` | `FUSION_UI_DB` | this app's SQLite file |
| `CACHE_DIR` | `FUSION_UI_CACHE` | netCDF result blobs |
| `MACHINE` | `FUSION_MACHINE` | defaults to `cmod` |

## Data and storage

- **The discharge DB is read-only.** Never write `plasma_discharges.json`, never
  mirror it into SQLite, and do not normalise its free-text `comment`.
- **Open the app database with `db.open_db()`**, never bare `sqlite3.connect`.
  Readers outside the app use `db.open_readonly()`.
- **Write `runs`, `scalars` and `param_sets` only through `core/store.py`.** Raw
  SQL there breaks the tie between the hash, the blob path and the ledger.
- **A schema change appends a function to `db.MIGRATIONS`.** The list index is
  the version it produces. Never edit `db.SCHEMA` or an existing migration.
- **`runs.code_version`, `input_mtime` and `upstream_run_id` never enter a cache
  key.**
- **Pages read the `shots` table, never the filesystem.** `catalog.rescan` is the
  only thing that walks the data tree, and it stats files without opening them.
- **The app imports nothing from `fusion_scripts`**
  (`tests/test_no_fusion_scripts.py`). New methods go to `imaging_methods`.
- **Not every `runs.plot` is a registered spec.** A ledger may hold rows of
  removed plots, and of a retired one-off import under `density_scan_import`
  (`core/seed.py`), until they are pruned.

## Streamlit

- Shared helpers live in `fusion_ui/ui.py`. A page cannot import `app.py`:
  Streamlit runs it as a script, so importing it would run the start page.
- **Get connections from `ui.get_connection()`**, which is per thread. A
  connection cached with `st.cache_resource` breaks when a second browser tab
  connects.
- Session state other pages rely on:
  - `selection`: `{"machine", "shot", "diagnostic", "preprocessed"}` or `None`;
  - `shot_selection`: a `list[int]` of shot numbers;
  - `fields.open`: `{"shot", "settings", "pixel"}`, set before switching to the
    Fields page, which reads it once and removes it (`fusion_ui/jump.py`);
  - `fields.pixel`: `(x, y)` or `None`;
  - `stats.basket`: a `list[TraceRef]` in plot order, and `stats.reference`.

## The `PlotSpec` contract

Every analysis is a `PlotSpec` registered in `fusion_ui/plots/`. No plot is
wired into a page: a page picks a spec out of the registry and knows nothing
else about it. The README lists the specs and has the recipe for a new one;
`docs/ARCHITECTURE.md` says why each field exists.

```python
@dataclass(frozen=True)
class PlotSpec:
    key:         str        # permanent: part of the cache key
    label:       str
    diagnostics: tuple      # strings: ("apd", "phantom")
    params:      type       # a dataclass: the form and the hash
    render:      Callable   # (result, params, target) -> go.Figure | None
    compute:     Callable | None = None   # (ds, params) -> xr.Dataset; None is live
    scalars:     Callable | None = None   # (result) -> dict
    overlay:     Callable | None = None   # (items, params, target) -> go.Figure
    choices:     Callable | None = None   # (ds, field_path, chosen) -> tuple | None
    requires:    str | None = None        # plot key of an upstream spec
    upstream_params: Callable | None = None   # (params) -> the upstream's params
    description: str = ""
    preprocessed: bool | None = None      # False raw only, True preprocessed only
    batch_only:  bool = False             # computed only by `fusion-ui precompute`
    whole_record: bool = False            # compute gets the whole record
    companions:  tuple = ()               # other diagnostics a live spec draws
```

- **`compute`, `render` and `scalars` never touch the database or the
  filesystem, and `compute` and `scalars` never touch Streamlit.** The store
  does the I/O. `render` returns a figure, or draws into Streamlit and returns
  `None` when it needs a slider, a click target or several figures. Prefer
  returning a figure.
- **A plot key is permanent**, and so are the module, class name, field names,
  annotations and defaults of its params dataclass: all are in the hash.
  `tests/test_products.py::test_the_default_keys_are_stable` and
  `::test_every_registered_specs_default_hash_is_stable` pin them. If one fails,
  the change is wrong, not the test.
- **`scalars` keys** are a `str` for a shot-level value or `(x, y, name)` for a
  pixel's. Write at the pixel whenever the parameters name one. A name a spec
  writes needs an entry in `fusion_ui/data/documentation.md` and in
  `documentation.SOURCES` (`tests/test_documentation.py`).
- **Diagnostics are strings** matching `catalog.DIAGNOSTICS`, never members of
  `experimental_database`'s enum.
- **`requires`**: `compute` becomes `(ds, params, upstream)`. The upstream must
  be registered first (the import order in `plots/__init__.py`), accept every
  diagnostic the downstream does, and agree on `whole_record`.
  `upstream_params` must read out of the downstream's parameters, never build
  fresh defaults, or the two cache keys drift apart.
- **`batch_only`**: cached specs only. Never call `store.result` or
  `store.compute_and_store` with `batch=True` from a page. A page reads such a
  spec with `store.lookup`, shows `precompute.command(...)` for what is missing,
  and computes a spec built on one only when `store.missing_batch_upstreams`
  returns `[]`.
- **`whole_record`**: cached specs only; the one exception to "never load a full
  time axis". The spec reads its window with `loader.discharge_window(ds)` and
  cuts the record itself. **Anything that cuts a dataset before `compute` calls
  `loader.input_for(ds, t_start, t_end, whole=spec.whole_record)`.**
- **`companions`**: live specs without an `overlay` only. `render` is called as
  `render(result, params, target, companions)` with `{diagnostic: dataset or
  None}`. The page opens the files lazily; `render` reads a companion's data
  only when it draws it.
- **`choices`** is for options only the opened file knows. Static choices go in
  `params_ui.CHOICES`.
- **`overlay`** draws many pixels on one axis; `items` is `[((x, y), result), …]`.
  Many-pixel mode (`multipixel.supported`) takes a cached spec with `refx` and
  `refy` in its params and no batch-only link, or a live spec with an `overlay`.

`register()` enforces most of this at import. Do not work around a refusal.

## Parameters

- **View state is never a parameter.** A frame index, a selected pixel, a zoom,
  the Fields page's cuts and lags live in `st.session_state`, keyed off
  `Target.key`. A slider drag must never mint a `param_sets` row.
- **The hash includes the plot key** (`params_ui.hash_params(plot_key, params)`).
- **Every leaf is coerced by its annotation, and an unknown type raises.** A new
  leaf type gets a rule in `params_ui`, never a silent pass-through.
- **A dataclass with no fields raises.** `velocity_estimation`'s
  `EstimationOptions`, `CAOptions` and `NeighbourOptions` carry `@dataclass` but
  declare their own `__init__`, so they would hash to nothing. Hold their
  settings in a real dataclass and build them inside `compute`.
- Compose `imaging_methods` parameter classes directly, so the form and the
  analysis agree on a knob's name. What upstream lacks (choices, optional
  fields, help) goes in `params_ui.CHOICES`, `OPTIONAL` and `HELP`.

## The products, the Fields page and statistics

- `pixel_averages` (batch only), `method_fields` and `blob_parameters` are thin
  adapters over `imaging_methods.decorrelation`. **`plots/_pipeline.py` is the
  only module that imports it** (a test fails on a second one).
- Each product's `SCALARS` maps a stored name to its blob variable. A dead pixel
  gets no row; a NaN at a live pixel is stored as NULL. A name there needs a
  label with a unit in `core/scalar_labels.py` (`tests/test_scalar_labels.py`).
- **The Fields page never computes.** It reads the ledger, loads blobs through
  a cache keyed on (path, mtime), and names the command that fills what is
  missing.
- **Builders in `fusion_ui/views/` are pure**: `build(bundle, **controls)`
  returns a figure, a table, a sentence or a list of them, and never touches
  Streamlit, the store, the database or the filesystem. `views/products.py` is
  the one module that reads, and only the ledger. A new figure is one builder
  with a test and one entry in `SHOT_VIEWS` or `PIXEL_VIEWS`; the page does not
  change.
- **`fusion_ui/api.py` only reads** (`db.open_readonly`, no compute). Its public
  names and signatures are what the group's scripts call: keep them stable.
- **Statistics** (`core/statistics.StatSpec`, registered in `fusion_ui/stats/`)
  are live and store nothing. `compute` is pure, and params use only `int`,
  `float`, `str` and `bool` leaves.

## Analysis conventions

- **Never load a full time axis.** An APD file is about 500 MB, and a probe
  file's ragged time axes are most of its gigabyte. Cut to the discharge DB's
  `t_start..t_end` (a centred 0.2 s window without metadata) and read one frame,
  pixel or channel at a time.
- **Decimate every 1-D trace before Plotly** with `core/decimate.py`'s min/max
  envelope; striding drops spikes. A decimated trace must resample on zoom
  (`decimate.zoomable_trace`).
- **Interactive plots are Plotly.** Movies are matplotlib, rendered to mp4 and
  served with `st.video`.
- **Develop on one small shot.** Never loop over the data tree to check
  something.
- **Nothing slow starts from a page.** Pages compute inside the process that
  serves the group. `fusion-ui precompute` is the only process pool.

## Testing

```bash
.venv/bin/pytest                  # all
.venv/bin/pytest -m "not slow"    # without the end-to-end analyses
```

- Tests are hermetic: an autouse fixture clears the `FUSION_*` variables, and
  the fixtures build a temporary tree. No test touches the real data. Pages are
  smoke-tested with `streamlit.testing.v1.AppTest`.
- Run Python from the root of the checkout or worktree you are editing
  (`python -m pytest`). From anywhere else the editable install imports the
  main checkout, and you test the wrong code without an error.
- Import `netCDF4` before `h5py` in a process that loads both, or netCDF reads
  fail.
- Never weaken a test or a tolerance to make it pass.

## The state directory has two writers

The service account, and whoever runs `fusion-ui precompute` or `rescan` by hand.

- Create the app's directories and files through `core/shared.py`, never with a
  bare `os.makedirs`, whose `mode` is masked by the umask.
- Write a blob aside and rename it into place (`store._write_blob`).
- An infrastructure failure (`PermissionError`, a read-only mount) is reported,
  never recorded as a `failed` row, which later fills would skip.
- The app only widens modes. The group is the operator's (`deploy/install.sh`).
