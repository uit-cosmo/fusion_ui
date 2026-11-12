# Shot Explorer

A browser UI over the group's fusion diagnostic data (Alcator C-Mod GPI imaging,
Langmuir probes and Thomson scattering). Pick a shot and a plot, get the plot;
pick a scalar and an x-axis, get it across every shot.

It runs as a hosted service on the group server, next to the data, and drives
the analysis code in [`imaging_methods`](https://github.com/uit-cosmo/imaging-methods)
and `velocity_estimation`. How it is built is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); how it is installed on the server
is in [`deploy/README.md`](deploy/README.md).

## Pages

- **Start page**: every run day (the first seven digits of a shot number), how
  many of its shots are curated and on disk, and what the day was run for.
- **Shot browser**: a filterable table of every shot on disk, joined to the
  discharge metadata (f_GW, I_p, n̄_e, confinement mode). It sets the shot, or
  the selection of shots, the other pages open on.
- **Single shot**: a shot, a diagnostic, a plot and its parameters in, a figure
  out. A cached plot is computed when **Compute** is pressed and kept; under the
  figure are the code versions it was made with and a **Recompute** button.
  Plots with a reference pixel can run on a rectangle of pixels (Pixels: Many).
- **Multi shot**: one stored scalar against shot number, f_GW, n̄_e or I_p,
  coloured by confinement mode. A scalar is picked by name and by source (the
  plot and parameter set that wrote it), and per-pixel values become one number
  per shot by a mean, median, maximum or a fixed pixel. Clicking a point opens
  that shot on the plot that made it.
- **Statistics**: a basket of traces (pixels of any imaging shot, channels of any
  probe shot) drawn together under one statistic (time trace, PDF, PSD, ACF, or
  CCF against a reference trace) over any time window. Nothing is stored.
- **Fields**: the velocity field of every method for one shot, on one scale, and
  for a clicked pixel the conditional average at chosen lags, the tracks and the
  numbers. It reads what batch jobs stored and never computes: what is missing
  it names, with the command that fills it.
- **Documentation**: what every quantity means and how it is computed
  ([`fusion_ui/data/documentation.md`](fusion_ui/data/documentation.md)), and
  how to read the results from Python.

## Plots

| Key | Shows | Files | Kind | Scalars |
|---|---|---|---|---|
| `raw_frames` | Frames, a pixel's trace, mp4 export | APD, phantom | live | |
| `probe_trace` | One quantity at one probe position | ASP, FSP | live | |
| `thomson_profiles` | n_e, T_e, p_e against ρ_pol or R_mid, with the MLP's profiles on request | Thomson scattering | live | |
| `mlp_profiles` | The six Mirror Langmuir Probe quantities against ρ, per plunge | ASP | live | |
| `dead_pixels` | The dead-pixel mask, with every pixel's PDF and spectrum | APD, raw only | cached | `dead`, `psd_ratio`, `number_dead` |
| `taud_psd` | Duration time from a fit to one pixel's spectrum | APD, phantom | cached | `taud_psd`, `lambda_psd` |
| `velocity_tde` | Velocity at one pixel by time delay estimation on the record | APD, phantom | cached | `vx_tde`, `vy_tde` |
| `pixel_averages` | The conditional average (2DCA) at every live pixel | APD, preprocessed only | cached, batch only | |
| `method_fields` | Seven methods' velocity fields, read off `pixel_averages` | APD, preprocessed only | cached | each method's v_R, v_Z and what they rest on |
| `blob_parameters` | Size, area and duration at every pixel, read off `pixel_averages` | APD, preprocessed only | cached | `area`, `lr`, `lz`, `taud_psd`, `lambda_psd` |

A live plot draws the file directly and stores nothing. A cached plot stores its
result and its scalars, which the Multi shot page reads. Each module in
`fusion_ui/plots/` says in its docstring what it computes and draws.

## Setup

```bash
python -m venv .venv
# The group's packages, editable, each from its own checkout:
.venv/bin/pip install -e ../imaging-methods -e ../experimental_database \
  -e ../velocity-estimation -e ../fpp-analysis-tools
.venv/bin/pip install -e ".[dev]"
cp .env.example .env            # then edit the paths for this machine
.venv/bin/fusion-ui init-db
.venv/bin/fusion-ui rescan      # index the data tree
.venv/bin/streamlit run fusion_ui/app.py
```

`.env.example` explains each variable. The app imports nothing from
`fusion_scripts`, which writes the preprocessed files it reads.

## Commands

| Command | Does |
|---|---|
| `fusion-ui init-db` | Create or migrate the app's SQLite database |
| `fusion-ui rescan` | Index the data tree into the `shots` table |
| `fusion-ui precompute PLOT [PLOT ...]` | Compute cached plots over the indexed shots, to fill the cache |
| `fusion-ui prune --plot KEY` | Count one plot's stored results; with `--yes`, delete them |
| `fusion-ui backfill-dt` | Measure each phantom file's frame interval into `shots.dt` |
| `fusion-ui status` | The resolved paths, the index and the stored results, by plot |

`fusion-ui --database PATH <command>` uses another SQLite file than
`$FUSION_UI_DB`. New shots appear in the browser only after a rescan: the server
runs it on cron, and the Shot browser has a button for it. `precompute` flags
(`fusion-ui precompute --help` has the full text):

| Flag | |
|---|---|
| `--shot N`, `--run-day D` | Only these shots, or the shots of these run days (seven digits). Repeatable, and they add up |
| `--workers N` | N processes over the shots, largest file first (default 1, in this process) |
| `--nice N` | Niceness of the computes (default 10 with several workers) |
| `--params-json PATH` | Other parameters than the defaults: one plot's complete set, when only that plot is named, or only the fields to change, such as `{"averages": {"window": 30}}`, applied to every plot named |
| `--pixel X Y` | The reference pixel, for a per-pixel plot |
| `--force` | Recompute even where a result is stored |
| `--retry-failed` | Recompute the runs recorded as failed, which are otherwise skipped |
| `--stale` | Recompute in place only the results whose input file or upstream changed since. Not with `--force` or `--retry-failed` |

Several plots named together run in dependency order within each shot, so a
shot's conditional averages are computed once. `prune --plot KEY --scalar NAME`
(repeatable) deletes only the scalars of those names under that plot, and
leaves its runs and blobs. The key need not be a registered plot. Without
`--yes` nothing is deleted and the exit status is 1.

## When to recompute

A result is computed once per shot and parameter set, and kept. The first fill
of a run day, on the server (8 cores, shared with the service, so at most 7
workers):

```bash
fusion-ui precompute method_fields blob_parameters --run-day 1160616 --workers 7
```

Both read `pixel_averages`, which this computes on the way. It takes 35 to 70
minutes a shot on one core, so it is **batch only**: no page computes it.
`method_fields` and `blob_parameters` take minutes, and can also be computed from
the Single shot page once the shot's averages are stored.

| What changed | Run | Cost |
|---|---|---|
| A shot was preprocessed | `fusion-ui rescan`, then `fusion-ui precompute method_fields blob_parameters --shot N` | up to about an hour a shot |
| Tracking, TDE or blob code (`imaging_methods.decorrelation`, `velocity_estimation`) | `fusion-ui precompute method_fields blob_parameters --force --workers 7`; the stored averages are reused | minutes a shot |
| The 2DCA itself (`imaging_methods.find_events_and_2dca`) | `fusion-ui precompute pixel_averages --force --workers 7`, then `fusion-ui precompute method_fields blob_parameters --stale --workers 7` | as a first fill |
| A setting | Not a recompute: other parameters give new results beside the old ones, through `--params-json` or the Single shot form | |
| One shot needs a shorter 2DCA window | `fusion-ui precompute method_fields blob_parameters --shot N --params-json short.json`, with `{"averages": {"window": 30}}` in the file | as a first fill, for that shot |
| A preprocessed file was rewritten | `fusion-ui rescan`, then `fusion-ui precompute pixel_averages method_fields blob_parameters --stale` | as a first fill |
| One result looks wrong | **Recompute** on the Single shot page, or `fusion-ui precompute PLOT --force --shot N` | |

`--stale` compares the file modification time each result recorded with the one
the index holds, so `rescan` comes first where a file changed. A shorter window
keeps more events, since events closer than one window apart are dropped, and it
narrows the lags the Fields page can show.

## After upgrading

Eight plots have been removed, seven of which stored results (`stored_mask` was
live), and `blob_parameters` writes five scalars where it wrote twelve. A
database filled before that still holds the old rows. They do no harm: they
only show as extra sources and names on the Multi shot page. To clear them
(`two_dca` last, since the others were built on it):

```bash
for key in velocity_contour fwhm_sizes gaussian_sizes velocity_2dca_tde \
           trajectories two_sided_exp two_dca; do
    fusion-ui prune --plot "$key" --yes
done
fusion-ui prune --plot blob_parameters --yes --scalar level \
    --scalar lx_c --scalar ly_c --scalar theta_c \
    --scalar lx_f --scalar ly_f --scalar theta_f
```

Leave out `--yes` to see the counts first. A key with nothing stored says so and
exits 1. `fusion-ui status` lists every plot key that has runs; one that is not
in the table above (`density_scan_import`, from a retired one-off import, is
another) is cleared the same way. Nothing needs a recompute.

## Reading results from Python

`fusion_ui.api` reads the stored products by shot and pixel. It only reads:
nothing is computed, and the database is opened read-only. Run the script with
the app's own Python (the checkout's `.venv/bin/python`), so that it finds the
app's database and cache.

```python
from fusion_ui import api

api.shots()                                    # shots with blob parameters stored
ds = api.blob_parameters(1160616009)           # xarray, every pixel on (y, x)
ds = api.velocity_fields(1160616009, pixels=[(6, 6)])   # pixels are (x, y)
df = api.table()                               # pandas, one row per shot and live pixel
```

A shot that is not stored raises `api.NotStored`, whose `command` is the
`fusion-ui precompute` line that fills it. The rest is in
[`fusion_ui/data/python_api.md`](fusion_ui/data/python_api.md), which the
Documentation page shows; `examples/garcia_velocity_2026_figs8_9.py` is the
example to start from.

## Extending

**A plot** is one file in `fusion_ui/plots/` and its import in that package's
`__init__.py`. No page changes; the form, the cache key, the stored result and
the scalars follow from the registration.

```python
@dataclass
class MyParams:
    """
    threshold: What counts as an event, in standard deviations.
    """
    threshold: float = 2.5

def compute(ds, params): ...              # -> xr.Dataset. Leave out for a live plot.
def render(result, params, target): ...   # -> go.Figure, or draw and return None
def scalars(result): ...                  # -> {name: value} or {(x, y, name): value}

SPEC = registry.register(
    registry.PlotSpec(
        key="my_plot",                    # permanent: it is part of the cache key
        label="My plot",
        diagnostics=("apd",),
        params=MyParams,
        render=render,
        compute=compute,
        scalars=scalars,
    )
)
```

Start from the module nearest to what you need: `spectra.py` for something
computed off the frames at one pixel, `method_fields.py` for something read off
the stored conditional averages of a whole shot, `raw.py` for a live view, and
`thomson_profiles.py` for a view that also draws another diagnostic of the shot.
The rules a plot must keep are in [`CLAUDE.md`](CLAUDE.md). A scalar a plot
writes needs an entry in `fusion_ui/data/documentation.md`; a test says so.

**A figure on the Fields page** is one builder in `fusion_ui/views/` and one
entry in `SHOT_VIEWS` or `PIXEL_VIEWS` in its `__init__.py`.

**A statistic** is one file in `fusion_ui/stats/` and its import in that
package's `__init__.py`, registered as a `statistics.StatSpec` with
`compute(trace, params)` and `render(items, params)`. `stats/pdf.py` is a short
one to copy, `stats/ccf.py` one that takes a reference trace.

**A run day** is a section in
[`fusion_ui/data/run_days.md`](fusion_ui/data/run_days.md), written by hand from
the day's C-Mod run page: a `## 1160616` heading, a bold line
`**MP800 - <title>**` quoting the miniproposal whose shot range covers the
shots, then a short summary. The sections already there show the form. A day
with no section is still listed, without a purpose.

## Tests

```bash
.venv/bin/pytest                  # everything
.venv/bin/pytest -m "not slow"    # without the end-to-end analyses
```

The suite builds a temporary tree of small files and never touches the real
data.
