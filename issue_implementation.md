# Statistics over arbitrary traces — implementation plan

## Context

`open_issues.md`:

> Certain statistic plots (pdf, psd, ccf) that take as input a time series (a trace), should be
> able to run on all current diagnostics, for any time series (any pixel in the case of
> apd/phantom, or any probe in the case of the others), and for any time window. Additionally it
> would be convenient to plot several at the same time, labelled maybe with the magnetic
> coordinates. […] A possibility is that we have a statistics plot type that, say pdf, that given
> a shot, allows you to throw in time series that you can select from all the diagnostics […]
> Additionally, it should have a time window that centers the traces.

Today nothing in the app takes "a trace" as its input. Every analysis is a `PlotSpec` bound to one
`Target` — one machine, one shot, one diagnostic, one preprocessed flag — and to the discharge DB's
window, which is not adjustable anywhere in the UI. The two ways a 1-D series is obtained are
unrelated: `loader.pixel_series(ds, iy, ix)` for imaging, `probes.load_trace(ds, quantity,
position)` for the ragged probe files. Comparing an APD pixel with a probe channel is not
expressible.

The statistics themselves are almost all already available and unused:

| | |
|---|---|
| PDF / histogram | `fppanalysis.distributions.get_hist(data, N)`, `distribution(data, N, kernel=…)` |
| PSD | `scipy.signal.welch`, wrapped by `imaging_methods.DurationTimeEstimator` |
| ACF and CCF | `fppanalysis.corr_fun(x, y, dt, norm=True, biased=…)` → `(lags, R)`, two-sided |
| τ_d fit | `im.DurationTimeEstimator`, curves `imaging_methods.utils.{autocorrelation, power_spectral_density}` |

`fppanalysis` is already a declared dependency (`pyproject.toml`) but `fusion_ui` has never
imported it directly — only through `imaging_methods`. That changes here, deliberately.

**Outcome:** a Statistics page where you build a basket of traces — pixels off any imaging shot,
channels off any probe shot — pick PDF, PSD, ACF or CCF, set a time window, and get every trace on
one axis, labelled with its magnetic coordinate.

### Decisions already made

| | |
|---|---|
| Statistics shipped | **PDF, PSD (with an optional τ_d fit), ACF, CCF against a reference trace** |
| Labels | **Magnetic coordinates**: `R − R_sep` for pixels, window-mean `ρ` for probe channels |
| Time window | **Absolute start and end**, two number inputs, defaulting to the discharge window |
| Basket scope | **Any shot, any diagnostic**, mixed freely |

The basket scope matters practically: no shot on this machine has both imaging and probe files (45
APD shots; one ASP shot, 1150618021, with no APD). A one-shot-at-a-time basket would make the
cross-diagnostic feature undemonstrable on the data we hold.

---

## The central design choice

**Statistics are live, and they live outside the `PlotSpec` registry.**

Two independent calls, both deliberate:

**Live, not cached.** A `PlotSpec` with a `compute` writes a `runs` row and a netCDF blob and goes
behind a Compute button. None of that earns its keep here: a Welch PSD or a histogram of a 583k
sample trace is tens of milliseconds, the dataset is already open and `st.cache_resource`-cached by
`loader.open_dataset`, and these views produce no scalar anyone wants on a multi-shot axis — that
is exactly what the cached `taud_psd` spec is *for*. So: no `runs` rows, no blobs, no schema
change, widgets take effect on the next rerun. Extraction is memoised with `st.cache_data` keyed on
`(path, mtime, ref, window)` — the trick `loader._cached_times` already uses — so dragging a widget
does not re-read the trace off disk.

**A parallel registry, not `PlotSpec`.** `PlotSpec.render` is `(result, params, target) → figure`:
one target, one result. A statistics view is many traces from many targets on one axis, and
`Target` cannot express that. Rather than bend the contract that eleven specs depend on, add a
small sibling — `StatSpec` — with the same purity rule (`compute` and `render` never touch
Streamlit, the database or the filesystem) and the same one-file-plus-one-import registration.
`taud_psd` is left untouched; it stays the cached, per-pixel, scalar-producing spectrum, and the
new `psd` statistic is its exploratory, cross-diagnostic cousin. Say so in the page caption so
nobody wonders which to trust.

**A trace is the new unit.** `core/traces.py` is the whole point of the change: one type that names
a 1-D series from any diagnostic, and one function that materialises it. Everything else follows.

---

## 1. New module: `fusion_ui/core/traces.py`

Pure — numpy and xarray only, no Streamlit, no database.

```python
IMAGING = ("apd", "phantom")
PROBE = ("asp", "fsp")


@dataclass(frozen=True)
class TraceRef:
    """Which 1-D series, on which file. Hashable, so it is a session-state value
    and a cache key."""
    machine: str
    shot: int
    diagnostic: str
    preprocessed: bool
    channel: tuple          # ("pixel", x, y) | ("probe", quantity, position)

    @property
    def target_key(self):   # "cmod_1160616027_apd_p" -- matches Target.key
    @property
    def key(self):          # "cmod_1160616027_apd_p_pixel_6_6"


@dataclass(frozen=True)
class Trace:
    ref: TraceRef
    time: np.ndarray
    value: np.ndarray
    dt: float
    coords: dict            # {"R","Z","rho","dr_sep"} -- floats or None
    label: str
```

```python
def channels(ds, diagnostic) -> list[tuple]
    """Every selectable channel in this file. All (x, y) for imaging, every
    (quantity, position) from probes.quantities_and_positions for a probe."""

def extract(ds, ref, t_start, t_end) -> Trace | None
    """The series over [t_start, t_end], or None when the record misses it."""

def label(trace, style) -> str
    """style in ("channel", "rz", "magnetic")."""

def common_grid(traces, reference) -> list[Trace]
    """Every trace linearly interpolated onto ``reference``'s time base.
    Only pairwise statistics need this."""
```

`extract` branches the way `pages/2_single_shot.py:open_target` already does, on `loader.TIME_DIM
in ds.dims`:

- **imaging** — `ds[loader.image_variable(ds)].isel(x=x, y=y).sel(time=slice(t0, t1)).load()`;
  `dt = im.get_dt(ds)`.
- **probe** — `probes.load_trace(ds, quantity, position)`, then a boolean mask on its own time
  array. `dt = float(np.median(np.diff(time)))`, since `im.get_dt` assumes the imaging layout.
  Probe channels are 107k–207k samples on independent axes; nothing here may assume a shared one.

`common_grid` is what makes a CCF between two different files possible at all. When every trace
already shares the reference's time base — the common case, several pixels of one APD file — it
must be a no-op, not a round-trip through `np.interp`. The page shows a caption when any trace was
actually resampled.

**One small refactor:** promote `spectra._image_variable` to `loader.image_variable(ds)` and have
`plots/spectra.py` call it. It is the only implementation of "the 3-D variable is `frames`, except
in the phantom files that spell it otherwise", and `traces.extract` needs the same rule. One
implementation, not two.

---

## 2. New module: `fusion_ui/core/geometry.py`

The magnetic-coordinate labels. Pure numpy; import `fusion_scripts` **inside the function**, not at
module top — it pulls in matplotlib, and every page import would pay for it.

```python
def separatrix_radius(ds, t, z) -> float | None
    """R [cm] of the outboard separatrix at height ``z``, at the EFIT slice
    nearest ``t``. None when the file carries no boundary or the leg is
    unusable."""

def pixel_dr_sep(ds, x, y, t) -> float | None
    """R(y, x) - separatrix_radius(ds, t, Z(y, x)) -- centimetres outboard of
    the separatrix, negative inside."""

def probe_rho(ds, quantity, position, t_start, t_end) -> float | None
    """Mean of the rho_<quantity>_<position> companion over the window."""
```

What the files actually hold, verified by `ncdump -h` on `apd_1110201007.nc` and
`asp_1150618021.nc`:

- **Imaging has no per-pixel flux coordinate.** It has the EFIT boundary: `rlcfs(xlcfs,
  efit_time)`, `zlcfs(ylcfs, efit_time)`, `efit_time(88)` at 20 ms from 0.06 s to 1.80 s — present
  in the preprocessed files too. `fusion_scripts.plotting_scripts.calculate_splinted_LCFS(t,
  efit_time, rlcfs, zlcfs)` picks the nearest slice, keeps the outboard leg (`rbbbs >= 86`) and
  cubic-splines R over `z ∈ linspace(-8, 1, 100)`. Reuse it and `np.interp` the pixel's Z into that
  curve. **Units line up: everything is centimetres** — the pixel grid is R ∈ [88.0, 91.1], Z ∈
  [−4.5, −1.1], comfortably inside the spline's range.
- **Probes have a real ρ**, `rho_<quantity>_<position>` on its own coarser `rho_time_…` base, and
  `probes.load_trace` already returns it. It is *time-varying* — the probe moves during its plunge
  — so the label is the mean over the chosen window, and the page says so.

Three things this must handle rather than crash on: a file with no `rlcfs`; a `z` outside the
spline's `[-8, 1]`; and `interpolate.interp1d` raising on a non-monotonic outboard leg. All three
return `None`, and the label falls back to `R, Z` with a caption.

**Be honest in the label.** `R − R_sep` evaluated at the pixel's own Z is not a flux coordinate; it
is a horizontal distance to the boundary. Write it as `R−R_sep` and put the caveat in the page's
caption and in the docstring. The hardcoded `>= 86` and `linspace(-8, 1)` inside
`calculate_splinted_LCFS` are C-Mod outboard-midplane numbers and will not carry to another
machine; note it where `machine` is read.

---

## 3. New module: `fusion_ui/core/statistics.py`

The registry, mirroring `core/registry.py` in shape and in its purity rule.

```python
@dataclass(frozen=True)
class StatSpec:
    key: str                # permanent: it keys the session state and the params prefix
    label: str
    params: type            # a dataclass -> params_ui walks it, same as PlotSpec
    compute: Callable       # (trace, params) -> xr.Dataset
                            # (trace, reference, params) -> xr.Dataset when pairwise
    render: Callable        # (items, params) -> go.Figure | None
                            # items = [(Trace, xr.Dataset), ...] in basket order
    pairwise: bool = False  # needs a reference trace, and a common time base
    description: str = ""

REGISTRY = {}
def register(spec)    # duplicate key raises, as registry.register does
def get(key)
def all_specs()
```

`compute`, `render` — like `PlotSpec`'s — never touch Streamlit, the database or the filesystem.
That is what makes them unit-testable against a synthetic trace with a known answer, which is how
every one of them is tested below.

---

## 4. New package: `fusion_ui/stats/`

Four modules plus an `__init__.py` that imports each, exactly like `fusion_ui/plots/`. Every params
dataclass uses only `int`/`float`/`str`/`bool` leaves — `params_ui._leaf` supports nothing else, and
this change must not need to extend it.

### `pdf.py` — `key="pdf"`

```python
@dataclass
class PdfParams:
    bins: int = 64
    estimator: str = "histogram"     # "histogram" | "kde"
    standardise: bool = True         # (x - mean) / std before binning
    log_y: bool = True
```

`fppanalysis.distributions.get_hist(v, bins)` for the histogram, `distribution(v, bins,
kernel=True)` for the KDE. Neither normalises its input, so `standardise` is applied here — and it
is on by default because a pixel in counts and a probe in m⁻³ are otherwise not on the same axis at
all. Result `xr.Dataset({"pdf": ("value", …)}, coords={"value": centres})`. `render` draws one step
line per trace, colour-cycled, log-y when asked.

Add `("PdfParams", "estimator"): ("histogram", "kde")` to `params_ui.CHOICES`.

### `psd.py` — `key="psd"`

```python
@dataclass
class PsdParams:
    nperseg: int = 2000              # what density_scan ran; see spectra.py
    cutoff: float = None             # optional
    fit: bool = True
```

Mirror `plots/spectra.py:compute` so a number here and a `taud_psd` number are the same number:
`im.DurationTimeEstimator(im.SecondOrderStatistic.PSD, im.Analytics.TwoSided)`,
`_get_second_order_statistic` for the plotted curve (the same private call and the same reason —
a second Welch call could drift out of step with the fit's normalisation), and
`estimate_duration_time` plus `imaging_methods.utils.power_spectral_density` for the fit. `render`
copies `spectra.overlay`: log-log, solid curve and dashed fit sharing one colour and one
`legendgroup`, τ_d in the legend name.

Add `("PsdParams", "cutoff")` to `params_ui.OPTIONAL` so it renders with an "automatic" checkbox
rather than a number field defaulting to 0.

### `acf.py` — `key="acf"`

```python
@dataclass
class AcfParams:
    max_lag: float = 1e-4            # seconds; the plotted lag range
    biased: bool = False
    fit: bool = False
```

`fppanalysis.corr_fun(v, v, dt, norm=True, biased=biased)` → `(lags, R)`, truncated to `|lag| ≤
max_lag`. With `fit`, the `SecondOrderStatistic.ACF` branch of `DurationTimeEstimator` plus
`imaging_methods.utils.autocorrelation`, drawn dashed like the PSD's.

### `ccf.py` — `key="ccf"`, `pairwise=True`

```python
@dataclass
class CcfParams:
    max_lag: float = 1e-4
    biased: bool = False
```

`compute(trace, reference, params)` → `corr_fun(trace.value, reference.value, dt, norm=True,
biased=…)`. The two arrays must be the same length on the same base, which is what
`traces.common_grid` guarantees before `compute` is called. Also emit the lag at the maximum and
the maximum itself into the result, and put them in the legend — that number is usually the reason
someone drew the plot.

---

## 5. New page: `fusion_ui/pages/4_statistics.py`

Numbered 4, after `3_multi_shot.py`; no existing page is renamed, so no URL moves.

State, both plain view state:

- `st.session_state["stats.basket"]` → `list[TraceRef]`, in plot order;
- `st.session_state["stats.reference"]` → the index into it used by a pairwise statistic.

The pixel selection reuses `multipixel.selector(ds, target)` **verbatim** — it already draws the
pixel map at `(R, Z)`, handles box and lasso, and works around the widget-remount bug with its
generation counter. It keys off `pixels.{target.key}`, so switching shots inside the picker keeps
each shot's rectangle.

**Sidebar**

1. Statistic selectbox over `statistics.all_specs()`.
2. `params_ui.form(spec.params, f"stats.params.{spec.key}", container=st.sidebar)` — loose widgets,
   no `st.form`, because there is no Compute gate on a live view.
3. Window: `t_start` / `t_end` number inputs plus a "Reset to the discharge window" button. The
   default comes from `loader.time_window(ds, discharge)` for the first shot added.
4. Label style radio: channel · R,Z · magnetic (default magnetic).

**Main**

1. An "Add traces" expander, open while the basket is empty: shot selectbox (default from
   `st.session_state["selection"]`, options from `ui.cached_shot_table()`), then a diagnostic radio
   limited to what that shot actually has, then raw/preprocessed. Imaging shots get
   `multipixel.selector` and an "Add N selected pixels" button; probe shots get a multiselect over
   `probes.quantities_and_positions(ds)` and an "Add N selected channels" button.
2. The basket as an `st.dataframe` — shot, diagnostic, channel, label, `R−R_sep` or `ρ`, samples in
   the window — with `on_select="rerun"`, multi-row, and a "Remove selected" and "Clear all" pair.
3. A reference selectbox when `spec.pairwise`.
4. The figure: extract → compute → `spec.render(items, params)` → `st.plotly_chart`.

**Open the dataset unsliced.** The page calls `loader.open_dataset(path)` and slices to the *user's*
window itself, rather than going through `open_target`'s discharge-window slice — that is the whole
point of "for any time window", and `.sel` is lazy, so no full time axis is ever read. Two guards
worth having: drop and flag a trace whose record does not intersect the window, and warn once when
the basket's total sample count crosses ~20 M (about 160 MB, roughly 32 full APD windows).

---

## 6. Tests

`tests/test_traces.py` — pure:

- `channels()` over the `apd_dataset_path` and `asp_dataset_path` fixtures;
- `extract()` on imaging: the right sample count for a sub-window, right `dt`;
- `extract()` on a probe: the ragged axis is respected, the mask is on that channel's own time;
- `None` when the window misses the record entirely;
- `common_grid` is a no-op when the bases already match, and interpolates when they do not;
- `label()` in all three styles, including the fallback when a coordinate is `None`.

`tests/test_geometry.py`:

- `separatrix_radius` against a synthetic boundary whose answer is known analytically;
- `None` for a dataset with no `rlcfs`, and for a `z` outside `[-8, 1]`;
- `pixel_dr_sep` sign convention: a pixel outboard of the boundary is positive.

  This needs `rlcfs` / `zlcfs` / `efit_time` on the `apd_dataset_path` fixture. **Extend that
  fixture** rather than adding a parallel one — every real APD file carries them, so the fixture
  becomes more faithful, and no existing test asserts its variable list.

`tests/test_stats_specs.py` — each `compute` against a trace with a planted answer, the
`blob_dataset_path` philosophy applied to statistics:

- PSD of a sine peaks at that sine's frequency;
- ACF at zero lag is 1, and the fitted τ_d recovers a planted correlation time;
- CCF of a trace against a shifted copy of itself peaks at the planted lag;
- PDF of standard-normal samples integrates to 1 and peaks near 0;
- `standardise` actually changes the axis, `bins` actually changes the length.

`tests/test_stats_app.py` — `AppTest`, with a new `statistics_deployment` fixture holding **an APD
file and an ASP file under the same shot number**. The real tree has no such shot, so the fixture is
the only place the cross-diagnostic path gets exercised; say that in its docstring.

- the page renders with an empty basket and does not raise;
- adding two pixels puts two rows in the basket and draws a figure;
- switching the statistic redraws without clearing the basket;
- CCF asks for a reference and draws against it;
- a mixed APD + ASP basket draws, and reports that the probe trace was resampled.

---

## 7. Docs

- **`CLAUDE.md`** — a new section, "Traces and the statistics page", after "The `PlotSpec`
  contract": what a `TraceRef` is, the `StatSpec` contract and its purity rule, why statistics are
  live while `taud_psd` is cached, that the basket is view state, and the `R−R_sep` caveat.
  Also note that `_image_variable` now lives in `loader`.
- **`README.md`** — a "Views" bullet for the Statistics page, and an "Adding a statistic" recipe
  next to "Adding a plot".
- **`docs/PLAN.md`** — a subsection under phase 05, like the multi-pixel one.
- **`open_issues.md`** — the issue struck, with the two follow-ups below recorded in its place.

---

## Verification

```bash
.venv/bin/pytest
.venv/bin/streamlit run fusion_ui/app.py
```

In the browser:

1. **Statistics** → add shot 1110201007 (apd, preprocessed) → drag a rectangle over ~6 pixels →
   **Add**. Six rows in the basket, each labelled with an `R−R_sep` in centimetres that increases
   left to right across the array.
2. **PDF**: six curves, standardised, log-y. Untick *standardise* — the axis changes to counts.
3. **PSD** with *fit*: six log-log spectra with dashed fits and τ_d in the legend. Cross-check one
   against the same pixel's cached **Duration time (PSD fit)** on the single-shot page: the τ_d
   values must agree.
4. **ACF**, then **CCF** with the centre pixel as reference: the CCF peak lag should grow with
   radial separation from it — this is the same physics `velocity_tde` measures, so the sign and
   rough magnitude are checkable against it.
5. Change the window to a 20 ms slice: every curve redraws immediately, no Compute button, and the
   sample count in the basket falls.
6. Add shot 1150618021 (asp) → channels `ne_0`…`ne_3` → they join the same axis, labelled with
   window-mean ρ, and a caption reports that they were resampled onto the reference's base.
7. `sqlite3 $FUSION_UI_DB 'select count(*) from runs'` — unchanged. The statistics page writes
   nothing.

## Out of scope

- Caching statistics in `runs`/`scalars`, or putting one on the multi-shot axis. `taud_psd` is the
  cached path and stays it.
- A true flux coordinate for imaging pixels. `R − R_sep` at the pixel's own Z is the approximation
  this ships; a real ρ needs a flux map the files do not carry.
- FSP and phantom: no local files, so both are handled by the same code paths as ASP and APD and
  neither is verified here.
- Moving compute off the Streamlit script thread — phase 05's process pool, and nothing here makes
  it harder, since `compute` and `render` stay pure.

## Follow-ups worth recording

- A **moments** statistic — mean, std, skewness, kurtosis per trace as a table — is about thirty
  lines on top of this machinery and is the obvious fifth `StatSpec`.
- The basket is the natural place to eventually carry a trace into a cached analysis: a "compute
  `taud_psd` for every trace in this basket" button would join the two halves of the app.
