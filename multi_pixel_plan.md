# Multi-pixel analysis — implementation plan

## Context

`open_issues.md`, "TODOS for later":

> Some diagnostics should be able to be run on several pixels at the same time. Have a
> pixel selection window, for example a plot of all the pixel locations where the pixels
> can be selected with a rectangle. Once they are selected, a plot type is chosen (for
> example, PSD fit), and then the method is applied to all selected pixels and presented
> in a single plot.

Today every per-pixel analysis is one pixel at a time. `taud_psd`, `velocity_tde` and the
whole 2DCA chain already carry `refx`/`refy` **in their params**, so each pixel is already
its own `param_sets` row, `runs` row and blob. Comparing twelve pixels means twelve visits
to the sidebar and twelve figures you cannot overlay.

The one existing many-pixel thing, `velocity_field`, solves it the other way: one spec that
sweeps *every* pixel internally, one blob, ~30 min, no subset concept, and its results share
nothing with the single-pixel runs.

**Outcome:** select a rectangle of pixels on the frame, press one button, get every selected
pixel's analysis on one axis — reusing the existing cache, so a pixel already computed in the
single-shot view is free.

### Decisions already made

| | |
|---|---|
| Presentation | **Overlaid curves**, one trace per pixel on a shared axis |
| Placement | **Inside `2_single_shot.py`**, as a mode |
| Cost guard | **Estimate + confirm**, no hard cap |

## The central design choice

**N separate runs, not one region-run.** The selected rectangle mints one params instance per
pixel (`refx`/`refy` stamped) and calls `store.result` once per pixel. Consequences:

- **no schema change, no `params_ui` change, no new spec.** `params_ui._leaf`
  (`fusion_ui/core/params_ui.py:236`) has no rule for a list or tuple, so a pixel *set* could
  not be a params field without extending `_leaf`, `_rebuild` and `_scalar_widget`. It doesn't
  need to be.
- **the cache is shared both ways.** A pixel computed in single-pixel mode is a hit here; a
  pixel computed here is a hit when you open it singly. Re-selecting an overlapping rectangle
  only pays for the new pixels.
- **the `requires` chain works untouched.** `upstream_params` carries `refx`/`refy`, so each
  pixel gets its own 2DCA run, stored once and shared by all five derived plots at that pixel.
- **the pixel set is view state**, in `st.session_state`, never in a hash — which is the
  registry's rule (`fusion_ui/core/registry.py:41-44`) and is *correct here* because the thing
  that actually varies the answer, the pixel, is in each individual run's own hash.

**Known limitation, state it and move on:** N pixel-runs are N distinct `params_hash` values, so
`multishot.distinct_sources` lists them as N separate sources on the Multi shot page. This
feature does not make the multi-shot scatter aggregate over a rectangle. A follow-up would be a
pixel-agnostic source grouping there; out of scope.

---

## 1. New module: `fusion_ui/core/multipixel.py`

Mirrors `core/multishot.py` (pure logic, unit-testable) with the `core/decimate.py`
`zoomable_trace` precedent for the few functions that draw (import `streamlit` *inside* the
function, never at module top).

### Pure

```python
PIXEL_FIELDS = ("refx", "refy")

def supported(spec) -> bool
    """True if spec is cached and a refx/refy pair exists in its params tree."""
```

Eligible today: `taud_psd`, `velocity_tde`, `two_dca`, `velocity_contour`, `fwhm_sizes`,
`gaussian_sizes`, `velocity_2dca_tde`, `trajectories`, `two_sided_exp`. Correctly excludes
`raw_frames` and `probe_trace` (live) and `velocity_field` (no `refx`/`refy` — deliberately,
see `fusion_ui/plots/velocity_field.py:81-90`).

```python
def with_pixel(params, x, y)
    """A deep copy of params with refx/refy stamped everywhere they appear."""
```

This is `precompute._set_pixel` (`fusion_ui/core/precompute.py:108-115`) promoted, plus a
`deepcopy` so the caller's instance is not mutated. **Refactor `precompute.default_params` to
call it** — one implementation, and `fusion-ui precompute --pixel X Y` keeps working unchanged.

```python
def pixels_from_points(points) -> list[tuple[int, int]]
    """(x, y) pairs off a Plotly selection event's customdata, de-duplicated, sorted."""

@dataclass(frozen=True)
class Estimate:
    total: int
    cached: int
    to_compute: int
    seconds_per_pixel: float | None
    eta_seconds: float | None

def median_seconds(conn, plot) -> float | None
    """Median runs.seconds over this plot's successful runs. None if it has never run."""

def estimate(conn, spec, target, params, pixels) -> Estimate
```

`estimate` checks the ledger only — `store.find_run` per pixel, no file opened, the same trick
`precompute.run` uses (`fusion_ui/core/precompute.py:137-145`). For a chained spec it **walks
`spec.requires` and adds each link's own median** for pixels that would miss it; without that
the 2DCA chain reads "~2 min" when it means "~30 min", because `runs.seconds` on the
downstream excludes its upstream by design (`fusion_ui/core/store.py:320-322`).

### Drawing

```python
def selector(ds, target, height=380) -> list[tuple[int, int]]
def run_all(conn, spec, target, params, ds, pixels) -> list[tuple[tuple, object, object]]
def view(conn, spec, target, params, ds) -> None
```

**`selector`** — one `go.Scatter` of markers at every pixel's `(R, Z)` from
`loader.pixel_grid(ds)` (index axes when there is no R/Z), `customdata=[x, y]` per marker,
`dragmode="select"`, `selection_mode=["box", "lasso"]`, `on_select="rerun"`.

Two things learned the hard way on `fix/selection-round-2` and that must be honoured here:

- a **markers** trace does report `selection["points"]` for a box, unlike a lines-only trace
  (`scatter/select.js` bails on a trace with neither markers nor text) — so
  `decimate.selection_points(event)` is the right reader and no y-range counterpart to
  `selected_shape_range` is needed;
- the widget's element id includes the figure spec, so the stored selection resets whenever
  the figure changes. On receiving a selection, **write the pixel set to session state and bump
  a generation counter in the chart key**, exactly as `decimate._apply_window` does
  (`fusion_ui/core/decimate.py:201-213`), or the previous rectangle will overwrite the new one.

Selected pixels are redrawn in a second highlighted trace. Alongside: the count, a **Select
all** and a **Clear** button.

State keys, both off `Target.key` per the convention:
`f"pixels.{target.key}"` → `list[(x, y)]`, `f"pixelgen.{target.key}"` → int.

**`run_all`** — the loop. Per pixel:
`store.result(conn, spec, target, with_pixel(params, x, y), ds)`, driving `st.progress` and a
"pixel 7/24 (x=6, y=4)" caption itself. A pixel whose run comes back `failed` is collected and
reported at the end, never fatal — the `velocity_field` philosophy ("one NaN pixel among many",
`fusion_ui/plots/velocity_field.py:110-127`) applied at the run level. Each pixel commits to the
DB as it finishes, so a browser refresh mid-run loses only the pixel in flight.

*Note:* this blocks the Streamlit script thread, as every Compute already does. Phase 05's
`ProcessPoolExecutor` is the real fix; nothing here makes it harder, since `compute` stays pure.

**`view`** — assembles the mode: selector → estimate line → **Run on N pixels** button →
`run_all` → figure. The built figure is stashed in session state under a key derived from
`(spec.key, target.key, params_hash, pixel tuple)` so ordinary reruns redraw without reloading
N blobs; a changed selection or parameter set invalidates it by key.

---

## 2. `PlotSpec` gains one optional field

`fusion_ui/core/registry.py`:

```python
overlay: Optional[Callable] = None   # (items, params, target) -> go.Figure
```

`items` is `[((x, y), result), …]` in selection order. The spec builds the whole figure — it
owns its axis titles, its log scales, its legend — and stays pure and testable, the same
contract `render` has. `register()` validates that `overlay` is only set on a cached spec
(a live spec has no per-pixel result to overlay).

Rejected: merging the traces out of N `spec.render` figures. Several specs draw into Streamlit
and return `None` (`two_dca`, `velocity_field`, and the slider-carrying renders), titles are
per-pixel, and colours would collide.

### `plots/spectra.py` gets the worked overlay

```python
def overlay(items, params, target):
    """Every selected pixel's PSD and its fit on one log-log axis."""
```

Per pixel: the Welch PSD solid and the fit dashed, sharing one colour from a cycle and one
`legendgroup`, so clicking the legend hides both. Legend name `"(x=6, y=4)  τ_d=3.1e-5"`.
Title `"Duration time at N pixels"`. ~25 lines, mirrors `spectra.render`
(`fusion_ui/plots/spectra.py:104-132`).

This is the only overlay written in this change. Adding another is one function plus one
`overlay=` line — `two_sided_exp` and `fwhm_sizes` are the obvious next two.

### Universal fallback for specs with no `overlay`

`multipixel.scalar_figure(items, spec)` — calls `spec.scalars(result)` per pixel (pure), keeps
the `(x, y, name)` entries, and draws value against pixel with a name selectbox. So all nine
eligible specs work from day one: `velocity_contour` gives you `vx_c` across the rectangle,
`fwhm_sizes` gives you `lr`/`lz`, without either needing an overlay function.

A spec with neither `overlay` nor `scalars` falls back to a table of which pixels succeeded,
with a pointer to single-pixel mode.

---

## 3. `fusion_ui/pages/2_single_shot.py`

The page stays thin — the whole change is about six lines in `main()`:

```python
spec = pick_spec(target.diagnostic)
...
window_caption(target)

if multipixel.supported(spec) and st.sidebar.radio(
        "Pixels", ["One", "Many"], horizontal=True, key=f"mode.{spec.key}") == "Many":
    params = params_ui.form(spec.params, f"params.{spec.key}",
                            container=st.sidebar, spec=spec, ds=ds)
    multipixel.view(ui.get_connection(), spec, target, params, ds)
    return

params, ready = params_ui.panel(spec, target, ds=ds)   # unchanged below here
```

Two points:

- **loose widgets, not `st.form`.** In many-pixel mode the gate is the "Run on N pixels" button,
  so the params form is drawn with `params_ui.form` (the shape `panel` already uses for live
  specs, `fusion_ui/core/params_ui.py:579-583`). The estimate then updates live as you drag the
  rectangle or change a parameter. The key prefix `params.{spec.key}` is shared with
  single-pixel mode, so parameters carry across the toggle — which is what you want.
- **`refx`/`refy` are still drawn and are overridden.** They stay visible (hiding them means
  special-casing the params walk) with a sidebar caption saying the pixel selection overrides
  them in this mode.

---

## 4. Tests

New `tests/test_multipixel.py` — pure, no Streamlit runtime:

- `supported()` over the real registry: the nine eligible specs in, `raw_frames` /
  `probe_trace` / `velocity_field` out.
- `with_pixel` on a top-level spelling (`TaudPsdParams`) and a nested one
  (`ContourVelocityParams.two_dca`); asserts the caller's instance is **not** mutated.
- `precompute.default_params` still stamps the pixel after the refactor (guards the CLI).
- `pixels_from_points` on a hand-built selection event, including de-duplication.
- `estimate` against a tmp DB with some runs pre-seeded: cached/to_compute counts, and that a
  chained spec's ETA includes its upstream.
- `spectra.overlay` on two synthetic result Datasets: trace count, legend groups, log axes.
- `scalar_figure` fallback on a spec with `scalars` but no `overlay`.

End-to-end in `tests/test_plots_roundtrip.py` style, using the `blob_dataset_path` fixture:
run `taud_psd` over four pixels through a tmp DB, assert four `runs` rows, eight `scalars` rows
(`taud_psd` + `lambda_psd` per pixel), and that a second pass computes nothing.

Smoke test in `tests/test_app.py`: `AppTest` on the single-shot page with the mode radio set to
"Many", asserting no exception and that the selector renders.

---

## 5. Docs

- `CLAUDE.md` — `PlotSpec` contract gains `overlay`; a line on multi-pixel mode under the
  Streamlit conventions, including that the pixel set is view state.
- `README.md` — one paragraph in "Adding a plot" on the optional `overlay`.
- `docs/PLAN.md` — a short subsection under phase 05.
- `open_issues.md` — the TODO moves out of "TODOS for later".

---

## Verification

```bash
.venv/bin/pytest                      # full suite, currently 251 passing
.venv/bin/streamlit run fusion_ui/app.py
```

In the browser, on shot 1160616027 (preprocessed, apd):

1. Single shot → plot **Duration time (PSD fit)** → Pixels: **Many**.
2. Drag a rectangle over ~6 pixels. Confirm the count and the highlight follow the drag, and
   that dragging a *second*, different rectangle replaces the first (the generation-counter bug).
3. The estimate reads "6 pixels · 0 cached · ~Ns". Press **Run on 6 pixels**; the progress bar
   advances; six PSDs land on one log-log axis with a per-pixel legend.
4. Press it again — "6 cached", instant, identical figure.
5. Switch to **One**, pick one of those six pixels: it draws from cache, no compute.
6. Switch the plot to **Blob velocity (contour tracking)**, same rectangle: the estimate should
   be visibly larger (it includes the 2DCA upstream) and the fallback scalar view draws `vx_c`
   across the rectangle.
7. `sqlite3 $FUSION_UI_DB 'select plot, count(*) from runs group by plot'` — six `taud_psd`
   rows, and for step 6 both `velocity_contour` and `two_dca` rows.

## Out of scope

- Making the Multi shot page aggregate a rectangle into one point (the N-sources issue above).
- Overlays for specs other than `taud_psd` — the mechanism plus the fallback covers them.
- Moving the loop off the script thread; that is phase 05's process pool.
- `open_issues.md` issue 1 (data overview + articles).

## Follow-up: the live pixel trace

After this plan shipped, the frame viewer (`raw_frames`) joined Many mode as
the one live spec with an `overlay`. `register()` no longer rejects `overlay`
on a live spec: its items carry the open time-sliced dataset (the same object
`render` gets) instead of stored results, and `multipixel.view` draws it
straight away with no estimate and no run button. `supported()` is true for a
live spec exactly when it defines one; the scalar fallback stays cached-only.
