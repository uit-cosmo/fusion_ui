<!--
The text of the Documentation page (fusion_ui/pages/6_documentation.py). The page reads this file at every
rerun, so an edit shows on the next one, with no restart. Reword freely; keep the markers below.

- A `## ` heading starts a section of the page, in this order. `### ` and deeper are headings inside it.
- A line holding only {{table:NAME}} is a table generated from the code: products, settings, scalars or cuts.
  {{diagram}} is the dependency diagram, and {{checked}} the commits the text was checked against.
- {{value:NAME}} inside a sentence is a number read from the code, and {{default:PATH}} the default of a
  setting, so that the text cannot drift from what runs. A name the page does not know shows as written and
  is flagged on the page; a test fails on it.
- A `### ` heading made only of scalar names in backticks is the entry of those names. The page puts each
  name's label, unit and sources under it. Every scalar name has exactly one entry, and a test checks it.
- Every account of a computation names the code it describes, as `file: function`. Where the code and a
  docstring or the paper disagree, the text says what the code does.
-->

What each quantity in Shot Explorer is, how it is computed, and what it depends on. Each account names the
code it describes, as `file: function`; where a docstring or the paper says otherwise, it follows the code.
The tables are generated from the code. The text is `fusion_ui/data/documentation.md`, and can be reworded
there.

{{checked}}

## Conventions

- **R and Z.** R is the major radius and Z the vertical position. *Radial* means along R, *poloidal* along Z.
- **Pixels.** A pixel is named by its column x and row y in the array, counted from 0: "pixel (5, 4)" is
  column 5, row 4. In the C-Mod APD files x grows with R and y with Z, so a row of the array runs radially and
  a column poloidally.
- **A structure's own axes.** The two ellipse fits (`lx_c`, `ly_c`, `theta_c` and `lx_f`, `ly_f`, `theta_f`)
  measure sizes along the ellipse's own axes, which its tilt turns away from R and Z. The tilt θ is the angle
  from the R axis to the lx axis, turning towards Z. Both fits order the axes so that lx is the shorter. Both
  give half-widths, a semi-axis or the distance at which a Gaussian falls to 1/e, where the FWHM sizes `lr`
  and `lz` are full widths along R and Z.
- **Three meanings of x and y.** A pixel's column and row; the axes of a fitted ellipse, which is what "size
  in x" and "size in y" mean in a label; and R and Z in the older names `vx_c`, `vy_c` and their kin.
- **Units.** SI throughout: m, s, m/s, m², rad. The APD files hold R and Z in centimetres. The three products
  convert them to metres as they read the record (`fusion_ui/plots/_pipeline.py: record`), so their
  velocities come out in m/s. The older single-shot specs compute in centimetres and divide their results by
  100. Lags are stored in seconds; the Fields page shows them in µs.
- **Lags and signs.** The lag τ is the time since an event peaked at the reference pixel. The conditional
  average at τ is the frame τ after each peak, averaged over the events, and the cross-correlation at τ pairs
  every pixel at time t + τ with the reference at t (`imaging_methods/cond_av.py: find_events_and_2dca`). A
  structure moving outward is therefore at larger R at positive lags. v_R is positive outward, towards larger
  R, and v_Z positive upward, towards larger Z.
- **Amplitudes.** The preprocessed record is in units of each pixel's running standard deviation (next
  section). The 2DCA threshold and the amplitude of a conditional average are in those units.

## From the raw file to the record

### The preprocessed file

The raw file `apd_<shot>.nc` holds the digitizer's voltage at every pixel, `frames` (y, x, time), and each
pixel's R and Z. The products read the preprocessed file, `apd_<shot>_preprocessed.nc`, which
`density_scan/preprocess.py: preprocess_shot` (fusion_scripts) makes from the raw file:

1. **Running normalisation** (`experimental_database/preprocessing.py: run_norm_ds`, through
   `fppanalysis.run_norm`). Each pixel's series has its running mean subtracted and is divided by its running
   standard deviation, both over {{value:preprocess.window}} samples, about 1 ms at the APD's 0.5 µs
   sampling. Fluctuations slower than that are taken out. The first and last {{value:preprocess.trimmed}}
   samples of the record have no complete window and are dropped. A sample that comes out not finite is set
   to 0.
2. **Dead pixels** (`experimental_database/preprocessing.py: interpolate_nans_3d`). Each pixel of the run
   day's mask (below) is replaced, sample by sample, by linear interpolation between the live pixels around
   it on the grid of pixel indices. A dead pixel outside the live pixels' hull takes its nearest live pixel's
   value.
3. The frames are stored as float32 and cut to the discharge window.
4. The result is cut again, to the analysis window (below). The normalisation in step 1 ran before this cut,
   so it saw the same neighbouring samples as without it.

Steps 1 to 3 are `PlasmaDischargeManager.preprocess_dataset` (`experimental_database/discharge_manager.py`),
run as it is; step 4 is `preprocess_shot`'s own.

Beside the frames the file keeps the mask it was made with (`dead`), the shot's own verdict before the run
day's rule (`dead_shot`), the evidence for it (`dead_evidence`, `dead_psd_ratio`), and attributes saying
how and when: `dead_mask_source`, `analysis_window`, `discharge_window`, `puff_rule`, `dead_thresholds`,
`preprocess_radius`, `fusion_scripts_commit` and `created`. The stored-mask view of the Single shot page
draws them.

**Older files.** The nine 1160616 files predate this step. They were made by
`density_scan/data_processing.py: preprocess_data`, through an earlier version of `preprocess_dataset` that
kept the frames in float64, with the hand-made 1160616 mask (`density_scan/dead_pixel_mask.py`), over the
discharge window and with no cut to the puff. They store no mask, and the analyses use the hand-made one for
them (`decorrelation/pipeline.py: dead_mask`). Any other file that stores no mask is refused: it may have
been made with another run day's mask, and must be preprocessed again.

### The analysis window

The discharge window, `t_start..t_end` in the discharge database, often starts before the gas puff, while
the live pixels still read the dark level; on many shots it is the whole record. That dark stretch distorts
every statistic taken over the window. `density_scan/puff.py: find_puff` cuts the window to the puff:

1. The signal is the mean over the pixels of the raw frames, smoothed by a running mean over
   {{value:puff.smooth}}.
2. The *dark level* is the median of the signal's first {{value:puff.baseline}}, and the *light level* its
   median over the discharge window. The threshold is halfway between them. Which side of the dark level the
   light is on is read off the data, as the side of the window's largest excursion; on C-Mod more light reads
   higher.
3. When the median is less than {{value:puff.clear_rise}} of the way from the dark level to that excursion,
   most of the window is dark after a short puff. The window's {{value:puff.fallback}} is then the light
   level, provided that at this level the rise is clear and the start of the record flat (step 5).
4. The puff runs from the first point at which the signal crosses the threshold and stays across it for at
   least {{value:puff.min_on}}, to the last point of the last such stretch. The **analysis window** is the
   overlap of the puff with the discharge window. A dip below the threshold inside it, as between two puffs,
   is reported in `puff_rule` and not cut. A cut shorter than {{value:puff.smooth}} is not made.
5. The discharge window is kept whole, and `puff_rule` says why, when the record does not start dark (its
   first {{value:puff.baseline}} vary by more than {{value:puff.flat_start}} of the rise from the dark to the
   light level), when there is no clear rise, when the signal never stays across the threshold for
   {{value:puff.min_on}}, or when the puff overlaps the window by less than {{value:puff.min_span}}.

**What the analyses read.** Shot Explorer slices every record to the discharge window as the discharge
database gives it now (`fusion_ui/core/loader.py: time_window`; a shot without one is read over a centred
{{value:loader.default_window}}). A file cut to the analysis window lies inside it, so the analyses read the
analysis window; the older 1160616 files are read over their discharge window.

The window a result was computed over is not stored with the result. If the discharge database's window is
edited later, the stored result and a recompute disagree, and nothing marks the stored one stale (L10 in
`docs/PHASE_06_DECORRELATION.md`).

### Dead pixels

A pixel is *dead* when its channel carries no plasma signal. `density_scan/dead_pixels.py: estimate_shot`
judges every pixel of a raw file over the shot's analysis window (`estimate`):

1. **Red spectrum.** The pixel's Welch power spectrum gives a ratio: its median over {{value:dead.red_band}}
   divided by its median over {{value:dead.noise_band}}, where only digitizer noise is left. Noise is white,
   a ratio near 1, and plasma fluctuations are stronger at low frequency. A ratio above {{value:dead.red}}
   makes the pixel live.
2. **Following a live neighbour.** A pixel that fails step 1 is still live when its {{value:dead.blob_band}}
   signal follows an already-live pixel among its eight neighbours: the regression gain of its band-passed
   signal on the neighbour's, their correlation times the ratio of their standard deviations, exceeds
   {{value:dead.gain}}. This repeats until no pixel joins. It keeps dim pixels whose own fluctuations are
   weak.
3. Every other pixel is dead, and so is a pixel whose samples are not all finite or do not vary.

The run day's mask (`consolidate`) marks a pixel dead on every shot of the day when it is dead on at least
{{value:dead.day_fraction}} of them, since the hardware does not change within a day. Run day 1160616 uses
the hand-made mask instead, which this estimate reproduces.

Two views of the Single shot page show the evidence. The dead-pixel view, on the raw file, draws every
pixel's PDF and spectrum against the shot's verdict, and the array-mean signal with the discharge and
analysis windows marked. The stored-mask view, on the preprocessed file, draws the mask the file was made
with. Both carry the method in full.

**What the analyses do with a dead pixel.**

- `pixel_averages` never takes it as the reference (`computed` is False there). `method_fields` and
  `blob_parameters` are NaN there and write no scalar for it. Both take the mask from the bank, with its
  source (`dead`, `dead_mask_source`), and never compute it again.
- It still appears in every frame, with the value preprocessing interpolated. The conditional averages,
  contours and maximum tracks pass over it smoothly; it adds nothing of its own.
- Neither TDE uses it as a neighbour.
- The Fields page marks it as dead (a grey square), apart from a live pixel whose method failed (a red
  cross).

### The record the products read

`pixel_averages`, `method_fields` and `blob_parameters` read preprocessed files only. Each takes the file
sliced to the discharge window, with R and Z in metres and the whole record loaded into memory
(`fusion_ui/plots/_pipeline.py: record`, which refuses an R that is not a major radius in centimetres). The
physics is `decorrelation/pipeline.py` in fusion_scripts. The paper's own runs
(`decorrelation/apd_check/cmod_scan.py`) call the same functions, so the app and the paper read the same
numbers.

## How the quantities depend on one another

Each arrow runs from an input to what is computed from it. Blue boxes are data, white boxes computations,
yellow notes settings, and the green boxes are where results are read. A setting belongs to the cache key of
every result it reaches, so changing one gives new results beside the old ones and never overwrites them.

{{diagram}}

## The products

Three results per shot, written to netCDF in the result store. The Fields page reads them, and the Multi
shot page reads the scalars they write. Their variable names are those of the paper's own files, so the
paper's code could read them as they are.

{{table:products}}

### pixel_averages

For each live pixel in turn as the reference, `decorrelation/pipeline.py: average` runs the
two-dimensional conditional average (2DCA), `imaging_methods/cond_av.py: find_events_and_2dca`, on the
record:

1. **Events.** An event is a run of consecutive samples in which the reference's signal exceeds
   `averages.threshold`. Its peak is the run's largest sample.
2. An event is dropped unless, at its peak, the reference is the largest pixel of the square
   `averages.check_max` pixels around it, cut at the array's border; 0 turns this off.
3. An event too near the start or end of the record for a whole window is dropped.
4. With `averages.single_counting`, events are taken highest peak first, and one that peaks less than a
   window from an event already taken is dropped.
5. Each event contributes the frames around its peak: `averages.window` samples, widened by one when even,
   so that a window of n samples gives the lags −⌊n/2⌋ to +⌊n/2⌋ samples.

The bank stores, for every reference, at every pixel and lag:

- `cond_av`, the **conditional average**: the mean over the events of the frame at each lag;
- `cond_repr`, the **conditional representativeness**: `cond_av`² divided by the mean over the events of
  the squared frame. It is 1 where every event has the same value, and smaller as they scatter;
- `cross_corr`, the **cross-correlation** of each pixel with the reference over the whole record, not over
  the events: both series standardised over the record, multiplied, summed and divided by the number of
  samples, at the same lags as the average.

Per reference it stores `nevents`, the number of events kept (0 for a live reference without any, whose
average is all NaN), and `computed`, False at dead references; and the mask, `dead`. The settings are
stamped in the attribute `averages`.

The bank is the expensive step: for every reference, `find_events_and_2dca` correlates every pixel with the
reference over the whole record. That is why it is computed once per shot and settings, in batch, and
everything else reads it back.

### method_fields

`decorrelation/pipeline.py: fields` reads seven velocities at every live pixel: three 2DCA tracks through
the pixel's own conditional average, and three- and two-point time-delay estimation (TDE) twice, off the
record and on the average. A pixel without events has no tracks and no TDE on the average; the TDE off the
record needs no events.

**The three tracks** (`pipeline.TRACKS`, `pipeline.track`) follow the structure through the reference's
average lag by lag, and fit a velocity to its path.

| suffix | field | position at each lag |
|---|---|---|
| `max` | `cond_av` | its maximum, refined to a fraction of a pixel |
| `com` | `cond_av` | the centroid of its contour at the level `level_com` |
| `2dcc` | `cross_corr` | its maximum, refined to a fraction of a pixel |

- **Maximum** (`imaging_methods/maximum_trajectory.py: compute_maximum_trajectory_da`). The brightest pixel
  of the frame. Away from the array's border, a parabola through it and its two neighbours moves the
  position by at most half a pixel, along the row and along the column separately, and only in a direction
  in which the brightest pixel exceeds both its neighbours. R and Z are interpolated linearly at the result.
  On the border the brightest pixel's own position is kept. A frame whose largest value is negative takes the
  position of the nearest lag that has one.
- **Centroid** (`imaging_methods/contours.py: get_contour_evolution`). The contour of `cond_av` at
  `level_com` times the largest value of `cond_av` over all lags and pixels: one absolute level for every
  lag, so the contour shrinks as the average decays and is gone once the average is below it. At each lag
  the contour is found by marching squares (`skimage.measure.find_contours`). Of several, the one enclosing
  the most signal is kept, the sum of `cond_av` at the pixels inside it (`compute_contour_mass`). The
  position is the centroid of the area the contour encloses: the geometric centre of the polygon, not
  weighted by the average's values. A contour cut by the array's border is closed by a straight line between
  its ends.
- **The contour level** (`twodca_manuscript/contour_level.py: neighbour_level`) is read off the pixel's own
  average, not set. In the frame at lag 0, take the brightest pixel and its four neighbours
  `tracking.neighbour_step` pixels away along the row and the column, fewer on the border. The level is the
  mean of the neighbours' values divided by the brightest pixel's. The contour at lag 0 therefore reaches, on
  average, one step out from the peak. The brightest pixel is normally the reference: `check_max` makes the
  reference the largest of its neighbourhood in every event.

**The fit** (`decorrelation/pipeline.py: track`, through
`imaging_methods/velocity_estimates.py: get_combined_mask` and `get_averaged_velocity_from_position`):

1. A lag at which the tracker found no position is filled by linear interpolation between the lags around
   it, extrapolated at the ends, and the track is smoothed over `tracking.position_filter.window_size` lags with
   a `tracking.position_filter.window_type` window (`imaging_methods/utils.py: smooth_da`). A window of 1
   leaves the track as it is.
2. The slope rests on the lags at which the position is within `tracking.position_filter.mask_distance`
   pixel pitches of the reference pixel (the mean pitch along R) and the field's largest value exceeds
   `tracking.position_filter.mask_signal_factor` times its largest value over all lags. For the 2DCC
   `tracking.cross_corr_mask_signal_factor` replaces the latter, because the cross-correlation sits on a
   pedestal well above zero. With `tracking.position_filter.require_within_boundaries` on, the centroid track
   also needs a contour that closes inside the array. Of the lags that pass, only the longest run of
   consecutive ones is kept, the earlier of two equally long.
3. The velocity is the slope of R and of Z against the lag over those lags: the least-squares slope when
   `tracking.velocity.estimator` is `lsq`, the mean of the centred differences when it is `central_diff`.
   It is NaN when no lag passes, and with `lsq` when fewer than two of them have a position.

For each track T the product stores `vr_T` and `vz_T`, `nlags_T` (the number of lags the slope rests on),
`level_T` (the contour level; NaN for the two maximum tracks), `pos_r_T` and `pos_z_T` (the position at every
lag, NaN where the tracker found nothing) and `fit_T` (true at the lags the slope rests on). A lag filled in
step 1 can be in `fit_T`, and counted in `nlags_T`, with no position.

**TDE off the record** (`decorrelation/pipeline.py: tde_fields`, with `velocity_estimation`'s
cross-correlation method, `TDEMethod.CC`). The delay from a pixel to a neighbour is the lag at which the
cross-correlation of the neighbour's series with the pixel's, over the whole record, peaks
(`velocity_estimation/time_delay_estimation.py: estimate_time_delay_ccf`):

- the cross-correlation is normalised, and its peak searched within 100 samples of zero lag;
- a peak below `tde.min_cc` gives no delay;
- with `tde.running_mean`, a cross-correlation with more than one local maximum above half the largest is
  smoothed by running means of 3, 5 and then 7 samples until it has one; if 7 do not do it, or it has no
  local maximum, there is no delay;
- with `tde.interpolate`, the delay is the maximum of a quartic spline through the cross-correlation rather
  than the sample at its peak.

The neighbours are the pixels on either side along the row and along the column, where they are on the
array and live. Each pair of one row neighbour and one column neighbour gives a velocity: four pairs inside the
array, two on an edge, one in a corner. The pixel's velocity is the mean over the pairs that gave one
(`velocity_estimation/two_dim_velocity_estimates.py: estimate_velocities_for_pixel`):

- **three-point** (`vr3_tde`, `vz3_tde`; `get_2d_velocities_from_time_delays`): the delays to the row
  neighbour (τ_r) and to the column neighbour (τ_z) are solved together for the velocity of one structure
  crossing the three pixels. On a rectangular grid with pitches Δr and Δz this reads
  v_R = Δr τ_r / (τ_r² + (Δr/Δz)² τ_z²) and v_Z = Δz τ_z / ((Δz/Δr)² τ_r² + τ_z²); the code uses the
  pixels' actual positions;
- **two-point** (`vr2_tde`, `vz2_tde`; `get_1d_velocities_from_time_delays`): each component on its own,
  v_R = Δr/τ_r and v_Z = Δz/τ_z, and 0 where a delay is exactly 0.

`cc_tde` is the mean, over the pairs used, of the smaller of the two cross-correlation peaks of a pair.

**TDE on the conditional average** (`decorrelation/pipeline.py: ca_tde`). The same two estimates, with each
delay read off the pixel's own `cond_av` instead of off the record: the lag at which `cond_av` peaks at the
pixel and at each live neighbour, the maximum of a quartic spline through the lags
(`density_scan/peak_time.py: get_maximum_time`). The delay along the row is the mean of t(x+1) − t(x) and
t(x) − t(x−1), over the neighbours there are; the delay along the column likewise. The pitches are read off
the array's first pixels. The two-point estimate is NaN, not 0, for a delay of exactly 0.

### blob_parameters

`decorrelation/pipeline.py: blobs` (`pixel_blob` at each pixel) gives thirteen numbers at every live pixel,
from the pixel's own conditional average at lag 0 and from its series in the record:

- `nevents`, the events behind the average, which `method_fields` writes as the scalar `number_events`;
- `level`, the contour level, read off the average as `level_com` is, with `blob_parameters`' own
  `neighbour_step`;
- `area`, `lx_c`, `ly_c`, `theta_c`, from the contour of `cond_av` at lag 0 at that level;
- `lr`, `lz`, the full widths at half maximum along the reference's row and column;
- `lx_f`, `ly_f`, `theta_f`, a tilted Gaussian fitted to `cond_av` at lag 0;
- `taud`, `lam`, the duration time and asymmetry of the pulses, fitted to the power spectrum of the pixel's
  series. They are written as the scalars `taud_psd` and `lambda_psd`.

Each is NaN where its estimate fails. The duration time is computed even at a live pixel without events. The
entries under *Every scalar name* say how each is computed.

### The older single-shot specs

Before phase 06, each quantity had a spec of its own on the Single shot page, computed at one reference
pixel, most of them from a `two_dca` run. Four write names the products also write (`two_dca`,
`taud_psd`, `fwhm_sizes`, `gaussian_sizes`), and so does the seed imported from `density_scan/results.json`
(`density_scan_import`).
They use the same functions with other settings, except where an entry below says otherwise. Those built on
`two_dca` use a 2DCA threshold of {{value:older.two_dca_threshold}} by default, where the products use
`averages.threshold` ({{default:averages.threshold}} by default), and `velocity_contour` and
`trajectories` read a contour at a fixed {{value:older.contour_level}} of the average's maximum rather than
at a level read off each average. The Multi shot page lists each as its own source of a name.

## The settings

A setting is a parameter of a product, and part of its cache key. Changing one gives new results beside the
old ones and never overwrites a result. A change to `averages` gives a new bank, and with it new
`method_fields` and `blob_parameters`; a change to anything else gives new results for one product,
computed in minutes from the bank already stored.

{{table:settings}}

The defaults are the twodca deck's C-Mod settings (`twodca_manuscript/datasets/cmod.py: params` and `SPEC`,
which `decorrelation/pipeline.py` reads), the same for every shot. A shot they do not suit shows up as
failed pixels; it is not retuned.

To compute with other settings, name the part that changes in a JSON file, such as
`{"averages": {"window": 30}}`, and run `fusion-ui precompute method_fields blob_parameters --shot N
--params-json params.json`. The Single shot page computes `method_fields` and `blob_parameters` with other
settings itself once their bank is stored. The Fields page offers every `method_fields` parameter set the
ledger has.

**Not settings.** The contour level is read off each pixel's average, and only `neighbour_step` is set. The
reference pixel is every live pixel in turn. The Fields page's cuts are view state: moving them recomputes
nothing.

## Every scalar name

A scalar is one number per pixel, or one per shot, kept in the ledger; the Multi shot page plots it across
shots. A name can be written by several sources, a product, an older spec or the seed, each with its own
settings, and the Multi shot page lists the sources apart. Its mean, median and maximum over pixels take
every pixel that has a value: the Fields page's cuts do not apply there. In the products a dead pixel has no
row, and a live pixel whose estimate failed has an empty value, which means tried and failed.

{{table:scalars}}

### `vr_max`, `vz_max`, `nlags_max`

The 2DCA maximum track: the slope of the position of the maximum of the pixel's own `cond_av` against the
lag, in R and in Z, and the number of lags the slope rests on (*method_fields* above). The maximum moves by
whole pixels, refined by at most half a pixel. Where a structure passes close to the centre of the
reference pixel, the maximum can stay on that pixel for several lags, and the slope then reads low. The
paper's figures and its `reliable()` cut use the centroid track for this reason
(`decorrelation/apd_check/figures.py: TRACK`).

### `vr_com`, `vz_com`, `nlags_com`, `level_com`

The 2DCA centroid track: the slope of the centroid of the contour of `cond_av` against the lag, in R and in
Z, the number of lags the slope rests on, and the level of the contour.

`level_com` is **not a setting**. It is read off the pixel's own average before tracking
(`contour_level.neighbour_level`): the mean value of the four pixels `tracking.neighbour_step` from the peak
of the frame at lag 0, as a fraction of the peak. Only `tracking.neighbour_step` is set
({{default:tracking.neighbour_step}} by default). `blob_parameters` writes the same quantity as `level`; the
two agree while both products use the same neighbour step, as they do at the defaults.

- The centroid is the centre of the area inside the contour, not weighted by the average. A contour cut by
  the array's border is closed by a straight line, and its centroid then follows the border rather than the
  structure. `tracking.position_filter.require_within_boundaries` drops such lags from the fit, and is off by
  default.
- `nlags_com` counts the lags at which the contour was lost and the position interpolated, as the paper
  does; `fit_com` includes them, and `pos_r_com`, `pos_z_com` are NaN there.
- These are not the older `velocity_contour`'s `vx_c`, `vy_c`, which track a contour at a fixed fraction of
  the maximum.

### `vr_2dcc`, `vz_2dcc`, `nlags_2dcc`

The 2DCC track: the slope of the maximum of the pixel's `cross_corr`, found as for the 2DCA maximum, and the
lags it rests on. The cross-correlation is taken over the whole record, not over the events, so it does not
depend on `averages.threshold`, `averages.check_max` or `averages.single_counting`; `averages.window` sets
the lags it spans. A pixel without events has no 2DCC all the same, since the bank keeps nothing for it.
Its fit uses `tracking.cross_corr_mask_signal_factor`
({{default:tracking.cross_corr_mask_signal_factor}} by default): the cross-correlation sits on a pedestal
well above zero, where a fraction chosen for the decaying average would never bind. The Fields page does not
cut it by events, and colours its arrows by the pixel's events all the same.

### `number_events`

The number of events in the pixel's conditional average: `nevents` in all three products, written once, by
`method_fields`. The older `two_dca` spec and the seed write the same count for averages made at their own
threshold: {{value:older.two_dca_threshold}} by default in `two_dca`, and 2 in the seed
(`density_scan/data_processing.py: compute_and_store_conditional_averages`), where the products use
`averages.threshold`, {{default:averages.threshold}} by default. The counts of two sources differ wherever
their thresholds do.

### `vr3_tde`, `vz3_tde`, `vr2_tde`, `vz2_tde`, `cc_tde`

TDE off the record: v_R and v_Z by three-point and by two-point time-delay estimation, from the
cross-correlations of the pixel's series with its neighbours' over the whole record (*method_fields* above),
and the cross-correlation peak they rest on.

- It conditions on nothing. Every fluctuation in the record counts, where the tracks of the conditional
  average follow only the events.
- The two-point estimate divides a pitch by a delay, so a component whose delay is near zero is very large,
  and at a delay of exactly 0 it is set to 0. The three-point estimate solves both delays together and stays
  finite; it is NaN at the whole pixel when any neighbour pair has both delays 0.
- `cc_tde` is the mean over the neighbour pairs used of the smaller of each pair's two cross-correlation
  peaks, after the running mean. A pair whose peak is below `tde.min_cc` ({{default:tde.min_cc}} by default)
  gives no delay, and does not count.
- On the array's edge fewer pairs are averaged, and a dead neighbour is not used.
- These are not the older `velocity_tde`'s `vx_tde`, `vy_tde`, which take their delays from a conditional
  average instead (below).

### `vr3_catde`, `vz3_catde`, `vr2_catde`, `vz2_catde`

TDE on the conditional average: the same three- and two-point estimates, with each delay read off the
pixel's own `cond_av`, as the lag at which it peaks at the pixel and at its live neighbours (*method_fields*
above). The average is the 2DCA tracks' own, so a difference from them comes from how the average is read,
not from which events went into it. The two-point estimate is NaN for a delay of exactly 0. The older
`velocity_2dca_tde` computes the three-point estimate the same way, on a `two_dca` average (below).

### `level`

The contour level of `blob_parameters`, read off the pixel's average by the same rule as `level_com`
(`contour_level.neighbour_level`), with `blob_parameters`' own `neighbour_step`
({{default:neighbour_step}} by default). Not a setting. It sets the contour that `area`, `lx_c`, `ly_c` and
`theta_c` are read from.

### `area`

The area enclosed by the contour of `cond_av` at lag 0, at `level` times the largest value of `cond_av` over
all lags (`get_contour_evolution`): the area of the polygon, in m². Of several contours, the one enclosing
the most signal is kept. A contour cut by the array's border is closed by a straight line, so a structure
larger than the view has the area of its part inside the view. This is not the older `area_c`, which is read
at a fixed {{value:older.contour_level}} of the maximum.

### `lx_c`, `ly_c`, `theta_c`

An ellipse fitted to the points of the contour behind `area` (`imaging_methods/contours.py:
fit_ellipses_to_contour_ds`, scikit-image's `EllipseModel`, a least-squares fit of the points' distances to
the ellipse): its semi-axes, the shorter first, and the angle from the R axis to `lx_c`, turning towards Z,
between 0 and π. These are half-lengths. The fit needs five contour points. A contour cut by the array's
border is fitted on its points inside the view, without the straight line that closes it.

### `lr`, `lz`

The full width at half maximum of `cond_av` at lag 0, `lr` along the reference's row (R) and `lz` along its
column (Z): the distance between the half-maximum points on the two sides of the reference
(`imaging_methods/parameter_estimation.py: estimate_fwhm_sizes`). The half maximum is half the reference
pixel's own value. On each side the profile is followed outward from the reference for as long as it does
not rise, and the half point is interpolated linearly along that falling stretch.

- **The interpolation leaves out the last point of the falling stretch.** A side whose profile has not
  fallen to half by the second-to-last point is held at that point, so its width is underestimated, and a
  side that falls for a single pixel and then rises counts as 0. A side that rises at once is NaN.
- The older `fwhm_sizes` and the seed use the same function.

### `lx_f`, `ly_f`, `theta_f`

A tilted Gaussian fitted to `cond_av` at lag 0, centred on the reference pixel
(`imaging_methods/parameter_estimation.py: fit_ellipse_to_event`, `fit_ellipse`): exp(−(x′/lx)² −
(y′/ly)²), with x′ and y′ the distances from the reference along axes turned by θ from R and Z. lx and ly are
the distances at which the Gaussian falls to 1/e, the shorter first, and θ is the angle from the R axis to
the lx axis, turning towards Z, between 0 and π.

- **A penalised fit.** It minimises the squared difference from the average plus the Gaussian's summed square
  times `size_penalty + tilt_penalty·θ² + aspect_penalty·(1 − lx/ly)²` (`blobs.gauss_fit`), by Nelder–Mead
  from four starting points, with lx and ly at most `blobs.gauss_fit.size_max` (unset: the larger extent of
  the array). The size penalty grows with the Gaussian's footprint, so the sizes are the penalty's answer as
  much as the average's: on a synthetic structure the default penalty gave less than half the unpenalised
  size (`docs/PLAN.md`, phase 03).
- **The amplitude is fixed at 1**, while `cond_av` is in running standard deviations and peaks above the
  2DCA threshold. The centre is the reference pixel's position, not fitted.
- **The seed's values under these names were not this fit, and were removed on 2026-10-08.**
  `density_scan/utils.py: analysis` stores the ellipse fitted to the contour at 0.3 of the maximum
  (`get_contour_parameters`) as `lx_f`, `ly_f` and `theta_f`, in every version that wrote `results.json`,
  while the docstring of `density_scan/discharge.py: BlobParameters` says Gaussian fit. Recomputed from the
  seed's own averages, its `lx_f` and `ly_f` matched that contour ellipse to a few percent and were far from
  the Gaussian fit, and its `theta_f` was in an older angle convention. Two quantities under one name would
  have shared an axis, so the three were deleted from the ledger (`fusion_ui/core/store.py:
  prune_scalars`, through `fusion-ui prune --plot density_scan_import --scalar lx_f --scalar ly_f --scalar
  theta_f`) and `fusion-ui import-results` no longer writes them (`fusion_ui/core/seed.py: NOT_IMPORTED`).
  The seed's other twelve names stay. On the Multi shot page these three names have no seed source any
  more.
- The older `gaussian_sizes` runs this fit, with the same defaults, on a `two_dca` average.

### `taud_psd`, `lambda_psd`

The duration time τ_d and the asymmetry λ of the pulses, from a fit to the power spectrum of the pixel's own
series over the whole record (`imaging_methods/duration_time_estimation.py:
DurationTimeEstimator.estimate_duration_time`; `blob_parameters` stores them as `taud` and `lam`):

1. The series is standardised and its Welch spectrum taken in segments of `blobs.taud_estimation.nperseg`
   samples. The zero frequency is dropped, and the frequencies are converted to angular frequency ω and kept
   below `blobs.taud_estimation.cutoff` rad/s.
2. The spectrum is fitted with that of a train of two-sided exponential pulses,
   S(ω) = 4τ_d / ([1 + ((1 − λ)τ_dω)²][1 + (λτ_dω)²]) (`imaging_methods/utils.py: power_spectral_density`),
   by least squares on its values rather than their logarithms, with differential evolution from a fixed
   seed. A pulse rises over λτ_d and decays over (1 − λ)τ_d.

- **λ and 1 − λ give the same spectrum**, so the fit cannot tell a fast rise from a fast decay. Read λ and
  1 − λ as the same answer.
- The whole record counts, not only the events, and the value is written even at a pixel without events.
- The running normalisation of preprocessing takes out fluctuations slower than its window of
  {{value:preprocess.window}} samples, and the fit cannot see them.
- The `taud_psd` spec and the seed use the same function with the same defaults. The `taud_psd` spec also
  runs on raw files, which are not normalised; the Multi shot page lists those results as a source of their
  own.

## Names only the single-shot specs and the seed write

These names come from the specs of the Single shot page and from the seed. No product writes them, and they
have no label: the Multi shot page shows the name.

### `vx_c`, `vy_c`, `area_c`

`velocity_contour` (`fusion_ui/plots/velocity_contour.py: compute`) and the seed: v_R, v_Z and the area of
the contour-centroid track of a `two_dca` average at one reference pixel. In these names x means R and y means
Z. They differ from `vr_com`, `vz_com` and `area` in their settings: the contour at a fixed
{{value:older.contour_level}} of the average's maximum (`contouring.threshold_factor`), the track smoothed
over {{value:older.window_size}} lags, the fit's mask at {{value:older.mask_signal_factor}} of the maximum,
the velocity as the mean of centred differences ({{value:older.estimator}}), and the older 2DCA threshold.
`area_c` is the contour's area at lag 0. The seed's come from `density_scan/utils.py:
get_contour_parameters`, the same computation, and agree with today's to a few percent: they predate a fix
to the contours on non-uniform grids in imaging_methods.

### `vx_2dca_tde`, `vy_2dca_tde`

`velocity_2dca_tde` (`fusion_ui/plots/velocity_2dca_tde.py: compute`) and the seed
(`density_scan/utils.py: get_2dca_tde_velocities`): the three-point TDE on a `two_dca` average at one
reference pixel, as `vr3_catde`, `vz3_catde`, but with the older 2DCA threshold and with every neighbour on
the array, dead or live.

### `vx_tde`, `vy_tde`

`velocity_tde` (`fusion_ui/plots/velocity_tde.py: compute`) and the seed
(`density_scan/utils.py: get_tde_velocities`): three-point TDE at one pixel, with each delay read off a
cross-conditional average (`velocity_estimation`'s `TDEMethod.CA`): the neighbour's series averaged around
the peaks at which the pixel's own exceeds {{value:older.tde_threshold}}, in the record's units. A neighbour
is used only where its cross-correlation with the pixel peaks at least {{value:older.tde_ccf_min_lag}} sample
from zero lag, and the hand-made 1160616 mask marks the dead ones, on every shot. These are not `vr3_tde`,
`vz3_tde`, which take their delays from the cross-correlation. How the conditional average picks its events
changed between `velocity_estimation` versions, so the seed's values need not match a recompute.

### `vx_2dca_lsq`, `vy_2dca_lsq`, `vx_ccf_lsq`, `vy_ccf_lsq`

`trajectories` (`fusion_ui/plots/trajectories.py: compute`): least-squares slopes of two tracks through a
`two_dca` average at one reference pixel. `_2dca_` is the centroid of the contour of `cond_av` at a fixed
{{value:older.contour_level}} of its maximum, and `_ccf_` the maximum of `cross_corr`. Both are smoothed over
{{value:older.window_size}} lags, with the fit's mask at {{value:older.mask_signal_factor}} of the maximum by
default. x means R and y means Z.

### `tau_prime`, `sigma_t`, `l_prime`, `sigma_sp`

`two_sided_exp` (`fusion_ui/plots/two_sided_exp.py: compute`): fits of an asymmetric two-sided exponential,
offset + (1 − offset)·exp(−|x|/ℓ), with ℓ = scale·(1 − σ) on the negative side and scale·σ on the positive
side (`waveform_analysis: fit_two_sided`), to two cuts of a `two_dca` average, each divided by the
reference's value at lag 0. `tau_prime` (s) and `sigma_t` are the scale and the split σ of the reference
pixel's average against the lag; `l_prime` (m) and `sigma_sp` those of the frame at lag 0 along the
reference's row. A cut that does not peak at the reference, or a fit resting on a bound, is NaN. The scale
depends on how far the cut extends, `tau_prime` on the 2DCA window and `l_prime` on the array, so neither
compares across settings or cameras.

### `dead`, `psd_ratio`, `number_dead`

`dead_pixels` (`fusion_ui/plots/dead_pixels.py: scalars`), on a raw file: at each pixel, 1 where the shot's
own verdict is dead and 0 where it is live, and the ratio of the red-spectrum test (NaN where the pixel has
no data); per shot, the number of dead pixels. They are the shot's verdict before the run day's rule (*Dead
pixels* above). The mask a preprocessed file was made with is in the file, not in the ledger.

## The Fields page's cuts

The Fields page draws every method's field for one shot from the stored products and computes nothing. Its
cuts choose which pixels the panels draw as arrows. They are view state, not settings: moving one recomputes
nothing and changes no stored number.

{{table:cuts}}

- **Minimum lags** applies to the three 2DCA tracks. A slope through two or three lags is a secant, not a
  fit.
- **Minimum events** applies to the methods read off the conditional average: the 2DCA maximum and centroid,
  and the TDEs on it. The 2DCC and the TDEs off the record do not depend on the events.
- **Interior pixels only** leaves the array's border out of every panel.
- **The paper's cut** keeps the same pixels in every panel: those whose centroid track rests on at least the
  minimum of lags and events, whose centroid and both TDEs off the record gave a velocity, and which are not
  on the border (`fusion_ui/views/reliable.py: rule`, which a test holds equal to
  `decorrelation/apd_check/figures.py: reliable`).

A pixel the cuts leave out keeps its value in the product and in the ledger, and the Multi shot page's means
include it. All seven panels share one arrow scale, set by the three 2DCA tracks. The arrows of the TDEs off
the record are coloured by `cc_tde`, and all others by the pixel's events.
