"""The MLP profiles: the reduction (``fusion_ui/core/mlp.py``), the view (``fusion_ui/plots/mlp_profiles.py``) and the
single-shot page that shows it.

What these tests hold them to, each on a synthetic file in the layout ``cmod_functions`` writes (``tests/mlp_fixtures.py``),
whose profiles are pure exponentials of known lengths:

- the plunges are found on the probe's depth, at their tips, and a bump of the parked probe or a dwell is no scan; every
  1 ms window, on whole milliseconds of the shot, is labelled with its plunge and the way the probe ran, and the windows
  where it held still make no profile;
- a pin's mean counts only where it recorded half the window, ρ's zeros are no mapping, and the four pins are combined as
  their mean, only where every pin recorded;
- the far-SOL fits give back the lengths the profiles were made with, and the shift to a separatrix temperature puts it at
  ρ = 0 and never extrapolates;
- the figure draws six rows on ρ in cm, a colour per plunge and a marker per way, the fits dashed and the separatrix
  dotted; a window with no plunge is said in words beside the probe's depth;
- the single-shot page draws it, with the discharge window on a probe file.
"""

import base64
import dataclasses
import inspect
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
import xarray as xr
from streamlit.testing.v1 import AppTest

import fusion_ui.plots  # noqa: F401 - registers the specs the page offers
from fusion_ui import documentation
from fusion_ui.core import catalog, db, loader, mlp, multipixel, params_ui, registry
from fusion_ui.plots import mlp_profiles as mp
from tests import mlp_fixtures as mf
from tests.test_thomson_profiles import reaches_fusion_scripts

REPO = Path(__file__).resolve().parent.parent
SINGLE_SHOT = str(REPO / "fusion_ui" / "pages" / "2_single_shot.py")

#: The discharge window of the fixture's shot, which holds the third plunge, in and out, as 1160616011's does.
WINDOW = (1.15, 1.45)


def profiles_of(window=WINDOW, te_sep=0.0, **options):
    return mlp.read(mf.asp_dataset(**options), *window, te_sep=te_sep)


def rows_of(fig):
    """``{row: [trace, …]}``, row 1 the top, in the order the traces were added."""
    found = {}
    for trace in fig.data:
        axis = trace.yaxis or "y"
        found.setdefault(1 if axis == "y" else int(axis[1:]), []).append(trace)
    return found


def markers(fig):
    return [trace for trace in fig.data if trace.mode == "markers"]


# ---------------------------------------------------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------------------------------------------------


def test_the_quantities_and_pins_are_read_off_the_names_and_a_rho_companion_is_none_of_them():
    ds = mf.asp_dataset()
    assert mlp.layout(ds) == {quantity: [0, 1, 2, 3] for quantity in mlp.QUANTITIES}
    assert mlp.pins(ds) == [0, 1, 2, 3]
    assert mlp.layout(mf.asp_dataset(quantities=("Te", "ne"))) == {"ne": [0, 1, 2, 3], "Te": [0, 1, 2, 3]}
    assert mlp.pins(mf.asp_dataset(drop_pin=2)) == [0, 1, 3]


def test_the_track_is_read_from_its_arrays_or_their_text_and_nothing_else_passes_for_one():
    time, depth = mlp.track(mf.asp_dataset())
    assert time.size == depth.size > 20000 and np.all(np.diff(time) > 0)
    assert depth.max() == pytest.approx(mf.TIP)
    text_time, text_depth = mlp.track(mf.asp_dataset(as_text=True))
    assert np.allclose(text_time, time, atol=1e-6) and np.allclose(text_depth, depth, atol=1e-8)

    assert mlp.track(mf.asp_dataset(track=False)) is None
    cut_short = mf.asp_dataset().assign_attrs(plunge_time="[0.1 0.2 ... 2.0]", plunge_depth="[0.0 0.1 ... 0.2]")
    assert mlp.track(cut_short) is None  # a repr cut short does not hold the array
    unpaired = mf.asp_dataset().assign_attrs(plunge_depth=np.zeros(10))
    assert mlp.track(unpaired) is None


def test_the_origin_is_r_z_and_phi_and_none_without_one():
    assert mlp.origin(mf.asp_dataset()) == pytest.approx((0.976224, 0.110719, 3.12), abs=1e-6)
    ds = mf.asp_dataset()
    del ds.attrs["probe_origin"]
    assert mlp.origin(ds) is None


# ---------------------------------------------------------------------------------------------------------------------
# The plunges
# ---------------------------------------------------------------------------------------------------------------------


def test_each_plunge_is_found_at_its_tip_between_the_shallowest_points_around_it():
    found = mlp.plunges(*mlp.track(mf.asp_dataset()))
    assert [p.number for p in found] == [1, 2, 3]
    assert [p.tip for p in found] == pytest.approx(mf.TIPS, abs=1e-3)
    assert all(p.depth == pytest.approx(mf.TIP) for p in found)
    assert found[0].start == pytest.approx(-0.04)  # the first runs from the start of the record
    assert found[-1].end == pytest.approx(2.0999, abs=1e-3)  # and the last to its end
    for before, after in zip(found, found[1:]):
        assert before.end == after.start  # they meet at the shallowest point between them
        assert before.tip < before.end < after.tip


def test_a_bump_of_the_parked_probe_is_no_plunge():
    """1150618021's probe settles at its parking position with bumps of 1.1 to 1.2 cm, which are not plunges."""
    time = np.arange(0.0, 1.0, 1e-4)
    knots = [0, 0.2, 0.3, 0.4, 0.6, 0.65, 0.7, 1.0]
    depth = np.interp(time, knots, [-0.014, -0.014, 0.094, -0.0195, -0.014, -0.0035, -0.014, -0.014])
    found = mlp.plunges(time, depth)
    assert [round(p.tip, 2) for p in found] == [0.3]


def test_a_dwell_is_one_plunge_whose_held_windows_are_on_no_stroke():
    ds = mf.asp_dataset(dwell=True)
    (plunge,) = mlp.plunges(*mlp.track(ds))
    found = mlp.means(ds, 0.2, 1.8)
    held = (found["time"].values > 0.55) & (found["time"].values < 1.45)
    assert set(found["stroke"].values[held]) == {""}
    assert set(found["stroke"].values[(found["time"].values > 0.31) & (found["time"].values < 0.5)]) == {"in"}
    assert set(found["stroke"].values[(found["time"].values > 1.51) & (found["time"].values < 1.7)]) == {"out"}
    assert set(found["plunge"].values) == {plunge.number}


# ---------------------------------------------------------------------------------------------------------------------
# The 1 ms means
# ---------------------------------------------------------------------------------------------------------------------


def test_the_windows_are_whole_milliseconds_of_the_shots_clock_inside_the_page_window():
    edges = mlp.edges(1.305, 1.45)
    assert edges.size == 146 and edges[0] == pytest.approx(1.305) and edges[-1] == pytest.approx(1.45)
    assert mlp.edges(1.3054, 1.45)[0] == pytest.approx(1.306)  # only whole windows, wherever the page's begins
    assert np.allclose(np.diff(edges), 1e-3)
    assert mlp.edges(1.0, 1.0005).size == 0


def test_each_window_is_labelled_with_its_plunge_and_the_way_the_probe_ran():
    found = mlp.means(mf.asp_dataset(), *WINDOW)
    time, stroke, plunge = found["time"].values, found["stroke"].values, found["plunge"].values
    going_in = (time > 1.22) & (time < 1.29)
    going_out = (time > 1.31) & (time < 1.38)
    assert set(stroke[going_in]) == {"in"} and set(stroke[going_out]) == {"out"}
    assert set(plunge[going_in | going_out]) == {3}
    assert set(stroke[(time > 1.16) & (time < 1.21)]) == {""}  # at the base depth between plunges, it holds still
    assert found["speed"].values[going_in] == pytest.approx(mf.SPEED, rel=1e-3)
    assert found["speed"].values[going_out] == pytest.approx(-mf.SPEED, rel=1e-3)


def window_of(found, middle):
    """The index of the window whose middle is ``middle``, and its two edges as :func:`fusion_ui.core.mlp.edges` makes
    them: a sample on the upper edge belongs to the next window."""
    k = int(np.flatnonzero(np.isclose(found["time"].values, middle))[0])
    edges = mlp.edges(*WINDOW)
    return k, edges[k], edges[k + 1]


def test_each_quantity_is_the_mean_of_the_window_and_rho_the_mean_at_its_samples_pin_by_pin():
    ds = mf.asp_dataset()
    found = mlp.means(ds, *WINDOW)
    k, low, high = window_of(found, 1.2505)
    for column, pin in enumerate(found["pin"].values):
        time = ds[f"time_ne_{pin}"].values.astype(float)
        inside = (time >= low) & (time < high)
        assert inside.sum() == 20  # every sample of the window, 50 µs apart
        expected = ds[f"ne_{pin}"].values[inside].astype(float).mean()
        assert found["ne"].values[k, column] == pytest.approx(expected, rel=1e-9)
        rho_time = ds[f"rho_time_ne_{pin}"].values.astype(float)
        rho = ds[f"rho_ne_{pin}"].values.astype(float)
        at_samples = np.interp(time[inside], rho_time, rho).mean()
        assert found["rho"].values[k, column] == pytest.approx(at_samples, rel=1e-9)
        # The companion's own samples sit 25 µs earlier on average, which on a stroke is 12 µm of ρ.
        own = rho[(rho_time >= low) & (rho_time < high)].mean()
        assert abs(own - at_samples) == pytest.approx(mf.RHO_PER_DEPTH * mf.SPEED * 25e-6, rel=0.05)
    # The fixture's pins sample the same profile: its mean at the mean ρ of the samples, to the curvature over a window.
    rho = found["rho"].values[k, 0]
    assert found["Te"].values[k, 0] == pytest.approx(mf.TE0 * np.exp(-rho / mf.L_T), rel=1e-4)


def test_where_a_pin_did_not_record_rho_is_the_mean_of_its_own_samples_in_the_window():
    ds = mf.asp_dataset()
    found = mlp.means(ds, *WINDOW)
    k, low, high = window_of(found, 1.1805)  # at the base depth between plunges, where the MLP does not record
    assert np.isnan(found["ne"].values[k]).all()
    rho_time = ds["rho_time_ne_0"].values.astype(float)
    inside = (rho_time >= low) & (rho_time < high)
    assert found["rho"].values[k, 0] == pytest.approx(ds["rho_ne_0"].values[inside].astype(float).mean())
    assert found["rho"].values[k, 0] == pytest.approx(mf.RHO_BASE)


def test_a_pin_that_recorded_less_than_half_a_window_has_no_mean_there():
    ds = mf.asp_dataset()
    pin = 3
    found = mlp.means(ds, *WINDOW)
    k, low, high = window_of(found, 1.2505)
    time = ds[f"time_ne_{pin}"].values.astype(float)
    lost = (time >= low + 0.0003) & (time < low + 0.0009)  # the pin recorded 0.4 ms of the window
    thinned = ds.assign({f"ne_{pin}": (f"time_ne_{pin}", np.where(lost, np.nan, ds[f"ne_{pin}"].values))})
    found = mlp.means(thinned, *WINDOW)
    assert np.isnan(found["ne"].values[k, pin]) and np.isfinite(found["ne"].values[k, 0])
    assert np.isfinite(found["ne"].values[k + 1, pin])
    # Half a window is enough.
    half = (time >= low + 0.0003) & (time < low + 0.0008)
    halved = ds.assign({f"ne_{pin}": (f"time_ne_{pin}", np.where(half, np.nan, ds[f"ne_{pin}"].values))})
    assert np.isfinite(mlp.means(halved, *WINDOW)["ne"].values[k, pin])


def test_rhos_zeros_are_no_mapping_and_are_left_out_of_a_windows_rho():
    ds = mf.asp_dataset()
    rho = ds["rho_ne_0"].values.copy()
    rho_time = ds["rho_time_ne_0"].values.astype(float)
    found = mlp.means(ds, *WINDOW)
    held, held_low, held_high = window_of(found, 1.1805)  # the pin does not record here
    moving, low, high = window_of(found, 1.2505)  # and here it does
    zeros = []
    for start, end in ((held_low, held_high), (low, high)):
        inside = np.flatnonzero((rho_time >= start) & (rho_time < end))
        zeros += list(inside[2:5])  # three of the window's ten samples read 0
    rho[zeros] = 0.0
    found = mlp.means(ds.assign({"rho_ne_0": ("rho_time_ne_0", rho)}), *WINDOW)

    kept = (rho != 0.0) & (rho_time >= held_low) & (rho_time < held_high)
    assert found["rho"].values[held, 0] == pytest.approx(rho[kept].astype(float).mean())
    time = ds["time_ne_0"].values.astype(float)
    samples = time[(time >= low) & (time < high)]
    mapped = rho != 0.0
    expected = np.interp(samples, rho_time[mapped], rho[mapped].astype(float)).mean()
    assert found["rho"].values[moving, 0] == pytest.approx(expected, rel=1e-9)
    assert found["rho"].values[moving, 0] == pytest.approx(mlp.means(ds, *WINDOW)["rho"].values[moving, 0], rel=1e-3)


def test_the_pins_are_combined_as_their_mean_only_where_every_pin_recorded():
    ds = mf.asp_dataset()
    found = mlp.means(ds, *WINDOW)
    combined = mlp.combine(found)
    every = np.isfinite(found["ne"].values).all(axis=1)
    some = np.isfinite(found["ne"].values).any(axis=1) & ~every
    assert some.any(), "the fixture's pins start and stop recording apart"
    assert np.isnan(combined["ne"].values[some]).all()
    assert combined["ne"].values[every] == pytest.approx(found["ne"].values[every].mean(axis=1))
    # The pins' factors average to one, so the mean of the four is the profile itself.
    rho = combined["rho"].values[every]
    assert combined["ne"].values[every] == pytest.approx(mf.NE0 * np.exp(-rho / mf.L_N), rel=2e-4)
    assert combined["Vf"].values[every] == pytest.approx(-0.5 * combined["Te"].values[every], rel=1e-4)
    assert set(combined["pins"].values[every]) == {4}


def test_a_quantity_that_fewer_pins_hold_is_the_mean_of_those():
    ds = mf.asp_dataset(drop_pin=2)
    combined = mlp.combine(mlp.means(ds, *WINDOW))
    drawn = mlp.on_strokes(combined)
    factor = np.mean([mf.PIN_FACTOR[pin] for pin in (0, 1, 3)])
    assert drawn["ne"].values == pytest.approx(factor * mf.NE0 * np.exp(-drawn["rho"].values / mf.L_N), rel=2e-4)


def test_only_the_windows_on_a_stroke_are_read_and_they_are_read_the_same():
    ds = mf.asp_dataset()
    every = mlp.means(ds, *WINDOW)
    some = mlp.means(ds, *WINDOW, strokes_only=True)
    on = some["stroke"].values != ""
    span = slice(np.flatnonzero(on)[0], np.flatnonzero(on)[-1] + 1)
    for quantity in mlp.QUANTITIES:
        assert np.array_equal(some[quantity].values[span], every[quantity].values[span], equal_nan=True)
    assert np.isnan(some["ne"].values[: span.start]).all()

    dwell = mlp.means(mf.asp_dataset(dwell=True), *WINDOW, strokes_only=True)
    assert not np.isfinite(dwell["ne"].values).any()  # nothing read: the probe never moved
    assert np.isfinite(dwell["rho"].values).all()  # where it was is still known


def test_the_windows_the_mlp_recorded_on_a_stroke_are_counted_with_those_some_pin_missed():
    found = mlp.means(mf.asp_dataset(), *WINDOW)
    recorded, partial = mlp.recording(found)
    counts = np.isfinite(found["ne"].values).sum(axis=1)
    moving = found["stroke"].values != ""
    assert recorded == int((moving & (counts > 0)).sum()) and 0 < partial < recorded
    assert partial == int((moving & (counts > 0) & (counts < 4)).sum())


# ---------------------------------------------------------------------------------------------------------------------
# The far-SOL e-folding lengths, and the separatrix
# ---------------------------------------------------------------------------------------------------------------------


def test_the_far_sol_fits_give_back_the_lengths_the_profiles_were_made_with():
    profiles = profiles_of()
    fits = mlp.far_sol(profiles.drawn)
    assert fits["ne"].length == pytest.approx(mf.L_N, rel=1e-3)
    assert fits["Te"].length == pytest.approx(mf.L_T, rel=1e-3)
    assert fits["ne"].amplitude == pytest.approx(mf.NE0, rel=1e-3)
    rho = profiles.drawn["rho"].values
    assert fits["ne"].points == int(((rho >= 0.005) & (rho <= 0.018)).sum()) > 20
    assert (fits["ne"].lo, fits["ne"].hi) == mlp.FAR_SOL == (0.005, 0.018)


def test_a_fit_needs_five_points_and_a_profile_that_falls_off():
    rho = np.linspace(0.006, 0.017, 4)
    few = mlp.efold(rho, np.exp(-rho / 0.03))
    assert not few.ok and np.isnan(few.length) and "4 points between ρ 0.5 and 1.8 cm" in few.reason
    rho = np.linspace(0.006, 0.017, 10)
    rising = mlp.efold(rho, np.exp(rho / 0.03))
    assert not rising.ok and "does not fall off" in rising.reason
    fit = mlp.efold(rho, 5.0 * np.exp(-rho / 0.03))
    assert fit.ok and fit.at(0.0) == pytest.approx(5.0) and fit.length == pytest.approx(0.03)


def test_the_shift_puts_the_separatrix_temperature_at_rho_zero():
    profiles = profiles_of(te_sep=30.0)
    expected = mf.L_T * np.log(30.0 / mf.TE0)  # T_e = 60 exp(-ρ/2 cm) reaches 30 eV at ρ = 1.39 cm
    assert profiles.shift.ok and profiles.shift.by == pytest.approx(expected, rel=1e-3)
    assert profiles.shift.points > 2 * mlp.MIN_SIDE


def test_the_shift_never_extrapolates_and_says_why_not():
    drawn = profiles_of().drawn
    te = drawn["Te"].values
    beyond = mlp.shift_to(drawn["rho"].values, te, float(te.max()) * 1.5)
    assert not beyond.ok and beyond.by == 0.0 and "never reaches" in beyond.reason
    below = mlp.shift_to(drawn["rho"].values, te, float(te.min()) * 0.9)
    assert not below.ok and "above" in below.reason and "everywhere" in below.reason
    sparse = mlp.shift_to(np.array([0.0, 0.001, 0.002, 0.003]), np.array([40.0, 35.0, 25.0, 20.0]), 30.0)
    assert not sparse.ok and "too few" in sparse.reason
    assert not mlp.shift_to(drawn["rho"].values, te, 0.0).ok


# ---------------------------------------------------------------------------------------------------------------------
# What another view needs of the reduction: whether the probe plunges in a window, from the track alone, and the same
# profiles from a file opened lazily (the Thomson scattering view draws them over its own)
# ---------------------------------------------------------------------------------------------------------------------


def test_the_strokes_in_a_window_are_those_the_means_label_judged_from_the_track_alone():
    ds = mf.asp_dataset()
    for window in (WINDOW, (0.6, 1.45), (1.305, 1.45), (1.33, 1.36)):
        assert mlp.strokes_in(ds, *window) == mlp.strokes(mlp.means(ds, *window)), window
    assert mlp.strokes_in(ds, *WINDOW) == [(3, "in"), (3, "out")]
    assert mlp.strokes_in(ds, 1.305, 1.45) == [(3, "out")]  # 1160616009's window begins after the tip
    # The attributes are all it reads: a dataset that holds nothing else gives the same answer.
    track_only = xr.Dataset(attrs=ds.attrs)
    assert mlp.strokes_in(track_only, *WINDOW) == [(3, "in"), (3, "out")]


def test_no_stroke_is_found_where_the_probe_holds_still_or_the_file_has_no_track():
    assert mlp.strokes_in(mf.asp_dataset(dwell=True), *WINDOW) == []
    assert mlp.strokes_in(mf.asp_dataset(track=False), *WINDOW) == []
    assert mlp.strokes_in(mf.asp_dataset(), 0.05, 0.25) == []  # parked, before the first plunge
    assert mlp.strokes_in(mf.asp_dataset(), 1.0, 1.0005) == []  # no whole millisecond


def test_without_a_window_the_strokes_of_the_whole_track_are_found():
    ways = [(n, way) for n in (1, 2, 3) for way in ("in", "out")]
    assert mlp.strokes_in(mf.asp_dataset()) == ways
    assert mlp.strokes_in(mf.asp_dataset(), float("nan"), 1.0) == ways


def test_a_file_opened_lazily_gives_the_profiles_one_opened_whole_gives(tmp_path):
    """Its axes are not indexed (``loader.lazily``), so the reduction reads each one to find the window in it, and then
    only the window of the values: what it finds must be what the index would have found."""
    path = mf.write(tmp_path, mf.asp_dataset())
    lazy = loader.lazily(str(path))
    assert not lazy.indexes
    with xr.open_dataset(path) as whole:
        assert whole.indexes  # every axis indexed, as the single-shot page opens a probe file of its own
        for window in (WINDOW, (1.305, 1.45)):
            ours, theirs = mlp.read(lazy, *window, te_sep=30.0), mlp.read(whole, *window, te_sep=30.0)
            assert ours.means.identical(theirs.means)
            assert ours.combined.identical(theirs.combined)
            assert ours.shift == theirs.shift and ours.shift.ok
        for name in ("ne_0", "rho_ne_0"):
            assert [a.tolist() for a in mlp._series(lazy, name, *WINDOW)] == [
                a.tolist() for a in mlp._series(whole, name, *WINDOW)
            ]
    lazy.close()


# ---------------------------------------------------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------------------------------------------------


def test_the_spec_is_registered_live_for_asp_files_after_the_probe_trace():
    spec = registry.get("mlp_profiles")
    assert spec.label == "MLP profiles" and spec.diagnostics == ("asp",)
    assert not spec.cached and spec.compute is None and spec.scalars is None and spec.overlay is None
    assert [s.key for s in registry.for_diagnostic("asp")] == ["probe_trace", "mlp_profiles"]
    for other in ("apd", "phantom", "fsp", "thomson_scattering"):
        assert spec not in registry.for_diagnostic(other)
    assert not multipixel.supported(spec)


def test_the_parameters_hash_and_their_fit_range_is_the_modules_far_sol():
    spec = registry.get("mlp_profiles")
    params = spec.params()
    assert [f.name for f in dataclasses.fields(params)] == ["te_sep", "fit_from", "fit_to"]
    assert (params.te_sep, params.fit_from, params.fit_to) == (0.0, 0.5, 1.8)
    digest, text = params_ui.hash_params(spec.key, params)
    assert params_ui.hash_params(spec.key, params_ui.from_canonical(spec.params, text, plot_key=spec.key))[0] == digest


def test_neither_module_imports_anything_from_fusion_scripts():
    for module in (mlp, mp):
        assert reaches_fusion_scripts(inspect.getsource(module)) == [], module.__name__


def test_the_reduction_never_imports_streamlit():
    assert "streamlit" not in inspect.getsource(mlp)


def test_every_quantity_and_both_lengths_have_a_unit_and_how_it_was_established():
    assert set(mlp.UNITS) == set(mlp.QUANTITIES) | {"rho", "depth"}
    assert all(unit and how for unit, how in mlp.UNITS.values())


# ---------------------------------------------------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------------------------------------------------


def test_six_rows_share_rho_and_each_stroke_is_one_trace_in_every_row():
    fig = mp.figure(profiles_of())
    rows = rows_of(fig)
    assert sorted(rows) == [1, 2, 3, 4, 5, 6]
    assert [fig.layout[f"yaxis{'' if k == 1 else k}"].title.text for k in rows] == [row[2] for row in mp.ROWS]
    for k, traces in rows.items():
        assert [t.name for t in traces if t.mode == "markers"] == ["plunge 3, in", "plunge 3, out"]
    shown = [t.name for t in fig.data if t.showlegend]
    assert shown == ["plunge 3, in", "plunge 3, out", "far-SOL fit"]
    assert fig.layout.xaxis6.title.text == "ρ [cm]"


def test_a_plunge_has_its_colour_on_every_shot_and_the_marker_points_the_way_the_probe_ran():
    fig = mp.figure(profiles_of(window=(0.6, 1.45)))
    seen = {(t.name, t.marker.color, t.marker.symbol) for t in markers(fig)}
    assert seen == {
        (f"plunge {n}, {way}", mp.PLUNGE_COLOURS[n], mp.STROKE_SYMBOLS[way]) for n in (1, 2, 3) for way in ("in", "out")
    }
    assert mp.STROKE_SYMBOLS == {"in": "triangle-left", "out": "triangle-right-open"}  # and filled in, open out


def test_the_values_are_drawn_in_the_units_of_the_axis_titles():
    profiles = profiles_of()
    drawn = profiles.drawn
    rows = rows_of(mp.figure(profiles))
    inward = drawn.isel(window=np.flatnonzero(drawn["stroke"].values == "in"))
    for k, (name, _, _, scale, _, _) in enumerate(mp.ROWS, start=1):
        trace = rows[k][0]
        assert np.allclose(trace.x, inward["rho"].values * 100)
        assert np.allclose(trace.y, inward[name].values * scale)


def test_the_density_temperature_and_currents_are_logarithmic_and_the_potentials_linear():
    fig = mp.figure(profiles_of())
    types = [fig.layout[f"yaxis{'' if k == 1 else k}"].type for k in range(1, 7)]
    assert types == ["log", "log", "log", "log", "linear", "linear"]


def test_the_far_sol_fits_are_drawn_dashed_over_the_points_they_rest_on_in_the_n_e_and_t_e_rows_only():
    profiles = profiles_of()
    rho = profiles.drawn["rho"].values * 100
    rows = rows_of(mp.figure(profiles))
    inside = rho[(rho >= 0.5) & (rho <= 1.8)]
    for k in (1, 2):
        (line,) = [t for t in rows[k] if t.mode == "lines"]
        assert line.line.dash == "dash"
        assert (min(line.x), max(line.x)) == pytest.approx((inside.min(), inside.max()))
        assert 0.5 <= min(line.x) < 0.55 and 1.75 < max(line.x) <= 1.8
    for k in (3, 4, 5, 6):
        assert not [t for t in rows[k] if t.mode == "lines"]
    other = mp.figure(profiles, mp.MlpProfilesParams(fit_from=0.2, fit_to=1.0))
    (line,) = [t for t in rows_of(other)[1] if t.mode == "lines"]
    inside = rho[(rho >= 0.2) & (rho <= 1.0)]
    assert (min(line.x), max(line.x)) == pytest.approx((inside.min(), inside.max()))


def test_a_fit_is_not_drawn_where_it_has_no_points():
    """1150618021's plunges reach in to ρ = 0.9 cm only, so its fit begins there and not at the range's 0.5 cm."""
    profiles = profiles_of()
    drawn = profiles.drawn
    far = drawn.isel(window=np.flatnonzero(drawn["rho"].values >= 0.009))
    fit = mlp.far_sol(far)["ne"]
    assert fit.ok and fit.first == pytest.approx(far["rho"].values.min()) and fit.first > 0.009 - 1e-12
    profiles.combined = profiles.combined.where(profiles.combined["rho"] >= 0.009)
    (line,) = [t for t in rows_of(mp.figure(profiles))[1] if t.mode == "lines"]
    assert min(line.x) >= 0.9


def test_the_separatrix_is_a_dotted_line_at_rho_zero_in_every_row_with_one_legend_entry():
    fig = mp.figure(profiles_of())
    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(lines) == 6 and {s.x0 for s in lines} == {0.0} and {s.line.dash for s in lines} == {"dot"}
    assert [s.name for s in lines if s.showlegend] == ["separatrix"]


def test_a_shift_moves_the_points_the_fits_and_the_title_but_not_the_points_the_fits_are_made_on():
    plain, moved = profiles_of(), profiles_of(te_sep=30.0)
    by = moved.shift.by * 100
    fig, fig_moved = mp.figure(plain), mp.figure(moved)
    assert np.allclose(markers(fig_moved)[0].x, np.asarray(markers(fig)[0].x) + by)
    (line,) = [t for t in rows_of(fig_moved)[1] if t.mode == "lines"]
    (unmoved,) = [t for t in rows_of(fig)[1] if t.mode == "lines"]
    assert np.allclose(line.x, np.asarray(unmoved.x) + by)
    assert 0.5 + by <= min(line.x) and max(line.x) <= 1.8 + by
    assert "moved by -1.39 cm to T<sub>e</sub> = 30 eV" in fig_moved.layout.xaxis6.title.text
    assert mlp.far_sol(moved.drawn)["ne"].length == mlp.far_sol(plain.drawn)["ne"].length


def test_a_hover_text_says_which_stroke_when_where_the_depth_and_the_value():
    profiles = profiles_of()
    fig = mp.figure(profiles)
    drawn = profiles.drawn
    inward = drawn.isel(window=np.flatnonzero(drawn["stroke"].values == "in"))
    text = markers(fig)[0].hovertext[10]
    assert text == (
        f"plunge 3, in · t = {inward['time'].values[10]:.4f} s<br>"
        f"ρ = {inward['rho'].values[10] * 100:.2f} cm (file: {inward['rho'].values[10] * 100:.2f} cm) · "
        f"depth {inward['depth'].values[10] * 100:.2f} cm<br>"
        f"n_e = {inward['ne'].values[10] * 1e-20:.3g} 10²⁰ m⁻³"
    )


# ---------------------------------------------------------------------------------------------------------------------
# What is said
# ---------------------------------------------------------------------------------------------------------------------


def test_a_window_that_holds_a_plunge_has_nothing_to_say_of_its_absence():
    assert mp.no_plunge_text(profiles_of()) is None


def test_a_dwell_is_said_in_words_with_the_probes_depth_and_where_rho_ran():
    text = mp.no_plunge_text(profiles_of(dwell=True))
    assert text.startswith("**No plunge in the window 1.150–1.450 s:**")
    rho = mf.rho_at(0.109) * 100
    assert f"held still at a depth of 10.9 cm, where ρ was {rho:.2f} cm, so there is no profile" in text
    assert "depth over the shot" in text


def test_a_file_without_a_track_says_its_plunges_cannot_be_found():
    text = mp.no_plunge_text(profiles_of(track=False))
    assert "no plunge depth" in text and "cannot be found" in text


def test_a_dwell_through_a_moving_plasma_says_how_far_rho_ran():
    ds = mf.asp_dataset(dwell=True)
    for name in [name for name in ds.data_vars if name.startswith("rho_")]:
        time = ds[ds[name].dims[0]].values.astype(float)
        ds[name] = (ds[name].dims, ds[name].values + np.float32(0.002) * np.sin(2 * np.pi * time / 0.1).astype("float32"))
    text = mp.no_plunge_text(mlp.read(ds, *WINDOW))
    assert "held still at a depth of 10.9 cm, where ρ ran from 0.2" in text and "cm as the plasma moved" in text


def test_a_plunge_that_makes_no_profile_says_why():
    ds = mf.asp_dataset()
    quiet = ds.assign({name: ds[name] * np.nan for name in ds.data_vars if not name.startswith("rho_")})
    assert "the MLP recorded nothing while it moved" in mp.no_plunge_text(mlp.read(quiet, *WINDOW))
    measured = [name for name in ds.data_vars if not name.startswith("rho_")]
    lone = ds.drop_vars([name for name in measured if name.endswith("_3")])
    lone = lone.assign({name: lone[name] * np.nan for name in measured if name.endswith("_2")})
    assert "did every pin record" in mp.no_plunge_text(mlp.read(lone, *WINDOW))
    unplaced = ds.assign({name: ds[name] * 0.0 for name in ds.data_vars if name.startswith("rho_")})
    assert "holds no ρ" in mp.no_plunge_text(mlp.read(unplaced, *WINDOW))


def test_a_window_that_begins_after_a_tip_draws_the_way_out_alone():
    """1160616009's discharge window begins at 1.305 s, after its third plunge's tip at 1.296 s."""
    strokes = mp.drawn_strokes(profiles_of(window=(1.305, 1.45)))
    assert [(number, way) for number, way, *_ in strokes] == [(3, "out")]


def test_with_no_window_the_whole_record_is_drawn_and_the_notes_say_so():
    profiles = profiles_of(window=(np.nan, np.nan))
    every = [(n, way) for n in (1, 2, 3) for way in ("in", "out")]
    assert [(n, way) for n, way, *_ in mp.drawn_strokes(profiles)] == every
    assert "over the whole record, which has no discharge-DB window" in mp.notes(profiles, mp.MlpProfilesParams())[0]


def test_the_notes_name_the_strokes_the_pins_rho_the_fits_and_the_probe():
    profiles = profiles_of()
    points, rho, fits, probe = mp.notes(profiles, mp.MlpProfilesParams())
    assert "in the window 1.150–1.450 s: plunge 3, in (1.2" in points and "plunge 3, out (1.3" in points
    assert "mean of the 4 pins (NE, SW, SE, NW)" in points and "not every pin recorded are left out" in points
    assert "mapped to the outboard midplane" in rho and "Documentation page" in rho and "te_sep" in rho
    assert "between ρ = 0.5 and 1.8 cm of the file's ρ" in fits and "L_n = 3.00 cm" in fits and "L_T = 2.00 cm" in fits
    assert probe.startswith("Mirror Langmuir Probe (MLP) on the horizontal scanning probe, at R = 0.976 m and Z = 0.111")


def test_the_notes_say_by_how_much_and_to_what_the_profiles_were_moved_or_why_not():
    moved = mp.notes(profiles_of(te_sep=30.0), mp.MlpProfilesParams(te_sep=30.0))
    assert "Moved by -1.39 cm, as the paper moves its profiles, so that T_e is 30 eV at ρ = 0" in moved[1]
    assert "between ρ = 0.5 and 1.8 cm of the file's ρ (-0.89 to 0.41 cm on the shifted axis): L_n = 3.00 cm" in moved[2]
    unmoved = mp.notes(profiles_of(te_sep=500.0), mp.MlpProfilesParams(te_sep=500.0))
    assert "Not moved to T_e = 500 eV at the separatrix: T_e never reaches 500 eV" in unmoved[1]


def test_the_depth_chart_shades_the_window_and_draws_the_depth_in_cm():
    profiles = profiles_of(dwell=True)
    fig = mp.depth_figure(profiles, mlp.track(mf.asp_dataset(dwell=True)))
    (trace,) = fig.data
    assert max(trace.y) == pytest.approx(10.9) and len(trace.x) <= 2000
    (shade,) = [s for s in fig.layout.shapes if s.type == "rect"]
    assert (shade.x0, shade.x1) == WINDOW
    assert mp.depth_figure(profiles, None).layout.annotations[0].text == "The file has no plunge depth."


# ---------------------------------------------------------------------------------------------------------------------
# The single-shot page
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def tree(monkeypatch, tmp_path):
    """A data tree for the page, and ``tree(shot, window=None, **options)`` to put one ASP file in it, with the shot's
    discharge window in a discharge DB when ``window`` is given."""
    import streamlit as st

    root = tmp_path / "alcator"
    database = tmp_path / "state" / "shot_explorer.sqlite"
    discharges = tmp_path / "plasma_discharges.json"
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(root))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(discharges))
    monkeypatch.setenv("FUSION_UI_DB", str(database))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")
    entries = []

    def install(shot=mf.SHOT, window=None, **options):
        mf.write(root, mf.asp_dataset(shot, **options), shot)
        if window is not None:
            entries.append(
                {
                    "shot_number": shot,
                    "plasma_current": 0.54,
                    "line_averaged_density": 1.07,
                    "greenwald_fraction": 0.3,
                    "t_start": window[0],
                    "t_end": window[1],
                    "mlp_mode": "dwell" if options.get("dwell") else "scan",
                    "comment": "L",
                }
            )
        discharges.write_text(json.dumps(entries))
        conn = db.open_db(database)
        catalog.rescan(conn, str(root), "cmod", str(discharges))
        conn.close()
        st.cache_data.clear()
        st.cache_resource.clear()
        return shot

    yield install
    st.cache_data.clear()
    st.cache_resource.clear()


def open_view(shot, **params):
    """The single-shot page on the MLP profiles of ``shot``'s ASP file, with any parameters set first."""
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60)
    app.session_state["selection"] = {"machine": "cmod", "shot": shot, "diagnostic": "asp", "preprocessed": False}
    app.session_state["spec.asp"] = registry.get("mlp_profiles")
    for name, value in params.items():
        app.session_state[f"params.mlp_profiles.{name}"] = value
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    return app


def decode(node):
    """A figure's JSON with Plotly's binary arrays, ``{"dtype", "bdata", "shape"}``, turned back into lists."""
    if isinstance(node, dict):
        if "bdata" in node:
            data = np.frombuffer(base64.b64decode(node["bdata"]), dtype=np.dtype(node["dtype"]))
            if "shape" in node:
                data = data.reshape([int(n) for n in node["shape"].split(",")])
            return data.tolist()
        return {key: decode(value) for key, value in node.items()}
    if isinstance(node, list):
        return [decode(value) for value in node]
    return node


def charts(app):
    return [go.Figure(decode(json.loads(c.proto.spec))) for c in app.get("plotly_chart")]


def captions(app):
    return [c.value for c in app.main.caption]


def test_the_page_draws_a_scans_profiles_in_one_chart_with_its_notes_and_the_window(tree):
    app = open_view(tree(window=WINDOW))
    assert [w.options for w in app.sidebar.selectbox if w.label == "Plot"] == [["Probe trace", "MLP profiles"]]
    (fig,) = charts(app)
    assert [t.name for t in fig.data if t.showlegend] == ["plunge 3, in", "plunge 3, out", "far-SOL fit"]
    shown = captions(app)
    window = "Window 1.1500–1.4500 s — from the discharge DB. A probe file has no shared time axis, so it is not cut to it."
    assert window in shown
    assert any(c.startswith("Points: the 1 ms means of the MLP over the plunges in the window 1.150–1.450 s") for c in shown)
    assert any("L_n = 3.00 cm" in c and "L_T = 2.00 cm" in c for c in shown)
    assert not app.warning and not app.info


def test_the_page_says_so_for_a_dwell_and_draws_the_probes_depth_instead(tree):
    app = open_view(tree(window=WINDOW, dwell=True))
    (said,) = app.info
    assert said.value.startswith("**No plunge in the window 1.150–1.450 s:**")
    (fig,) = charts(app)
    assert fig.layout.yaxis.title.text == "probe depth [cm]"


def test_the_page_moves_the_profiles_to_a_separatrix_temperature_and_warns_when_it_cannot(tree):
    shot = tree(window=WINDOW)
    app = open_view(shot, te_sep=30.0)
    assert any(c.startswith("ρ is each pin's distance") and "Moved by -1.39 cm" in c for c in captions(app))
    app = open_view(shot, te_sep=500.0)
    (warned,) = app.warning
    assert warned.value.startswith("Not moved to T_e = 500 eV: T_e never reaches 500 eV")


def test_a_shot_with_no_discharge_window_draws_every_plunge_and_says_no_window(tree):
    app = open_view(tree())
    assert not [c for c in captions(app) if c.startswith("Window")]
    (fig,) = charts(app)
    assert {t.name for t in fig.data if t.mode == "markers"} == {
        f"plunge {n}, {way}" for n in (1, 2, 3) for way in ("in", "out")
    }


def test_the_page_gives_a_probe_file_the_discharge_window_and_no_default_without_one():
    class Entry:
        t_start, t_end = 1.15, 1.45

    class Empty:
        t_start = t_end = float("nan")

    assert loader.metadata_window(Entry()) == (1.15, 1.45, "metadata")
    start, end, source = loader.metadata_window(Empty())
    assert source == "none" and np.isnan(start) and np.isnan(end)
    assert loader.metadata_window(None)[2] == "none"


# ---------------------------------------------------------------------------------------------------------------------
# The Documentation page
# ---------------------------------------------------------------------------------------------------------------------


def test_the_documentation_has_a_section_on_the_mlp_profiles_that_says_how_each_unit_was_established():
    doc = documentation.load()
    (section,) = [s for s in doc.sections if s.title == "MLP profiles"]
    text, unknown = documentation.fill(" ".join(block.text for block in section.blocks))
    assert not unknown
    for unit in ("m⁻³", "eV", "A m⁻²", "V"):
        assert unit in text
    for said in ("fusion_ui/core/mlp.py: means", "fusion_ui/core/mlp.py: shift_to", "fusion_ui/core/mlp.py: efold"):
        assert said in text
    values = documentation.facts()
    assert values["mlp.window"] == "1 ms" and values["mlp.far_sol"] == "0.5 to 1.8 cm"
