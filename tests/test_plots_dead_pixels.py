"""The dead-pixel view: the mask's evidence, every pixel's PDF and spectrum, raw files only, judged over the gas puff.

Two kinds of record. ``record`` is stationary red noise: it has no puff, so the analysis window is the discharge
window, and what is tested is the mask, the PDFs and the figures as they were. ``tests/puff_fixtures`` has the records
with a puff in them, whose discharge windows tell the whole record from a cut of it.
"""

from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import pytest
import xarray as xr
from scipy import signal
from streamlit.testing.v1 import AppTest

import fusion_ui.plots  # noqa: F401 - registers every spec
from fusion_ui.core import fusion_scripts, loader, registry
from fusion_ui.plots import dead_pixels
from tests import puff_fixtures as pf

STEP = 1e-3  # digitizer step the synthetic record is quantised to, as the raw files are
DEAD = (0, 3)
EMPTY = (2, 0)
N, DT = 60_000, 5e-7


@pytest.fixture(scope="module")
def record():
    """3 x 4 pixels sharing one red signal, but for one white-noise pixel and one with no samples."""
    rng = np.random.default_rng(1)
    phi = np.exp(-DT / 20e-6)
    shared = signal.lfilter([1], [1, -phi], rng.normal(size=N) * np.sqrt(1 - phi**2))
    frames = 0.9 * shared + 0.02 * rng.normal(size=(3, 4, N))
    frames[DEAD] = 0.01 * rng.normal(size=N)
    frames = np.round(frames / STEP) * STEP
    frames[EMPTY] = np.nan
    return xr.Dataset(
        {"frames": (("y", "x", "time"), frames.astype("float32"))},
        coords={
            "time": np.arange(N) * DT,
            "R": (("y", "x"), np.tile(88.0 + 0.4 * np.arange(4), (3, 1))),
            "Z": (("y", "x"), np.tile(0.4 * np.arange(3), (4, 1)).T),
        },
    )


@pytest.fixture(scope="module")
def result(record):
    """The view's result on the whole of the stationary record, as the page hands it: its own span is the window."""
    spec = registry.get("dead_pixels")
    whole = loader.whole_record(record, float(record.time[0]), float(record.time[-1]))
    return spec.compute(whole, spec.params())


def test_the_planted_dead_pixels_are_found(result):
    expected = np.zeros((3, 4), bool)
    expected[DEAD] = expected[EMPTY] = True
    np.testing.assert_array_equal(result.dead.values, expected)


def test_a_pdf_has_one_bin_per_digitizer_level(result):
    """A dead pixel spans a few tens of levels: binned finer it would draw as a comb."""
    centres = result.pdf_volts.values[DEAD]
    steps = np.diff(centres[np.isfinite(centres)])
    np.testing.assert_allclose(steps, STEP, rtol=1e-3)


@pytest.mark.parametrize("view", dead_pixels.VIEWS)
def test_every_view_draws_one_panel_per_pixel(result, view):
    fig = dead_pixels.figure(result, view)
    assert isinstance(fig, go.Figure)
    panels = [t for t in fig.data if t.showlegend is False]
    assert len(panels) == 12
    assert len(fig.layout.shapes) == (2 * 12 if view == "Spectrum" else 0)


def test_scalars_mark_every_pixel(result):
    out = registry.get("dead_pixels").scalars(result)
    assert out["number_dead"] == 2
    assert out[(DEAD[1], DEAD[0], "dead")] == 1.0
    assert out[(1, 1, "dead")] == 0.0
    assert sum(1 for k in out if isinstance(k, tuple) and k[2] == "psd_ratio") == 12


def test_the_summary_names_the_dead_pixels(result):
    text = dead_pixels.summary(result)
    assert text.startswith("2 of 12 pixels dead")
    assert "(0, 3)" in text


def test_it_is_offered_on_raw_files_only():
    spec = registry.get("dead_pixels")
    assert spec.preprocessed is False
    assert spec in registry.for_diagnostic("apd", preprocessed=False)
    assert spec not in registry.for_diagnostic("apd", preprocessed=True)


# ---------------------------------------------------------------------------
# The window. The puff is found on the whole record, against the dark level at its start, and the mask, the PDFs and
# the spectra are judged over what the cut leaves: the analysis window.
# ---------------------------------------------------------------------------

#: Whether the record cut to the discharge window first would have given another analysis window.
CUT_GIVES_ANOTHER = {"two puffs": True, "just before the rise": True, "from the start": False}


def test_the_dead_pixel_view_is_the_one_spec_that_needs_the_whole_record():
    """The default is unchanged for every other spec: a 500 MB record stays out of memory."""
    assert registry.get("dead_pixels").whole_record
    assert {key for key, spec in registry.REGISTRY.items() if spec.whole_record} == {"dead_pixels"}


def test_its_parameters_are_unchanged_so_the_cached_results_keep_their_keys():
    """The results cached on the server (111 shots) are keyed by this hash, and it is the hash of these two knobs
    under this class name. A parameter added, renamed or moved would orphan them all: the puff window needed none,
    being a property of the record and not a choice."""
    import json

    from fusion_ui.core import params_ui

    digest, body = params_ui.hash_params("dead_pixels", dead_pixels.DeadPixelParams())
    assert digest == "c87bebadae3c0c04ac973bf608a4d6ea0571133b"
    assert json.loads(body)["params"]["values"] == {"gain": 0.1, "red": 150.0}


def test_a_record_cut_to_its_window_is_refused(record):
    """Not guessed at: the record's own span would pass for the discharge window, and the window found from it
    would be wrong without a word."""
    spec = registry.get("dead_pixels")
    with pytest.raises(ValueError, match="carries no discharge window"):
        spec.compute(record, spec.params())


@pytest.fixture(scope="module")
def scenarios():
    """Each scenario of ``tests/puff_fixtures``, as the page hands it over (the whole record, with the discharge
    window) and computed, with the puff window found on the whole record and on the record cut to the window."""
    fusion_scripts.import_config()
    from density_scan import puff

    spec = registry.get("dead_pixels")
    out = {}
    for name, case in pf.SCENARIOS.items():
        ds = pf.record(case)
        t0, t1 = case.window
        out[name] = SimpleNamespace(
            name=name,
            case=case,
            record=ds,
            result=spec.compute(loader.input_for(ds, t0, t1, whole=spec.whole_record), spec.params()),
            whole=puff.puff_window(ds, case.window),
            cut=puff.puff_window(ds.sel(time=slice(t0, t1)), case.window),
        )
    return out


NAMES = list(pf.SCENARIOS)


@pytest.mark.parametrize("name", NAMES)
def test_the_analysis_window_is_the_one_found_on_the_whole_record(scenarios, name):
    s = scenarios[name]
    found = dead_pixels.puff_of(s.result)
    assert found.analysis_window == s.whole.analysis_window
    assert found.discharge_window == s.case.window
    assert found.found == s.whole.found
    assert found.note == s.whole.note and found.rule == s.whole.rule
    assert (found.start, found.end) == pytest.approx((s.whole.start, s.whole.end), nan_ok=True)
    assert found.dips == s.whole.dips
    # The attributes carry the names ``density_scan.dead_pixels.estimate_shot`` gives its own.
    assert list(s.result.attrs["analysis_window"]) == list(s.whole.analysis_window)
    assert list(s.result.attrs["discharge_window"]) == list(s.case.window)
    assert s.result.attrs["puff_found"] == int(s.whole.found) and s.result.attrs["puff_rule"] == s.whole.rule


@pytest.mark.parametrize("name", NAMES)
def test_a_record_cut_to_the_window_first_would_have_given_another_window(scenarios, name):
    """The whole point of handing the spec the whole record. Where the discharge window starts after the record
    does, the cut record's start is not the record's: the dark level is read from the wrong stretch, or a puff that
    came and went before the window does not exist. The control is a window that starts with the record."""
    s = scenarios[name]
    ours = dead_pixels.puff_of(s.result).analysis_window
    assert (s.cut.analysis_window != ours) is CUT_GIVES_ANOTHER[name]
    if CUT_GIVES_ANOTHER[name]:
        assert s.cut.analysis_window != s.whole.analysis_window


def test_a_window_that_starts_in_the_dark_gap_keeps_the_gap_as_a_dip(scenarios):
    """A dip below the threshold inside the window, as between two puffs, is reported and not cut. The record cut
    to the window first would not know of the first puff, and would take the gap for the dark before the puff."""
    s = scenarios["two puffs"]
    found = dead_pixels.puff_of(s.result)
    assert found.analysis_window == (0.035, 0.115)  # the discharge window: the first puff came and went before it
    assert found.dips and found.dips[0][0] == 0.035 and 0.049 < found.dips[0][1] < 0.051
    assert s.cut.analysis_window[0] == pytest.approx(0.050, abs=1e-3)  # cut to the window, the gap reads as dark


@pytest.mark.parametrize("name", NAMES)
def test_the_mask_and_its_evidence_are_those_of_the_analysis_window(scenarios, name):
    """``density_scan.dead_pixels.estimate`` on the analysis window of the same record: what ``estimate_shot`` does."""
    from density_scan import dead_pixels as scan

    s = scenarios[name]
    expected = scan.estimate(s.record.sel(time=slice(*s.whole.analysis_window)), keep_psd=True)
    for variable in ("dead", "evidence", "red_ratio", "gain"):
        np.testing.assert_array_equal(s.result[variable].values, expected[variable].values, err_msg=variable)
    # The spectra drawn are the ones the verdict used, at the log-spaced frequencies the result keeps.
    np.testing.assert_array_equal(s.result.psd.values, expected.psd.sel(frequency=s.result.frequency.values).values)
    planted = np.zeros(pf.SHAPE, bool)
    planted[pf.DEAD] = planted[pf.EMPTY] = True
    np.testing.assert_array_equal(s.result.dead.values, planted)


@pytest.mark.parametrize("name", NAMES)
def test_the_pdfs_and_their_moments_are_over_the_analysis_window_only(scenarios, name):
    s = scenarios[name]
    frames = s.record.frames.sel(time=slice(*s.whole.analysis_window)).values
    for pixel in ((1, 1), (2, 3), pf.DEAD):
        centres, density = dead_pixels._pdf(frames[pixel])
        np.testing.assert_array_equal(s.result.pdf_volts.values[pixel], centres)
        np.testing.assert_array_equal(s.result.pdf.values[pixel], density)
        assert float(s.result["mean"].values[pixel]) == pytest.approx(float(frames[pixel].mean()), rel=1e-5)
        assert float(s.result["std"].values[pixel]) == pytest.approx(float(frames[pixel].std()), rel=1e-4)


def test_the_dark_stretch_before_the_puff_is_left_out_of_the_judgement(scenarios):
    """On the control the window starts 40 ms before the puff. Judged over the whole discharge window, the live
    pixels' spectra carry that stretch and the PSD ratio differs; the view judges what the puff window leaves."""
    from density_scan import dead_pixels as scan

    s = scenarios["from the start"]
    assert s.whole.analysis_window[0] == pytest.approx(0.040, abs=1e-3)
    over_the_window = scan.estimate(s.record.sel(time=slice(*s.case.window)))
    live = ~s.result.dead.values
    assert not np.allclose(s.result.red_ratio.values[live], over_the_window.red_ratio.values[live], rtol=0.01)
    assert float(s.result["mean"].values[1, 1]) > float(
        s.record.frames.sel(time=slice(*s.case.window)).values[1, 1].mean()
    )  # the dark stretch pulls the discharge window's mean down


def test_a_record_without_a_puff_is_judged_over_the_whole_discharge_window(result, record):
    found = dead_pixels.puff_of(result)
    assert not found.found
    assert found.analysis_window == found.discharge_window == (float(record.time[0]), float(record.time[-1]))
    assert np.isnan(found.start) and np.isnan(found.end)
    assert found.note and found.rule.startswith("Discharge window kept: ")
    assert result.pdf.shape[:2] == (3, 4)


# ---------------------------------------------------------------------------
# What the result stores of the search, and a result that predates it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_the_search_for_the_puff_survives_netcdf(scenarios, name, tmp_path):
    """The page reads the blob back off disk on every visit after the first: nothing it draws may depend on a type
    netCDF cannot hold -- a bool, or an empty attribute (no dips: no attribute)."""
    stored = scenarios[name].result
    stored.to_netcdf(tmp_path / "blob.nc")
    with xr.open_dataset(tmp_path / "blob.nc") as back:
        back = back.load()
    first, again = dead_pixels.puff_of(stored), dead_pixels.puff_of(back)
    for field in ("found", "analysis_window", "discharge_window", "record_window", "level_quantile", "note", "rule"):
        assert getattr(again, field) == getattr(first, field), field
    for field in ("start", "end", "baseline", "level", "threshold"):
        assert getattr(again, field) == pytest.approx(getattr(first, field), nan_ok=True), field
    assert again.dips == first.dips
    np.testing.assert_array_equal(again.time, first.time)
    np.testing.assert_array_equal(again.signal, first.signal)
    assert ("puff_dips" in back.attrs) == bool(first.dips)


def test_a_puff_that_was_not_found_survives_netcdf_with_its_nan_times(result, tmp_path):
    """No puff: ``start`` and ``end`` are NaN, which a netCDF attribute holds, and ``found`` is an int, not a bool."""
    result.to_netcdf(tmp_path / "blob.nc")
    with xr.open_dataset(tmp_path / "blob.nc") as back:
        back = back.load()
    found = dead_pixels.puff_of(back)
    assert found.found is False and np.isnan(found.start) and np.isnan(found.end)
    assert found.analysis_window == found.discharge_window and found.dips == ()
    assert found.rule.startswith("Discharge window kept: ")
    assert "puff_dips" not in back.attrs
    assert dead_pixels.puff_figure(found).data  # and it still draws


def test_the_smoothed_array_mean_is_kept_over_the_whole_record_at_a_drawable_size(scenarios):
    s = scenarios["two puffs"]
    assert s.result.puff_signal.dims == ("puff_time",)
    assert 0 < s.result.puff_time.size <= 4000
    assert s.result.puff_time.values[0] < 0.001 and s.result.puff_time.values[-1] > 0.119  # all of it, not the window


def test_a_result_that_predates_the_puff_window_has_no_search_to_read(scenarios):
    """The 111 cached results, until they are computed again: neither the signal nor the attributes."""
    stored = scenarios["two puffs"].result
    assert dead_pixels.puff_of(stored) is not None
    assert dead_pixels.puff_of(pf.before_the_puff_window(stored)) is None
    # Part of it is as good as none: a figure of the cut needs all of it.
    assert dead_pixels.puff_of(stored.drop_vars(["puff_signal", "puff_time"])) is None
    for name in ("puff_note", "puff_threshold", "analysis_window"):
        broken = stored.copy()
        del broken.attrs[name]
        assert dead_pixels.puff_of(broken) is None, name
    broken = stored.copy()
    broken.attrs["analysis_window"] = "the puff"
    assert dead_pixels.puff_of(broken) is None
    # None of that touched the result itself.
    assert dead_pixels.puff_of(stored) is not None


# ---------------------------------------------------------------------------
# The figure of the cut
# ---------------------------------------------------------------------------


def strips(fig):
    """The dips as the figure marks them: the shapes that run along the bottom of the axes."""
    return [shape for shape in fig.layout.shapes if shape.y1 == 0.05]


def line_traces(fig):
    return {t.name: t for t in fig.data if t.mode == "lines" and t.x is not None and None not in list(t.x)}


def test_the_figure_draws_the_array_mean_the_windows_the_three_levels_and_where_the_puff_comes_and_goes(scenarios):
    s = scenarios["from the start"]
    found = dead_pixels.puff_of(s.result)
    fig = dead_pixels.puff_figure(found)
    signal_trace, *_ = fig.data
    np.testing.assert_array_equal(signal_trace.x, found.time)
    np.testing.assert_array_equal(signal_trace.y, found.signal)

    lines = line_traces(fig)
    by_value = {name.split()[0]: np.unique(t.y)[0] for name, t in lines.items() if name != signal_trace.name}
    assert by_value["baseline"] == pytest.approx(found.baseline)
    assert by_value["level"] == pytest.approx(found.level)
    assert by_value["threshold"] == pytest.approx(found.threshold)
    assert found.threshold == pytest.approx(0.5 * (found.baseline + found.level))
    for trace in lines.values():
        if trace is not signal_trace:
            assert (trace.x[0], trace.x[-1]) == found.record_window  # across the whole record

    (marks,) = [t for t in fig.data if t.mode == "markers"]
    assert list(marks.x) == [found.start, found.end]
    assert list(marks.customdata) == ["on", "off"]
    assert list(marks.y) == pytest.approx([found.threshold] * 2, abs=0.2)  # on the curve, where it crosses

    # Two bands -- the discharge window, and the analysis window over it -- each over the whole height.
    bands = list(fig.layout.shapes)
    assert [(b.x0, b.x1) for b in bands] == [found.discharge_window, found.analysis_window]
    assert all((b.y0, b.y1) == (0.0, 1.0) for b in bands)
    assert tuple(fig.layout.xaxis.range) == found.record_window


def test_a_puff_still_on_when_the_record_ends_is_marked_on_but_not_off(scenarios):
    """Its "end" is then the record's last sample, which is no event: a triangle at the edge of the axes would pass
    for one."""
    found = dead_pixels.puff_of(scenarios["from the start"].result)
    assert found.end < found.time[-1]  # this one goes off inside the record
    found.end = float(found.time[-1])
    (marks,) = [t for t in dead_pixels.puff_figure(found).data if t.mode == "markers"]
    assert (list(marks.x), list(marks.customdata), marks.name) == ([found.start], ["on"], "puff on")


def test_the_legend_names_the_bands_the_lines_and_their_values(scenarios):
    found = dead_pixels.puff_of(scenarios["from the start"].result)
    names = [t.name for t in dead_pixels.puff_figure(found).data]
    assert names == [
        "array mean, 1 ms mean",
        f"baseline {found.baseline:.3f} V",
        f"level (median) {found.level:.3f} V",
        f"threshold {found.threshold:.3f} V",
        "puff on, off",
        "discharge window",
        "analysis window",
    ]


def test_a_dip_the_cut_leaves_in_is_marked_along_the_bottom_of_the_window(scenarios):
    found = dead_pixels.puff_of(scenarios["two puffs"].result)
    fig = dead_pixels.puff_figure(found)
    assert [(b.x0, b.x1) for b in strips(fig)] == list(found.dips)
    assert fig.data[-1].name == "dip below the threshold, not cut"
    # And there is no strip where there is no dip.
    without = dead_pixels.puff_of(scenarios["from the start"].result)
    assert not strips(dead_pixels.puff_figure(without))


def test_a_puff_that_was_not_found_is_still_drawn_with_the_lines_the_rule_had_to_stand_on(result):
    found = dead_pixels.puff_of(result)
    fig = dead_pixels.puff_figure(found)
    assert not [t for t in fig.data if t.mode == "markers"]  # nothing to mark
    assert {"baseline", "level", "threshold"} <= {name.split()[0] for name in line_traces(fig)}
    assert [(b.x0, b.x1) for b in fig.layout.shapes] == [found.discharge_window, found.analysis_window]
    assert found.analysis_window == found.discharge_window  # the kept window: the two bands coincide


def test_the_light_level_is_named_for_the_quantile_it_is():
    """Where most of the window is dark after a short puff, the rule takes the 90th percentile as the light."""
    assert dead_pixels.light_level(0.5) == "median"
    assert dead_pixels.light_level(0.9) == "90th percentile"
    assert dead_pixels.light_level(0.1) == "10th percentile"
    assert dead_pixels.light_level(0.01) == "1st percentile" and dead_pixels.light_level(0.22) == "22nd percentile"
    assert dead_pixels.light_level(0.13) == "13th percentile"


def test_a_fallback_level_says_so_in_the_legend(scenarios):
    found = dead_pixels.puff_of(scenarios["from the start"].result)
    found.level_quantile = 0.9
    names = [t.name for t in dead_pixels.puff_figure(found).data]
    assert f"level (90th percentile) {found.level:.3f} V" in names


# ---------------------------------------------------------------------------
# The words, and what the view draws
# ---------------------------------------------------------------------------


def test_the_window_line_sets_the_analysis_window_beside_the_discharge_window_and_the_rule():
    line = dead_pixels.window_line("0.0400–0.1150 s", "0.0000–0.1150 s", "the puff rises")
    assert line == "Analysis window 0.0400–0.1150 s · discharge window 0.0000–0.1150 s · puff rule: the puff rises"
    assert dead_pixels.window_line(None, "0.9500–1.4000 s", None) == "Discharge window 0.9500–1.4000 s"
    assert dead_pixels.window_line(None, None, "the puff rises") == "Puff rule: the puff rises"
    assert dead_pixels.window_line() is None
    assert dead_pixels.seconds((1.0734, 1.4)) == "1.0734–1.4000 s"


def drawn(result, params, target):
    from fusion_ui.plots import dead_pixels

    dead_pixels.render(result, params, target)


def draw(result):
    target = registry.Target("cmod", 1234, "apd", False, "unused", 0.0, 0.12)
    app = AppTest.from_function(drawn, args=(result, dead_pixels.DeadPixelParams(), target), default_timeout=60).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def kinds(app):
    return [element.type for element in app.main.children.values()]


def test_the_view_draws_the_cut_above_the_grid_with_the_window_line_under_it(scenarios):
    s = scenarios["two puffs"]
    app = draw(s.result)
    column = kinds(app)
    # The counts, the plain words and the method, then the figure of the cut and its line, then the toggle and grid.
    assert column.index("expander") < column.index("plotly_chart") < column.index("radio")
    assert column.count("plotly_chart") == 2
    assert column[column.index("plotly_chart") + 1] == "caption"
    found = dead_pixels.puff_of(s.result)
    line = [c.value for c in app.caption if c.value.startswith("Analysis window")]
    seconds = dead_pixels.seconds
    expected = dead_pixels.window_line(seconds(found.analysis_window), seconds(found.discharge_window), found.rule)
    assert line == [expected]
    assert "0.0350–0.1150 s" in line[0] and "dip below threshold at 0.035-0.050 s" in line[0]
    assert not app.info and not app.warning


def test_a_result_that_predates_the_puff_window_still_draws_its_grid_and_says_so(scenarios):
    app = draw(pf.before_the_puff_window(scenarios["two puffs"].result))
    assert [i.value for i in app.info] == [dead_pixels.PREDATES]
    assert "predates the puff window" in app.info[0].value and "Recompute" in app.info[0].value
    column = kinds(app)
    assert column.count("plotly_chart") == 1  # the grid; no figure of the cut to draw
    assert column.index("expander") < column.index("radio") < column.index("plotly_chart")
    assert not [c for c in app.caption if c.value.startswith("Analysis window")]
    assert [r.options for r in app.radio] == [list(dead_pixels.VIEWS)]


def test_every_view_of_the_grid_still_draws_for_a_result_with_a_puff_window(scenarios):
    result = scenarios["two puffs"].result
    for view in dead_pixels.VIEWS:
        assert len([t for t in dead_pixels.figure(result, view).data if t.showlegend is False]) == 12


# ---------------------------------------------------------------------------
# The method, in words that are true on both views
# ---------------------------------------------------------------------------


def test_the_method_says_the_analysis_window_where_it_said_the_discharge_window():
    """The spectra are judged over the analysis window, and the text that says so is the same on both views."""
    text = dead_pixels.METHOD.format(red=150.0, gain=0.1)
    assert "Welch PSD of each pixel over the analysis window" in text
    assert "Welch PSD of each pixel over the discharge window" not in text
    assert "*analysis window*" in text and "cut to\nthe gas puff" in text
    for fact in ("1 ms", "halfway", "5 ms", "90th percentile", "keeps the discharge window", "not cut"):
        assert fact in text, fact


def test_the_method_is_true_of_both_views_not_just_the_raw_one():
    """It used to end "This view shows the single shot", which the stored-mask view, drawing the day's mask, is not."""
    text = dead_pixels.METHOD
    assert "This view shows the single shot" not in text
    assert "raw file's view shows this shot's own verdict, before that rule" in text
    assert "preprocessed file's view shows the mask the file was made with, which is the day's" in text
    assert "overrides the shot's own verdict" in text
