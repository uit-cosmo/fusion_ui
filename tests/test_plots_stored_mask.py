"""The stored-mask view: what a preprocessed file says about its dead pixels, with and without the variables.

The file is the one ``density_scan.preprocess`` (J6, in fusion_scripts) writes, built here by hand from the format in
docs/PHASE_06_DECORRELATION.md: ``dead``, ``dead_shot``, ``dead_evidence`` and ``dead_psd_ratio`` beside the frames, and
the attributes ``dead_mask_source``, ``dead_thresholds``, ``preprocess_radius``, ``fusion_scripts_commit``,
``created``, ``analysis_window``, ``discharge_window`` and ``puff_rule``. Nothing J6 adds is imported: the view reads
the file, and is tested on files written the way J6 writes them, through xarray and netCDF.

The hand-made 1160616 mask the view falls back on is compared against ``fields_fixtures.DEAD``, an independent
transcription of it, so a drawing that agreed with itself would still fail.
"""

import base64
import json
from collections import Counter
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
import xarray as xr
from streamlit.testing.v1 import AppTest

import fusion_ui.plots  # noqa: F401 - registers every spec
from fusion_ui.core import catalog, db, multipixel, registry
from fusion_ui.plots import dead_pixels, stored_mask
from tests import fields_fixtures as ff

REPO_ROOT = Path(__file__).resolve().parent.parent
SINGLE_SHOT = str(REPO_ROOT / "fusion_ui" / "pages" / "2_single_shot.py")

NY, NX, N_TIME = ff.NY, ff.NX, 40
HAND_MADE_SHOT = 1160616027
ESTIMATED_SHOT = 1140827019

#: Four pixels, as (y, x), that the evidence classes need: two where this shot's own verdict and the run day's mask
#: part ways, one with no samples, and one that is live only because it follows a neighbour.
KILLED = (0, 2)  # dead in the day's mask, though this shot saw it live (through a neighbour)
SPARED = (1, 4)  # live in the day's mask, though this shot saw it dead
EMPTY = (2, 1)  # dead, and nothing in the record to say otherwise
FOLLOWS = (0, 0)  # live, and only because it follows a live neighbour


def day_mask():
    """The 1140827 run day's mask: the 2016 hand-made one without the four pixels that died after 2014."""
    mask = ff.DEAD.copy()
    for y, x in ((4, 8), (8, 7), (8, 8), (9, 7)):
        mask[y, x] = False
    return mask


def verdicts(dead):
    """``(dead_shot, evidence, ratio)`` as the file stores them for a day's mask ``dead``.

    The shot agrees with the mask but at the four pixels above. Live pixels have a large PSD ratio, dead ones about
    1 (white noise), and the pixel with no data none.
    """
    shot = dead.copy()
    evidence = np.where(dead, 0, 2).astype(np.int8)
    ratio = np.where(dead, 1.3, 800.0)

    shot[KILLED], evidence[KILLED], ratio[KILLED] = False, 1, 61.0
    shot[SPARED], evidence[SPARED], ratio[SPARED] = True, 0, 2.1
    shot[EMPTY], evidence[EMPTY], ratio[EMPTY] = True, -1, np.nan
    evidence[FOLLOWS], ratio[FOLLOWS] = 1, 47.0
    return shot, evidence, ratio


ATTRS = dict(
    dead_mask_source="estimated, run day 1140827, 8 shots",
    dead_thresholds="red 150, gain 0.1, day fraction 1/3",
    preprocess_radius=1000,
    fusion_scripts_commit="1a2b3c4",
    created="2026-10-09T12:00:00",
    analysis_window=[1.0734, 1.4],
    discharge_window=[0.95, 1.4],
    puff_rule="the puff rises through half its height at 1.0734 s and stays up: kept from there",
)


def preprocessed(shot, dead=None, *, variables=True, attrs=ATTRS, drop=()):
    """A tiny preprocessed APD record: frames on (y, x, time) and, with ``variables``, what J6 adds to it.

    ``dead`` is the mask the file was made with (the 1140827 day's by default); ``drop`` names variables or
    attributes to leave out, for a file that stores only part of it. The record is a few frames: what the view
    reads is the (y, x) variables and the attributes, and ``analysis_window`` is only an attribute of it.
    """
    dead = day_mask() if dead is None else dead
    frames = np.random.default_rng(shot).normal(size=(NY, NX, N_TIME)).astype("float32")
    ds = xr.Dataset(
        {"frames": (("y", "x", "time"), frames)},
        coords={
            "R": (("y", "x"), np.tile(88.0 + 0.4 * np.arange(NX), (NY, 1)).astype("float32")),
            "Z": (("y", "x"), np.tile(-2.0 + 0.4 * np.arange(NY), (NX, 1)).T.astype("float32")),
            "time": ("time", np.linspace(1.0734, 1.0739, N_TIME)),
        },
        attrs={"shot_number": shot},
    )
    if variables:
        shot_verdict, evidence, ratio = verdicts(dead)
        stored = {
            "dead": dead,
            "dead_shot": shot_verdict,
            "dead_evidence": evidence,
            "dead_psd_ratio": ratio,
        }
        for name, values in stored.items():
            if name not in drop:
                ds[name] = (("y", "x"), values)
        ds.attrs.update({k: v for k, v in attrs.items() if k not in drop})
    return ds


def agreeing(shot, attrs=ATTRS):
    """A file whose mask is the hand-made 1160616 one and whose own verdict, evidence and ratio agree with it."""
    ds = preprocessed(shot, ff.DEAD, attrs=attrs)
    ds["dead_shot"] = ds["dead"]
    evidence = np.where(ff.DEAD, 0, 2).astype(np.int8)
    ds["dead_evidence"] = (("y", "x"), evidence)
    ds["dead_psd_ratio"] = (("y", "x"), np.where(ff.DEAD, 1.2, 2400.0))
    return ds


def target_for(shot):
    return registry.Target("cmod", shot, "apd", True, "unused", 1.0, 1.1)


def read(ds, shot=ESTIMATED_SHOT):
    return stored_mask.read(ds, target_for(shot))


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


def test_it_is_a_live_spec_for_preprocessed_files_only():
    spec = registry.get("stored_mask")
    assert spec.label == "Dead-pixel mask stored at preprocessing"
    assert not spec.cached and spec.preprocessed is True
    assert spec in registry.for_diagnostic("apd", preprocessed=True)
    assert spec not in registry.for_diagnostic("apd", preprocessed=False)
    assert not multipixel.supported(spec)  # nothing to stamp a pixel into


def test_its_one_knob_is_a_form_leaf_the_store_would_hash():
    """A live spec is never hashed, but its parameters must walk like every other spec's."""
    from fusion_ui.core import params_ui

    digest, body = params_ui.hash_params("stored_mask", stored_mask.StoredMaskParams())
    assert len(digest) == 40 and '"show_ratio": true' in body


# ---------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------


def test_a_stored_mask_is_read_off_the_file_as_it_is():
    mask = read(preprocessed(ESTIMATED_SHOT))
    assert mask.origin == "stored"
    assert mask.source == "estimated, run day 1140827, 8 shots"
    np.testing.assert_array_equal(mask.dead, day_mask())
    shot, evidence, ratio = verdicts(day_mask())
    np.testing.assert_array_equal(mask.dead_shot, shot)
    np.testing.assert_array_equal(mask.evidence, evidence)
    np.testing.assert_array_equal(mask.ratio, ratio)
    assert mask.R.shape == mask.Z.shape == (NY, NX)
    assert mask.analysis_window == "1.0734–1.4000 s"
    assert mask.discharge_window == "0.9500–1.4000 s"
    assert mask.puff_rule.startswith("the puff rises")
    assert mask.made == (
        "created 2026-10-09T12:00:00 · fusion_scripts 1a2b3c4 · running-normalisation radius 1000"
        " · dead-pixel thresholds red 150, gain 0.1, day fraction 1/3"
    )


def test_a_file_that_stores_no_mask_is_on_the_hand_made_one_for_1160616():
    mask = read(preprocessed(HAND_MADE_SHOT, variables=False), HAND_MADE_SHOT)
    assert mask.origin == "hand-made"
    assert mask.source == "hand-made mask, 1160616"
    np.testing.assert_array_equal(mask.dead, ff.DEAD)
    assert mask.dead_shot is mask.evidence is mask.ratio is None
    assert mask.analysis_window is mask.discharge_window is mask.puff_rule is mask.made is None


def test_a_file_that_stores_no_mask_is_unknown_anywhere_else():
    mask = read(preprocessed(ESTIMATED_SHOT, variables=False))
    assert mask.origin == "missing" and mask.dead is None and mask.source is None


def test_a_hand_made_mask_that_does_not_fit_the_array_is_not_drawn():
    """The hand-made mask is 10 x 9: on any other array it would be a guess."""
    ds = preprocessed(HAND_MADE_SHOT, variables=False).isel(y=slice(0, 4), x=slice(0, 5))
    assert read(ds, HAND_MADE_SHOT).dead is None


def test_a_mask_with_no_recorded_source_says_so_in_the_pipelines_words():
    """``decorrelation.pipeline.dead_mask`` calls it "stored in the preprocessed file"."""
    assert read(preprocessed(ESTIMATED_SHOT, drop=("dead_mask_source",))).source == "stored in the preprocessed file"


def test_a_file_that_stores_part_of_it_gives_what_it_has():
    mask = read(preprocessed(ESTIMATED_SHOT, drop=("dead_shot", "dead_evidence", "dead_psd_ratio", *ATTRS)))
    assert mask.dead.sum() == 18
    assert mask.dead_shot is mask.evidence is mask.ratio is None
    assert mask.analysis_window is mask.discharge_window is mask.puff_rule is mask.made is None
    assert mask.source == "stored in the preprocessed file"


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.array([1.0734, 1.4]), (1.0734, 1.4)),
        ([0.95, 1.4], (0.95, 1.4)),
        ("[0.95, 1.4]", (0.95, 1.4)),
        ((1, 2), (1.0, 2.0)),
        ("1.0734 to 1.4", None),
        ([1.0, 2.0, 3.0], None),
        ([np.nan, 1.4], None),
        (None, None),
    ],
)
def test_a_window_attribute_is_a_start_and_an_end_or_nothing(value, expected):
    assert stored_mask.window(value) == expected


def test_a_window_attribute_that_is_not_one_is_shown_as_stored():
    ds = preprocessed(ESTIMATED_SHOT)
    ds.attrs["analysis_window"] = "from the rise to the end"
    mask = read(ds)
    assert mask.analysis_window == "from the rise to the end"
    assert "Analysis window from the rise to the end" in stored_mask.window_line(mask)


def test_an_empty_attribute_is_left_out_of_the_line_on_how_the_file_was_made():
    ds = preprocessed(ESTIMATED_SHOT, drop=("dead_thresholds", "fusion_scripts_commit"))
    ds.attrs["created"] = ""
    ds.attrs["puff_rule"] = ""
    mask = read(ds)
    assert mask.made == "running-normalisation radius 1000"
    assert mask.puff_rule is None


def test_attributes_of_any_netcdf_type_read_as_one_line_of_text():
    assert stored_mask._text(np.array([150.0, 0.1])) == "150, 0.1"
    assert stored_mask._text(np.float64(0.25)) == "0.25"
    assert stored_mask._text(np.int64(1000)) == "1000"
    assert stored_mask._text(b"abc") == "abc"
    assert stored_mask._text("red 150") == "red 150"


# ---------------------------------------------------------------------------
# What there is to say about it
# ---------------------------------------------------------------------------


def test_the_classes_follow_the_evidence_and_where_the_day_overrode_the_shot():
    mask = read(preprocessed(ESTIMATED_SHOT))
    classes = stored_mask.classify(mask.dead, mask.dead_shot, mask.evidence)
    assert classes[KILLED] == stored_mask.DEAD_BY_DAY
    assert classes[SPARED] == stored_mask.LIVE_BY_DAY
    assert classes[EMPTY] == stored_mask.NO_DATA
    assert classes[FOLLOWS] == stored_mask.NEIGHBOUR
    others = np.ones((NY, NX), bool)
    for pixel in (KILLED, SPARED, EMPTY, FOLLOWS):
        others[pixel] = False
    expected = np.where(day_mask(), stored_mask.DEAD, stored_mask.RED)
    np.testing.assert_array_equal(classes[others], expected[others])


def test_without_the_shots_own_verdict_it_is_the_evidences():
    """``dead_evidence`` says it too: dead at 0 or below. The day's mask overrides it where they differ."""
    mask = read(preprocessed(ESTIMATED_SHOT, drop=("dead_shot",)))
    classes = stored_mask.classify(mask.dead, None, mask.evidence)
    assert classes[KILLED] == stored_mask.DEAD_BY_DAY
    assert classes[SPARED] == stored_mask.LIVE_BY_DAY
    assert classes[EMPTY] == stored_mask.NO_DATA


def test_without_evidence_the_shots_verdict_still_shows_where_it_was_overridden():
    mask = read(preprocessed(ESTIMATED_SHOT, drop=("dead_evidence",)))
    classes = stored_mask.classify(mask.dead, mask.dead_shot, None)
    assert Counter(classes.ravel().tolist()) == {
        stored_mask.DEAD: 17,
        stored_mask.LIVE: 71,
        stored_mask.DEAD_BY_DAY: 1,
        stored_mask.LIVE_BY_DAY: 1,
    }


def test_without_evidence_or_a_verdict_a_pixel_is_dead_or_live_and_nothing_more():
    classes = stored_mask.classify(day_mask())
    assert set(np.unique(classes)) == {stored_mask.DEAD, stored_mask.LIVE}
    np.testing.assert_array_equal(classes == stored_mask.DEAD, day_mask())


def test_a_nan_in_the_evidence_is_no_evidence():
    evidence = np.full((2, 2), np.nan)
    classes = stored_mask.classify(np.array([[True, False], [False, True]]), None, evidence)
    dead, live = stored_mask.DEAD, stored_mask.LIVE
    np.testing.assert_array_equal(classes, [[dead, live], [live, dead]])


def test_the_evidence_classes_keep_the_dead_pixel_views_words_and_colours():
    for code in (stored_mask.NO_DATA, stored_mask.DEAD, stored_mask.NEIGHBOUR, stored_mask.RED):
        assert stored_mask.LABELS[code] == dead_pixels.LABELS[code]
        assert stored_mask.COLOURS[code] == dead_pixels.COLOURS[code]
    # One colour for each thing a cell can mean. The plain "live" is the red-spectrum colour: a file that does not
    # say how its live pixels were found never has both.
    meanings = set(stored_mask.LABELS) - {stored_mask.LIVE}
    assert len({stored_mask.COLOURS[code] for code in meanings}) == len(meanings)


@pytest.fixture(scope="module")
def estimated():
    """``density_scan.dead_pixels``, and what ``estimate`` returns for a 3 x 4 record with one white-noise pixel
    (0, 3) and one with no samples (2, 0), the rest sharing one red signal: the file's evidence as the estimator
    writes it, and the estimator's own codes."""
    from scipy import signal

    from fusion_ui.core import fusion_scripts

    fusion_scripts.import_config()
    from density_scan import dead_pixels as scan

    rng = np.random.default_rng(1)
    n, dt = 60_000, 5e-7
    phi = np.exp(-dt / 20e-6)
    shared = signal.lfilter([1], [1, -phi], rng.normal(size=n) * np.sqrt(1 - phi**2))
    frames = 0.9 * shared + 0.02 * rng.normal(size=(3, 4, n))
    frames[0, 3] = 0.01 * rng.normal(size=n)
    frames[2, 0] = np.nan
    record = xr.Dataset({"frames": (("y", "x", "time"), frames.astype("float32"))}, coords={"time": np.arange(n) * dt})
    return scan, scan.estimate(record)


def test_the_codes_are_the_estimators(estimated):
    scan, _ = estimated
    assert (stored_mask.NO_DATA, stored_mask.DEAD, stored_mask.NEIGHBOUR, stored_mask.RED) == (
        scan.NO_DATA,
        scan.NO_EVIDENCE,
        scan.FOLLOWS_NEIGHBOUR,
        scan.RED_SPECTRUM,
    )


def test_what_the_estimator_returns_is_classed_and_drawn_as_planted(estimated):
    """The file's variables are the estimator's ``dead``, ``evidence`` and ``red_ratio``, stored as they come."""
    _, result = estimated
    mask = stored_mask.StoredMask(
        dead=result.dead.values,
        origin="stored",
        source="estimated",
        dead_shot=result.dead.values,
        evidence=result.evidence.values.astype(float),
        ratio=result.red_ratio.values,
    )
    classes = stored_mask.classes_of(mask)
    assert classes[0, 3] == stored_mask.DEAD and classes[2, 0] == stored_mask.NO_DATA
    assert (np.delete(classes.ravel(), [3, 8]) == stored_mask.RED).all()
    shown = {(a.x, a.y): a.text for a in stored_mask.figure(mask).layout.annotations}
    assert shown[(0, 2)] == "–"  # no samples, so no ratio
    assert float(shown[(3, 0)]) < 10 and int(shown[(1, 1)]) > 150  # white noise, and a red pixel above the threshold


def test_the_counts_say_what_is_dead_and_where_the_day_overrode_the_shot():
    text = stored_mask.summary(read(preprocessed(ESTIMATED_SHOT)))
    assert text.startswith("18 of 90 pixels dead: ")
    assert "(0, 2)" in text and "(2, 1)" in text and "(8, 8)" not in text
    assert "1 live only through a neighbour: (0, 0)" in text
    assert "dead in the mask, live on this shot at (0, 2)" in text
    assert "live in the mask, dead on this shot at (1, 4)" in text


def test_the_counts_say_so_when_the_mask_agrees_with_the_shot_everywhere():
    text = stored_mask.summary(read(agreeing(HAND_MADE_SHOT), HAND_MADE_SHOT))
    assert text.startswith("22 of 90 pixels dead: ")
    assert text.endswith("the mask agrees with this shot's own verdict at every pixel")
    assert "overrides" not in text


def test_the_counts_of_a_mask_with_nothing_to_compare_it_with_are_the_masks_alone():
    text = stored_mask.summary(read(preprocessed(HAND_MADE_SHOT, variables=False), HAND_MADE_SHOT))
    assert text.startswith("22 of 90 pixels dead: ")
    assert "verdict" not in text


def test_the_window_line_sets_the_analysis_window_beside_the_discharge_window_and_the_puff_rule():
    mask = read(preprocessed(ESTIMATED_SHOT))
    assert stored_mask.window_line(mask) == (
        "Analysis window 1.0734–1.4000 s · discharge window 0.9500–1.4000 s"
        " · puff rule: the puff rises through half its height at 1.0734 s and stays up: kept from there"
    )
    mask.puff_rule = None
    assert stored_mask.window_line(mask) == "Analysis window 1.0734–1.4000 s · discharge window 0.9500–1.4000 s"
    mask.discharge_window = None
    assert stored_mask.window_line(mask) == "Analysis window 1.0734–1.4000 s"
    mask.analysis_window = None
    assert stored_mask.window_line(mask) is None


def test_a_window_line_that_starts_with_the_rule_still_starts_with_a_capital():
    mask = read(preprocessed(ESTIMATED_SHOT, drop=("analysis_window", "discharge_window")))
    assert stored_mask.window_line(mask).startswith("Puff rule: the puff rises")


def test_the_line_on_how_the_file_was_made_quotes_the_file():
    mask = read(preprocessed(ESTIMATED_SHOT))
    line = stored_mask.made_line(mask)
    assert line.startswith("Preprocessing: created 2026-10-09T12:00:00 · fusion_scripts 1a2b3c4")
    bare = read(preprocessed(ESTIMATED_SHOT, drop=("puff_rule", *ATTRS)))
    assert stored_mask.made_line(bare) is None


# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def drawn():
    mask = read(preprocessed(ESTIMATED_SHOT))
    return mask, stored_mask.figure(mask)


def heatmap_of(fig):
    (heat,) = [t for t in fig.data if isinstance(t, go.Heatmap)]
    return heat


def heatmap_classes(fig):
    """The class code of every cell, read back off the heatmap by its position in the colour order."""
    order = list(stored_mask.LABELS)
    return np.vectorize(order.__getitem__)(np.asarray(heatmap_of(fig).z))


def test_the_map_has_one_cell_per_pixel_as_the_array_is_laid_out(drawn):
    mask, fig = drawn
    assert isinstance(fig, go.Figure)
    heat = heatmap_of(fig)
    assert np.asarray(heat.z).shape == (NY, NX)
    # y = 0 is the first row, drawn at the bottom: Z up, as in the dead-pixel view.
    np.testing.assert_array_equal(heatmap_classes(fig), stored_mask.classes_of(mask))
    np.testing.assert_array_equal(heat.y, np.arange(NY))
    assert fig.layout.yaxis.autorange is None  # not reversed


def test_the_map_colours_the_cells_by_what_the_file_says_of_each(drawn):
    """Counted per class, from what the fixture plants: 18 dead, of which one the shot saw live and one with no data."""
    _, fig = drawn
    assert Counter(heatmap_classes(fig).ravel().tolist()) == {
        stored_mask.DEAD: 16,
        stored_mask.NO_DATA: 1,
        stored_mask.DEAD_BY_DAY: 1,
        stored_mask.LIVE_BY_DAY: 1,
        stored_mask.NEIGHBOUR: 1,
        stored_mask.RED: 70,
    }
    colours = [step[1] for step in heatmap_of(fig).colorscale][::2]
    assert colours == [stored_mask.COLOURS[code] for code in stored_mask.LABELS]


def test_the_legend_lists_the_classes_that_are_there_and_no_others(drawn):
    _, fig = drawn
    legend = [t for t in fig.data if isinstance(t, go.Scatter)]
    present = (
        stored_mask.NO_DATA,
        stored_mask.DEAD,
        stored_mask.NEIGHBOUR,
        stored_mask.RED,
        stored_mask.DEAD_BY_DAY,
        stored_mask.LIVE_BY_DAY,
    )
    assert [t.name for t in legend] == [stored_mask.LABELS[code] for code in present]
    assert [t.marker.color for t in legend] == [stored_mask.COLOURS[code] for code in present]


def test_each_cell_carries_its_ratio_and_a_hover_that_names_the_pixel(drawn):
    _, fig = drawn
    shown = {(a.x, a.y): a.text for a in fig.layout.annotations}
    assert len(shown) == NY * NX
    assert shown[(0, 0)] == "47"  # (y, x) = (0, 0): live only through a neighbour
    assert shown[(1, 1)] == "1.3"  # dead, a ratio near 1
    assert shown[(3, 1)] == "800"  # live by its spectrum
    assert shown[(1, 2)] == "–"  # no data: no ratio
    hover = np.asarray(heatmap_of(fig).text)[KILLED]
    assert "pixel (y=0, x=2)" in hover and "R 88.80 cm, Z -2.00 cm" in hover
    assert stored_mask.LABELS[stored_mask.DEAD_BY_DAY] in hover and "PSD ratio 61" in hover


def test_the_numbers_can_be_left_off(drawn):
    mask, _ = drawn
    fig = stored_mask.figure(mask, show_ratio=False)
    assert not fig.layout.annotations
    assert "PSD ratio" not in fig.layout.title.text


def test_a_cell_is_labelled_in_a_colour_that_reads_on_its_fill():
    assert stored_mask._ink(stored_mask.COLOURS[stored_mask.DEAD]) == "#ffffff"
    assert stored_mask._ink(stored_mask.COLOURS[stored_mask.RED]) == "#ffffff"
    assert stored_mask._ink(stored_mask.COLOURS[stored_mask.NEIGHBOUR]) == "#1f1f1f"
    assert stored_mask._ink(stored_mask.COLOURS[stored_mask.NO_DATA]) == "#1f1f1f"


def test_a_map_without_a_ratio_or_evidence_still_draws():
    mask = stored_mask.StoredMask(dead=day_mask(), origin="stored", source="stored in the preprocessed file")
    fig = stored_mask.figure(mask)
    assert not fig.layout.annotations
    assert [t.name for t in fig.data if isinstance(t, go.Scatter)] == ["dead", "live"]
    assert "PSD ratio" not in np.asarray(heatmap_of(fig).text)[0, 0]


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


@pytest.fixture
def tree(monkeypatch, tmp_path):
    """A data tree for the single-shot page, and ``install(dataset)`` to put one preprocessed APD file in it."""
    import streamlit as st

    folder = tmp_path / "alcator"
    database = tmp_path / "state" / "shot_explorer.sqlite"
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(folder))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(tmp_path / "no_such_discharges.json"))
    monkeypatch.setenv("FUSION_UI_DB", str(database))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")
    st.cache_data.clear()
    st.cache_resource.clear()

    def install(ds):
        shot = ds.attrs["shot_number"]
        (folder / "apd").mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(folder / "apd" / f"apd_{shot}_preprocessed.nc")
        conn = db.open_db(database)
        catalog.rescan(conn, str(folder), "cmod", None)
        conn.close()
        st.cache_data.clear()
        st.cache_resource.clear()
        return shot

    yield install
    st.cache_data.clear()
    st.cache_resource.clear()


def open_view(shot):
    """The single-shot page on the preprocessed file of ``shot``, with this view picked."""
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60)
    app.session_state["selection"] = {
        "machine": "cmod",
        "shot": shot,
        "diagnostic": "apd",
        "preprocessed": True,
    }
    app.session_state["spec.apd"] = registry.get("stored_mask")
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    return app


def markdown(app):
    return [m.value for m in app.markdown]


def captions(app):
    return [c.value for c in app.caption]


def column(app):
    """What the page drew in its main column, top to bottom: ``(kind, text)`` of each element."""
    return [
        (element.type, getattr(element, "value", getattr(element, "label", None)))
        for element in app.main.children.values()
    ]


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
    """Each chart's figure, rebuilt from the JSON Streamlit was given."""
    return [go.Figure(decode(json.loads(c.proto.spec))) for c in app.get("plotly_chart")]


def test_a_stored_mask_is_drawn_with_the_plain_words_its_source_and_its_window(tree):
    app = open_view(tree(preprocessed(ESTIMATED_SHOT)))
    assert not app.warning

    # The counts, the words, the method in its expander, the map, and then what the file says about the map.
    assert dead_pixels.plain_summary() in markdown(app)
    assert [e.label for e in app.expander] == [dead_pixels.METHOD_TITLE]
    kinds = [kind for kind, _ in column(app)]
    assert kinds.index("markdown") + 1 == kinds.index("expander")  # the summary sits directly above the expander
    assert kinds.index("expander") + 1 == kinds.index("plotly_chart")

    (chart,) = charts(app)
    assert Counter(heatmap_classes(chart).ravel().tolist())[stored_mask.DEAD] == 16  # through netCDF and JSON
    shown = captions(app)
    assert any(c.startswith("18 of 90 pixels dead: ") for c in shown)
    under = [value for kind, value in column(app)[kinds.index("plotly_chart") + 1 :] if kind == "caption"]
    assert under == [
        "Mask source: **estimated, run day 1140827, 8 shots**.",
        "Analysis window 1.0734–1.4000 s · discharge window 0.9500–1.4000 s"
        " · puff rule: the puff rises through half its height at 1.0734 s and stays up: kept from there",
        "Preprocessing: created 2026-10-09T12:00:00 · fusion_scripts 1a2b3c4 · running-normalisation radius 1000"
        " · dead-pixel thresholds red 150, gain 0.1, day fraction 1/3",
    ]


def test_the_ratio_numbers_are_a_sidebar_choice(tree):
    app = open_view(tree(preprocessed(ESTIMATED_SHOT)))
    assert len(charts(app)[0].layout.annotations) == NY * NX
    (box,) = [w for w in app.checkbox if w.label == "show ratio"]
    box.uncheck().run()
    assert not app.exception
    assert not charts(app)[0].layout.annotations


def test_the_hand_made_mask_stands_in_for_a_1160616_file_that_stores_none(tree):
    app = open_view(tree(preprocessed(HAND_MADE_SHOT, variables=False)))
    assert not app.warning
    assert dead_pixels.plain_summary() in markdown(app)
    (chart,) = charts(app)
    classes = heatmap_classes(chart)
    np.testing.assert_array_equal(classes == stored_mask.DEAD, ff.DEAD)
    assert set(np.unique(classes)) == {stored_mask.DEAD, stored_mask.LIVE}
    shown = captions(app)
    assert any(c.startswith("22 of 90 pixels dead: ") for c in shown)
    (source,) = [c for c in shown if c.startswith("Mask source:")]
    assert "**hand-made mask, 1160616**" in source and "stores no mask" in source
    assert not any(c.startswith(("Analysis window", "Preprocessing")) for c in shown)


def test_a_file_that_stores_none_elsewhere_gets_a_warning_and_no_map(tree):
    app = open_view(tree(preprocessed(ESTIMATED_SHOT, variables=False)))
    (warning,) = [w.value for w in app.warning]
    assert "predates stored masks" in warning and "preprocessed again" in warning
    assert "run day 1140827" in warning
    assert not charts(app)
    assert not any(c.startswith("Mask source") for c in captions(app))
    # The words are there all the same: the view always explains itself.
    assert dead_pixels.plain_summary() in markdown(app)
    assert [e.label for e in app.expander] == [dead_pixels.METHOD_TITLE]


def test_a_1160616_file_that_stores_its_mask_says_where_it_came_from(tree):
    """The hand-made mask stored by preprocessing, with the detector's own verdict beside it: they agree."""
    app = open_view(tree(agreeing(HAND_MADE_SHOT, {**ATTRS, "dead_mask_source": "hand-made, 1160616"})))
    shown = captions(app)
    assert "Mask source: **hand-made, 1160616**." in shown
    assert any(c.endswith("the mask agrees with this shot's own verdict at every pixel") for c in shown)
    classes = heatmap_classes(charts(app)[0])
    np.testing.assert_array_equal(classes == stored_mask.DEAD, ff.DEAD)
    assert set(np.unique(classes)) == {stored_mask.DEAD, stored_mask.RED}


def test_a_file_with_the_mask_alone_draws_dead_and_live(tree):
    only_the_mask = preprocessed(ESTIMATED_SHOT, drop=("dead_shot", "dead_evidence", "dead_psd_ratio", *ATTRS))
    app = open_view(tree(only_the_mask))
    (chart,) = charts(app)
    assert set(np.unique(heatmap_classes(chart))) == {stored_mask.DEAD, stored_mask.LIVE}
    assert not chart.layout.annotations  # no ratio to print
    assert "Mask source: **stored in the preprocessed file**." in captions(app)
    assert not any(c.startswith(("Analysis window", "Preprocessing")) for c in captions(app))


def test_a_file_without_a_window_draws_the_mask_and_no_window_line(tree):
    app = open_view(tree(preprocessed(ESTIMATED_SHOT, drop=("analysis_window", "discharge_window", "puff_rule"))))
    assert charts(app)
    assert not any(c.startswith(("Analysis window", "Puff rule", "Discharge window")) for c in captions(app))
    assert "Mask source: **estimated, run day 1140827, 8 shots**." in captions(app)
    assert any(c.startswith("Preprocessing: ") for c in captions(app))


# ---------------------------------------------------------------------------
# The words, written once
# ---------------------------------------------------------------------------


def window_captions(app):
    return [c for c in captions(app) if c.startswith("Window ")]


def test_the_windows_caption_gives_the_analysis_window_a_cropped_file_stores(tree):
    """A preprocessed file is cut to its analysis window: that is the window of its data, so that is what the page
    says. With no discharge-DB entry to set beside it, it says no more."""
    app = open_view(tree(preprocessed(ESTIMATED_SHOT)))
    assert window_captions(app) == ["Window 1.0734–1.4000 s — the analysis window this file is cropped to."]


def test_the_windows_caption_sets_the_discharge_dbs_window_beside_it(tree, monkeypatch, tmp_path):
    import json

    from tests import puff_fixtures as pf

    descriptor = tmp_path / "plasma_discharges.json"
    descriptor.write_text(json.dumps([pf.discharge_entry(ESTIMATED_SHOT, (0.95, 1.4))]))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(descriptor))
    app = open_view(tree(preprocessed(ESTIMATED_SHOT)))
    assert window_captions(app) == [
        "Window 1.0734–1.4000 s — the analysis window this file is cropped to"
        " (the discharge DB's window is 0.9500–1.4000 s)."
    ]


#: What the page says of a file that gives it no analysis window to quote, on a record the 0.2 s default makes sense of.
DEFAULT_CAPTION = "Window 0.9000–1.1000 s — no discharge-DB entry yet, showing a centred 0.2 s default."


def over_a_default_window(ds):
    return ds.assign_coords(time=np.linspace(0.9, 1.1, N_TIME))


def test_the_windows_caption_is_the_old_one_for_a_file_that_stores_no_analysis_window(tree):
    ds = over_a_default_window(preprocessed(ESTIMATED_SHOT, drop=("analysis_window",)))
    assert window_captions(open_view(tree(ds))) == [DEFAULT_CAPTION]


def test_an_analysis_window_that_is_not_a_window_is_not_quoted(tree):
    ds = over_a_default_window(preprocessed(ESTIMATED_SHOT))
    ds.attrs["analysis_window"] = "the puff"
    assert window_captions(open_view(tree(ds))) == [DEFAULT_CAPTION]


def test_the_view_reads_the_one_file_the_user_rewords(tree, monkeypatch, tmp_path):
    """Edit the plain summary's file and the stored-mask view follows on its next rerun: it keeps no copy."""
    reworded = tmp_path / "reworded.md"
    reworded.write_text("A sentence the user wrote later.\n", encoding="utf-8")
    monkeypatch.setattr(dead_pixels, "PLAIN_SUMMARY_PATH", str(reworded))
    app = open_view(tree(preprocessed(ESTIMATED_SHOT)))
    assert "A sentence the user wrote later." in markdown(app)
    assert not any("In plain words" in m for m in markdown(app))

    reworded.write_text("And another, after the first edit.\n", encoding="utf-8")
    app.run()
    assert "And another, after the first edit." in markdown(app)
    assert not any("A sentence the user wrote later." in m for m in markdown(app))
