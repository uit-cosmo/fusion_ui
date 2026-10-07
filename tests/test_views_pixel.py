"""The pixel level's builders: the lag strip, the large frame, the tracks and the numbers.

Synthetic products again (``tests/fields_fixtures``). The contour rules are also checked on a
hand-made field whose answers are known without running anything.
"""

import time

import numpy as np
import plotly.graph_objects as go
import pytest
import xarray as xr

from fusion_ui.views import PIXEL_VIEWS, frame, lag_strip, numbers, pixel, tracks
from fusion_ui.views.bundle import Bundle
from fusion_ui.views.methods import METHODS
from fusion_ui.views.overlays import contours, neighbour_level
from tests import fields_fixtures as ff

US = 1e6


@pytest.fixture(scope="module")
def products():
    return ff.make_products(
        failed=((7, 4),),
        no_events=((6, 7),),
        fit_half_width_us={(5, 4): 4.0, (3, 3): 2.0},
    )


def bundle_of(products, pixel=(5, 4), **kwargs):
    return Bundle(
        shot=1160616027,
        bank=products.bank,
        fields=products.fields,
        blobs=products.blobs,
        pixel=pixel,
        **kwargs,
    )


def heatmaps(figure):
    return [t for t in figure.data if isinstance(t, go.Heatmap)]


def message_of(figure):
    assert not figure.data, "a message figure has nothing drawn in it"
    return " ".join(a.text for a in figure.layout.annotations)


# -- the contour rules, on a field whose answers are known ---------------------------------------


def bump(peaks, sigma=1.2):
    """A (y, x, time) field of 7 x 7 pixels: a Gaussian bump of height ``peaks[k]`` at lag ``k``."""
    y, x = np.mgrid[0:7, 0:7]
    frames = [
        peak * np.exp(-((x - 3) ** 2 + (y - 3) ** 2) / (2 * sigma**2)) for peak in peaks
    ]
    return xr.DataArray(
        np.stack(frames, axis=-1),
        dims=("y", "x", "time"),
        coords={
            "R": (("y", "x"), 0.88 + 0.004 * x),
            "Z": (("y", "x"), -0.04 + 0.004 * y),
            "time": np.arange(len(peaks)) * 1e-6,
        },
    )


def test_one_absolute_level_is_used_for_every_lag():
    field = bump([10.0, 6.0, 2.0])
    # Half of the maximum over all lags is 5: the first two lags reach it, the third does not.
    assert sorted(contours(field, [0, 1, 2], 0.5)) == [0, 1]
    assert sorted(contours(field, [0, 1, 2], 0.7)) == [0]
    # Asking for fewer lags must not move the level to the ones asked for: lag 1 alone, or without
    # the peak lag, is still contoured at 5 and not at half of its own peak.
    assert sorted(contours(field, [1, 2], 0.5)) == [1]
    assert sorted(contours(field, [2], 0.5)) == []
    assert sorted(contours(field, [1], 0.7)) == []


def library_outline(field, index, level):
    """What ``get_contour_evolution`` gives for one lag at ``level`` times the field's maximum."""
    import imaging_methods as im

    threshold = level * float(field.max())
    frame = field.isel(time=[index])
    evolution = im.get_contour_evolution(
        frame, threshold / float(frame.max()), max_displacement_threshold=None
    )
    points = evolution["contours"].isel(time=0).values
    points = points[np.isfinite(points).all(axis=1)]
    return points[:, 0], points[:, 1]


def test_the_outline_is_the_one_the_library_computes_for_the_stored_tracks(products):
    """The point of using the same primitives: what is drawn is what the centroid track was fitted on."""
    bundle = bundle_of(products)
    for name, level in (("cond_av", 0.49), ("cross_corr", 0.81), ("cond_repr", 0.6)):
        field = pixel.reference_field(bundle, name)
        mine = contours(field, range(0, 61, 3), level)
        assert mine, name
        for index, (r, z) in mine.items():
            expected = library_outline(field, index, level)
            np.testing.assert_allclose(r, expected[0], rtol=0, atol=1e-12)
            np.testing.assert_allclose(z, expected[1], rtol=0, atol=1e-12)


def test_with_two_contours_the_one_that_encloses_the_most_is_drawn_as_the_library_chooses():
    y, x = np.mgrid[0:9, 0:9]
    frames = 10 * np.exp(-((x - 2) ** 2 + (y - 2) ** 2) / (2 * 0.8**2)) + 7 * np.exp(
        -((x - 6) ** 2 + (y - 6) ** 2) / (2 * 1.6**2)
    )
    field = xr.DataArray(
        frames[:, :, None],
        dims=("y", "x", "time"),
        coords={
            "R": (("y", "x"), 0.88 + 0.004 * x),
            "Z": (("y", "x"), -0.04 + 0.004 * y),
            "time": [0.0],
        },
    )
    r, z = contours(field, [0], 0.5)[0]
    expected = library_outline(field, 0, 0.5)
    np.testing.assert_allclose(r, expected[0], atol=1e-12)
    np.testing.assert_allclose(z, expected[1], atol=1e-12)
    # It is the wide, weaker blob at (6, 6) that encloses the most, not the sharper one at (2, 2).
    assert r.mean() == pytest.approx(0.88 + 6 * 0.004, abs=0.002)


def test_a_lag_that_cannot_be_contoured_loses_its_own_outline_and_no_other(monkeypatch):
    from skimage import measure

    field = bump([10.0, 9.0, 8.0])
    real = measure.find_contours
    calls = []

    def flaky(frame, threshold, *args, **kwargs):
        calls.append(float(frame.max()))
        if len(calls) == 2:  # the second lag
            raise ValueError("array must be two-dimensional")
        return real(frame, threshold, *args, **kwargs)

    monkeypatch.setattr(measure, "find_contours", flaky)
    assert sorted(contours(field, [0, 1, 2], 0.5)) == [0, 2]
    assert len(calls) == 3
    # A lag that has decayed below the level is not even looked at.
    calls.clear()
    assert sorted(contours(bump([10.0, 2.0]), [0, 1], 0.5)) == [0]
    assert len(calls) == 1


def test_the_contour_is_at_the_level_times_the_maximum():
    field = bump([10.0])
    r, z = contours(field, [0], 0.5)[0]
    # Half maximum of a Gaussian of sigma 1.2 pixels sits at sigma * sqrt(2 ln 2) from its centre.
    radius = np.hypot(r - 0.88 - 3 * 0.004, z + 0.04 - 3 * 0.004)
    assert radius.mean() == pytest.approx(
        1.2 * np.sqrt(2 * np.log(2)) * 0.004, rel=0.08
    )
    # Higher level, smaller contour.
    r2, z2 = contours(field, [0], 0.8)[0]
    assert np.hypot(r2 - 0.892, z2 + 0.028).mean() < radius.mean()


def test_the_contour_level_is_the_peaks_neighbours_over_the_peak():
    frame_ = np.zeros((5, 5, 3))
    frame_[2, 2, 1] = 10.0
    frame_[1, 2, 1], frame_[3, 2, 1], frame_[2, 1, 1], frame_[2, 3, 1] = (
        6.0,
        4.0,
        8.0,
        2.0,
    )
    time = np.array([-1.0, 0.0, 1.0])
    assert neighbour_level(frame_, time) == pytest.approx(0.5)
    # A peak on the border averages the neighbours it has.
    frame_[:, :, 1] = 0
    frame_[0, 0, 1], frame_[1, 0, 1], frame_[0, 1, 1] = 10.0, 5.0, 3.0
    assert neighbour_level(frame_, time) == pytest.approx(0.4)
    # Further out when the view is sampled finer than the structure.
    frame_[:, :, 1] = 0
    frame_[2, 2, 1], frame_[0, 2, 1] = 10.0, 1.0
    assert neighbour_level(frame_, time, step=2) == pytest.approx(0.025)
    # Nothing positive to take a level from.
    assert np.isnan(neighbour_level(np.zeros((3, 3, 3)), time))


# -- the lag strip --------------------------------------------------------------------------------


def test_the_strip_is_a_row_per_field_and_a_column_per_lag(products):
    figure = lag_strip.lag_strip(bundle_of(products), n=5)
    assert len(heatmaps(figure)) == 2 * 5
    assert len(figure.layout.meta["lags"]) == 5
    assert figure.layout.meta["lags"] == sorted(figure.layout.meta["lags"])
    # Seven columns when asked for seven.
    assert len(heatmaps(lag_strip.lag_strip(bundle_of(products), n=7))) == 14


def test_each_row_keeps_one_colour_scale_across_its_lags(products):
    bundle = bundle_of(products)
    figure = lag_strip.lag_strip(bundle, n=5)
    for row, name in enumerate(("cond_av", "cross_corr")):
        axis = "coloraxis" if row == 0 else "coloraxis2"
        row_maps = [h for h in heatmaps(figure) if h.coloraxis == axis]
        assert len(row_maps) == 5
        field = pixel.reference_field(bundle, name).values
        assert figure.layout[axis].cmax == pytest.approx(
            np.nanmax(field)
        )  # over every lag, not the ones shown
        assert figure.layout[axis].cmin == 0.0
        # A per-frame scale would put every frame's peak at the top. The outer lags have decayed.
        peaks = [np.nanmax(h.z) for h in row_maps]
        assert max(peaks) == pytest.approx(np.nanmax(field), rel=0.2) and min(
            peaks
        ) < 0.9 * max(peaks)


def test_the_default_span_is_the_fitted_lags_widened_by_a_quarter(products):
    # (5, 4) is fitted on +-4 us, a width of 8: a quarter of it either side makes +-6.
    choice = lag_strip.choose_lags(bundle_of(products, pixel=(5, 4)))
    assert choice.source == "fit" and choice.span_us == pytest.approx(6.0)
    assert [
        products.bank["time"].values[i] * US for i in choice.indices
    ] == pytest.approx([-6, -3, 0, 3, 6])
    # (3, 3) on +-2 us: +-3.
    assert lag_strip.default_span_us(bundle_of(products, pixel=(3, 3))) == (
        pytest.approx(3.0),
        "fit",
    )
    # An ordinary pixel is on the fixture's +-4.5.
    assert lag_strip.default_span_us(bundle_of(products, pixel=(1, 3)))[
        0
    ] == pytest.approx(7.0)


def test_the_deck_lags_when_the_track_did_not_fit(products):
    # (7, 4): the centroid track fitted nothing, so there is no span of the pixel's own.
    span, source = lag_strip.default_span_us(bundle_of(products, pixel=(7, 4)))
    assert source == "deck" and span == pytest.approx(
        10.0
    )  # the deck's 20 frames of 0.5 us
    # No method_fields at all: the same.
    bank_only = Bundle(bank=products.bank, pixel=(5, 4))
    assert lag_strip.default_span_us(bank_only) == (pytest.approx(10.0), "deck")
    assert len(heatmaps(lag_strip.lag_strip(bank_only))) == 10


def test_a_span_or_typed_lags_replace_the_default(products):
    bundle = bundle_of(products)
    choice = lag_strip.choose_lags(bundle, n=3, span_us=9.0)
    assert choice.source == "span"
    assert [
        products.bank["time"].values[i] * US for i in choice.indices
    ] == pytest.approx([-9, 0, 9])

    typed = lag_strip.choose_lags(bundle, n=9, span_us=9.0, typed="-6, −2, 0; 2 6")
    assert typed.source == "typed" and typed.span_us is None
    assert [
        products.bank["time"].values[i] * US for i in typed.indices
    ] == pytest.approx([-6, -2, 0, 2, 6])
    # Lags snap to the lag axis.
    near = lag_strip.choose_lags(bundle, typed="2.3 -2.2")
    assert [
        products.bank["time"].values[i] * US for i in near.indices
    ] == pytest.approx([-2.0, 2.5])


def test_the_lags_offered_stop_at_the_banks_window(products):
    bundle = bundle_of(products)
    assert lag_strip.choose_lags(bundle, span_us=100.0).span_us == pytest.approx(15.0)
    choice = lag_strip.choose_lags(bundle, typed="-30 -5 0 5 30")
    assert choice.dropped == (-30.0, 30.0)
    assert [
        products.bank["time"].values[i] * US for i in choice.indices
    ] == pytest.approx([-5, 0, 5])

    # A shot with a shorter window has fewer lags, and its default span stays inside them.
    short = ff.make_products(window=40)
    short_bundle = bundle_of(short)
    assert short.bank.sizes["time"] == 41
    assert lag_strip.choose_lags(short_bundle, span_us=100.0).span_us == pytest.approx(
        10.0
    )
    assert lag_strip.default_span_us(short_bundle)[0] <= 10.0
    # Typed lags that are all outside leave the default in place and say why, rather than drawing nothing.
    nothing = lag_strip.choose_lags(bundle, typed="40 50")
    assert nothing.error and len(nothing.indices) > 1 and nothing.source == "fit"
    garbled = lag_strip.choose_lags(bundle, typed="two, three")
    assert "two" in garbled.error and len(garbled.indices) > 1
    with pytest.raises(ValueError, match="cannot read"):
        lag_strip.parse_lags("1, x")


def test_the_contour_is_drawn_where_the_average_has_not_decayed(products):
    bundle = bundle_of(products)
    figure = lag_strip.lag_strip(bundle, lags="-15, -3, 0, 3, 15")
    contour = [
        t for t in figure.data if t.name and t.name.startswith("cond_av contour")
    ]
    # At +-15 us the average has decayed below the level: no contour, and that is not a fault.
    assert len(contour) == 3
    level = float(products.fields["level_com"].values[4, 5])
    assert f"{level:.2f}" in contour[0].name
    assert figure.layout.meta["levels"]["cond_av"] == pytest.approx(level)
    # The stored level is the one the centroid track used; cross_corr has none and is read off its own average.
    cross = pixel.reference_field(bundle, "cross_corr")
    assert figure.layout.meta["levels"]["cross_corr"] == pytest.approx(
        neighbour_level(cross.values, products.bank["time"].values)
    )


def test_the_reference_pixel_and_each_tracks_position_are_marked_at_every_lag(products):
    bundle = bundle_of(products)
    figure = lag_strip.lag_strip(bundle, n=5)
    R, Z = bundle.grid
    reference = [t for t in figure.data if t.name == "reference pixel"]
    assert len(reference) == 10  # one in every panel
    assert all(
        t.x[0] == pytest.approx(R[4, 5]) and t.y[0] == pytest.approx(Z[4, 5])
        for t in reference
    )

    lags_index = [
        int(np.argmin(np.abs(products.bank["time"].values - t)))
        for t in figure.layout.meta["lags"]
    ]
    for name, key in (("2DCA max", "max"), ("2DCA centroid", "com"), ("2DCC", "2dcc")):
        marks = sorted(
            (t for t in figure.data if t.name == name),
            key=lambda t: int(t.xaxis[1:] or 1),
        )
        assert len(marks) == 5, name
        expected_r = products.fields[f"pos_r_{key}"].values[4, 5, lags_index]
        expected_z = products.fields[f"pos_z_{key}"].values[4, 5, lags_index]
        # The 2DCC sits in the second row, whose panels are numbered after the first row's five.
        np.testing.assert_allclose([t.x[0] for t in marks], expected_r)
        np.testing.assert_allclose([t.y[0] for t in marks], expected_z)
    # A lag a track did not follow draws nothing: lag +-15 us is outside the 12 us the fixture tracks.
    wide = lag_strip.lag_strip(bundle, lags="-15 0 15")
    assert len([t for t in wide.data if t.name == "2DCA centroid"]) == 1


def test_a_dead_pixel_or_one_without_events_says_so_instead_of_drawing(products):
    dead = lag_strip.lag_strip(bundle_of(products, pixel=(2, 0)))
    assert "dead" in message_of(dead)
    empty = lag_strip.lag_strip(bundle_of(products, pixel=(6, 7)))
    assert "No events" in message_of(empty)
    outside = lag_strip.lag_strip(bundle_of(products, pixel=(40, 2)))
    assert "outside" in message_of(outside)
    assert "No pixel" in message_of(
        lag_strip.lag_strip(bundle_of(products, pixel=None))
    )
    assert "not available" in message_of(
        lag_strip.lag_strip(Bundle(fields=products.fields, pixel=(5, 4)))
    )


def test_a_failed_pixel_still_draws_its_average(products):
    figure = lag_strip.lag_strip(bundle_of(products, pixel=(7, 4)))
    assert len(heatmaps(figure)) == 10
    assert figure.layout.meta["source"] == "deck"


# -- the large frame ------------------------------------------------------------------------------


def test_the_frame_keeps_one_colour_scale_across_the_lags(products):
    bundle = bundle_of(products)
    first, second = frame.frame_figure(bundle, "cond_av", -10.0), frame.frame_figure(
        bundle, "cond_av", 0.0
    )
    assert first.layout.coloraxis.cmin == second.layout.coloraxis.cmin == 0.0
    assert first.layout.coloraxis.cmax == second.layout.coloraxis.cmax
    field = pixel.reference_field(bundle, "cond_av").values
    assert second.layout.coloraxis.cmax == pytest.approx(np.nanmax(field))
    # The frames differ, the scale does not.
    assert not np.allclose(heatmaps(first)[0].z, heatmaps(second)[0].z)


def test_the_frame_is_the_nearest_lag_and_says_which(products):
    bundle = bundle_of(products)
    figure = frame.frame_figure(bundle, "cond_av", 3.2)
    assert figure.layout.meta["lag"] * US == pytest.approx(3.0)
    assert "+3.0" in figure.layout.title.text
    index = figure.layout.meta["index"]
    np.testing.assert_allclose(
        heatmaps(figure)[0].z,
        pixel.reference_field(bundle, "cond_av").values[:, :, index],
    )
    assert frame.lag_index(bundle, 1000.0) == products.bank.sizes["time"] - 1


def test_each_field_has_its_own_tracks_and_unknown_fields_are_refused(products):
    bundle = bundle_of(products)
    marked = lambda name: {
        t.name for t in frame.frame_figure(bundle, name, 0.0).data
    }  # noqa: E731
    assert {"2DCA max", "2DCA centroid"} <= marked("cond_av") and "2DCC" not in marked(
        "cond_av"
    )
    assert "2DCC" in marked("cross_corr") and "2DCA max" not in marked("cross_corr")
    assert not {"2DCA max", "2DCA centroid", "2DCC"} & marked("cond_repr")
    with pytest.raises(ValueError, match="field"):
        frame.frame_figure(bundle, "frames")
    line = frame.trace_figure(bundle, "cond_repr", 4.0)
    assert line.layout.shapes[0].x0 == pytest.approx(4.0)
    assert "dead" in message_of(frame.frame_figure(bundle_of(products, pixel=(2, 0))))


# -- the tracks -----------------------------------------------------------------------------------


def named(figure, name, axis):
    return [t for t in figure.data if t.name == name and t.yaxis == axis]


def test_each_track_is_drawn_as_a_displacement_from_the_reference_with_its_fitted_lags_highlighted(
    products,
):
    bundle = bundle_of(products)
    figure = tracks.tracks_figure(bundle)
    R, Z = bundle.grid
    lag = products.fields["time"].values * US
    for key, label in (("max", "2DCA max"), ("com", "2DCA centroid"), ("2dcc", "2DCC")):
        fit = products.fields[f"fit_{key}"].values[4, 5]
        for axis, comp, ref in (("y", "r", R[4, 5]), ("y2", "z", Z[4, 5])):
            position = (products.fields[f"pos_{comp}_{key}"].values[4, 5] - ref) * 1e3
            (all_lags,) = named(figure, label, axis)
            valid = np.isfinite(position)
            np.testing.assert_allclose(all_lags.x, lag[valid])
            np.testing.assert_allclose(all_lags.y, position[valid])
            (highlight,) = named(figure, f"{label}: lags of the fit", axis)
            np.testing.assert_allclose(
                highlight.x, lag[fit]
            )  # exactly the lags the slope rests on
            np.testing.assert_allclose(highlight.y, position[fit])
            assert highlight.line.width > 2 and all_lags.marker.opacity < 1


def test_the_fitted_line_is_the_stored_velocity(products):
    bundle = bundle_of(products)
    figure = tracks.tracks_figure(bundle)
    lag = products.fields["time"].values * US
    R, Z = bundle.grid
    for key in ("max", "com", "2dcc"):
        for axis, comp, ref in (("y", "r", R[4, 5]), ("y2", "z", Z[4, 5])):
            line = next(
                t
                for t in figure.data
                if t.mode == "lines"
                and t.line.dash == "dash"
                and t.yaxis == axis
                and t.legendgroup == key
            )
            velocity = products.fields[
                f"v{comp.replace('r', 'r').replace('z', 'z')}_{key}"
            ].values[4, 5]
            # m/s is 1e-3 mm per microsecond; the line is the velocity, whatever it is.
            assert np.polyfit(line.x, line.y, 1)[0] == pytest.approx(
                velocity / 1e3, rel=1e-9
            )
            # Through the mean of the fitted points: the least-squares line itself, for lsq.
            fit = products.fields[f"fit_{key}"].values[4, 5]
            position = (products.fields[f"pos_{comp}_{key}"].values[4, 5] - ref) * 1e3
            assert np.mean(line.y) == pytest.approx(position[fit].mean())
            assert np.mean(line.x) == pytest.approx(lag[fit].mean())
    text = " ".join(a.text for a in figure.layout.annotations)
    for key in ("max", "com", "2dcc"):
        assert f"v_R = {products.fields[f'vr_{key}'].values[4, 5]:.0f} m/s" in text
    assert "events" in text


def test_the_tde_velocities_are_drawn_as_the_straight_lines_they_imply(products):
    figure = tracks.tracks_figure(bundle_of(products))
    for points, name in (("3", "3TDE (CC)"), ("2", "2TDE (CC)")):
        for axis, comp in (("y", "r"), ("y2", "z")):
            (line,) = named(figure, name, axis)
            velocity = products.fields[f"v{comp}{points}_tde"].values[4, 5]
            assert line.line.dash == "dashdot"
            assert (line.y[1] - line.y[0]) / (line.x[1] - line.x[0]) == pytest.approx(
                velocity / 1e3
            )
            assert line.y[0] == pytest.approx(
                velocity * line.x[0] / 1e3
            )  # through the reference at zero lag
    # They are not allowed to set the axis: the range is the tracks'.
    spans = [
        np.ptp(
            np.concatenate(
                [t.y for t in figure.data if t.yaxis == "y" and t.mode != "lines"]
            )
        )
    ]
    assert (
        tuple(figure.layout.yaxis.range)[1] - tuple(figure.layout.yaxis.range)[0]
        < 4 * spans[0]
    )


def test_a_dead_pixel_or_one_without_events_has_no_tracks_to_draw(products):
    assert "dead" in message_of(tracks.tracks_figure(bundle_of(products, pixel=(2, 0))))
    assert "No events" in message_of(
        tracks.tracks_figure(bundle_of(products, pixel=(6, 7)))
    )
    assert "not computed" in message_of(
        tracks.tracks_figure(Bundle(bank=products.bank, pixel=(5, 4)))
    )
    # A pixel whose fit failed still shows what the tracks did, and no fitted line.
    failed = tracks.tracks_figure(bundle_of(products, pixel=(7, 4)))
    assert isinstance(failed, go.Figure) and failed.data
    assert not [
        t
        for t in failed.data
        if t.mode == "lines" and t.line.dash == "dash" and t.legendgroup == "com"
    ]


# -- the numbers ----------------------------------------------------------------------------------


def test_the_methods_table_has_every_methods_numbers(products):
    methods_table, blobs_table = numbers.numbers(bundle_of(products))
    assert list(methods_table["method"]) == [m.label for m in METHODS]
    fields = products.fields
    row = methods_table.set_index("method").loc["2DCA centroid"]
    assert row["v_R [m/s]"] == pytest.approx(fields["vr_com"].values[4, 5])
    assert row["v_Z [m/s]"] == pytest.approx(fields["vz_com"].values[4, 5])
    assert row["|v| [m/s]"] == pytest.approx(
        np.hypot(row["v_R [m/s]"], row["v_Z [m/s]"])
    )
    assert (
        row["lags"] == fields["nlags_com"].values[4, 5]
        and row["level"] == fields["level_com"].values[4, 5]
    )
    assert row["events"] == fields["nevents"].values[4, 5] and np.isnan(row["CC"])
    tde = methods_table.set_index("method").loc["3TDE (CC)"]
    assert tde["CC"] == pytest.approx(fields["cc_tde"].values[4, 5]) and np.isnan(
        tde["lags"]
    )
    # Numbers stay numbers, so the columns sort as numbers.
    assert all(methods_table[c].dtype.kind == "f" for c in methods_table.columns[1:])


def test_the_blob_parameters_come_with_units_when_they_are_computed(products):
    _, blobs_table = numbers.numbers(bundle_of(products))
    assert list(blobs_table["parameter"]) == list(ff.BLOB_NAMES)
    assert blobs_table.set_index("parameter").loc["lr", "value"] == pytest.approx(
        products.blobs["lr"].values[4, 5]
    )
    assert blobs_table.set_index("parameter").loc["area", "unit"] == "m²"
    assert blobs_table.set_index("parameter").loc["taud", "unit"] == "s"
    assert blobs_table.set_index("parameter").loc["theta_f", "unit"] == "rad"
    without = numbers.numbers(
        Bundle(fields=products.fields, bank=products.bank, pixel=(5, 4))
    )
    assert len(without) == 1  # the blob table waits for blob_parameters


def test_a_dead_pixel_has_a_sentence_where_the_tables_would_be(products):
    assert "dead" in numbers.numbers(bundle_of(products, pixel=(2, 0)))
    assert "No events" in numbers.numbers(bundle_of(products, pixel=(6, 7)))
    assert numbers.method_table(Bundle(bank=products.bank, pixel=(5, 4))).empty
    assert numbers.blob_table(Bundle(pixel=(5, 4))) is None


# -- every figure builder returns a figure, whatever the pixel --------------------------------------


@pytest.mark.parametrize(
    "pixel_in_view",
    [(5, 4), (7, 4), (6, 7), (2, 0), (0, 0), (8, 9), (40, 40), None],
    ids=[
        "ordinary",
        "failed fit",
        "no events",
        "dead",
        "corner",
        "other corner",
        "off the array",
        "none",
    ],
)
def test_every_figure_builder_returns_a_figure_whatever_the_pixel(
    products, pixel_in_view
):
    bundle = bundle_of(products, pixel=pixel_in_view)
    bank_only = Bundle(bank=products.bank, pixel=pixel_in_view)
    fields_only = Bundle(fields=products.fields, pixel=pixel_in_view)
    for built in (
        lag_strip.lag_strip(bundle),
        lag_strip.lag_strip(bank_only),
        frame.frame_figure(bundle),
        frame.frame_figure(bank_only, "cross_corr", 5.0),
        frame.trace_figure(bundle),
        tracks.tracks_figure(bundle),
        tracks.tracks_figure(fields_only),
    ):
        assert isinstance(built, go.Figure)
        built.to_json()  # and every one can be sent


# -- the lists the page is made from --------------------------------------------------------------


def test_the_pixel_views_are_entries_that_name_what_they_read(products):
    assert [v.key for v in PIXEL_VIEWS] == ["lag_strip", "frame", "tracks", "numbers"]
    reads = {v.key: v.reads for v in PIXEL_VIEWS}
    assert reads["lag_strip"] == ("pixel_averages",) and reads["tracks"] == (
        "method_fields",
    )
    bundle = bundle_of(products)
    for view in PIXEL_VIEWS:
        controls = view.controls(bundle) if view.controls else ()
        values = {c.key: c.default for c in controls}
        result = view.build(bundle, **values)
        for item in result if isinstance(result, list) else [result]:
            assert isinstance(item, (go.Figure, str)) or hasattr(item, "columns")
    strip_controls = {c.key: c for c in PIXEL_VIEWS[0].controls(bundle)}
    assert (
        strip_controls["span_us"].default is None
        and "±6" in strip_controls["span_us"].placeholder
    )
    assert strip_controls["span_us"].bounds[1] == pytest.approx(15.0)
    frame_controls = {c.key: c for c in PIXEL_VIEWS[1].controls(bundle)}
    assert frame_controls["lag"].bounds == (
        pytest.approx(-15.0),
        pytest.approx(15.0),
        pytest.approx(0.5),
    )


# -- speed ----------------------------------------------------------------------------------------


def build_pixel_level(bundle):
    """Every pixel-level view, and the JSON the browser is sent for its figures."""
    for view in PIXEL_VIEWS:
        controls = view.controls(bundle) if view.controls else ()
        result = view.build(bundle, **{c.key: c.default for c in controls})
        for item in result if isinstance(result, list) else [result]:
            if isinstance(item, go.Figure):
                item.to_json()


def test_building_the_pixel_level_from_a_loaded_bank_takes_well_under_a_second(
    products, capsys
):
    bundle = bundle_of(products)
    build_pixel_level(
        bundle
    )  # the first call pays for imaging_methods' import, which the page has done already
    wall, cpu = time.perf_counter(), time.process_time()
    build_pixel_level(bundle)
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    with capsys.disabled():
        print(
            f"\npixel level from a loaded bank: {wall * 1000:.0f} ms wall, {cpu * 1000:.0f} ms CPU (builders and JSON)"
        )
    # CPU time, because a shared machine delays wall time without making the builders any slower.
    assert cpu < 1.0
