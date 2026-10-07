"""The array as the figures draw it: axes, reading order, and a click read back off a chart."""

import types

import numpy as np
import pytest

from fusion_ui.views import geometry
from tests import fields_fixtures as ff


@pytest.fixture(scope="module")
def grid():
    geo = ff.make_geometry()
    return geo.R, geo.Z, geo.dead


def test_a_heatmap_axis_is_the_mean_of_each_column_and_row(grid):
    R, Z, _ = grid
    x_axis, y_axis = geometry.grid_axes(R, Z)
    assert x_axis.shape == (9,) and y_axis.shape == (10,)
    np.testing.assert_allclose(x_axis, R.mean(axis=0))
    np.testing.assert_allclose(y_axis, Z.mean(axis=1))
    assert (np.diff(x_axis) > 0).all() and (np.diff(y_axis) > 0).all()
    # The pitch of the fixture's array: a little under four millimetres.
    assert geometry.pitch(R, Z) == pytest.approx(0.0039, rel=0.05)


def test_a_grid_a_heatmap_cannot_place_falls_back_to_the_pixel_indices(grid):
    R, Z, _ = grid
    scrambled = R.copy()
    scrambled[:, 3] = scrambled[:, 6]  # columns that no longer increase
    x_axis, y_axis = geometry.grid_axes(scrambled, Z)
    np.testing.assert_array_equal(x_axis, np.arange(9))
    np.testing.assert_allclose(y_axis, Z.mean(axis=1))
    nan = np.full((3, 4), np.nan)
    x_axis, y_axis = geometry.grid_axes(nan, nan)
    np.testing.assert_array_equal(x_axis, np.arange(4))
    np.testing.assert_array_equal(y_axis, np.arange(3))


def test_reading_order_is_the_top_row_first_left_to_right(grid):
    R, Z, _ = grid
    order = geometry.reading_order(R, Z)
    assert len(order) == 90 and len(set(order)) == 90
    assert (
        order[0] == (0, 9)
        and order[8] == (8, 9)
        and order[9] == (0, 8)
        and order[-1] == (8, 0)
    )
    # Whichever way the rows run: the top row is the one with the largest Z.
    flipped = geometry.reading_order(R, Z[::-1])
    assert flipped[0] == (0, 0) and flipped[-1] == (8, 9)


def test_previous_and_next_walk_the_live_pixels_and_skip_the_dead(grid):
    R, Z, dead = grid
    order = geometry.reading_order(R, Z)
    live = [p for p in order if not dead[p[1], p[0]]]
    assert len(live) == 68
    # Forward from the first live pixel visits every live pixel once, in reading order, and wraps.
    seen, here = [live[0]], live[0]
    for _ in range(67):
        here = geometry.step_pixel(order, dead, here, +1)
        seen.append(here)
    assert seen == live
    assert geometry.step_pixel(order, dead, live[-1], +1) == live[0]
    assert geometry.step_pixel(order, dead, live[0], -1) == live[-1]
    # Back is the inverse of forward.
    assert geometry.step_pixel(order, dead, live[10], -1) == live[9]
    # (0, 9) is dead, as is (1, 9): the first live pixel of the top row is (2, 9).
    assert live[0] == (2, 9)


def test_stepping_from_a_dead_pixel_goes_to_the_neighbouring_live_one(grid):
    R, Z, dead = grid
    order = geometry.reading_order(R, Z)
    assert dead[9, 0] and geometry.step_pixel(order, dead, (0, 9), +1) == (2, 9)
    assert geometry.step_pixel(order, dead, (3, 9), -1) == (2, 9)
    assert geometry.step_pixel(order, dead, (3, 9), +1) == (4, 9)
    # A pixel that is not on the array starts from the nearest end.
    assert geometry.step_pixel(order, dead, (40, 40), +1) == (2, 9)
    assert not dead[0, 8] and geometry.step_pixel(order, dead, (40, 40), -1) == (8, 0)
    # With nothing live there is nowhere to go.
    assert geometry.step_pixel(order, np.ones_like(dead), (4, 4), +1) == (4, 4)


# -- reading a click ------------------------------------------------------------------------------


def event(*custom):
    return {
        "selection": {
            "points": [{"customdata": c} for c in custom],
            "box": [],
            "lasso": [],
        }
    }


def test_a_click_names_the_pixel_in_its_customdata():
    assert geometry.pixel_from_event(event([5, 4, 400.0, 100.0])) == (5, 4)
    assert geometry.pixel_from_event(event([5.0, 4.0])) == (
        5,
        4,
    )  # JSON has one number type
    assert geometry.pixel_from_event(event((2, 7))) == (2, 7)
    assert geometry.pixel_from_event(event([5, 4], [3, 9], [3, 2])) == (3, 2)


def test_a_typed_array_that_went_through_json_is_still_a_click():
    """The browser hands a Float64Array back as an object keyed "0", "1", ... : the failure the
    first real click found. Figures now send lists, and this reads the object anyway."""
    assert geometry.pixel_from_event(event({"0": 5.0, "1": 4.0, "2": 400.0})) == (5, 4)
    assert geometry.pixel_from_event(event({0: 1.0, 1: 2.0})) == (1, 2)


def test_anything_that_is_not_a_pixel_is_no_click_and_not_an_error():
    for custom in (None, [], [5], "ab", {"x": 1}, [None, 4], ["a", "b"]):
        assert geometry.pixel_from_event(event(custom)) is None, custom
    assert geometry.pixel_from_event(None) is None
    assert geometry.pixel_from_event({}) is None
    assert geometry.pixel_from_event({"selection": {"points": []}}) is None
    assert geometry.pixel_from_event({"selection": {}}) is None
    # One malformed point does not hide a good one.
    assert geometry.pixel_from_event(event(None, [6, 1])) == (6, 1)


def test_the_event_may_be_an_object_with_attributes():
    state = types.SimpleNamespace(
        selection=types.SimpleNamespace(
            points=[types.SimpleNamespace(customdata=[4, 4])]
        )
    )
    assert geometry.pixel_from_event(state) == (4, 4)
