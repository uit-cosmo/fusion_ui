"""Click-to-pixel mapping for the frame viewer -- pure logic, no Streamlit runtime."""

import numpy as np

from fusion_ui.core import decimate
from fusion_ui.plots import raw


def test_physical_click_maps_to_nearest_pixel():
    x_axis = np.array([88.0, 88.4, 88.8])
    y_axis = np.array([-4.5, -4.1, -3.7])
    assert raw._point_to_pixel({"x": 88.41, "y": -4.09}, x_axis, y_axis, (3, 3)) == (
        1,
        1,
    )
    assert raw._point_to_pixel({"x": 88.0, "y": -4.5}, x_axis, y_axis, (3, 3)) == (
        0,
        0,
    )


def test_cell_indices_win_over_coordinates_when_in_bounds():
    x_axis = np.array([88.0, 88.4, 88.8])
    y_axis = np.array([-4.5, -4.1, -3.7])
    # Coordinates point at (0, 0) but the backend also reports cell (2, 1).
    point = {"x": 88.0, "y": -4.5, "point_number": [2, 1]}
    assert raw._point_to_pixel(point, x_axis, y_axis, (3, 3)) == (2, 1)


def test_out_of_bounds_cell_falls_back_to_coordinates():
    x_axis = np.array([88.0, 88.4, 88.8])
    y_axis = np.array([-4.5, -4.1, -3.7])
    point = {"x": 88.8, "y": -3.7, "point_number": [99, 99]}
    assert raw._point_to_pixel(point, x_axis, y_axis, (3, 3)) == (2, 2)


def test_unmappable_point_is_none_not_an_exception():
    assert (
        raw._point_to_pixel({"curve_number": 1}, np.arange(3), np.arange(3), (3, 3))
        is None
    )


def test_selection_points_accepts_dict_and_attribute_spellings():
    dict_event = {"selection": {"points": [{"x": 1.0, "y": 2.0}]}}
    assert decimate.selection_points(dict_event) == [{"x": 1.0, "y": 2.0}]

    class AttrSelection:
        points = [{"x": 3.0, "y": 4.0}]

    class AttrEvent:
        selection = AttrSelection()

    assert decimate.selection_points(AttrEvent()) == [{"x": 3.0, "y": 4.0}]
    assert decimate.selection_points(None) == []
    assert decimate.selection_points(object()) == []
