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
        raw._point_to_pixel({"curve_number": 2}, np.arange(3), np.arange(3), (3, 3))
        is None
    )


def test_click_grid_customdata_maps_exactly():
    # What the invisible click-grid overlay reports: its own [iy, ix] stamp,
    # whatever the axes look like.
    x_axis = np.array([88.0, 88.4, 88.8])
    y_axis = np.array([-4.5, -4.1, -3.7])
    point = {"curve_number": 1, "point_number": 7, "customdata": [2, 1]}
    assert raw._point_to_pixel(point, x_axis, y_axis, (3, 3)) == (2, 1)


def test_scalar_point_number_is_a_row_major_cell_index():
    x_axis = np.array([88.0, 88.4, 88.8])
    y_axis = np.array([-4.5, -4.1, -3.7])
    assert raw._point_to_pixel({"point_number": 0}, x_axis, y_axis, (3, 3)) == (0, 0)
    assert raw._point_to_pixel({"point_number": 8}, x_axis, y_axis, (3, 3)) == (2, 2)


def test_out_of_bounds_customdata_falls_back_to_coordinates():
    x_axis = np.array([88.0, 88.4, 88.8])
    y_axis = np.array([-4.5, -4.1, -3.7])
    point = {"x": 88.8, "y": -3.7, "customdata": [99, 99]}
    assert raw._point_to_pixel(point, x_axis, y_axis, (3, 3)) == (2, 2)


def test_frame_figure_carries_a_click_grid_covering_every_cell():
    import numpy as np

    values = np.zeros((4, 5))
    x_axis = np.arange(5).astype(float)
    y_axis = np.arange(4).astype(float)
    figure = raw._frame_figure(values, x_axis, y_axis, ("x", "y"), (1, 2), "Plasma")

    grid = figure.data[1]
    assert len(grid.x) == 20
    cells = {(int(c[0]), int(c[1])) for c in grid.customdata}
    assert cells == {(iy, ix) for iy in range(4) for ix in range(5)}
    assert grid.marker.opacity == 0


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


def test_click_grid_customdata_carries_the_cell_value_too():
    x_axis = np.array([88.0, 88.4, 88.8])
    y_axis = np.array([-4.5, -4.1, -3.7])
    point = {"curve_number": 1, "point_number": 7, "customdata": [2, 1, 0.37]}
    assert raw._point_to_pixel(point, x_axis, y_axis, (3, 3)) == (2, 1)


def test_the_click_grid_is_the_only_hoverable_trace():
    """Clicks reach Streamlit only as Plotly *selections*, and
    ``selectOnClick`` starts from the hover data -- so the click grid must
    not be ``hoverinfo="skip"``, and the heatmap under it must be, or it
    steals the hover and yields a point no trace can select."""
    values = np.zeros((4, 5))
    figure = raw._frame_figure(
        values, np.arange(5).astype(float), np.arange(4).astype(float),
        ("x", "y"), (1, 2), "Plasma",
    )
    heatmap, grid, marker = figure.data
    assert heatmap.hoverinfo == "skip"
    assert grid.hoverinfo != "skip"
    assert grid.hovertemplate
    assert marker.hoverinfo == "skip"
    # No cutoff on the hover search: a click between two cells must still
    # find the nearest one instead of being dropped.
    assert figure.layout.hoverdistance == -1
    assert figure.layout.clickmode == "event+select"
    # Streamlit forces clickmode back to plain "event" -- no select-on-click
    # -- whenever the dragmode is select or lasso.
    assert figure.layout.dragmode == "pan"
