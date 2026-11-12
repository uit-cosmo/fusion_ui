"""Click-to-pixel mapping for the frame viewer: pure logic, no Streamlit runtime."""

import numpy as np

from fusion_ui.plots import raw

X_AXIS = np.array([88.0, 88.4, 88.8])
Y_AXIS = np.array([-4.5, -4.1, -3.7])


def to_pixel(point):
    return raw._point_to_pixel(point, X_AXIS, Y_AXIS, (3, 3))


def test_physical_click_maps_to_nearest_pixel():
    assert to_pixel({"x": 88.41, "y": -4.09}) == (1, 1)
    assert to_pixel({"x": 88.0, "y": -4.5}) == (0, 0)


def test_unmappable_point_is_none_not_an_exception():
    assert to_pixel({"curve_number": 2}) is None


def test_click_grid_customdata_maps_exactly():
    """What the invisible click grid reports: its own ``[iy, ix, value]``,
    whatever the coordinates say."""
    assert to_pixel({"x": 88.0, "y": -4.5, "customdata": [2, 1]}) == (2, 1)
    assert to_pixel({"curve_number": 1, "point_number": 7, "customdata": [2, 1, 0.37]}) == (2, 1)


def test_out_of_bounds_customdata_falls_back_to_coordinates():
    assert to_pixel({"x": 88.8, "y": -3.7, "customdata": [99, 99]}) == (2, 2)


def frame_figure():
    values = np.zeros((4, 5))
    x_axis, y_axis = np.arange(5).astype(float), np.arange(4).astype(float)
    return raw._frame_figure(values, x_axis, y_axis, ("x", "y"), (1, 2), "Plasma")


def test_frame_figure_carries_a_click_grid_covering_every_cell():
    grid = frame_figure().data[1]
    assert len(grid.x) == 20
    cells = {(int(c[0]), int(c[1])) for c in grid.customdata}
    assert cells == {(iy, ix) for iy in range(4) for ix in range(5)}
    assert grid.marker.opacity == 0


def test_the_click_grid_is_the_only_hoverable_trace():
    """Clicks reach Streamlit only as Plotly selections, and select-on-click
    starts from the hover data. So the click grid must not be
    ``hoverinfo="skip"``, and the heatmap under it must be, or it takes the
    hover and yields a point no trace can select."""
    figure = frame_figure()
    heatmap, grid, marker = figure.data
    assert heatmap.hoverinfo == "skip"
    assert grid.hoverinfo != "skip"
    assert grid.hovertemplate
    assert marker.hoverinfo == "skip"
    # No cutoff on the hover search: a click between two cells finds the nearest.
    assert figure.layout.hoverdistance == -1
    assert figure.layout.clickmode == "event+select"
    # Streamlit turns select-on-click off when the dragmode is select or lasso.
    assert figure.layout.dragmode == "pan"
