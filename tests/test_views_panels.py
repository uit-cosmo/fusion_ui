"""The shot level's builders: seven panels on one scale, and what every pixel of them is.

Built on synthetic products (``tests/fields_fixtures``): a blob of known velocity crossing a 10 x 9
array with the 22 dead pixels of the hand-made 1160616 mask.
"""

import json

import numpy as np
import plotly.graph_objects as go
import pytest

from fusion_ui.views import arrows, methods, panels
from fusion_ui.views.bundle import Bundle, Cuts
from fusion_ui.views.methods import METHODS, Status
from tests import fields_fixtures as ff


@pytest.fixture(scope="module")
def products():
    return ff.make_products(failed=((7, 4),), no_events=((6, 7),))


def bundle_of(products, **kwargs):
    return Bundle(
        shot=1160616027,
        bank=products.bank,
        fields=products.fields,
        blobs=products.blobs,
        **kwargs,
    )


def traces(figure, kind, panel=None):
    found = [t for t in figure.data if t.meta and t.meta.get("kind") == kind]
    return [t for t in found if panel is None or t.meta.get("panel") == panel]


def pairs(trace):
    """The ``(x, y)`` pixels a marker trace carries as customdata."""
    return {(int(c[0]), int(c[1])) for c in trace.customdata}


def argwhere_xy(mask):
    return {(int(x), int(y)) for y, x in np.argwhere(mask)}


# -- every pixel is one of four things ------------------------------------------------------------


def test_every_pixel_of_a_panel_is_exactly_one_of_ok_failed_cut_or_dead(products):
    bundle = bundle_of(products, cuts=Cuts(min_lags=8, min_events=450))
    for method in METHODS:
        p = methods.panel(bundle, method)
        assert set(np.unique(p.status)) <= {int(s) for s in Status}
        assert (p.status == Status.DEAD).sum() == 22
        assert ((p.status == Status.DEAD) == products.geometry.dead).all()
        # Whatever is not OK says why, and OK pixels say nothing.
        assert all(p.reason[p.status != Status.OK])
        assert not any(p.reason[p.status == Status.OK])


def test_a_live_pixel_with_no_estimate_is_a_failed_fit_and_not_a_dead_pixel(products):
    bundle = bundle_of(products)
    p = methods.panel(bundle, methods.METHOD_BY_KEY["com"])
    assert p.status[4, 7] == Status.FAILED  # the pixel whose fit rests on one lag
    assert "1 lag" in p.reason[4, 7]
    assert p.status[7, 6] == Status.FAILED  # no events at its reference
    assert "no events" in p.reason[7, 6]
    assert p.status[0, 2] == Status.DEAD and "mask" in p.reason[0, 2]


def test_the_view_cuts_apply_where_they_mean_something(products):
    lags = methods.panel(
        bundle_of(products, cuts=Cuts(min_lags=20)), methods.METHOD_BY_KEY["com"]
    )
    # Every fit in the fixture rests on 19 lags: all live, fitted pixels fall under a minimum of 20.
    assert (lags.status == Status.CUT).sum() == 90 - 22 - 2
    assert "19 lags < 20" in lags.reason[4, 5]
    # A TDE has no lags to cut on.
    tde = methods.panel(
        bundle_of(products, cuts=Cuts(min_lags=20)), methods.METHOD_BY_KEY["tde3"]
    )
    assert (tde.status == Status.CUT).sum() == 0

    events = Cuts(min_events=10_000)
    for key, cut in (("com", True), ("catde3", True), ("tde3", False)):
        p = methods.panel(bundle_of(products, cuts=events), methods.METHOD_BY_KEY[key])
        assert bool((p.status == Status.CUT).any()) is cut, key

    edge = methods.panel(
        bundle_of(products, cuts=Cuts(interior_only=True)),
        methods.METHOD_BY_KEY["tde2"],
    )
    cut_pixels = argwhere_xy(edge.status == Status.CUT)
    assert cut_pixels and all(x in (0, 8) or y in (0, 9) for x, y in cut_pixels)
    assert all("edge pixel" in edge.reason[y, x] for x, y in cut_pixels)


def test_a_missing_variable_is_all_failed_not_an_error(products):
    bundle = bundle_of(products)
    fields = products.fields.drop_vars(["vr_2dcc", "vz_2dcc", "nlags_2dcc"])
    p = methods.panel(
        Bundle(fields=fields, bank=products.bank), methods.METHOD_BY_KEY["2dcc"]
    )
    assert (p.status == Status.FAILED).sum() == 90 - 22
    figure = panels.velocity_panels(Bundle(fields=fields, bank=products.bank))
    assert isinstance(figure, go.Figure)
    assert bundle.dead.sum() == 22


# -- the figure ---------------------------------------------------------------------------------


def test_each_mode_returns_a_figure(products):
    for mode in panels.MODES:
        assert isinstance(panels.velocity_panels(bundle_of(products), mode), go.Figure)
    with pytest.raises(ValueError, match="mode"):
        panels.velocity_panels(bundle_of(products), "quiver")


def test_all_panels_share_one_arrow_scale(products):
    figure = panels.velocity_panels(bundle_of(products))
    scale = figure.layout.meta["arrow_scale"]
    for method in METHODS:
        pixels, shafts = (
            traces(figure, "pixels", method.key)[0],
            traces(figure, "shafts", method.key)[0],
        )
        sx, sy = np.asarray(shafts.x).reshape(-1, 3), np.asarray(shafts.y).reshape(
            -1, 3
        )
        length = np.hypot(sx[:, 1] - sx[:, 0], sy[:, 1] - sy[:, 0])
        speed = np.array(pixels.customdata)[:, 4]
        # Every arrow of every panel is its speed over the one scale.
        np.testing.assert_allclose(length * scale, speed, rtol=1e-9)

    # And that scale is the figure's: the 90th-percentile speed over every panel at once.
    pooled = np.concatenate(
        [
            methods.panel(bundle_of(products), m).speed[
                methods.panel(bundle_of(products), m).ok
            ]
            for m in METHODS
        ]
    )
    R, _ = bundle_of(products).grid
    expected = arrows.arrow_scale(pooled, R.max() - R.min())
    assert scale == pytest.approx(expected)
    assert scale == pytest.approx(
        np.percentile(pooled, 90) / (0.18 * (R.max() - R.min()))
    )


def test_there_is_one_key_and_it_is_drawn_at_the_scale(products):
    figure = panels.velocity_panels(bundle_of(products))
    scale, key_speed = (
        figure.layout.meta["arrow_scale"],
        figure.layout.meta["key_speed"],
    )
    (shaft,) = traces(figure, "key-shaft")
    assert len(traces(figure, "key-head")) == 1
    assert shaft.x[1] - shaft.x[0] == pytest.approx(key_speed / scale)
    # A round number: 1, 2 or 5 times a power of ten.
    mantissa = key_speed / 10 ** np.floor(np.log10(key_speed))
    assert mantissa in (1.0, 2.0, 5.0)
    assert any(f"{key_speed:g} m/s" in a.text for a in figure.layout.annotations)


def test_the_arrow_gain_moves_the_one_scale_and_not_a_panel(products):
    base = panels.velocity_panels(bundle_of(products))
    longer = panels.velocity_panels(bundle_of(products), arrow_gain=2.0)
    assert longer.layout.meta["arrow_scale"] == pytest.approx(
        base.layout.meta["arrow_scale"] / 2
    )
    assert longer.layout.meta["key_speed"] == base.layout.meta["key_speed"]

    def lengths(figure, panel):
        shafts = traces(figure, "shafts", panel)[0]
        x, y = np.asarray(shafts.x).reshape(-1, 3), np.asarray(shafts.y).reshape(-1, 3)
        return np.hypot(x[:, 1] - x[:, 0], y[:, 1] - y[:, 0])

    for (
        method
    ) in (
        METHODS
    ):  # every arrow of every panel is twice as long, so the scale is still one
        np.testing.assert_allclose(
            lengths(longer, method.key), 2 * lengths(base, method.key)
        )
    (key,), (key_longer,) = traces(base, "key-shaft"), traces(longer, "key-shaft")
    assert key_longer.x[1] - key_longer.x[0] == pytest.approx(2 * (key.x[1] - key.x[0]))
    # Maps have no arrows to lengthen, and a gain that is not a length is refused.
    np.testing.assert_array_equal(
        traces(panels.velocity_panels(bundle_of(products), "vr", 3.0), "map", "com")[
            0
        ].z,
        traces(panels.velocity_panels(bundle_of(products), "vr"), "map", "com")[0].z,
    )
    for bad in (0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="arrow_gain"):
            panels.velocity_panels(bundle_of(products), arrow_gain=bad)


def test_every_cell_shows_the_same_stretch_of_the_array_at_equal_aspect(products):
    layout = panels.velocity_panels(bundle_of(products)).layout
    ranges = set()
    for k in range(1, 9):
        x_axis = layout["xaxis" if k == 1 else f"xaxis{k}"]
        y_axis = layout["yaxis" if k == 1 else f"yaxis{k}"]
        assert (
            y_axis.scaleanchor == ("x" if k == 1 else f"x{k}")
            and y_axis.scaleratio == 1
        )
        assert x_axis.constrain == "domain" and y_axis.constrain == "domain"
        ranges.add((tuple(x_axis.range), tuple(y_axis.range)))
    # The key cell included: a metre is as long in it as in any panel, so the key can be held up to them.
    assert len(ranges) == 1


def test_dead_pixels_and_failed_fits_are_marked_differently(products):
    figure = panels.velocity_panels(bundle_of(products))
    dead, failed = traces(figure, "dead", "com")[0], traces(figure, "failed", "com")[0]
    assert dead.marker.symbol != failed.marker.symbol
    assert dead.marker.color != failed.marker.color
    assert dead.name != failed.name
    # The marks sit on the right pixels, whichever panel.
    assert pairs(dead) == argwhere_xy(products.geometry.dead)
    assert pairs(failed) == {(7, 4), (6, 7)}
    # Hovering says which it is.
    assert "dead pixel" in dead.text[0] and "no fit" in failed.text[0]


def test_each_mark_is_in_the_legend_once(products):
    figure = panels.velocity_panels(bundle_of(products, cuts=Cuts(min_events=450)))
    names = [t.name for t in figure.data if t.showlegend]
    assert sorted(names) == ["cut by the view", "dead pixel (mask)", "no fit"]


def test_nan_everywhere_still_draws_and_marks_every_live_pixel_as_failed(products):
    fields = products.fields.copy(deep=True)
    for name in list(fields.data_vars):
        if name.startswith(("vr", "vz")):
            fields[name][:] = np.nan
    bundle = Bundle(fields=fields, bank=products.bank)
    figure = panels.velocity_panels(bundle)
    assert isinstance(figure, go.Figure)
    assert figure.layout.meta["arrow_scale"] is None  # nothing to scale off
    assert not traces(figure, "shafts") and not traces(figure, "key-shaft")
    assert len(pairs(traces(figure, "failed", "com")[0])) == 90 - 22
    json.loads(figure.to_json())  # NaN must not leave the figure unserialisable


def test_the_view_cuts_move_pixels_out_of_the_arrows_and_mark_them(products):
    plain = panels.velocity_panels(bundle_of(products))
    cut = panels.velocity_panels(bundle_of(products, cuts=Cuts(min_events=600)))
    assert not traces(plain, "cut")
    marked = traces(cut, "cut", "com")[0]
    shown = pairs(traces(cut, "pixels", "com")[0])
    events = products.fields["nevents"].values
    assert pairs(marked) == {
        (x, y) for y, x in np.argwhere(events < 600) if not products.geometry.dead[y, x]
    } - {(7, 4)} - {(6, 7)}
    assert not (shown & pairs(marked))
    assert "events < 600" in marked.text[0]
    # The CC-based TDE does not rest on events, so no pixel of it is cut by them.
    assert not traces(cut, "cut", "tde3")


def test_the_maps_share_one_diverging_scale(products):
    figure = panels.velocity_panels(bundle_of(products), "vr")
    axis = figure.layout.coloraxis
    assert axis.cmin == -axis.cmax and axis.cmid == 0
    maps = traces(figure, "map")
    assert len(maps) == 7 and all(m.coloraxis == "coloraxis" for m in maps)
    # NaN where a pixel is not a number, so it shows blank under its mark.
    com = traces(figure, "map", "com")[0]
    z = np.asarray(com.z)
    panel = methods.panel(bundle_of(products), methods.METHOD_BY_KEY["com"])
    assert np.isnan(z[~panel.ok]).all() and np.isfinite(z[panel.ok]).all()
    shown = np.concatenate(
        [np.asarray(m.z)[np.isfinite(np.asarray(m.z))] for m in maps]
    )
    assert axis.cmax == pytest.approx(np.percentile(np.abs(shown), 95))
    # v_Z has its own limit on the same machinery.
    assert (
        panels.velocity_panels(bundle_of(products), "vz").layout.meta["colour_limit"]
        != figure.layout.meta["colour_limit"]
    )


def test_hovering_a_pixel_of_any_panel_shows_its_numbers(products):
    figure = panels.velocity_panels(bundle_of(products))
    for method in METHODS:
        (dots,) = traces(figure, "pixels", method.key)
        assert "v_R = %{customdata[2]:.0f} m/s" in dots.hovertemplate
        assert method.label in dots.hovertemplate
    # A track says how many lags, a TDE how well correlated, the rest how many events.
    assert "lags" in traces(figure, "pixels", "com")[0].hovertemplate
    assert "CC" in traces(figure, "pixels", "tde3")[0].hovertemplate
    assert "events" in traces(figure, "pixels", "catde2")[0].hovertemplate
    # The numbers are the pixel's own.
    row = next(
        r
        for r in traces(figure, "pixels", "com")[0].customdata
        if (r[0], r[1]) == (5, 4)
    )
    assert row[2] == pytest.approx(products.fields["vr_com"].values[4, 5])
    assert row[3] == pytest.approx(products.fields["vz_com"].values[4, 5])
    assert row[5] == products.fields["nlags_com"].values[4, 5]


def test_every_marker_carries_its_pixel_as_plain_lists_that_survive_json(products):
    """A click is read back from ``customdata``. Plotly 6+ writes a numpy array as a typed array,
    which the browser hands back as an object, not as ``[x, y]``: the figure must send lists.
    """
    sent = json.loads(panels.velocity_panels(bundle_of(products), "arrows").to_json())
    kinds = {
        t["meta"]["kind"]: t for t in sent["data"] if "meta" in t and "customdata" in t
    }
    assert {"pixels", "dead", "failed"} <= set(kinds)
    for kind in ("pixels", "dead", "failed"):
        rows = kinds[kind]["customdata"]
        assert isinstance(rows, list) and all(
            isinstance(row, list) for row in rows
        ), kind
        assert all(
            isinstance(row[0], (int, float)) and isinstance(row[1], (int, float))
            for row in rows
        )
    # The dead pixels' [x, y] are exactly the mask's.
    assert {tuple(row) for row in kinds["dead"]["customdata"]} == argwhere_xy(
        products.geometry.dead
    )


def test_the_pixel_in_view_is_ringed_in_every_panel(products):
    figure = panels.velocity_panels(bundle_of(products, pixel=(5, 4)))
    rings = traces(figure, "selected")
    assert len(rings) == 7
    R, Z = bundle_of(products).grid
    assert all(
        r.x[0] == pytest.approx(R[4, 5]) and r.y[0] == pytest.approx(Z[4, 5])
        for r in rings
    )
    assert not traces(panels.velocity_panels(bundle_of(products)), "selected")
    # A pixel that is not on this array is not drawn, and is not an error.
    assert not traces(
        panels.velocity_panels(bundle_of(products, pixel=(40, 40))), "selected"
    )


def test_without_the_fields_the_figure_says_so(products):
    figure = panels.velocity_panels(Bundle(bank=products.bank))
    assert isinstance(figure, go.Figure)
    assert "not computed" in figure.layout.annotations[0].text


def test_arrow_helpers():
    assert arrows.arrow_scale([100.0, 200.0, 300.0], 0.03) == pytest.approx(
        np.percentile([100, 200, 300], 90) / (0.18 * 0.03)
    )
    assert (
        arrows.arrow_scale([], 0.03) is None
        and arrows.arrow_scale([np.nan, 0.0, -1.0], 0.03) is None
    )
    assert arrows.arrow_scale([100.0], 0.0) is None
    assert (
        arrows.nice_speed([480.0, 510.0]) == 500.0
        and arrows.nice_speed([1100.0]) == 1000.0
    )
    assert arrows.nice_speed([]) is None
    # Heads point along the shaft, in degrees clockwise from up.
    np.testing.assert_allclose(
        arrows.head_angles([0, 1, 0, -1], [1, 0, -1, 0]), [0, 90, 180, -90]
    )
    xs, ys = arrows.shafts([0.0, 1.0], [0.0, 1.0], [2.0, 3.0], [0.0, 1.0])
    assert np.isnan(xs[2]) and np.isnan(xs[5]) and list(xs[:2]) == [0.0, 2.0]
