"""The Thomson scattering profiles view: reading the file, the figure, the words around it, and the pages that show it.

Each test runs on a synthetic file in the layout of the real ones (``tests/thomson_fixtures.py``).
"""

import base64
import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

import fusion_ui.plots  # noqa: F401 - registers the specs the page offers
from fusion_ui.core import catalog, db, loader, multipixel, params_ui, registry, traces
from fusion_ui.plots import thomson_profiles as tp
from tests import thomson_fixtures as tf

REPO = Path(__file__).resolve().parent.parent
SINGLE_SHOT = str(REPO / "fusion_ui" / "pages" / "2_single_shot.py")
BROWSER = str(REPO / "fusion_ui" / "pages" / "1_shot_browser.py")
STATISTICS = str(REPO / "fusion_ui" / "pages" / "4_statistics.py")

#: The label of the fit in the legend when it was made over the fixture's window.
FIT = "fit, 1–1.4 s"
ROWS = ["edge TS", "core TS", FIT]
AXES = ("rho_pol", "R_mid")
#: The layout's names for the rows n_e, T_e and p_e, top to bottom.
YAXES = ("yaxis", "yaxis2", "yaxis3")


def cut(ds, window=tf.WINDOW):
    """``ds`` cut to a window the way the single-shot page cuts the file it opens."""
    return loader.sliced(ds, *window)


def profiles_of(window=tf.WINDOW, **options):
    return tp.read(cut(tf.thomson_dataset(**options), window))


def figure_of(axis="rho_pol", **options):
    return tp.figure(profiles_of(**options), axis)


def in_row(fig, n):
    """The traces of row ``n`` (1 is the top), in the order they were added."""
    axis = "y" if n == 1 else f"y{n}"
    return [trace for trace in fig.data if trace.yaxis == axis]


def names_in_row(fig, n):
    return [trace.name for trace in in_row(fig, n)]


class Target:
    """What ``notes`` reads of the page's target: the window the points are cut to."""

    def __init__(self, t_start, t_end):
        self.t_start, self.t_end = t_start, t_end


FITS_WINDOW = Target(*tf.WINDOW)


def notes_of(found, axis="rho_pol", target=FITS_WINDOW):
    return tp.notes(found, target, axis)


# ---------------------------------------------------------------------------
# The spec and the catalog
# ---------------------------------------------------------------------------


def test_the_spec_is_live_for_the_raw_thomson_scattering_file_with_the_axis_as_its_one_parameter():
    spec = registry.get("thomson_profiles")
    assert spec.label == "Thomson scattering profiles"
    assert not spec.cached and spec.compute is None and spec.scalars is None and spec.overlay is None
    assert registry.for_diagnostic("thomson_scattering", False) == [spec]
    assert registry.for_diagnostic("thomson_scattering", True) == []
    assert all(spec not in registry.for_diagnostic(other, False) for other in ("apd", "phantom", "asp", "fsp"))
    assert not multipixel.supported(spec)
    assert spec.params().axis == "rho_pol"
    assert params_ui.CHOICES[("ThomsonProfilesParams", "axis")] == AXES


def test_the_catalog_indexes_the_thomson_scattering_folder_raw_only(conn, tmp_path, monkeypatch):
    root = tmp_path / "alcator"
    tf.write_thomson(root, 1160616027)
    tf.write_thomson(root, 1140827019)
    stats = catalog.rescan(conn, str(root), "cmod", None)
    assert stats.inserted == 2 and stats.skipped == 0 and "thomson_scattering" not in stats.missing_folders

    found = {(row["shot"], row["diagnostic"], row["preprocessed"]): row for row in conn.execute("SELECT * FROM shots")}
    assert set(found) == {(1160616027, "thomson_scattering", 0), (1140827019, "thomson_scattering", 0)}
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(root))
    path = loader.dataset_path("cmod", 1160616027, "thomson_scattering", False)
    assert path == found[(1160616027, "thomson_scattering", 0)]["path"]
    assert path.endswith("thomson_scattering/thomson_scattering_1160616027.nc")

    table = catalog.shot_table(conn, None).set_index("shot")
    assert list(table["thomson_scattering"]) == ["R", "R"]
    assert list(table["apd"]) == ["", ""]  # a shot with only this file lists no other diagnostic


def test_the_file_the_fixture_writes_opens_the_way_the_page_opens_the_real_ones(tmp_path):
    """netCDF3 as the real files are, ``time`` in seconds and not decoded."""
    ds = loader.open_dataset(str(tf.write_thomson(tmp_path)))
    assert ds["time"].dtype == np.float64 and ds["time"].attrs["units"] == "s"
    assert set(ds.dims) == {"time", "rho_pol", "edge_channel", "core_channel"}
    window = loader.sliced(ds, *tf.WINDOW)
    assert 0 < window.sizes["time"] < ds.sizes["time"] and window.sizes["rho_pol"] == ds.sizes["rho_pol"]


# ---------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------


def test_only_the_readings_with_a_measurement_are_kept_and_their_junk_errors_never_are():
    ds = cut(tf.thomson_dataset())
    found = tp.read(ds)
    assert set(found.points) == {"edge", "core"} and found.pulses == ds.sizes["time"]
    for system, points in found.points.items():
        ne = ds[f"{system}_ne"].values
        assert (ne == 0).any(), "the fixture plants readings of 0"
        assert points.readings == int(np.isfinite(ne).sum())  # a pulse the system did not record is no reading
        assert points.drawn == int((ne > 0).sum()) < points.readings
        assert (points.ne > 0).all() and (points.Te > 0).all()
        # The tree's error at a reading of 0 is 2e20 m^-3 and 1000 eV: none of it is kept.
        assert np.nanmax(points.ne_err) < 1e20 and np.nanmax(points.Te_err) < 500


def test_a_point_carries_where_and_when_it_was_read():
    ds = cut(tf.thomson_dataset())
    points = tp.read(ds).points["edge"]
    i = 5
    c = int(points.channel[i])
    row = int(np.argmin(abs(ds["time"].values - points.time[i])))
    assert ds["edge_ne"].values[row, c] == points.ne[i]
    assert ds["edge_rho_pol"].values[row, c] == points.rho_pol[i]
    assert ds["edge_R_mid"].values[row, c] == points.R_mid[i]
    assert ds["edge_Z"].values[c] == points.Z[i] and ds["edge_R"].values[c] == points.R[i]


def test_a_temperature_of_0_beside_a_density_is_not_drawn_and_a_bad_error_leaves_no_bar():
    ds = cut(tf.thomson_dataset(zeros=False))
    ds["edge_Te"][2, 3] = 0.0  # the density at this reading stays
    ds["edge_ne_err"][4, 1] = -1.0
    ds["edge_Te_err"][4, 1] = np.nan
    ds["edge_ne_err"][6, 2] = 0.0
    ds["edge_Te_err"][8, 0] = 0.0  # an error of exactly 0 is no error either
    ds["edge_ne"][3, 2] = 0.0  # a density of 0 beside a temperature that reads: the rule is n_e > 0
    points = tp.read(ds).points["edge"]
    assert points.drawn == ds["edge_ne"].size - 1

    def at(row, channel):
        found = np.flatnonzero((points.time == ds["time"].values[row]) & (points.channel == channel))
        return int(found[0]) if found.size else None

    assert at(3, 2) is None  # not drawn, though its T_e is not 0
    assert np.isnan(points.Te[at(2, 3)]) and points.ne[at(2, 3)] > 0 and np.isnan(points.pe()[at(2, 3)])
    assert np.isnan(points.ne_err[at(4, 1)]) and np.isnan(points.Te_err[at(4, 1)])
    assert np.isnan(points.ne_err[at(6, 2)])
    assert np.isnan(points.Te_err[at(8, 0)]) and np.isfinite(points.ne_err[at(8, 0)])
    assert np.isfinite(points.ne_err[at(5, 1)])


def test_the_fit_the_scalars_and_the_attributes_are_read_and_each_is_optional():
    found = profiles_of()
    assert found.fit.window == tf.WINDOW and found.fit.rho_pol[0] == 0.0 and found.fit.rho_pol[-1] == 1.1
    assert found.fit.pe is not None and found.fit.R_mid is not None
    assert found.scalars == pytest.approx({"R_mid_sep": 0.89, "Te_sep_2pm": 36.0, "lambda_q_2pm": 0.002})
    assert found.attrs["fit_check"] == tf.PASSED and found.attrs["shot"] == tf.SHOT
    assert profiles_of(pe=False).fit.pe is None
    assert profiles_of(fit=False).fit is None
    assert profiles_of(scalars=False).scalars == {}
    assert set(profiles_of(core=False).points) == {"edge"} and set(profiles_of(edge=False).points) == {"core"}


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------


def test_each_row_has_both_systems_and_the_fit_on_one_shared_axis_and_the_legend_lists_each_once():
    fig = figure_of()
    assert [names_in_row(fig, n) for n in (1, 2, 3)] == [ROWS] * 3
    # One shared axis: the upper two follow the bottom one, which carries the title.
    assert fig.layout.xaxis.matches == "x3" and fig.layout.xaxis2.matches == "x3"
    assert fig.layout.xaxis3.title.text == "ρ<sub>pol</sub>" and not fig.layout.xaxis.title.text
    assert [fig.layout[k].title.text for k in YAXES] == [
        "n<sub>e</sub> [10<sup>20</sup> m<sup>-3</sup>]",
        "T<sub>e</sub> [eV]",
        "p<sub>e</sub> [kPa]",
    ]
    # The two systems differ by marker and by colour.
    edge, core = in_row(fig, 1)[:2]
    assert (edge.marker.symbol, core.marker.symbol) == ("circle", "diamond") and edge.marker.color != core.marker.color
    # A legend entry appears once and hides its trace in every row; the separatrix is a line of every row.
    assert [trace.name for trace in fig.data if trace.showlegend is not False] == ROWS
    assert [trace.legendgroup for trace in in_row(fig, 3)] == ["edge", "core", "fit"]
    assert [(s.name, s.legendgroup, s.showlegend, s.x0, s.line.dash) for s in fig.layout.shapes] == [
        ("separatrix", "separatrix", shown, 1.0, "dash") for shown in (True, False, False)
    ]


def test_the_values_and_their_bars_are_drawn_in_the_units_of_the_axis_titles():
    found = profiles_of()
    points = found.points["core"]
    fig = tp.figure(found)
    ne, te, pe = (in_row(fig, n)[1] for n in (1, 2, 3))
    np.testing.assert_allclose(ne.y, points.ne * 1e-20)  # 10^20 m^-3
    np.testing.assert_allclose(te.y, points.Te)  # eV
    e = 1.602176634e-19
    np.testing.assert_allclose(pe.y, e * points.ne * points.Te * 1e-3)  # kPa
    np.testing.assert_allclose(ne.error_y.array, points.ne_err * 1e-20)
    np.testing.assert_allclose(te.error_y.array, points.Te_err)
    propagated = e * np.sqrt((points.Te * points.ne_err) ** 2 + (points.ne * points.Te_err) ** 2) * 1e-3
    np.testing.assert_allclose(pe.error_y.array, propagated)
    np.testing.assert_allclose(in_row(fig, 3)[2].y, found.fit.pe * 1e-3)


def test_the_fit_is_drawn_over_its_whole_range_without_a_band():
    found = profiles_of()
    fig = tp.figure(found)
    for n in (1, 2, 3):
        fit = in_row(fig, n)[2]
        assert fit.mode == "lines" and fit.x[0] == 0.0 and fit.x[-1] == 1.1 and len(fit.x) == found.fit.rho_pol.size
        assert fit.error_y.array is None
    assert not [trace for trace in fig.data if trace.fill not in (None, "none")]
    assert {trace.name for trace in fig.data if trace.mode == "lines"} == {FIT}  # no edge of a band either


def test_on_the_r_mid_axis_the_points_the_fit_and_the_separatrix_are_in_r_mid():
    found = profiles_of()
    fig = tp.figure(found, "R_mid")
    np.testing.assert_allclose(in_row(fig, 1)[0].x, found.points["edge"].R_mid)
    np.testing.assert_allclose(in_row(fig, 1)[2].x, found.fit.R_mid)
    assert fig.layout.xaxis3.title.text == "R<sub>mid</sub> [m]"
    assert len(fig.layout.shapes) == 3 and all(s.x0 == pytest.approx(0.89) for s in fig.layout.shapes)


def test_the_separatrix_on_the_r_mid_axis_is_the_scalar_else_the_fits_at_rho_pol_1_else_none():
    assert tp.separatrix(profiles_of(), "R_mid") == pytest.approx(0.89)
    assert tp.separatrix(profiles_of(scalars=False, fit=False), "rho_pol") == 1.0
    # A file without the scalar reads it off the fit's own R_mid at rho_pol = 1.
    assert tp.separatrix(profiles_of(scalars=False), "R_mid") == pytest.approx(tf.r_mid(1.0), abs=1e-9)
    # Neither: no line.
    none = profiles_of(scalars=False, fit=False)
    assert tp.separatrix(none, "R_mid") is None and not tp.figure(none, "R_mid").layout.shapes
    # The scalar comes first, even where the fit's own R_mid would say another number.
    found = profiles_of()
    found.scalars["R_mid_sep"] = 0.9
    assert tp.separatrix(found, "R_mid") == 0.9
    # A fit_R_mid that is NaN at rho_pol = 1 says nothing: no separatrix, not a line at NaN.
    ds = tf.thomson_dataset(scalars=False)
    ds["fit_R_mid"] = ds["fit_R_mid"].where(~((ds["rho_pol"] > 0.9) & (ds["rho_pol"] < 1.05)))
    nan = tp.read(cut(ds))
    assert tp.separatrix(nan, "R_mid") is None and not tp.figure(nan, "R_mid").layout.shapes
    assert notes_of(nan, "R_mid")[2].endswith("The file gives no separatrix.")


def test_only_what_the_file_holds_is_drawn():
    fig = figure_of(fit=False)
    assert [names_in_row(fig, n) for n in (1, 2, 3)] == [["edge TS", "core TS"]] * 3  # p_e is e n_e T_e of the points
    fig = figure_of(pe=False)  # a fit without a pressure fit leaves the pressure row to the points
    assert [names_in_row(fig, n) for n in (1, 2, 3)] == [ROWS, ROWS, ["edge TS", "core TS"]]
    assert {trace.name for trace in figure_of(core=False).data} == {"edge TS", FIT}
    assert {trace.name for trace in figure_of(edge=False, fit=False).data} == {"core TS"}
    # A system whose points have no place is left out, and the other is drawn.
    found = profiles_of()
    found.points["edge"].rho_pol[:] = np.nan
    assert {trace.name for trace in tp.figure(found).data} == {"core TS", FIT}


def test_a_window_with_no_pulse_still_draws_the_fit_and_a_file_with_nothing_says_so():
    empty = profiles_of(window=(5.0, 6.0))
    assert empty.pulses == 0 and all(p.drawn == 0 for p in empty.points.values())
    fig = tp.figure(empty)
    assert {trace.name for trace in fig.data} == {FIT} and len(fig.layout.shapes) == 3

    fig = tp.figure(profiles_of(window=(5.0, 6.0), fit=False))
    assert not fig.data and "No Thomson scattering measurement" in fig.layout.annotations[0].text


def test_a_file_whose_equilibrium_could_not_be_opened_has_nothing_to_place_and_says_why():
    found = profiles_of(efit=False, fit=False, fit_status="failed: ValueError: no EFIT tree")
    assert found.points["edge"].drawn and np.isnan(found.points["edge"].rho_pol).all()
    measured = sum(p.drawn for p in found.points.values())
    for axis in AXES:
        fig = tp.figure(found, axis)
        assert not fig.data
        assert fig.layout.annotations[0].text.startswith(f"None of the {measured} measured points can be placed on")
    note, fit_note, axis_note = notes_of(found)[:3]
    assert "have no ρ_pol, since no EFIT tree could be opened (TreeNODATA): they are not drawn." in note
    assert fit_note == "No fit in this file (failed: ValueError: no EFIT tree)."
    assert "from the EFIT tree" not in axis_note  # there is none to name
    assert tp.separatrix(found, "R_mid") is None


def test_a_fit_without_an_r_mid_is_not_drawn_on_the_r_mid_axis_and_the_caption_says_so():
    found = profiles_of(fit_r_mid=False)
    assert FIT in names_in_row(tp.figure(found, "rho_pol"), 1)
    fig = tp.figure(found, "R_mid")
    assert names_in_row(fig, 1) == ["edge TS", "core TS"]
    assert "The file has no R_mid for the fit, so it is not drawn on this axis." in notes_of(found, "R_mid")[1]
    assert "no R_mid for the fit" not in notes_of(found)[1]
    assert {s.x0 for s in fig.layout.shapes} == {found.scalars["R_mid_sep"]}  # the separatrix still has the scalar


def test_an_unknown_axis_is_refused():
    with pytest.raises(ValueError, match="rho_pol, R_mid"):
        tp.figure(profiles_of(), "psi_n")


def test_points_without_a_flux_mapping_are_not_placed_and_the_notes_count_them():
    found = profiles_of(unmapped=True)
    expected = sum(int(np.isnan(p.rho_pol).sum()) for p in found.points.values())
    assert expected > 0
    fig = tp.figure(found)
    assert np.isnan(in_row(fig, 1)[0].x).sum() == int(np.isnan(found.points["edge"].rho_pol).sum())
    assert f"{expected} of the points have no ρ_pol (the pulse is outside the EFIT time base)" in notes_of(found)[0]
    assert f"{expected} of the points have no R_mid" in notes_of(found, "R_mid")[0]  # named for the axis
    assert "no ρ_pol" not in notes_of(profiles_of())[0]


def test_a_hover_text_says_which_system_when_where_and_what_the_value_is():
    found = profiles_of()
    points = found.points["edge"]
    hover = in_row(tp.figure(found), 2)[0].hovertext  # the T_e row
    assert len(hover) == points.drawn
    first = hover[0]
    assert first.startswith("edge TS · t = ") and f"channel {int(points.channel[0])}" in first
    assert f"ρ_pol = {points.rho_pol[0]:.3f}" in first and "T_e = " in first and " ± " in first and " eV" in first
    assert f"R = {points.R[0]:.4f} m, Z = {points.Z[0]:+.4f} m" in first


@pytest.mark.parametrize(
    "value, digits, text",
    [(2140.0, 3, "2140"), (234.0, 2, "234"), (1.234, 3, "1.23"), (0.00012, 2, "0.00012"), (2.5e7, 3, "2.5e+07")],
)
def test_hover_numbers_have_no_exponent_below_a_million(value, digits, text):
    assert tp._number(value, digits) == text


# ---------------------------------------------------------------------------
# The range of a row
# ---------------------------------------------------------------------------


def toy(x, value, *, error=None, fit=None, r_mid=None, fit_r_mid=None):
    """Profiles whose only points are edge T_e readings of ``value`` eV at ρ_pol ``x``, and whose fit is ``(ρ_pol, T_e)``
    when one is given, so that the rule can be checked against numbers worked out by hand.

    Each point's error is 10% of its value unless ``error`` says otherwise; ``r_mid`` and ``fit_r_mid`` are where the
    points and the fit are on the R_mid axis, the same as on ρ_pol unless given.
    """
    x, value = np.asarray(x, float), np.asarray(value, float)
    error = 0.1 * value if error is None else np.asarray(error, float)
    n, nan = x.size, np.full(x.size, np.nan)
    points = tp.Points(
        system="edge",
        pulses=1,
        readings=n,
        time=np.zeros(n),
        channel=np.arange(n),
        R=nan,
        Z=nan,
        rho_pol=x,
        R_mid=x if r_mid is None else np.asarray(r_mid, float),
        ne=np.ones(n),
        ne_err=nan,
        Te=value,
        Te_err=error,
    )
    found = None
    if fit is not None:
        rho, te = (np.asarray(a, float) for a in fit)
        found = tp.Fit(
            rho_pol=rho,
            ne=te,
            Te=te,
            pe=None,
            R_mid=rho if fit_r_mid is None else np.asarray(fit_r_mid, float),
            window=None,
        )
    return tp.Profiles(points={"edge": points}, fit=found, pulses=1, scalars={}, attrs={})


def top_of(found, axis="rho_pol"):
    return tp.y_limits(found, "Te", axis)[1]


def test_the_top_is_1_1_times_the_98th_percentile_of_the_points_and_not_their_largest():
    found = toy(np.linspace(0.5, 1.0, 100), np.arange(1.0, 101.0))
    bottom, top = tp.y_limits(found, "Te", "rho_pol")
    assert bottom == 0.0
    assert top == pytest.approx(1.1 * 98.02)  # 1 + 0.98 * 99: not 1.1 * 100
    assert tp.points_above(found, "Te", "rho_pol", top) == 0


def test_a_point_measured_to_worse_than_2_sigma_does_not_set_the_top_and_is_counted_above_it():
    value = np.append(np.arange(1.0, 101.0), [1000.0] * 3)
    error = np.append(0.1 * np.arange(1.0, 101.0), [800.0] * 3)  # an error of 800 on 1000 is worse than half the value
    found = toy(np.linspace(0.5, 1.0, 103), value, error=error)
    assert top_of(found) == pytest.approx(1.1 * 98.02)  # as if the three were not there
    assert tp.points_above(found, "Te", "rho_pol", top_of(found)) == 3


def test_a_point_with_no_error_is_not_measured_to_better_than_2_sigma():
    value = np.arange(1.0, 101.0)
    error = np.where(value > 90, np.nan, 0.1 * value)  # the last ten have none
    found = toy(np.linspace(0.5, 1.0, 100), value, error=error)
    assert top_of(found) == pytest.approx(1.1 * 88.22)  # the 98th percentile of 1 to 90


def test_with_fewer_than_5_well_measured_points_all_the_points_set_the_percentile():
    def top(well):
        value = np.array([10.0] * well + [20.0] * (20 - well))
        error = np.array([1.0] * well + [20.0] * (20 - well))  # the others are measured no better than their value
        return top_of(toy(np.linspace(0.5, 1.0, 20), value, error=error))

    assert top(4) == pytest.approx(1.1 * 20)  # four are not enough to go by: all twenty set it
    assert top(5) == pytest.approx(1.1 * 10)  # five are, and the others no longer count


def test_a_fit_within_3_times_the_points_sets_the_top_and_one_that_runs_away_does_not():
    x, value = np.linspace(0.5, 1.0, 40), np.full(40, 100.0)
    rho = np.linspace(0.0, 1.1, 111)

    def top(fit):
        return top_of(toy(x, value, fit=(rho, fit)))

    assert top_of(toy(x, value)) == pytest.approx(110.0)
    assert top(np.full(rho.size, 250.0)) == pytest.approx(1.1 * 250)  # within 3 times the points: it sets the top
    assert top(np.full(rho.size, 300.0)) == pytest.approx(1.1 * 300)  # exactly 3 times counts
    assert top(np.full(rho.size, 301.0)) == pytest.approx(1.1 * 100)  # a little beyond does not: the points do
    assert top(np.where(rho <= 1.0, 100.0, 1e5)) == pytest.approx(110.0)  # follows the points, and then leaves them
    # The bottom stays at zero where the fit goes below it.
    below = toy(x, value, fit=(rho, np.where(rho > 0.5, -200.0, 100.0)))
    assert tp.y_limits(below, "Te", "rho_pol") == pytest.approx((0.0, 110.0))


def test_a_fit_is_judged_against_the_points_beside_it_and_not_against_the_largest_ones():
    x = np.r_[np.linspace(0.1, 0.3, 20), np.linspace(0.9, 1.0, 20)]
    value = np.r_[np.full(20, 100.0), np.full(20, 1000.0)]  # low points inside, high ones at the edge

    def top(at):
        return top_of(toy(x, value, fit=([at], [2000.0])))

    assert top(0.2) == pytest.approx(1.1 * 1000)  # 2000 beside points of 100 is not plausible: the points set it
    assert top(1.0) == pytest.approx(1.1 * 2000)  # beside points of 1000 it is, and it sets the top


def test_a_row_with_no_points_takes_the_whole_fit():
    rho = np.linspace(0.0, 1.1, 111)
    fit_only = toy([], [], fit=(rho, np.where(rho <= 1.0, 100.0, 1e5)))
    assert tp.y_limits(fit_only, "Te", "rho_pol") == pytest.approx((0.0, 1.1e5))  # no points to hold the fit to
    nothing = tp.Profiles(points={}, fit=fit_only.fit, pulses=0, scalars={}, attrs={})  # not even a system
    assert tp.y_limits(nothing, "Te", "rho_pol") == pytest.approx((0.0, 1.1e5))


def test_the_points_and_the_fit_are_judged_on_the_axis_drawn():
    """The same points and fit give another top on R_mid than on ρ_pol when the points beside a sample of the fit are
    other points there: here the two groups of points trade places."""
    x = np.r_[np.linspace(0.5, 0.7, 20), np.linspace(0.9, 1.0, 20)]
    value = np.r_[np.full(20, 100.0), np.full(20, 1000.0)]
    r_mid = np.r_[np.linspace(0.90, 1.00, 20), np.linspace(0.50, 0.60, 20)]
    found = toy(x, value, r_mid=r_mid, fit=([0.6, 1.05], [50.0, 2500.0]), fit_r_mid=[0.55, 0.95])
    assert top_of(found, "rho_pol") == pytest.approx(1.1 * 2500)  # beside points of 1000, at ρ_pol 1.05
    assert top_of(found, "R_mid") == pytest.approx(1.1 * 1000)  # beside points of 100, at R_mid 0.95


def test_a_row_with_nothing_above_zero_has_no_top_and_is_left_to_autorange_from_zero():
    rho = np.linspace(0.0, 1.1, 111)
    found = toy([], [], fit=(rho, np.zeros(rho.size)))
    assert tp.y_limits(found, "Te", "rho_pol") is None
    fig = tp.figure(found)
    for key in ("yaxis", "yaxis2"):  # the n_e and the T_e row: the fit holds both
        options = fig.layout[key].autorangeoptions
        assert fig.layout[key].range is None and options.minallowed == 0 and options.maxallowed is None
    assert not fig.layout.annotations


def test_each_row_opens_on_zero_to_its_top_and_a_runaway_fit_leaves_through_it():
    found = profiles_of(runaway=True)
    plain = tp.figure(profiles_of())
    for axis in AXES:
        fig = tp.figure(found, axis)
        for n, (key, (name, _label, _title, scale, _unit)) in enumerate(zip(YAXES, tp.QUANTITIES), start=1):
            _, top = tp.y_limits(found, name, axis, scale)
            row, options = fig.layout[key], fig.layout[key].autorangeoptions
            assert tuple(row.range) == (0.0, top)  # it opens on it
            assert (options.minallowed, options.maxallowed) == (0.0, top)  # and autoscale is given the same
            fit = in_row(fig, n)[2]
            assert len(fit.x) == found.fit.rho_pol.size and np.max(fit.y) > 50 * top  # drawn whole, far above the top
            if axis == "rho_pol":
                assert row.range == plain.layout[key].range  # where the points put it, as without the runaway


def stray(n, removed=False):
    """The fixture's profiles with ``n`` of its edge readings in the window made 10 keV, measured to no better than
    that. With ``removed`` those readings are instead no measurement at all: the same points without the strays."""
    ds = tf.thomson_dataset()
    times = ds["time"].values
    in_window = (times >= tf.WINDOW[0]) & (times <= tf.WINDOW[1])
    rows, channels = np.nonzero((ds["edge_ne"].values > 0) & in_window[:, None])
    for i, c in list(zip(rows, channels))[:n]:
        if removed:
            ds["edge_ne"][int(i), int(c)] = 0.0
            ds["edge_Te"][int(i), int(c)] = 0.0
        else:
            ds["edge_Te"][int(i), int(c)] = 1e4
            ds["edge_Te_err"][int(i), int(c)] = 1e4
    return tp.read(cut(ds))


def corner_notes(fig):
    """What each row says in its corner: ``{its y axis's domain: text}``."""
    return {note.yref: note.text for note in fig.layout.annotations}


def test_a_row_says_in_its_corner_how_many_points_lie_above_its_top_and_a_row_with_none_says_nothing():
    assert not tp.figure(profiles_of()).layout.annotations  # no point of the fixture is above the top of its row
    fig = tp.figure(stray(3))
    # T_e and p_e (the same readings), and not n_e, which they leave as it was.
    assert corner_notes(fig) == {"y2 domain": "▲ 3 points above", "y3 domain": "▲ 3 points above"}
    # The strays do not set the top: it is where it is with those readings gone.
    assert fig.layout.yaxis2.range == tp.figure(stray(3, removed=True)).layout.yaxis2.range
    assert corner_notes(tp.figure(stray(1))) == {"y2 domain": "▲ 1 point above", "y3 domain": "▲ 1 point above"}
    assert corner_notes(tp.figure(stray(3), "R_mid")) == corner_notes(fig)


@pytest.mark.slow
def test_in_a_browser_a_row_opens_on_its_range_and_autoscale_and_reset_axes_return_to_it():
    """That autoscale keeps the range is a claim about the renderer, so it is read off one. Skipped where playwright or
    its browser is not there."""
    sync_api = pytest.importorskip("playwright.sync_api")
    fig = tp.figure(profiles_of(runaway=True))
    expected = pytest.approx([end for key in YAXES for end in (0.0, fig.layout[key].range[1])])
    read = (
        "() => ['yaxis', 'yaxis2', 'yaxis3'].flatMap("
        "k => document.querySelector('.js-plotly-plot')._fullLayout[k].range)"
    )
    zoom = (
        "() => Plotly.relayout(document.querySelector('.js-plotly-plot'), "
        "{'yaxis.range': [0.5, 0.6], 'yaxis2.range': [100, 110], 'yaxis3.range': [1, 2]})"
    )
    with sync_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception as error:  # no browser installed for playwright here
            pytest.skip(f"no chromium to drive: {error}")
        try:
            page = browser.new_page(viewport={"width": 1000, "height": 1000})
            page.set_content(fig.to_html(include_plotlyjs=True, full_html=True))
            page.wait_for_selector(".js-plotly-plot .nsewdrag")
            page.wait_for_timeout(500)
            assert page.evaluate(read) == expected  # it opens on it, far below the runaway fit
            for button in ("Autoscale", "Reset axes"):
                page.evaluate(zoom)
                assert page.evaluate(read) != expected  # zoomed away from it
                page.hover(".js-plotly-plot .nsewdrag")
                page.click(f'[data-title="{button}"]')
                page.wait_for_timeout(300)
                assert page.evaluate(read) == expected, button
        finally:
            browser.close()


# ---------------------------------------------------------------------------
# The words
# ---------------------------------------------------------------------------

CORE = "Its core is an extrapolation, not a description of the core points"
OUTER = "Its outer end is not credible"
SILENT = "It may have failed silently: read the points, not the curve."
SPELT = f"{tf.CHECK_TEXT}: FAILED: Te on axis is 134 eV"


@pytest.mark.parametrize(
    "text, outcome",
    [
        (tf.PASSED, "passed"),
        (tf.FAILED, "failed: Te on axis is 134 eV"),
        (SPELT, "FAILED: Te on axis is 134 eV"),
        (f"{tf.CHECK_TEXT}:   passed \n", "passed"),
        ("passed", "passed"),  # an outcome without the rule's description
        ("no fit", ""),
        ("   ", ""),
        (None, ""),
        (tf.CHECK_TEXT, tf.CHECK_TEXT),  # the description alone records no outcome: it reads as one no rule knows
    ],
)
def test_the_checks_outcome_is_what_follows_the_rules_description(text, outcome):
    assert tp.check_outcome(text) == outcome


@pytest.mark.parametrize(
    "check, verb, shown, advice, ending",
    [
        # The advice follows the clause that failed.
        (tf.FAILED, "fails", "“failed: Te on axis is 134 eV”", [CORE], f"The file's `fit_check`: “{tf.FAILED}”"),
        (tf.FAILED_OUTER, "fails", "“failed: Te at rho_pol 1.10 is 140 eV”", [OUTER], f"“{tf.FAILED_OUTER}”"),
        (tf.FAILED_BOTH, "fails", "“failed: Te at rho_pol 1.10 is 140 eV; Te on axis is 134 eV”", [CORE, OUTER], "”"),
        # A verdict spelt another way is still a verdict.
        (SPELT, "fails", "“FAILED: Te on axis is 134 eV”", [CORE], f"“{SPELT}”"),
        # An outcome nobody wrote a rule for is not "passed", and names no clause.
        (f"{tf.CHECK_TEXT}: PASSED", "does not pass", "“PASSED”", [SILENT], "”"),
        # The rule's description alone says "on axis" as well, and is no verdict that something failed there.
        (tf.CHECK_TEXT, "does not pass", f"“{tf.CHECK_TEXT}”", [SILENT], "”"),
        # No outcome recorded.
        ("", "does not pass", "the file records no result", [SILENT], "The file's `fit_check` is empty."),
        (None, "does not pass", "the file records no result", [SILENT], "The file has no `fit_check`."),
        ("no fit", "does not pass", "the file records no result", [SILENT], "The file's `fit_check`: “no fit”"),
    ],
)
def test_a_fit_whose_outcome_is_anything_but_passed_is_warned_about_quoting_the_check(
    check, verb, shown, advice, ending
):
    text = tp.warning(profiles_of(check=check))
    assert text.startswith(f"**This fit {verb} run_db.py's own check: {shown}.** ")
    assert [said for said in (CORE, OUTER, SILENT) if said in text] == advice
    assert text.endswith(ending)


def test_there_is_no_warning_for_a_fit_that_passed_or_for_a_file_with_no_fit_to_warn_about():
    assert tp.warning(profiles_of(check=tf.PASSED)) is None
    assert tp.warning(profiles_of(check="passed")) is None
    assert tp.warning(profiles_of(fit=False)) is None  # fit_check reads "no fit": nothing is drawn to warn about


def test_a_note_that_calls_the_core_an_extrapolation_gets_a_caption_even_when_the_check_passed():
    assert tp.extrapolation({"fit_note": f"{tf.FIT_NOTE} {tf.EXTRAPOLATION}"}) == tf.EXTRAPOLATION  # among decimals
    assert tp.extrapolation({"fit_note": f"{tf.FIT_NOTE} {tf.WINDOW_NOTE}"}) == "" and tp.extrapolation({}) == ""
    found = profiles_of(extrapolation=True)
    assert tp.warning(found) is None  # the check passed
    assert notes_of(found)[2] == (
        f"The fit's core is an extrapolation: the file's `fit_note` says “{tf.EXTRAPOLATION}” "
        "Read the points there, not the curve."
    )
    # None for a note that does not say so, and none for a file with no fit to be an extrapolation of.
    assert not [n for n in notes_of(profiles_of()) if "extrapolation" in n]
    assert not [n for n in notes_of(profiles_of(fit=False, extrapolation=True)) if "extrapolation" in n]


def test_the_points_note_counts_what_each_system_read():
    found = profiles_of()
    note = notes_of(found)[0]
    edge, core = found.points["edge"], found.points["core"]
    assert f"the {found.pulses} laser pulses in the window" in note
    assert f"edge {edge.drawn} of {edge.readings} readings" in note
    assert f"core {core.drawn} of {core.readings} readings" in note
    assert "no core TS in this file" in notes_of(profiles_of(core=False))[0]


def test_the_fit_note_names_its_window_its_whole_range_and_why_it_has_no_band():
    fit_note = notes_of(profiles_of())[1]
    assert "over 1–1.4 s, one profile for the whole window" in fit_note
    assert "drawn over its whole range (ρ_pol 0–1.1)" in fit_note
    assert "20% of the fit by construction" in fit_note
    # A fit made with another error model is not said to have 20%.
    other = notes_of(profiles_of(frac_err=False))[1]
    assert "No band is drawn around it." in other and "20%" not in other


def test_the_caption_says_when_the_pages_window_is_not_the_one_the_fit_was_made_over():
    found = profiles_of()
    other = notes_of(found, target=Target(1.1, 1.3))[1]
    assert "The page's window is 1.1–1.3 s, so the points and the fit are not over the same pulses." in other
    for same in (Target(1.0, 1.4), Target(1.0 + 5e-8, 1.4), None, Target(float("nan"), float("nan"))):
        assert "page's window" not in notes_of(found, target=same)[1]


def test_attribute_text_is_escaped_for_markdown_in_the_captions():
    """``fit_status`` and ``efit`` are exceptions' texts, which can hold brackets, asterisks, dollars and backticks."""
    assert tp._escape(r"\electrons::top.yag_new.results *x* `y` [z] <a> $b") == (
        r"\\electrons::top.yag\_new.results \*x\* \`y\` \[z\] \<a\> \$b"
    )
    status = "failed: KeyError: 'rho_pol' $a$ and $b$ *x* `y`"
    assert notes_of(profiles_of(fit=False, fit_status=status))[1] == f"No fit in this file ({tp._escape(status)})."
    found = profiles_of(efit=False, fit=False)
    found.attrs["efit"] = "no EFIT tree could be opened (TreeNODATA [*] $x$)"
    assert "since no EFIT tree could be opened (TreeNODATA \\[\\*\\] \\$x\\$): they are not drawn" in notes_of(found)[0]


def test_the_axis_note_says_what_the_axis_is_and_where_the_separatrix_is():
    found = profiles_of()
    rho = notes_of(found)[2]
    assert "ρ_pol = √ψ_N" in rho and "`EFIT20`" in rho and "The separatrix is ρ_pol = 1." in rho
    rmid = notes_of(found, "R_mid")[2]
    assert "outboard midplane" in rmid and "The separatrix is at R_mid = 0.8900 m." in rmid
    assert "The file gives no separatrix." in notes_of(profiles_of(scalars=False, fit=False), "R_mid")[2]


def test_the_diagnosticians_caveat_ends_the_notes_from_the_file_or_else_from_the_module():
    assert notes_of(profiles_of())[-1] == tf.CAVEAT
    assert notes_of(profiles_of(caveat="Ask Jerry first."))[-1] == "Ask Jerry first."
    assert notes_of(profiles_of(caveat=None))[-1] == tp.CAVEAT  # a file made without one is never shown without it


def test_the_about_list_quotes_how_the_file_was_made_and_the_scalars():
    rows = dict(tp.about(profiles_of()))
    assert list(rows)[:2] == ["created", "script_commit"]  # in the order the module lists them
    assert rows["created"] == "2026-10-09T08:37:26Z" and rows["cmod_analysis_commit"].startswith("d6a6466")
    assert rows["fit_check"] == tf.PASSED and rows["fit_function"] == "cmod_tools.get_cmod_kin_profs"
    assert rows["Te_sep_2pm"] == "36 eV" and rows["lambda_q_2pm"] == "2 mm" and rows["R_mid_sep"] == "0.89 m"
    bare = dict(tp.about(profiles_of(scalars=False, fit=False)))
    assert "Te_sep_2pm" not in bare and bare["fit_check"] == "no fit"


# ---------------------------------------------------------------------------
# A fit that was moved to a separatrix (``fit_rho_shift``), and a file without one
# ---------------------------------------------------------------------------


def head_of(window=tf.WINDOW, shift=tf.SHIFT, **options):
    """The profiles of a file that holds ``fit_rho_shift``, whose fit was moved by ``shift`` (0: not moved)."""
    return profiles_of(window=window, commit="24e1499", shift=shift, **options)


def kept(ds, name, system="edge"):
    """The file's ``<system>_<name>`` at the points the view keeps (n_e > 0), in the order it keeps them."""
    return ds[f"{system}_{name}"].values[ds[f"{system}_ne"].values > 0]


def test_the_file_gives_its_fits_shift_and_its_points_move_with_a_fit_that_moved():
    moved = head_of()
    assert moved.fit.rho_shift == tf.SHIFT and moved.fit.moved
    assert moved.fit.rho_pol[0] == tf.SHIFT and moved.fit.rho_pol[-1] == pytest.approx(1.1 + tf.SHIFT)  # the moved grid
    assert moved.scalars["Te_sep_fit"] == np.interp(1.0, moved.fit.rho_pol, moved.fit.Te)
    assert all(points.moved and points.rho_shift == tf.SHIFT for points in moved.points.values())
    # The finder failed: the file says the fit was not moved, and its points stay where EFIT puts them.
    still = head_of(shift=0.0)
    assert still.fit.rho_shift == 0.0 and not still.fit.moved and "Te_sep_fit" not in still.scalars
    assert not any(points.moved for points in still.points.values())
    # A file without the variable has no shift at all.
    old = profiles_of()
    assert old.fit.rho_shift is None and not old.fit.moved and not any(p.moved for p in old.points.values())


@pytest.mark.parametrize("shift", [tf.SHIFT, -0.006])
def test_the_points_of_a_moved_fit_are_drawn_where_the_fit_saw_them(shift):
    ds = cut(tf.thomson_dataset(commit="24e1499", shift=shift))
    found = tp.read(ds)
    grid, fit_r_mid = ds["rho_pol"].values, ds["fit_R_mid"].values
    on_rho, on_r_mid = tp.figure(found, "rho_pol"), tp.figure(found, "R_mid")
    for i, system in enumerate(("edge", "core")):
        moved = kept(ds, "rho_pol", system) + shift
        # On ρ_pol at their ρ_pol plus the shift, and on R_mid at the fit's own R_mid of that, not at their own pulse's.
        np.testing.assert_array_equal(in_row(on_rho, 1)[i].x, moved)
        x = np.asarray(in_row(on_r_mid, 1)[i].x, dtype=float)
        np.testing.assert_array_equal(x, np.interp(moved, grid, fit_r_mid, left=np.nan, right=np.nan))
        np.testing.assert_allclose(x - kept(ds, "R_mid", system), 0.21 * shift, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(in_row(on_rho, 1)[2].x, grid)  # the fit on its own grid, which is the moved one
    np.testing.assert_array_equal(in_row(on_r_mid, 1)[2].x, fit_r_mid)
    # What the file says of each point stays as EFIT puts it: only where it is drawn moves.
    np.testing.assert_array_equal(found.points["edge"].rho_pol, kept(ds, "rho_pol"))
    np.testing.assert_array_equal(found.points["edge"].R_mid, kept(ds, "R_mid"))


@pytest.mark.parametrize("shift", [tf.SHIFT, -0.006])
def test_the_separatrix_of_a_moved_fit_is_its_own_rho_pol_1_where_its_t_e_is_te_sep_fit(shift):
    found = head_of(shift=shift)
    for axis in AXES:
        fig = tp.figure(found, axis)
        (x,) = {shape.x0 for shape in fig.layout.shapes}  # the same line in every row
        fit = in_row(fig, 2)[2]  # the T_e row's fit, on the axis drawn
        assert np.interp(x, fit.x, fit.y) == pytest.approx(found.scalars["Te_sep_fit"], rel=1e-12), axis
    assert tp.separatrix(found, "rho_pol") == 1.0
    assert tp.separatrix(found, "R_mid") == np.interp(1.0, found.fit.rho_pol, found.fit.R_mid)


def test_the_separatrix_of_a_moved_fit_on_r_mid_is_its_own_before_the_files_scalar():
    found = head_of()
    found.scalars["R_mid_sep"] = 0.95  # another number than the fit's own R_mid at its ρ_pol 1
    assert tp.separatrix(found, "R_mid") == pytest.approx(tf.r_mid(1.0), rel=0, abs=1e-12)
    # Without the fit's R_mid, the scalar stands in.
    assert tp.separatrix(head_of(fit_r_mid=False), "R_mid") == pytest.approx(tf.r_mid(1.0))
    # A fit that was not moved keeps the rule of every other file: the scalar first.
    still = head_of(shift=0.0)
    still.scalars["R_mid_sep"] = 0.95
    assert tp.separatrix(still, "R_mid") == 0.95


@pytest.mark.parametrize("axis", AXES)
def test_the_range_of_a_row_is_set_by_the_points_as_drawn(axis):
    """A moved fit is judged against its points as they are drawn, moved with it: points that sit, unmoved, where the
    moved ones are drawn give the same tops and counts."""
    found = head_of(runaway=True)
    in_place = {
        system: dataclasses.replace(points, rho_pol=points.on("rho_pol"), R_mid=points.on("R_mid"), rho_shift=0.0)
        for system, points in found.points.items()
    }
    drawn = tp.Profiles(points=in_place, fit=found.fit, pulses=found.pulses, scalars=found.scalars, attrs=found.attrs)
    fig = tp.figure(found, axis)
    for key, (name, _label, _title, scale, _unit) in zip(YAXES, tp.QUANTITIES):
        limits = tp.y_limits(found, name, axis, scale)
        assert limits == tp.y_limits(drawn, name, axis, scale) and tuple(fig.layout[key].range) == limits
        top = limits[1]
        assert tp.points_above(found, name, axis, top, scale) == tp.points_above(drawn, name, axis, top, scale)


def test_a_moved_point_the_fits_r_mid_does_not_reach_is_left_off_r_mid_and_counted():
    ds = cut(tf.thomson_dataset(commit="24e1499", shift=tf.SHIFT))
    rows, channels = np.nonzero(ds["edge_ne"].values > 0)
    ds["edge_rho_pol"][int(rows[0]), int(channels[0])] = 1.15  # 1.16 moved, beyond the moved grid's end at 1.11
    found = tp.read(ds)
    edge = found.points["edge"]
    assert np.isnan(edge.on("R_mid")[0]) and edge.on("rho_pol")[0] == 1.15 + tf.SHIFT
    note = notes_of(found, "R_mid")[0]
    assert note.endswith(
        " 1 of the points moved with the fit lie beyond its grid, where `fit_R_mid` gives them no R_mid, and are not "
        "drawn."
    )
    assert "have no R_mid" not in note  # EFIT gave it a place: it is the fit's R_mid that does not reach it
    assert "beyond its grid" not in notes_of(found, "rho_pol")[0]
    # Where the fit's R_mid is NaN, the points are counted as such.
    ds["fit_R_mid"][:] = np.nan
    found = tp.read(ds)
    every = sum(points.drawn for points in found.points.values())
    assert f" {every - 1} of the points moved with the fit fall where `fit_R_mid` is NaN" in notes_of(found, "R_mid")[0]


def test_a_moved_point_with_no_rho_pol_from_efit_is_counted_as_every_other_file_counts_it():
    found = head_of(unmapped=True)
    expected = sum(int(np.isnan(p.rho_pol).sum()) for p in found.points.values())
    assert expected > 0
    for axis in AXES:
        note = notes_of(found, axis)[0]
        assert f"{expected} of the points have no {tp.AXES[axis][1]} (the pulse is outside the EFIT time base)" in note
        assert "beyond its grid" not in note


def test_a_moved_fit_without_an_r_mid_places_neither_itself_nor_its_points_on_r_mid():
    found = head_of(fit_r_mid=False)
    fig = tp.figure(found, "R_mid")
    assert not fig.data and "None of the" in fig.layout.annotations[0].text
    assert "so neither it nor the points moved with it are drawn on this axis." in notes_of(found, "R_mid")[1]
    assert names_in_row(tp.figure(found, "rho_pol"), 1) == ROWS


def test_a_moved_fit_is_captioned_with_the_move_the_te_sep_and_the_files_own_words():
    found = head_of()
    te_sep = found.scalars["Te_sep_fit"]
    lines = notes_of(found)
    assert [line.split(" ")[0] for line in lines] == ["Points:", "Fit:", "The", "Fictitious", "ρ_pol", "From"]
    move = lines[2]
    assert "by +0.0100 in ρ_pol" in move and f"T_e,sep = {te_sep:.1f} eV (`Te_sep_fit`)" in move
    assert move.endswith(f"The file's `fit_separatrix`: “{tp._escape(tf.fit_separatrix(tf.SHIFT, te_sep))}”")
    assert lines[4].endswith("moved by +0.0100 with the fit. The separatrix is the fit's ρ_pol = 1.")
    r_mid = notes_of(found, "R_mid")
    assert r_mid[2] == move  # the same move on either axis
    assert "each is drawn at the fit's own R_mid of its moved ρ_pol" in r_mid[4]
    assert r_mid[4].endswith(f"The separatrix, the fit's ρ_pol = 1, is at R_mid = {tf.r_mid(1.0):.4f} m.")
    assert "by -0.0060 in ρ_pol" in tp.move_note(head_of(shift=-0.006))  # a fit moved inward says so with its sign
    found.scalars.pop("Te_sep_fit")
    assert "though the file records no T_e,sep that it was moved to." in tp.move_note(found)


def test_a_fit_held_by_fictitious_points_says_so_and_that_the_p_e_fit_is_not():
    for found in (head_of(), head_of(shift=0.0)):
        (held,) = [line for line in notes_of(found) if line.startswith("Fictitious points")]
        assert "The p_e fit is not held, and can run away in the SOL." in held
        assert held.endswith(f"The file's `fit_note` says “{tp._escape(tf.FORCE_TO_ZERO_NOTE)}”")
    assert tp.held_note(profiles_of().attrs) == ""  # force_to_zero is off
    no_fit = head_of(fit=False, fit_status="failed: ValueError: No edge Thomson data!")
    assert not [line for line in notes_of(no_fit) if line.startswith("Fictitious")]  # with no fit, nothing is held
    # A note that does not say how leaves the caption without the quote.
    assert tp.held_note(dict(head_of().attrs, fit_note=tf.FIT_NOTE)).endswith("can run away in the SOL.")


def test_a_moved_point_says_in_its_hover_where_it_is_drawn_and_where_efit_puts_it():
    found = head_of()
    points = found.points["edge"]
    first = in_row(tp.figure(found), 2)[0].hovertext[0]
    assert (
        f"ρ_pol = {points.rho_pol[0] + tf.SHIFT:.3f} · R_mid = {points.on('R_mid')[0]:.4f} m, moved with the fit "
        "(+0.0100)<br>"
    ) in first
    assert f"<br>EFIT: ρ_pol = {points.rho_pol[0]:.3f} · R_mid = {points.R_mid[0]:.4f} m<br>" in first


def test_the_about_list_gives_te_sep_fit_beside_te_sep_2pm_the_move_and_where_the_separatrix_code_is():
    found = head_of()
    rows = tp.about(found)
    names = [name for name, _ in rows]
    assert names[names.index("Te_sep_2pm") + 1] == "Te_sep_fit"
    text = dict(rows)
    assert text["Te_sep_fit"] == f"{found.scalars['Te_sep_fit']:.4g} eV"
    assert text["fit_separatrix"] == tf.fit_separatrix(tf.SHIFT, found.scalars["Te_sep_fit"])
    assert text["separatrix_code"] == (
        f"{tf.SEPARATRIX_CODE}: outside C-Mod_Analysis' repository, so no commit pins it; the file keeps the sha256 of "
        "each of the 2 files of it that were loaded, in separatrix_code_sha256."
    )
    assert text["fit_rho_shift"] == "+0.01 in ρ_pol"
    still = dict(tp.about(head_of(shift=0.0)))
    assert still["fit_rho_shift"] == "0: the fit was not moved" and "Te_sep_fit" not in still
    assert tp.separatrix_code({"separatrix_code": "/x"}).endswith("the file records no sha256 of it.")
    old = dict(tp.about(profiles_of()))
    assert not {"Te_sep_fit", "fit_rho_shift", "fit_separatrix", "separatrix_code"} & set(old)


def test_a_file_without_a_shift_draws_every_point_where_efit_puts_it():
    """No ``fit_rho_shift``: each point is drawn where EFIT put it at its own pulse, the separatrix is at 1 and at
    ``R_mid_sep``, and nothing speaks of a move or of fictitious points."""
    ds = cut(tf.thomson_dataset())
    assert "fit_rho_shift" not in ds and json.loads(ds.attrs["fit_settings"])["force_to_zero"] is False
    found = tp.read(ds)
    for axis, fit_x, sep in (
        ("rho_pol", ds["rho_pol"].values, 1.0),
        ("R_mid", ds["fit_R_mid"].values, float(ds["R_mid_sep"])),
    ):
        fig = tp.figure(found, axis)
        for n in (1, 2, 3):
            edge, core, fit = in_row(fig, n)
            np.testing.assert_array_equal(edge.x, kept(ds, axis, "edge"))
            np.testing.assert_array_equal(core.x, kept(ds, axis, "core"))
            np.testing.assert_array_equal(fit.x, fit_x)
        assert {shape.x0 for shape in fig.layout.shapes} == {sep}
        hovers = [text for trace in fig.data for text in (trace.hovertext or ())]  # the fit's is a template
        assert hovers and not [text for text in hovers if "moved" in text or "EFIT:" in text]
        lines = notes_of(found, axis)
        assert [line.split(" ")[0] for line in lines] == ["Points:", "Fit:", tp.AXES[axis][1], "From"]
        assert not [line for line in lines if "moved" in line or "Fictitious" in line]
    # A fit that was not moved draws the same figure, and has no caption of a move: only a move changes anything.
    still = head_of(shift=0.0)
    assert tp.move_note(still) == "" and tp.move_note(found) == ""
    for axis in AXES:
        assert tp.figure(still, axis).to_json() == tp.figure(found, axis).to_json()
        assert not [line for line in notes_of(still, axis) if "moved" in line]


# ---------------------------------------------------------------------------
# The check of the fit's own points, and a mapping that failed
# ---------------------------------------------------------------------------

POINTS_WARNING = (
    "**The points drawn may not be where the fit saw them.** The file's `fit_points_check`, which compares the edge "
    "points the fit was made on with this file's, "
)


@pytest.mark.parametrize(
    "said, shown",
    [
        (tf.POINTS_DIFFER, f"says “{tp._escape(tf.POINTS_DIFFER)}”"),
        ("EQUAL", "says “EQUAL”"),  # not the word the file writes
        ("the check could not run", "says “the check could not run”"),
        ("   ", "is empty"),
    ],
)
def test_a_points_check_that_says_anything_but_equal_is_warned_about_quoting_it(said, shown):
    found = head_of(points_check=said)
    assert tp.points_warning(found) == f"{POINTS_WARNING}{shown}."
    assert tp.warning(found) is None  # the fit's own check passed: that is another warning


def test_there_is_no_points_warning_when_the_check_says_equal_or_is_absent_or_there_is_no_fit():
    assert tp.points_warning(head_of(points_check="equal")) is None
    assert tp.points_warning(head_of(points_check="equal: the fit's edge points are this file's")) is None
    assert tp.points_warning(head_of()) is None and tp.points_warning(profiles_of()) is None  # no attribute
    no_fit = head_of(fit=False, fit_status="failed: ValueError: No edge Thomson data!", points_check=tf.POINTS_DIFFER)
    assert tp.points_warning(no_fit) is None  # no fit to have seen the points anywhere


def edge_lost(**options):
    """Profiles whose edge points have no ρ_pol and no R_mid at any pulse, as where mapping them failed."""
    return profiles_of(failed=(("edge", "rho_pol"), ("edge", "R_mid")), **options)


@pytest.mark.parametrize("commit, shift", [("d6a6466", 0.0), ("24e1499", tf.SHIFT)])
@pytest.mark.parametrize("axis", AXES)
def test_a_system_whose_mapping_failed_at_every_pulse_is_left_out_and_counted(commit, shift, axis):
    found = edge_lost(commit=commit, shift=shift, mapping_failure=tf.MAPPING_FAILURE)
    edge = found.points["edge"]
    assert edge.drawn and np.isnan(edge.on(axis)).all()
    fig = tp.figure(found, axis)
    assert [names_in_row(fig, n) for n in (1, 2, 3)] == [["core TS", FIT]] * 3
    note = notes_of(found, axis)[0]
    assert note.endswith(
        f" {edge.drawn} of the points have no {tp.AXES[axis][1]} and are not drawn: the file's `mapping_failure` "
        f"says “{tp._escape(tf.MAPPING_FAILURE)}”."
    )
    # The range is the core's and the fit's alone, as without the edge.
    no_edge = tp.Profiles(
        points={"core": found.points["core"]}, fit=found.fit, pulses=found.pulses, scalars=found.scalars, attrs={}
    )
    for name, _label, _title, scale, _unit in tp.QUANTITIES:
        assert tp.y_limits(found, name, axis, scale) == tp.y_limits(no_edge, name, axis, scale)


def test_a_file_without_mapping_failure_keeps_the_time_base_as_the_reason():
    found = edge_lost()
    note = notes_of(found)[0]
    lost = found.points["edge"].drawn
    assert note.endswith(f" {lost} of the points have no ρ_pol (the pulse is outside the EFIT time base) and are not drawn.")


def test_a_failed_r_mid_leaves_the_points_on_rho_pol_and_off_r_mid_unless_the_fit_moved_them():
    """Every point has its ρ_pol and none an R_mid. On ρ_pol all are drawn; on R_mid they are left out and counted,
    unless the fit was moved, whose own R_mid then places them."""
    lost = (("edge", "R_mid"), ("core", "R_mid"))
    still = profiles_of(commit="24e1499", failed=lost, mapping_failure=tf.MAPPING_FAILURE)
    assert names_in_row(tp.figure(still, "rho_pol"), 1) == ROWS
    assert "have no ρ_pol" not in notes_of(still)[0]
    assert names_in_row(tp.figure(still, "R_mid"), 1) == [FIT]
    every = sum(points.drawn for points in still.points.values())
    assert f" {every} of the points have no R_mid and are not drawn: the file's `mapping_failure`" in (
        notes_of(still, "R_mid")[0]
    )
    moved = profiles_of(commit="24e1499", shift=tf.SHIFT, failed=lost, mapping_failure=tf.MAPPING_FAILURE)
    assert names_in_row(tp.figure(moved, "R_mid"), 1) == ROWS
    assert "of the points have no R_mid" not in notes_of(moved, "R_mid")[0]


# ---------------------------------------------------------------------------
# The single-shot page
# ---------------------------------------------------------------------------


@pytest.fixture
def tree(monkeypatch, tmp_path):
    """A data tree for the pages, and ``tree(shot, window=None, **options)`` to put one Thomson scattering file in it.

    A ``window`` is the file's and, written into a discharge DB, the discharge window of the shot; without one the shot
    has no entry, and the page cuts a centred 0.2 s. ``tree.root`` is the data folder and ``tree.database`` the ledger.
    """
    import streamlit as st

    root = tmp_path / "alcator"
    database = tmp_path / "state" / "shot_explorer.sqlite"
    discharges = tmp_path / "plasma_discharges.json"
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(root))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(discharges))
    monkeypatch.setenv("FUSION_UI_DB", str(database))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")
    st.cache_data.clear()
    st.cache_resource.clear()
    entries = []

    def install(shot=tf.SHOT, window=None, **options):
        tf.write_thomson(root, shot, window=window or tf.WINDOW, **options)
        if window is not None:
            entries.append(
                {
                    "shot_number": shot,
                    "plasma_current": 0.55,
                    "line_averaged_density": 1.0,
                    "greenwald_fraction": 0.5,
                    "t_start": window[0],
                    "t_end": window[1],
                    "mlp_mode": "",
                    "comment": "",
                }
            )
            discharges.write_text(json.dumps(entries))
        reindex()
        return shot

    def reindex():
        conn = db.open_db(database)
        catalog.rescan(conn, str(root), "cmod", str(discharges) if entries else None)
        conn.close()
        st.cache_data.clear()
        st.cache_resource.clear()

    install.root, install.database, install.reindex = root, database, reindex
    yield install
    st.cache_data.clear()
    st.cache_resource.clear()


def open_view(shot):
    """The single-shot page on the Thomson scattering file of ``shot``."""
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60)
    app.session_state["selection"] = {
        "machine": "cmod",
        "shot": shot,
        "diagnostic": "thomson_scattering",
        "preprocessed": False,
    }
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    return app


def widget(app, kind, label):
    matches = [w for w in getattr(app, kind) if w.label == label]
    assert matches, f"no {kind} labelled {label!r}: {[w.label for w in getattr(app, kind)]}"
    return matches[0]


def captions(app):
    """The captions of the main column: the sidebar's, under the plot picker, are not the figure's."""
    return [c.value for c in app.main.caption]


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


def test_the_page_offers_only_this_plot_for_the_file_and_draws_it_with_its_notes(tree):
    app = open_view(tree(window=tf.WINDOW))
    assert widget(app, "selectbox", "Plot").options == ["Thomson scattering profiles"]
    assert widget(app, "selectbox", "Diagnostic").options == ["thomson_scattering"]
    assert not app.warning  # the fit passed its check

    (fig,) = charts(app)
    assert [t.name for t in fig.data if t.showlegend is not False] == ROWS
    assert [s.name for s in fig.layout.shapes if s.showlegend] == ["separatrix"]

    shown = captions(app)
    assert any(c.startswith("Window 1.0000–1.4000 s") for c in shown)
    notes = shown[next(i for i, c in enumerate(shown) if c.startswith("Points: ")) :]
    assert [n.split(":")[0] for n in notes[:2]] == ["Points", "Fit"]
    assert notes[2].startswith("ρ_pol = √ψ_N") and notes[-1] == tf.CAVEAT
    assert not [c for c in shown if "page's window" in c]  # the page's window is the fit's

    (expander,) = app.expander
    assert expander.label == "About this file"
    text = "\n".join(m.value for m in expander.markdown)
    assert "- `created`: 2026-10-09T08:37:26Z" in text and "- `R_mid_sep`: 0.89 m" in text

    conn = db.connect(tree.database)  # a live spec writes nothing to the ledger
    try:
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    finally:
        conn.close()


def test_the_page_says_when_its_window_is_not_the_fits_and_cuts_the_points_to_its_own(tree):
    app = open_view(tree())  # no discharge DB entry: the page cuts a centred 0.2 s
    assert any("so the points and the fit are not over the same pulses" in c for c in captions(app))
    (fig,) = charts(app)
    times = [float(re.search(r"t = ([\d.]+) s", text).group(1)) for text in in_row(fig, 1)[0].hovertext]
    assert times and all(0.89 <= t <= 1.09 for t in times)  # the pulses of the centred window, not of the fit's


def test_the_warnings_sit_above_the_figure_the_fits_own_check_first_and_the_captions_below_it(tree):
    app = open_view(tree(window=tf.WINDOW, commit="24e1499", check=tf.FAILED, points_check=tf.POINTS_DIFFER))
    first, second = [w.value for w in app.warning]
    assert first.startswith("**This fit fails run_db.py's own check: “failed: Te on axis is 134 eV”.** ")
    assert CORE in first and tf.FAILED in first
    assert second == f"{POINTS_WARNING}says “{tp._escape(tf.POINTS_DIFFER)}”."
    kinds = [element.type for element in app.main.children.values()]
    chart = kinds.index("plotly_chart")
    assert kinds.index("warning") < chart < kinds.index("caption", chart)
    assert FIT in [trace.name for trace in charts(app)[0].data]  # the fit is drawn all the same


def test_the_axis_is_a_sidebar_choice_and_drawing_against_r_mid_redraws_the_figure(tree):
    app = open_view(tree(window=tf.WINDOW))
    axis = widget(app, "selectbox", "axis")
    assert axis.options == ["rho_pol", "R_mid"] and axis.value == "rho_pol"
    assert charts(app)[0].layout.xaxis3.title.text == "ρ<sub>pol</sub>"
    axis.set_value("R_mid").run()
    assert not app.exception
    (fig,) = charts(app)
    assert fig.layout.xaxis3.title.text == "R<sub>mid</sub> [m]"
    assert any(c.startswith("R_mid is the major radius") and "0.8900 m" in c for c in captions(app))


BOTH = {"edge TS", "core TS"}


@pytest.mark.parametrize(
    "shot, options, systems, fit",
    [
        (1160616027, dict(window=(1.15, 1.45), check=tf.FAILED), BOTH, True),  # a fit that failed its check
        (1120217022, dict(window=(1.25, 1.55), edge=False, fit=False, fit_status="failed: no edge"), {"core TS"}, False),
        (1091216009, dict(window=(1.3, 1.55), core=False), {"edge TS"}, True),  # a file with one system
        (1160616026, dict(window=(1.15, 1.45), commit="24e1499"), BOTH, True),  # the separatrix finder failed
        (1110201022, dict(window=(0.9, 1.3), commit="24e1499", shift=-0.0111), BOTH, True),  # a fit moved inward
        (1140827019, dict(window=(0.6, 1.3), efit=False, fit=False, fit_status="failed: no EFIT tree"), set(), False),
    ],
)
def test_the_page_draws_each_kind_of_file_it_can_meet(tree, shot, options, systems, fit):
    app = open_view(tree(shot, **options))
    (fig,) = charts(app)
    names = {trace.name for trace in fig.data}
    assert {name for name in names if not name.startswith("fit")} == systems
    assert any(name.startswith("fit") for name in names) == fit
    assert bool(app.warning) == (options.get("check") == tf.FAILED)


def test_the_page_draws_a_moved_fit_with_its_points_on_it_and_its_captions_on_both_axes(tree):
    app = open_view(tree(window=tf.WINDOW, commit="24e1499", shift=tf.SHIFT))
    assert not app.warning  # the fit passed its check
    (fig,) = charts(app)
    file = cut(tf.thomson_dataset(commit="24e1499", shift=tf.SHIFT))
    np.testing.assert_allclose(in_row(fig, 1)[0].x, kept(file, "rho_pol") + tf.SHIFT, rtol=0, atol=0)
    assert {shape.x0 for shape in fig.layout.shapes} == {1.0}
    shown = captions(app)
    assert any(c.startswith("The fit was moved to a separatrix") and "by +0.0100 in ρ_pol" in c for c in shown)
    assert any(c.startswith("Fictitious points hold the outer side of the n_e and T_e fits") for c in shown)
    (expander,) = app.expander
    text = "\n".join(m.value for m in expander.markdown)
    assert "- `Te_sep_2pm`: 36 eV\n- `Te_sep_fit`: " in text  # beside the two-point model's
    assert "- `fit_rho_shift`: +0.01 in ρ\\_pol" in text

    widget(app, "selectbox", "axis").set_value("R_mid").run()
    assert not app.exception, [e.value for e in app.exception]
    (on_r_mid,) = charts(app)
    (sep,) = {shape.x0 for shape in on_r_mid.layout.shapes}
    assert sep == pytest.approx(tf.r_mid(1.0), rel=0, abs=1e-12)  # the fit's own R_mid at its ρ_pol 1
    assert any("each is drawn at the fit's own R_mid of its moved ρ_pol" in c for c in captions(app))


# ---------------------------------------------------------------------------
# The other pages: a Thomson scattering file is listed, and has no traces to offer
# ---------------------------------------------------------------------------


def test_the_shot_browser_lists_the_file_in_its_own_column(tree):
    tree(1160616027, window=tf.WINDOW)
    tree(1140827019, window=(0.6, 1.3))
    app = AppTest.from_file(BROWSER, default_timeout=60).run()
    assert not app.exception, [e.value for e in app.exception]
    table = app.dataframe[0].value
    assert set(table.set_index("shot")["thomson_scattering"]) == {"R"}
    (has_diagnostic,) = [w for w in app.sidebar.multiselect if w.label == "Has diagnostic"]
    assert "thomson_scattering" in has_diagnostic.options


@pytest.fixture
def statistics_tree(tree, apd_dataset_path):
    """Shot 1234 has an APD file and a Thomson scattering file, shot 1235 only the latter."""
    assert apd_dataset_path.parent.parent == tree.root  # the APD fixture writes into the same data folder
    tf.write_thomson(tree.root, 1234)
    tf.write_thomson(tree.root, 1235)
    tree.reindex()
    return tree


def test_the_statistics_page_offers_no_trace_from_a_thomson_scattering_file(statistics_tree):
    """A shot with only a Thomson scattering file is not offered, even when it is the shot the browser selected."""
    app = AppTest.from_file(STATISTICS, default_timeout=60)
    app.session_state["selection"] = {
        "machine": "cmod",
        "shot": 1235,
        "diagnostic": "thomson_scattering",
        "preprocessed": False,
    }
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    (shots,) = [w for w in app.selectbox if w.label == "Shot"]
    assert shots.options == ["1234"] and shots.value == 1234
    assert [w.options for w in app.radio if w.label == "Diagnostic"] == [["apd"]]  # the shot's other file is not offered


def test_the_statistics_page_says_so_when_no_shot_has_a_file_with_traces(tree):
    tree(window=tf.WINDOW)  # the only file in the tree is a Thomson scattering one
    app = AppTest.from_file(STATISTICS, default_timeout=60).run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("No shot with traces" in w.value for w in app.warning)
    assert not [w for w in app.selectbox if w.label == "Shot"]


def test_a_diagnostic_without_traces_is_refused_not_answered_with_no_channels():
    assert "thomson_scattering" not in traces.SOURCES
    with pytest.raises(ValueError, match="'thomson_scattering' has no traces to take"):
        traces.channels(tf.thomson_dataset(), "thomson_scattering")


def test_the_status_table_lines_up_its_columns_for_the_longest_diagnostic_name(tree, apd_dataset_path, capsys):
    from fusion_ui import cli

    tree(window=tf.WINDOW)  # a Thomson scattering file beside the fixture's APD one: both are indexed
    cli.main(["status"])
    rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith("  cmod ")]
    assert {line.split()[1] for line in rows} == {"apd", "thomson_scattering"}
    assert len({line.index(" raw ") for line in rows}) == 1  # the kind column starts in one place
