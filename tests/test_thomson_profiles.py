"""The Thomson scattering profiles view: reading the file, the figure, the words around it, and the pages that show it.

What these tests hold the view to, each on a synthetic file in the layout fusion_scripts writes (``tests/thomson_fixtures.py``):

- the catalog indexes ``thomson_scattering/thomson_scattering_<shot>.nc``, raw only, and the spec is offered for that diagnostic
  and no other;
- only the points with n_e > 0 are drawn, a 0 and its junk errors never reach the figure, and a missing or non-positive error
  leaves a point without a bar;
- n_e, T_e and p_e share one axis, ρ_pol or R_mid; the edge and core points differ by marker; the fit is drawn over its whole
  range, labelled with its window and with no band; the separatrix is at ρ_pol = 1, or at the file's R_mid;
- a file with no fit, with no core, or with nothing in the window still draws what it has;
- a fit that fails ``fit_check`` gets a warning that quotes it; the caption carries the caveat, the page's window when it is not
  the fit's, and why there is no band;
- the single-shot page draws it, the Statistics page does not offer it, the shot browser lists it, and the Documentation page
  describes it.
"""

import ast
import base64
import inspect
import json
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

import fusion_ui.plots  # noqa: F401 - registers the specs the page offers
from fusion_ui import documentation
from fusion_ui.core import catalog, db, loader, multipixel, params_ui, registry, traces
from fusion_ui.plots import thomson_profiles as tp
from tests import thomson_fixtures as tf

REPO = Path(__file__).resolve().parent.parent
SINGLE_SHOT = str(REPO / "fusion_ui" / "pages" / "2_single_shot.py")
BROWSER = str(REPO / "fusion_ui" / "pages" / "1_shot_browser.py")
STATISTICS = str(REPO / "fusion_ui" / "pages" / "4_statistics.py")

#: The label of the fit in the legend when it was made over the fixture's window.
FIT = "fit, 1–1.4 s"


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


# ---------------------------------------------------------------------------
# The spec and the catalog
# ---------------------------------------------------------------------------


def test_the_spec_is_registered_live_and_for_the_raw_thomson_scattering_file_only():
    spec = registry.get("thomson_profiles")
    assert spec.label == "Thomson scattering profiles"
    assert spec.diagnostics == ("thomson_scattering",)
    assert not spec.cached and spec.compute is None  # nothing to compute, nothing to cache
    assert spec.preprocessed is False
    assert spec.scalars is None and spec.overlay is None
    assert spec in registry.for_diagnostic("thomson_scattering", False)
    assert spec not in registry.for_diagnostic("thomson_scattering", True)
    for other in ("apd", "phantom", "asp", "fsp"):
        assert spec not in registry.for_diagnostic(other, False)
    assert registry.for_diagnostic("thomson_scattering") == [spec]  # the only plot of its diagnostic


def test_the_spec_has_one_parameter_the_axis_and_it_is_not_a_many_pixel_view():
    spec = registry.get("thomson_profiles")
    assert list(spec.params.__dataclass_fields__) == ["axis"]
    assert spec.params().axis == "rho_pol"
    assert params_ui.CHOICES[("ThomsonProfilesParams", "axis")] == ("rho_pol", "R_mid")
    assert not multipixel.supported(spec)
    digest, text = params_ui.hash_params(spec.key, spec.params())
    back = params_ui.from_canonical(spec.params, text, plot_key=spec.key)
    assert params_ui.hash_params(spec.key, back)[0] == digest


#: Where fusion_scripts' code is reached from: its own top-level packages, and the app's one door to them.
FUSION_SCRIPTS = {"fusion_scripts", "density_scan", "decorrelation", "plotting_scripts", "tools"}


def imported_names(source):
    """Every name a module's source imports, dotted in full: ``from a.b import c`` is ``a.b`` and ``a.b.c``, so that
    ``from fusion_ui.core import fusion_scripts`` is seen as the import of ``fusion_ui.core.fusion_scripts``."""
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            names += [module] + [f"{module}.{alias.name}".lstrip(".") for alias in node.names]
    return names


def reaches_fusion_scripts(source):
    """The imports of ``source`` that go to fusion_scripts, directly or through the app's ``core.fusion_scripts``."""
    return [name for name in imported_names(source) if FUSION_SCRIPTS & set(name.split("."))]


def test_the_module_imports_nothing_from_fusion_scripts():
    """L14: the app is moving off fusion_scripts, so new code reads the file and nothing else."""
    assert reaches_fusion_scripts(inspect.getsource(tp)) == []


@pytest.mark.parametrize(
    "line",
    [
        "import fusion_scripts",
        "import density_scan.dead_pixels",
        "from decorrelation import pipeline",
        "from fusion_ui.core import fusion_scripts, loader",  # the way the codebase reaches it
        "from fusion_ui.core.fusion_scripts import import_config",
        "from ..core import fusion_scripts",
    ],
)
def test_the_guard_sees_each_way_of_reaching_fusion_scripts(line):
    assert reaches_fusion_scripts(line), line
    assert reaches_fusion_scripts("from fusion_ui.core import loader, registry\nimport numpy as np") == []


def test_the_catalog_indexes_the_thomson_scattering_folder_raw_only(conn, tmp_path):
    root = tmp_path / "alcator"
    tf.write_thomson(root, 1160616027)
    tf.write_thomson(root, 1140827019)
    stats = catalog.rescan(conn, str(root), "cmod", None)
    assert stats.inserted == 2 and stats.skipped == 0
    assert "thomson_scattering" in catalog.DIAGNOSTICS
    assert "thomson_scattering" not in stats.missing_folders

    found = {(row["shot"], row["diagnostic"], row["preprocessed"]): row for row in conn.execute("SELECT * FROM shots")}
    assert set(found) == {(1160616027, "thomson_scattering", 0), (1140827019, "thomson_scattering", 0)}
    assert found[(1160616027, "thomson_scattering", 0)]["path"].endswith(
        "thomson_scattering/thomson_scattering_1160616027.nc"
    )

    table = catalog.shot_table(conn, None).set_index("shot")
    assert list(table["thomson_scattering"]) == ["R", "R"]
    assert list(table["apd"]) == ["", ""]  # a shot with only this file lists no other diagnostic


def test_the_dataset_path_is_the_diagnostics_folder_and_the_shot(tmp_path, monkeypatch):
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(tmp_path))
    path = loader.dataset_path("cmod", 1160616027, "thomson_scattering", False)
    assert path.endswith("thomson_scattering/thomson_scattering_1160616027.nc")


def test_the_file_the_fixture_writes_opens_the_way_the_page_opens_the_real_ones(tmp_path):
    """netCDF3 as the real files are, read with xarray's own choice of engine, ``time`` in seconds and not decoded."""
    path = tf.write_thomson(tmp_path)
    ds = loader.open_dataset(str(path))
    assert ds["time"].dtype == np.float64 and ds["time"].attrs["units"] == "s"
    assert ds["edge_channel"].dtype == np.int32
    assert ds.attrs["fit_status"] == "ok" and set(ds.dims) == {"time", "rho_pol", "edge_channel", "core_channel"}
    window = loader.sliced(ds, *tf.WINDOW)
    assert 0 < window.sizes["time"] < ds.sizes["time"] and window.sizes["rho_pol"] == ds.sizes["rho_pol"]


# ---------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------


def test_only_the_readings_with_a_measurement_are_kept_and_their_junk_errors_never_are():
    ds = cut(tf.thomson_dataset())
    found = tp.read(ds)
    assert set(found.points) == {"edge", "core"}
    assert found.pulses == ds.sizes["time"]
    for system, points in found.points.items():
        ne = ds[f"{system}_ne"].values
        assert (ne == 0).any(), "the fixture plants readings of 0"
        assert points.pulses == ds.sizes["time"]
        assert points.readings == int(np.isfinite(ne).sum())  # a pulse the system did not record is no reading
        assert points.drawn == int((ne > 0).sum()) < points.readings
        assert (points.ne > 0).all() and (points.Te > 0).all()
        # The tree's error at a reading of 0 is 2e20 m^-3 and 1000 eV: none of it is kept.
        assert np.nanmax(points.ne_err) < 1e20 and np.nanmax(points.Te_err) < 500
        assert len({points.ne.size, points.Te.size, points.time.size, points.channel.size, points.rho_pol.size}) == 1


def test_a_point_carries_where_and_when_it_was_read():
    ds = cut(tf.thomson_dataset())
    points = tp.read(ds).points["edge"]
    i = 5
    t, c = points.time[i], int(points.channel[i])
    row = int(np.argmin(abs(ds["time"].values - t)))
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
    ds["edge_ne"][3, 2] = 0.0  # a density of 0 beside a temperature that reads: the plan's rule is n_e > 0
    points = tp.read(ds).points["edge"]
    assert points.drawn == ds["edge_ne"].size - 1  # n_e > 0 everywhere but at the one reading of 0

    def at(row, channel):
        found = np.flatnonzero((points.time == ds["time"].values[row]) & (points.channel == channel))
        return int(found[0]) if found.size else None

    assert at(3, 2) is None  # not drawn, though its T_e is not 0
    assert np.isnan(points.Te[at(2, 3)]) and points.ne[at(2, 3)] > 0
    assert np.isnan(points.pe()[at(2, 3)])
    assert np.isnan(points.ne_err[at(4, 1)]) and np.isnan(points.Te_err[at(4, 1)])
    assert np.isnan(points.ne_err[at(6, 2)])
    assert np.isnan(points.Te_err[at(8, 0)]) and np.isfinite(points.ne_err[at(8, 0)])
    assert np.isfinite(points.ne_err[at(5, 1)])


def test_a_file_that_holds_only_one_system_reads_that_one():
    assert set(profiles_of(core=False).points) == {"edge"}
    assert set(profiles_of(edge=False).points) == {"core"}


def test_the_fit_is_read_with_its_window_and_without_a_pressure_fit_when_the_file_has_none():
    fit = profiles_of().fit
    assert fit.window == tf.WINDOW and fit.rho_pol[0] == 0.0 and fit.rho_pol[-1] == 1.1
    assert fit.pe is not None and fit.R_mid is not None
    assert profiles_of(pe=False).fit.pe is None
    assert profiles_of(fit=False).fit is None  # a failed fit has no rho_pol and no fit_*


def test_the_files_scalars_and_attributes_come_back_as_plain_python():
    found = profiles_of()
    assert found.scalars == pytest.approx({"R_mid_sep": 0.89, "Te_sep_2pm": 36.0, "lambda_q_2pm": 0.002})
    assert found.attrs["fit_check"] == tf.PASSED and found.attrs["shot"] == tf.SHOT
    assert found.attrs["window_t_start"] == tf.WINDOW[0]
    assert profiles_of(scalars=False).scalars == {}


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------


def test_n_e_t_e_and_p_e_share_one_axis_and_each_row_has_both_systems_and_the_fit():
    fig = figure_of()
    assert names_in_row(fig, 1) == ["edge TS", "core TS", FIT]
    assert names_in_row(fig, 2) == ["edge TS", "core TS", FIT]
    assert names_in_row(fig, 3) == ["edge TS", "core TS", FIT]
    # One shared axis: the upper two follow the bottom one, which carries the title.
    assert fig.layout.xaxis.matches == "x3" and fig.layout.xaxis2.matches == "x3"
    assert fig.layout.xaxis3.title.text == "ρ<sub>pol</sub>" and not fig.layout.xaxis.title.text
    assert [fig.layout[k].title.text for k in ("yaxis", "yaxis2", "yaxis3")] == [
        "n<sub>e</sub> [10<sup>20</sup> m<sup>-3</sup>]",
        "T<sub>e</sub> [eV]",
        "p<sub>e</sub> [kPa]",
    ]
    # Each profile is read from zero, and an error bar that reaches below it ends at the axis, not below it.
    assert {fig.layout[k].autorangeoptions.minallowed for k in ("yaxis", "yaxis2", "yaxis3")} == {0}


def test_the_two_systems_differ_by_marker_and_by_colour_and_the_legend_lists_each_once():
    fig = figure_of()
    edge, core = in_row(fig, 1)[:2]
    assert (edge.marker.symbol, core.marker.symbol) == ("circle", "diamond")
    assert edge.marker.color != core.marker.color
    assert edge.mode == core.mode == "markers"
    assert [trace.name for trace in fig.data if trace.showlegend is not False] == ["edge TS", "core TS", FIT]
    # Toggling a legend entry hides that system in every row.
    for n in (1, 2, 3):
        assert [trace.legendgroup for trace in in_row(fig, n)][:3] == ["edge", "core", "fit"]
    # The separatrix is a line in every row, one legend entry for the three, which hides them all.
    shapes = fig.layout.shapes
    assert [(s.name, s.legendgroup, s.showlegend) for s in shapes] == [
        ("separatrix", "separatrix", True),
        ("separatrix", "separatrix", False),
        ("separatrix", "separatrix", False),
    ]


def test_the_values_are_drawn_in_the_units_of_the_axis_titles():
    found = profiles_of()
    fig = tp.figure(found)
    points = found.points["edge"]
    ne, te, pe = (in_row(fig, n)[0] for n in (1, 2, 3))
    np.testing.assert_allclose(ne.y, points.ne * 1e-20)  # 10^20 m^-3
    np.testing.assert_allclose(te.y, points.Te)  # eV
    np.testing.assert_allclose(pe.y, 1.602176634e-19 * points.ne * points.Te * 1e-3)  # kPa
    fit = in_row(fig, 3)[2]
    np.testing.assert_allclose(fit.y, found.fit.pe * 1e-3)
    np.testing.assert_allclose(fit.x, found.fit.rho_pol)


def test_the_error_bars_are_the_trees_and_the_pressure_points_carry_the_propagated_error():
    found = profiles_of()
    points = found.points["core"]
    fig = tp.figure(found)
    ne, te, pe = (in_row(fig, n)[1] for n in (1, 2, 3))
    np.testing.assert_allclose(ne.error_y.array, points.ne_err * 1e-20)
    np.testing.assert_allclose(te.error_y.array, points.Te_err)
    e = 1.602176634e-19
    expected = e * np.sqrt((points.Te * points.ne_err) ** 2 + (points.ne * points.Te_err) ** 2) * 1e-3
    np.testing.assert_allclose(pe.error_y.array, expected)
    assert ne.error_y.visible and ne.error_y.type == "data"


def test_a_reading_of_zero_and_its_junk_error_never_reach_the_figure():
    markers = [trace for trace in figure_of().data if trace.mode == "markers"]
    assert len(markers) == 6  # edge and core, in each of the three rows
    for trace in markers:
        y, err = np.asarray(trace.y), np.asarray(trace.error_y.array)
        assert np.nanmin(y) > 0
        # The fixture's errors are 10% of the value and a reading of 0 has 2e20 m^-3 and 1000 eV, which no point shows.
        assert np.all(err[np.isfinite(err)] < 0.5 * y[np.isfinite(err)])


def test_the_fit_is_drawn_over_its_whole_range_without_a_band():
    found = profiles_of()
    fig = tp.figure(found)
    for n in (1, 2, 3):
        fit = in_row(fig, n)[2]
        assert fit.mode == "lines" and len(fit.x) == found.fit.rho_pol.size  # the whole grid, 0 to 1.1
        assert fit.x[0] == 0.0 and fit.x[-1] == 1.1
        assert fit.error_y.array is None
    assert not [trace for trace in fig.data if trace.fill not in (None, "none")]  # no filled band anywhere
    lines = {trace.name for trace in fig.data if trace.mode == "lines"}
    assert lines == {FIT}  # nothing else is a curve: no upper or lower edge of a band


def test_the_separatrix_is_rho_pol_1_in_every_row():
    fig = figure_of("rho_pol")
    assert [(s.x0, s.x1, s.line.dash) for s in fig.layout.shapes] == [(1.0, 1.0, "dash")] * 3
    assert tp.separatrix(profiles_of(), "rho_pol") == 1.0
    assert tp.separatrix(profiles_of(scalars=False, fit=False), "rho_pol") == 1.0


def test_on_the_r_mid_axis_the_points_the_fit_and_the_separatrix_are_in_r_mid():
    found = profiles_of()
    fig = tp.figure(found, "R_mid")
    np.testing.assert_allclose(in_row(fig, 1)[0].x, found.points["edge"].R_mid)
    np.testing.assert_allclose(in_row(fig, 1)[2].x, found.fit.R_mid)
    assert fig.layout.xaxis3.title.text == "R<sub>mid</sub> [m]"
    assert len(fig.layout.shapes) == 3 and all(s.x0 == pytest.approx(0.89) for s in fig.layout.shapes)


def test_the_separatrix_on_the_r_mid_axis_is_the_scalar_else_the_fits_at_rho_pol_1_else_none():
    assert tp.separatrix(profiles_of(), "R_mid") == pytest.approx(0.89)
    # A file without the scalar reads it off the fit's own R_mid at rho_pol = 1 (the grid has no point exactly there).
    from_fit = tp.separatrix(profiles_of(scalars=False), "R_mid")
    assert from_fit == pytest.approx(tf.r_mid(1.0), abs=1e-9)
    # Neither: no line, and so no legend entry for one.
    none = profiles_of(scalars=False, fit=False)
    assert tp.separatrix(none, "R_mid") is None
    assert not tp.figure(none, "R_mid").layout.shapes
    # The scalar comes first, even where the fit's own R_mid would say another number.
    found = profiles_of()
    found.scalars["R_mid_sep"] = 0.9
    assert tp.separatrix(found, "R_mid") == 0.9
    # A fit_R_mid that is NaN at rho_pol = 1 says nothing: no separatrix, not a line at NaN.
    ds = tf.thomson_dataset(scalars=False)
    ds["fit_R_mid"] = ds["fit_R_mid"].where(~((ds["rho_pol"] > 0.9) & (ds["rho_pol"] < 1.05)))
    nan = tp.read(cut(ds))
    assert tp.separatrix(nan, "R_mid") is None
    assert not tp.figure(nan, "R_mid").layout.shapes
    assert tp.notes(nan, FITS_WINDOW, "R_mid")[2].endswith("The file gives no separatrix.")


def test_a_file_without_a_fit_draws_the_points_and_no_fit():
    fig = figure_of(fit=False)
    for n in (1, 2, 3):  # p_e is still drawn: e n_e T_e of the points
        assert names_in_row(fig, n) == ["edge TS", "core TS"]
    assert not [trace for trace in fig.data if trace.name.startswith("fit")]


def test_a_fit_without_a_pressure_fit_leaves_the_pressure_row_to_the_points():
    fig = figure_of(pe=False)
    assert names_in_row(fig, 1)[2] == FIT and names_in_row(fig, 2)[2] == FIT
    assert names_in_row(fig, 3) == ["edge TS", "core TS"]


def test_a_file_with_no_core_draws_the_edge_alone():
    assert {trace.name for trace in figure_of(core=False).data} == {"edge TS", FIT}
    assert {trace.name for trace in figure_of(edge=False, fit=False).data} == {"core TS"}


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
    for axis in ("rho_pol", "R_mid"):
        fig = tp.figure(found, axis)
        assert not fig.data
        measured = sum(p.drawn for p in found.points.values())
        assert fig.layout.annotations[0].text == (
            f"None of the {measured} measured points can be placed on {tp.AXES[axis][1]}, and there is no fit to draw."
        )
    note, fit_note, axis_note = tp.notes(found, FITS_WINDOW, "rho_pol")[:3]
    assert "have no ρ_pol, since no EFIT tree could be opened (TreeNODATA): they are not drawn." in note
    assert "outside the EFIT time base" not in note
    assert fit_note == "No fit in this file (failed: ValueError: no EFIT tree)."
    assert "from the EFIT tree" not in axis_note  # there is none to name
    assert tp.separatrix(found, "R_mid") is None


def test_a_system_whose_points_have_no_place_is_left_out_and_the_other_is_drawn():
    found = profiles_of()
    found.points["edge"].rho_pol[:] = np.nan
    names = [trace.name for trace in tp.figure(found).data]
    assert "edge TS" not in names and "core TS" in names


def test_a_fit_without_an_r_mid_is_not_drawn_on_the_r_mid_axis_and_the_caption_says_so():
    found = profiles_of(fit_r_mid=False)
    assert found.fit.R_mid is None
    assert FIT in names_in_row(tp.figure(found, "rho_pol"), 1)
    fig = tp.figure(found, "R_mid")
    assert not [trace for trace in fig.data if trace.name.startswith("fit")]
    assert (
        "The file has no R_mid for the fit, so it is not drawn on this axis."
        in tp.notes(found, FITS_WINDOW, "R_mid")[1]
    )
    assert "no R_mid for the fit" not in tp.notes(found, FITS_WINDOW, "rho_pol")[1]
    assert {s.x0 for s in fig.layout.shapes} == {
        found.scalars["R_mid_sep"]
    }  # the separatrix still comes from the scalar


def test_an_unknown_axis_is_refused():
    with pytest.raises(ValueError, match="rho_pol, R_mid"):
        tp.figure(profiles_of(), "psi_n")


def test_points_without_a_flux_mapping_are_not_placed_and_the_notes_count_them():
    found = profiles_of(unmapped=True)
    expected = sum(int(np.isnan(p.rho_pol).sum()) for p in found.points.values())
    assert expected > 0
    fig = tp.figure(found)  # still builds
    assert np.isnan(in_row(fig, 1)[0].x).sum() == int(np.isnan(found.points["edge"].rho_pol).sum())
    assert f"{expected} of the points have no ρ_pol" in tp.notes(found, FITS_WINDOW, "rho_pol")[0]
    assert f"{expected} of the points have no R_mid" in tp.notes(found, FITS_WINDOW, "R_mid")[0]  # named for the axis
    assert "no ρ_pol" not in tp.notes(profiles_of(), FITS_WINDOW, "rho_pol")[0]


def test_a_hover_text_says_which_system_when_where_and_what_the_value_is():
    found = profiles_of()
    points = found.points["edge"]
    hover = in_row(tp.figure(found), 2)[0].hovertext  # the T_e row
    assert len(hover) == points.drawn
    first = hover[0]
    assert first.startswith("edge TS · t = ") and f"channel {int(points.channel[0])}" in first
    assert f"ρ_pol = {points.rho_pol[0]:.3f}" in first and "T_e = " in first and " eV" in first
    assert " ± " in first
    # Where the channel looks: R, then Z (and not the other way round).
    assert f"R = {points.R[0]:.4f} m, Z = {points.Z[0]:+.4f} m" in first


@pytest.mark.parametrize(
    "value, digits, text",
    [
        (2140.0, 3, "2140"),  # not 2.14e+03
        (234.0, 2, "234"),  # not 2.3e+02
        (51517.3, 3, "51517"),
        (0.512, 3, "0.512"),
        (1.234, 3, "1.23"),
        (0.00012, 2, "0.00012"),
        (2.5e7, 3, "2.5e+07"),  # a million and up keeps its exponent
    ],
)
def test_hover_numbers_have_no_exponent_below_a_million(value, digits, text):
    assert tp._number(value, digits) == text


def test_no_hover_text_of_a_real_looking_core_temperature_carries_an_exponent():
    """The core T_e of a shot is above 1 keV, and its error 100 eV or more: the numbers the hover is read for."""
    found = profiles_of()
    core_te = in_row(tp.figure(found), 2)[1].hovertext
    assert any(float(re.search(r"T_e = (\d+)", text).group(1)) >= 1000 for text in core_te)  # the fixture has some
    for row in (1, 2, 3):
        for trace in in_row(tp.figure(found), row)[:2]:
            assert not [text for text in trace.hovertext if "e+" in text], trace.name


# ---------------------------------------------------------------------------
# The range of a row
# ---------------------------------------------------------------------------

#: The layout's names for the rows n_e, T_e and p_e, top to bottom.
YAXES = ("yaxis", "yaxis2", "yaxis3")


def toy(x, value, *, error=None, fit=None, r_mid=None, fit_r_mid=None):
    """Profiles whose only points are edge T_e readings of ``value`` eV at ρ_pol ``x``, and whose fit is ``(ρ_pol, T_e)``
    when one is given, so that the rule that sets a row's top can be checked against numbers worked out by hand.

    The row is ``"Te"``, which is drawn in the unit it is in (scale 1). Each point's error is 10% of its value unless
    ``error`` says otherwise; ``r_mid`` is where each point is on the R_mid axis and ``fit_r_mid`` where each sample of the
    fit is, both the same as on ρ_pol unless given.
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


def test_the_top_is_1_1_times_the_98th_percentile_of_the_points_and_not_their_largest():
    found = toy(np.linspace(0.5, 1.0, 100), np.arange(1.0, 101.0))  # no fit; each point has a 10% error
    bottom, top = tp.y_limits(found, "Te", "rho_pol")
    assert bottom == 0.0
    assert top == pytest.approx(1.1 * 98.02)  # 1 + 0.98 * 99, between the 98th and the 99th of them: not 1.1 * 100
    assert tp.points_above(found, "Te", "rho_pol", top) == 0


def test_a_point_measured_to_worse_than_2_sigma_does_not_set_the_top_and_is_counted_above_it():
    value = np.append(np.arange(1.0, 101.0), [1000.0] * 3)
    error = np.append(0.1 * np.arange(1.0, 101.0), [800.0] * 3)  # an error of 800 on 1000 is worse than half the value
    found = toy(np.linspace(0.5, 1.0, 103), value, error=error)
    top = tp.y_limits(found, "Te", "rho_pol")[1]
    assert top == pytest.approx(1.1 * 98.02)  # as if the three were not there
    assert tp.points_above(found, "Te", "rho_pol", top) == 3  # and they are said to be, not left out silently


def test_a_point_with_no_error_is_not_measured_to_better_than_2_sigma():
    value = np.arange(1.0, 101.0)
    error = np.where(value > 90, np.nan, 0.1 * value)  # the last ten have none
    found = toy(np.linspace(0.5, 1.0, 100), value, error=error)
    assert tp.y_limits(found, "Te", "rho_pol")[1] == pytest.approx(1.1 * 88.22)  # the 98th percentile of 1 to 90


def test_with_fewer_than_5_well_measured_points_all_the_points_set_the_percentile():
    def top(well):
        value = np.array([10.0] * well + [20.0] * (20 - well))
        error = np.array([1.0] * well + [20.0] * (20 - well))  # the others are measured no better than their value
        return tp.y_limits(toy(np.linspace(0.5, 1.0, 20), value, error=error), "Te", "rho_pol")[1]

    assert top(4) == pytest.approx(1.1 * 20)  # four are not enough to go by: all twenty set it
    assert top(5) == pytest.approx(1.1 * 10)  # five are, and the others no longer count


def test_a_fit_that_runs_away_leaves_the_top_where_the_points_put_it():
    x, value = np.linspace(0.5, 1.0, 40), np.full(40, 100.0)
    rho = np.linspace(0.0, 1.1, 111)
    without = tp.y_limits(toy(x, value), "Te", "rho_pol")
    assert without == pytest.approx((0.0, 110.0))
    runaway = np.where(rho <= 1.0, 100.0, 1e5)  # follows the points, and then leaves them
    assert tp.y_limits(toy(x, value, fit=(rho, runaway)), "Te", "rho_pol") == without


def test_a_fit_within_3_times_the_points_sets_the_top_and_one_beyond_that_does_not():
    x, value = np.linspace(0.5, 1.0, 40), np.full(40, 100.0)
    rho = np.linspace(0.0, 1.1, 111)

    def top(level):
        return tp.y_limits(toy(x, value, fit=(rho, np.full(rho.size, level))), "Te", "rho_pol")[1]

    assert top(250.0) == pytest.approx(1.1 * 250)  # within 3 times the points: it sets the top
    assert top(300.0) == pytest.approx(1.1 * 300)  # exactly 3 times counts
    assert top(301.0) == pytest.approx(1.1 * 100)  # a little beyond does not: the points do


def test_a_fit_is_judged_against_the_points_beside_it_and_not_against_the_largest_ones():
    x = np.r_[np.linspace(0.1, 0.3, 20), np.linspace(0.9, 1.0, 20)]
    value = np.r_[np.full(20, 100.0), np.full(20, 1000.0)]  # low points inside, high ones at the edge

    def top(at):
        return tp.y_limits(toy(x, value, fit=([at], [2000.0])), "Te", "rho_pol")[1]

    assert top(0.2) == pytest.approx(1.1 * 1000)  # 2000 beside points of 100 is not plausible: the points set it
    assert top(1.0) == pytest.approx(1.1 * 2000)  # beside points of 1000 it is, and it sets the top


def test_a_row_with_no_points_takes_the_whole_fit_and_a_row_with_no_fit_is_set_by_the_points_alone():
    rho = np.linspace(0.0, 1.1, 111)
    fit_only = toy([], [], fit=(rho, np.where(rho <= 1.0, 100.0, 1e5)))
    assert tp.y_limits(fit_only, "Te", "rho_pol") == pytest.approx((0.0, 1.1e5))  # no points to hold the fit to
    nothing = tp.Profiles(points={}, fit=fit_only.fit, pulses=0, scalars={}, attrs={})  # not even a system
    assert tp.y_limits(nothing, "Te", "rho_pol") == pytest.approx((0.0, 1.1e5))
    no_fit = toy(np.linspace(0.5, 1.0, 40), np.full(40, 100.0))
    assert no_fit.fit is None and tp.y_limits(no_fit, "Te", "rho_pol") == pytest.approx((0.0, 110.0))


def test_the_points_and_the_fit_are_judged_on_the_axis_drawn():
    """The same points and fit give another top on R_mid than on ρ_pol when the points beside a sample of the fit are other
    points there: here the two groups of points trade places."""
    x = np.r_[np.linspace(0.5, 0.7, 20), np.linspace(0.9, 1.0, 20)]
    value = np.r_[np.full(20, 100.0), np.full(20, 1000.0)]
    r_mid = np.r_[np.linspace(0.90, 1.00, 20), np.linspace(0.50, 0.60, 20)]
    found = toy(x, value, r_mid=r_mid, fit=([0.6, 1.05], [50.0, 2500.0]), fit_r_mid=[0.55, 0.95])
    assert tp.y_limits(found, "Te", "rho_pol")[1] == pytest.approx(1.1 * 2500)  # beside points of 1000, at ρ_pol 1.05
    assert tp.y_limits(found, "Te", "R_mid")[1] == pytest.approx(1.1 * 1000)  # beside points of 100, at R_mid 0.95


def test_the_bottom_stays_at_zero_where_the_fit_goes_below_it():
    rho = np.linspace(0.0, 1.1, 111)
    found = toy(np.linspace(0.5, 1.0, 40), np.full(40, 100.0), fit=(rho, np.where(rho > 0.5, -200.0, 100.0)))
    assert tp.y_limits(found, "Te", "rho_pol") == pytest.approx((0.0, 110.0))  # the PDF's bottom would be -220 here


def test_a_row_with_nothing_above_zero_has_no_top_and_is_left_to_autorange_from_zero():
    rho = np.linspace(0.0, 1.1, 111)
    found = toy([], [], fit=(rho, np.zeros(rho.size)))
    assert tp.y_limits(found, "Te", "rho_pol") is None
    fig = tp.figure(found)
    for key in ("yaxis", "yaxis2"):  # the n_e and the T_e row: the fit holds both
        options = fig.layout[key].autorangeoptions
        assert fig.layout[key].range is None and options.minallowed == 0 and options.maxallowed is None
    assert not fig.layout.annotations


def test_each_row_opens_on_zero_to_its_top_and_autoscale_is_given_no_other_range_to_return_to():
    found = profiles_of(runaway=True)
    for axis in ("rho_pol", "R_mid"):
        fig = tp.figure(found, axis)
        for key, (name, _label, _title, scale, _unit) in zip(YAXES, tp.QUANTITIES):
            _, top = tp.y_limits(found, name, axis, scale)
            row, options = fig.layout[key], fig.layout[key].autorangeoptions
            assert tuple(row.range) == (0.0, top)  # it opens on it
            assert (options.minallowed, options.maxallowed) == (0.0, top)  # and autoscale is given the same


def test_a_runaway_fit_is_drawn_whole_and_leaves_through_the_top_without_moving_the_range():
    plain, runaway = tp.figure(profiles_of()), tp.figure(profiles_of(runaway=True))
    for n, key in enumerate(YAXES, start=1):
        fit = in_row(runaway, n)[2]
        assert fit.name == FIT and fit.x[0] == 0.0 and fit.x[-1] == 1.1  # the whole grid, to its last point
        assert np.max(fit.y) > 50 * runaway.layout[key].range[1]  # far above the top of its row
        assert runaway.layout[key].range == plain.layout[key].range  # and the range is where the points put it


def stray(n, removed=False):
    """The fixture's profiles with ``n`` of its edge readings in the window made 10 keV, measured to no better than that:
    points far above the rest, which are not well measured and so do not set the top. With ``removed`` those readings are
    instead no measurement at all (0, which is not drawn): the same population of points without the strays."""
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
    plain = tp.figure(profiles_of())
    assert not plain.layout.annotations  # no point of the fixture is above the top of its row
    fig = tp.figure(stray(3))
    # T_e and p_e (the same readings), and not n_e, which they leave as it was.
    assert corner_notes(fig) == {"y2 domain": "▲ 3 points above", "y3 domain": "▲ 3 points above"}
    # The strays do not set the top: it is where it is with those readings gone (the three were well-measured ones).
    assert fig.layout.yaxis2.range == tp.figure(stray(3, removed=True)).layout.yaxis2.range
    note = fig.layout.annotations[0]  # in the top right corner of its row, with no arrow
    assert (note.xref, note.x, note.xanchor) == ("paper", 0.99, "right")
    assert (note.y, note.yanchor, note.showarrow) == (0.985, "top", False)
    assert corner_notes(tp.figure(stray(1))) == {"y2 domain": "▲ 1 point above", "y3 domain": "▲ 1 point above"}
    assert corner_notes(tp.figure(stray(3), "R_mid")) == corner_notes(fig)  # on the other axis too


@pytest.mark.slow
def test_in_a_browser_a_row_opens_on_its_range_and_autoscale_and_reset_axes_return_to_it():
    """That autoscale keeps the range is a claim about the renderer, so it is read off one: the figure of a runaway fit is
    opened in chromium, zoomed, and autoscaled, as the modebar does it, and reset. Skipped where playwright or its browser
    is not there."""
    sync_api = pytest.importorskip("playwright.sync_api")
    fig = tp.figure(profiles_of(runaway=True))
    # The two ends of each row, in one list.
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


@pytest.mark.parametrize(
    "text, outcome",
    [
        (tf.PASSED, "passed"),
        (tf.FAILED, "failed: Te on axis is 134 eV"),
        (tf.FAILED_OUTER, "failed: Te at rho_pol 1.10 is 140 eV"),
        (f"{tf.CHECK_TEXT}: FAILED: Te on axis is 134 eV", "FAILED: Te on axis is 134 eV"),  # spelt another way
        (f"{tf.CHECK_TEXT}: PASSED", "PASSED"),  # which is not the outcome "passed" either
        (f"{tf.CHECK_TEXT}: failed:", "failed:"),
        (f"{tf.CHECK_TEXT}:   passed \n", "passed"),
        ("passed", "passed"),  # an outcome without the rule's description
        ("no fit", ""),
        ("   ", ""),
        ("", ""),
        (None, ""),
        # The description alone records no outcome: it reads as one, which no rule knows.
        (tf.CHECK_TEXT, tf.CHECK_TEXT),
    ],
)
def test_the_checks_outcome_is_what_follows_the_rules_description(text, outcome):
    assert tp.check_outcome(text) == outcome


def test_a_fit_that_fails_on_axis_is_warned_about_as_an_extrapolated_core_and_the_check_is_quoted():
    text = tp.warning(profiles_of(check=tf.FAILED))
    assert text.startswith("**This fit fails run_db.py's own check: “failed: Te on axis is 134 eV”.** ")
    assert (
        "Its core is an extrapolation, not a description of the core points: read the points there, not the curve."
        in text
    )
    assert "Its outer end is not credible" not in text  # (the quoted rule says "outer end" as well)
    assert text.endswith(f"The file's `fit_check`: “{tf.FAILED}”")  # the attribute itself, word for word


def test_a_fit_that_fails_at_its_outer_end_is_warned_about_there_and_its_core_is_not_called_an_extrapolation():
    text = tp.warning(profiles_of(check=tf.FAILED_OUTER))
    assert text.startswith("**This fit fails run_db.py's own check: “failed: Te at rho_pol 1.10 is 140 eV”.** ")
    assert "Its outer end is not credible: its T_e there can lie several times above the points" in text
    assert "extrapolation" not in text
    assert text.endswith(f"The file's `fit_check`: “{tf.FAILED_OUTER}”")


def test_a_fit_that_fails_both_clauses_gets_both_pieces_of_advice():
    text = tp.warning(profiles_of(check=tf.FAILED_BOTH))
    assert "Its core is an extrapolation" in text and "Its outer end is not credible" in text
    assert text.index("Its core is an extrapolation") < text.index("Its outer end is not credible")


CORE = "Its core is an extrapolation, not a description of the core points"
SILENT = "It may have failed silently: read the points, not the curve."


@pytest.mark.parametrize(
    "check, shown, verb, advice",
    [
        # A verdict spelt another way is still a verdict, and its clause still says what it says.
        (f"{tf.CHECK_TEXT}: FAILED: Te on axis is 134 eV", "FAILED: Te on axis is 134 eV", "fails", CORE),
        # An outcome nobody wrote a rule for, even one that reads well, is not "passed", and names no clause.
        (f"{tf.CHECK_TEXT}: PASSED", "PASSED", "does not pass", SILENT),
        (f"{tf.CHECK_TEXT}: ok", "ok", "does not pass", SILENT),
        # The rule's description alone says "on axis" as well, and is no verdict that something failed there.
        (tf.CHECK_TEXT, tf.CHECK_TEXT, "does not pass", SILENT),
    ],
)
def test_a_fit_whose_outcome_is_anything_but_passed_is_warned_about(check, shown, verb, advice):
    text = tp.warning(profiles_of(check=check))
    assert text.startswith(f"**This fit {verb} run_db.py's own check: “{shown}”.** ")
    assert advice in text
    assert text.endswith(f"The file's `fit_check`: “{check}”")
    assert (CORE in text) == (advice == CORE)  # the other advice is not given as well


@pytest.mark.parametrize(
    "check, ending",
    [
        ("", "The file's `fit_check` is empty."),
        ("   ", "The file's `fit_check` is empty."),
        (None, "The file has no `fit_check`."),
        ("no fit", "The file's `fit_check`: “no fit”"),  # the words of a file with no fit, on a file that has one
    ],
)
def test_a_fit_with_no_recorded_outcome_is_warned_about_and_does_not_pass(check, ending):
    text = tp.warning(profiles_of(check=check))
    assert text.startswith("**This fit does not pass run_db.py's own check: the file records no result.** ")
    assert "It may have failed silently: read the points, not the curve." in text
    assert text.endswith(ending)


def test_there_is_no_warning_for_a_fit_that_passed_or_for_a_file_with_no_fit_to_warn_about():
    assert tp.warning(profiles_of(check=tf.PASSED)) is None
    assert tp.warning(profiles_of(check="passed")) is None
    assert tp.warning(profiles_of(fit=False)) is None  # fit_check reads "no fit": nothing is drawn to warn about
    assert not tp.check_failed(profiles_of(fit=False))
    assert tp.check_failed(profiles_of(check=tf.FAILED)) and not tp.check_failed(profiles_of(check=tf.PASSED))


def test_the_sentence_of_the_note_that_calls_the_core_an_extrapolation_is_found_among_decimals():
    assert tp.extrapolation({"fit_note": f"{tf.FIT_NOTE} {tf.EXTRAPOLATION}"}) == tf.EXTRAPOLATION
    assert tp.extrapolation({"fit_note": f"{tf.FIT_NOTE} {tf.WINDOW_NOTE}"}) == ""
    assert tp.extrapolation({}) == "" and tp.extrapolation({"fit_note": None}) == ""


def test_a_note_that_calls_the_core_an_extrapolation_gets_a_caption_even_when_the_check_passed():
    found = profiles_of(extrapolation=True)
    assert tp.warning(found) is None  # the check passed
    notes = tp.notes(found, FITS_WINDOW, "rho_pol")
    assert notes[1].startswith("Fit: ") and notes[3].startswith("ρ_pol = ")
    assert notes[2] == (
        f"The fit's core is an extrapolation: the file's `fit_note` says “{tf.EXTRAPOLATION}” "
        "Read the points there, not the curve."
    )
    # None for a note that does not say so, and none for a file with no fit to be an extrapolation of.
    assert not [n for n in tp.notes(profiles_of(), FITS_WINDOW, "rho_pol") if "extrapolation" in n]
    nothing = tp.notes(profiles_of(fit=False, extrapolation=True), FITS_WINDOW, "rho_pol")
    assert not [n for n in nothing if "extrapolation" in n]


def test_a_fit_that_failed_at_its_outer_end_and_whose_core_is_an_extrapolation_hears_both():
    found = profiles_of(check=tf.FAILED_OUTER, extrapolation=True)
    assert "Its outer end is not credible" in tp.warning(found)
    assert any(n.startswith("The fit's core is an extrapolation") for n in tp.notes(found, FITS_WINDOW, "rho_pol"))


def test_the_points_note_counts_what_each_system_read_and_says_a_zero_is_no_measurement():
    found = profiles_of()
    note = tp.notes(found, FITS_WINDOW, "rho_pol")[0]
    edge, core = found.points["edge"], found.points["core"]
    assert f"the {found.pulses} laser pulses in the window" in note
    assert f"edge {edge.drawn} of {edge.readings} readings" in note
    assert f"core {core.drawn} of {core.readings} readings" in note
    assert "A reading of 0 is no measurement and is not drawn" in note
    assert "e·n_e·T_e" in note
    assert "no core TS in this file" in tp.notes(profiles_of(core=False), FITS_WINDOW, "rho_pol")[0]


def test_the_fit_note_names_its_window_its_whole_range_and_why_it_has_no_band():
    fit_note = tp.notes(profiles_of(), FITS_WINDOW, "rho_pol")[1]
    assert "over 1–1.4 s, one profile for the whole window" in fit_note
    assert "drawn over its whole range (ρ_pol 0–1.1)" in fit_note
    assert "No band is drawn around it" in fit_note
    assert "20% of the fit by construction" in fit_note and "`fit_pe_err` adds a density to a temperature" in fit_note
    assert "page's window" not in fit_note  # the page's window is the fit's


def test_the_fit_note_does_not_claim_20_percent_for_a_fit_made_with_another_error_model():
    fit_note = tp.notes(profiles_of(frac_err=False), FITS_WINDOW, "rho_pol")[1]
    assert "No band is drawn around it." in fit_note and "20%" not in fit_note


def test_the_caption_says_when_the_pages_window_is_not_the_one_the_fit_was_made_over():
    found = profiles_of()
    other = tp.notes(found, Target(1.1, 1.3), "rho_pol")[1]
    assert "The page's window is 1.1–1.3 s, so the points and the fit are not over the same pulses." in other
    for same in (Target(1.0, 1.4), Target(1.0 + 5e-8, 1.4), None, Target(float("nan"), float("nan"))):
        assert "page's window" not in tp.notes(found, same, "rho_pol")[1]


def test_the_text_of_an_exception_in_the_file_is_escaped_for_markdown_in_the_captions():
    """``fit_status`` and ``efit`` are exceptions' texts, which can hold brackets, asterisks, dollars and backticks."""
    status = "failed: KeyError: 'rho_pol' $a$ and $b$ *x* `y`"
    note = tp.notes(profiles_of(fit=False, fit_status=status), FITS_WINDOW, "rho_pol")[1]
    assert note == f"No fit in this file ({tp._escape(status)})."
    assert all(f"\\{char}" in note for char in "$*`_")

    found = profiles_of(efit=False, fit=False)
    found.attrs["efit"] = "no EFIT tree could be opened (TreeNODATA [*] $x$)"
    first = tp.notes(found, FITS_WINDOW, "rho_pol")[0]
    assert "since no EFIT tree could be opened (TreeNODATA \\[\\*\\] \\$x\\$): they are not drawn" in first


def test_a_file_with_no_fit_says_why_in_the_files_words():
    found = profiles_of(fit=False, fit_status="failed: ValueError: No edge Thomson data!")
    assert (
        tp.notes(found, FITS_WINDOW, "rho_pol")[1] == "No fit in this file (failed: ValueError: No edge Thomson data!)."
    )


def test_the_axis_note_says_what_the_axis_is_and_where_the_separatrix_is():
    found = profiles_of()
    rho = tp.notes(found, FITS_WINDOW, "rho_pol")[2]
    assert "ρ_pol = √ψ_N" in rho and "`EFIT20`" in rho and "The separatrix is ρ_pol = 1." in rho
    rmid = tp.notes(found, FITS_WINDOW, "R_mid")[2]
    assert "outboard midplane" in rmid and "window's middle time" in rmid
    assert "The separatrix is at R_mid = 0.8900 m." in rmid
    assert "The file gives no separatrix." in tp.notes(profiles_of(scalars=False, fit=False), FITS_WINDOW, "R_mid")[2]


def test_the_diagnosticians_caveat_ends_the_notes_from_the_file_or_else_from_the_module():
    assert tp.notes(profiles_of(), FITS_WINDOW, "rho_pol")[-1] == tf.CAVEAT
    assert tp.notes(profiles_of(caveat="Ask Jerry first."), FITS_WINDOW, "rho_pol")[-1] == "Ask Jerry first."
    # A file made without one is never shown without it.
    last = tp.notes(profiles_of(caveat=None), FITS_WINDOW, "rho_pol")[-1]
    assert last == tp.CAVEAT and "Jerry Hughes" in last


def test_the_about_list_quotes_how_the_file_was_made_and_the_scalars():
    rows = dict(tp.about(profiles_of()))
    assert rows["created"] == "2026-10-09T08:37:26Z"
    assert rows["cmod_analysis_commit"].startswith("d6a6466")
    assert rows["fit_check"] == tf.PASSED and rows["fit_function"] == "cmod_tools.get_cmod_kin_profs"
    assert rows["Te_sep_2pm"] == "36 eV" and rows["lambda_q_2pm"] == "2 mm" and rows["R_mid_sep"] == "0.89 m"
    assert list(rows)[:2] == ["created", "script_commit"]  # in the order the module lists them
    assert not [name for name in tp.ABOUT if name not in rows]
    bare = dict(tp.about(profiles_of(scalars=False, fit=False)))
    assert "Te_sep_2pm" not in bare and bare["fit_check"] == "no fit"


def test_attribute_text_is_escaped_for_markdown():
    assert tp._escape(r"\electrons::top.yag_new.results *x* `y` [z] <a> $b") == (
        r"\\electrons::top.yag\_new.results \*x\* \`y\` \[z\] \<a\> \$b"
    )


# ---------------------------------------------------------------------------
# The single-shot page
# ---------------------------------------------------------------------------


@pytest.fixture
def tree(monkeypatch, tmp_path):
    """A data tree for the pages, and ``tree(shot, window=None, **options)`` to put one Thomson scattering file in it.

    Installing a file rescans the tree and clears the caches, so the page sees it. A ``window`` is the file's and, written
    into a discharge DB, the discharge window of the shot; without one the shot has no entry, and the page cuts a centred
    0.2 s. ``tree.root`` is the data folder and ``tree.database`` the ledger.
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


def pulse_times(fig):
    """The ``t = …`` of every point of the figure's top row, from the hover texts."""
    times = []
    for trace in fig.data:
        if trace.yaxis in (None, "y") and trace.mode == "markers":
            times += [float(re.search(r"t = ([\d.]+) s", text).group(1)) for text in trace.hovertext]
    return times


def test_the_page_offers_only_this_plot_for_the_file_and_draws_it_with_its_notes(tree):
    app = open_view(tree(window=tf.WINDOW))
    assert widget(app, "selectbox", "Plot").options == ["Thomson scattering profiles"]
    assert widget(app, "selectbox", "Diagnostic").options == ["thomson_scattering"]
    assert not app.warning  # the fit passed its check

    (fig,) = charts(app)
    assert [t.name for t in fig.data if t.showlegend is not False] == ["edge TS", "core TS", FIT]
    assert [s.name for s in fig.layout.shapes if s.showlegend] == ["separatrix"]  # and the line of every row is hers

    shown = captions(app)
    assert any(c.startswith("Window 1.0000–1.4000 s") for c in shown)
    first = next(i for i, c in enumerate(shown) if c.startswith("Points: "))
    notes = shown[first:]
    assert [n.split(":")[0] for n in notes[:2]] == ["Points", "Fit"]
    assert notes[2].startswith("ρ_pol = √ψ_N")
    assert notes[-1] == tf.CAVEAT
    assert not [c for c in shown if "page's window" in c]  # the page's window is the fit's


def test_the_page_says_when_its_window_is_not_the_fits_and_cuts_the_points_to_its_own(tree):
    app = open_view(tree())  # no discharge DB entry: the page cuts a centred 0.2 s
    window = next(c for c in captions(app) if c.startswith("Window"))
    assert "no discharge-DB entry yet" in window
    assert any("so the points and the fit are not over the same pulses" in c for c in captions(app))
    (fig,) = charts(app)
    times = pulse_times(fig)
    assert times and all(0.89 <= t <= 1.09 for t in times)  # the pulses of the centred window, not of the fit's


def test_a_fit_that_fails_its_check_is_drawn_under_a_warning_that_quotes_the_check(tree):
    app = open_view(tree(window=tf.WINDOW, check=tf.FAILED))
    (warning,) = [w.value for w in app.warning]
    assert warning.startswith("**This fit fails run_db.py's own check: “failed: Te on axis is 134 eV”.** ")
    assert CORE in warning and tf.FAILED in warning
    (fig,) = charts(app)
    assert FIT in [trace.name for trace in fig.data]


def test_a_fit_that_fails_at_its_outer_end_is_warned_about_there_on_the_page(tree):
    app = open_view(tree(window=tf.WINDOW, check=tf.FAILED_OUTER))
    (warning,) = [w.value for w in app.warning]
    assert "Its outer end is not credible" in warning and "extrapolation" not in warning
    assert tf.FAILED_OUTER in warning


def test_a_verdict_spelt_another_way_an_unknown_outcome_and_an_empty_one_are_all_warned_about_on_the_page(tree):
    """The page does not take what it cannot read for a pass: a file with a fit is warned about unless it says
    ``passed``."""
    app = open_view(tree(window=tf.WINDOW, check=f"{tf.CHECK_TEXT}: FAILED: Te on axis is 134 eV"))
    (warning,) = [w.value for w in app.warning]
    assert warning.startswith("**This fit fails run_db.py's own check: “FAILED: Te on axis is 134 eV”.** ")
    assert CORE in warning

    app = open_view(tree(window=tf.WINDOW, check=f"{tf.CHECK_TEXT}: ok"))
    (warning,) = [w.value for w in app.warning]
    assert "does not pass run_db.py's own check: “ok”" in warning and SILENT in warning

    app = open_view(tree(window=tf.WINDOW, check=""))  # the attribute is there, and empty
    (warning,) = [w.value for w in app.warning]
    assert warning.startswith("**This fit does not pass run_db.py's own check: the file records no result.** ")
    assert warning.endswith("The file's `fit_check` is empty.")


def test_a_fit_that_passed_but_whose_core_is_an_extrapolation_is_captioned_and_not_warned_about(tree):
    app = open_view(tree(window=tf.WINDOW, extrapolation=True))
    assert not app.warning
    assert f"The fit's core is an extrapolation: the file's `fit_note` says “{tf.EXTRAPOLATION}” " in " ".join(
        captions(app)
    )


def test_the_warning_sits_above_the_figure_and_the_caption_below_it(tree):
    app = open_view(tree(window=tf.WINDOW, check=tf.FAILED))
    kinds = [element.type for element in app.main.children.values()]
    chart = kinds.index("plotly_chart")
    assert kinds.index("warning") < chart < kinds.index("caption", chart)


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


def test_a_shot_whose_fit_failed_draws_the_points_and_says_there_is_no_fit(tree):
    app = open_view(
        tree(window=tf.WINDOW, edge=False, fit=False, fit_status="failed: ValueError: No edge Thomson data!")
    )
    assert not app.warning
    (fig,) = charts(app)
    assert {trace.name for trace in fig.data} == {"core TS"} and len(fig.layout.shapes) == 3
    assert "No fit in this file (failed: ValueError: No edge Thomson data!)." in captions(app)
    assert any("no edge TS in this file" in c for c in captions(app))


def test_a_file_with_no_core_draws(tree):
    app = open_view(tree(window=tf.WINDOW, core=False))
    (fig,) = charts(app)
    assert {trace.name for trace in fig.data} == {"edge TS", FIT}
    assert any("no core TS in this file" in c for c in captions(app))


def test_a_file_whose_equilibrium_could_not_be_opened_says_so_where_the_figure_would_be(tree):
    app = open_view(tree(window=tf.WINDOW, efit=False, fit=False, fit_status="failed: ValueError: no EFIT tree"))
    (fig,) = charts(app)
    assert not fig.data and "None of the" in fig.layout.annotations[0].text
    assert any("since no EFIT tree could be opened (TreeNODATA)" in c for c in captions(app))
    assert not app.warning


def test_the_page_lists_what_the_file_says_of_itself_in_an_expander(tree):
    app = open_view(tree(window=tf.WINDOW))
    (expander,) = app.expander
    assert expander.label == "About this file"
    text = "\n".join(m.value for m in expander.markdown)
    assert "- `created`: 2026-10-09T08:37:26Z" in text
    assert "- `cmod_analysis_commit`: d6a646625e92399dd4c8b534e9c93001754a7c91" in text
    assert "- `R_mid_sep`: 0.89 m" in text


def test_the_page_writes_nothing_to_the_ledger(tree):
    open_view(tree(window=tf.WINDOW))
    conn = db.connect(tree.database)
    try:
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM scalars").fetchone()[0] == 0
    finally:
        conn.close()


@pytest.mark.parametrize(
    "shot, options",
    [
        (1140827019, dict(window=(0.6, 1.3))),  # a fit that passed its check
        (1160616027, dict(window=(1.15, 1.45), check=tf.FAILED)),  # one that failed it
        (1120217022, dict(window=(1.25, 1.55), edge=False, fit=False, fit_status="failed: ValueError: no edge")),
        (1091216009, dict(window=(1.3, 1.55), core=False)),  # a file with one system
    ],
)
def test_the_page_draws_each_kind_of_file_it_can_meet(tree, shot, options):
    app = open_view(tree(shot, **options))
    assert len(charts(app)) == 1
    assert bool(app.warning) == (options.get("check") == tf.FAILED)


def test_a_runaway_fit_does_not_set_the_range_the_page_draws(tree):
    (plain,) = charts(open_view(tree(window=tf.WINDOW)))
    app = open_view(tree(window=tf.WINDOW, runaway=True, check=tf.FAILED_OUTER))
    assert app.warning  # the fit fails its outer-end check, and the page says so above the figure
    (fig,) = charts(app)
    for n, key in enumerate(YAXES, start=1):
        row = fig.layout[key]
        assert np.max(in_row(fig, n)[2].y) > 50 * row.range[1]  # the fit runs far above the top of its row...
        assert row.range == plain.layout[key].range  # ...and the range is where the points put it, as without it
        assert row.autorangeoptions.maxallowed == row.range[1]  # autoscale returns to that range
    widget(app, "selectbox", "axis").set_value("R_mid").run()
    assert not app.exception, [e.value for e in app.exception]
    (on_r_mid,) = charts(app)
    # Judged on R_mid the rule gives the same tops here: the fixture's R_mid is a straight line of ρ_pol, which keeps
    # the points nearest each sample of the fit the same ones.
    assert all(on_r_mid.layout[key].range == fig.layout[key].range for key in YAXES)


# ---------------------------------------------------------------------------
# The other pages
# ---------------------------------------------------------------------------


def test_the_shot_browser_lists_the_file_in_its_own_column(tree):
    tree(1160616027, window=tf.WINDOW)
    tree(1140827019, window=(0.6, 1.3))
    app = AppTest.from_file(BROWSER, default_timeout=60).run()
    assert not app.exception, [e.value for e in app.exception]
    table = app.dataframe[0].value
    assert "thomson_scattering" in table.columns
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
    app = AppTest.from_file(STATISTICS, default_timeout=60).run()
    assert not app.exception, [e.value for e in app.exception]
    (radio,) = [w for w in app.radio if w.label == "Diagnostic"]
    assert radio.options == ["apd"]  # the shot has the other file too, and it is not offered


def test_the_statistics_page_lists_only_the_shots_that_have_a_file_with_traces(statistics_tree):
    """A shot with only a Thomson scattering file has nothing to add there, and is not offered, even when it is the
    shot the browser selected: the page opens on one that has."""
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
    assert [w.options for w in app.radio if w.label == "Diagnostic"] == [["apd"]]


def test_the_statistics_page_says_so_when_no_shot_has_a_file_with_traces(tree):
    tree(window=tf.WINDOW)  # the only file in the tree is a Thomson scattering one
    app = AppTest.from_file(STATISTICS, default_timeout=60).run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("No shot with traces" in w.value for w in app.warning)
    assert not [w for w in app.selectbox if w.label == "Shot"]


def test_the_traces_module_names_the_diagnostics_a_trace_can_come_from():
    assert traces.SOURCES == ("apd", "phantom", "asp", "fsp")
    assert set(traces.SOURCES) < set(catalog.DIAGNOSTICS) and "thomson_scattering" not in traces.SOURCES


def test_a_diagnostic_without_traces_is_refused_not_answered_with_no_channels():
    with pytest.raises(ValueError, match="'thomson_scattering' has no traces to take"):
        traces.channels(tf.thomson_dataset(), "thomson_scattering")


def test_the_status_table_lines_up_its_columns_for_the_longest_diagnostic_name(tree, apd_dataset_path, capsys):
    from fusion_ui import cli

    tree(window=tf.WINDOW)  # a Thomson scattering file beside the fixture's APD one: both are indexed
    cli.main(["status"])
    rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith("  cmod ")]
    assert {line.split()[1] for line in rows} == {"apd", "thomson_scattering"}
    assert len({line.index(" raw ") for line in rows}) == 1  # the kind column starts in one place


# ---------------------------------------------------------------------------
# The Documentation page
# ---------------------------------------------------------------------------

#: What the section quotes of the file by name; the fixture has each, in the layout fusion_scripts writes.
QUOTED = (
    "edge_Z",
    "core_Z",
    "edge_ne",
    "edge_ne_err",
    "edge_Te",
    "edge_Te_err",
    "time",
    "efit_tree",
    "edge_R",
    "R_mid_sep",
    "fit_settings",
    "window_t_start",
    "window_t_end",
    "fit_status",
    "fit_note",
    "fit_check",
    "fit_pe_err",
    "fit_pe",
    "fit_reg",
    "Te_sep_2pm",
    "lambda_q_2pm",
    "two_point_model",
    "caveat",
)


def section():
    (found,) = [s for s in documentation.load().sections if s.title == "Thomson scattering profiles"]
    return found


def test_the_documentation_describes_what_is_measured_fitted_and_each_axis():
    found = section()
    headings = [m.group(1) for b in found.blocks for m in re.finditer(r"^### (.+)$", b.text, re.M)]
    assert headings == ["What is measured", "What each axis means", "What is fitted", "The diagnostician's caveat"]
    text = " ".join(b.text for b in found.blocks)
    assert ".py: " in text  # each account names its code, as the other sections do
    assert "`fusion_ui/plots/thomson_profiles.py: figure`" in text
    assert "A 0 is no measurement" in text and "√ψ_N" in text


def test_every_file_name_the_section_quotes_is_one_the_files_layout_has():
    text = " ".join(b.text for b in section().blocks)
    ds = tf.thomson_dataset()
    held = set(ds.variables) | set(ds.attrs)
    for name in QUOTED:
        assert f"`{name}`" in text, f"the section no longer names {name}"
        assert name in held, f"{name} is not in the layout of the files"


def test_the_section_says_what_the_page_draws_as_the_page_does():
    text = " ".join(" ".join(b.text.split()) for b in section().blocks)
    assert "The page draws the fit over its whole range, with no band." in text
    assert "only the points with n_e > 0" in text
    assert "a warning above the figure that quotes the check" in text
    assert "10 to 20 ms apart" in text


def test_the_section_says_how_a_rows_range_is_set_with_the_numbers_of_the_rule():
    """The sentence quotes the rule's constants as placeholders, so that the text cannot drift from what runs, and says
    that a runaway fit leaves through the top."""
    text, unknown = documentation.fill(" ".join(" ".join(b.text.split()) for b in section().blocks))
    assert not unknown
    assert (
        "Each row's range is set by the points and not by the fit (`fusion_ui/plots/thomson_profiles.py: y_limits`, "
        "the rule of the density scan's PDF)"
    ) in text
    ordinal = documentation._ordinal(tp.WHOLE_PERCENTILE)
    assert (
        f"runs from 0 to {tp.MARGIN:g} times the larger of the {ordinal} percentile of the points measured to better than "
        f"{1 / tp.WELL_MEASURED:g}σ (all of them where fewer than {tp.MIN_WELL} are) and the fit's largest value where it "
        f"stays within {tp.FIT_FACTOR:g} times the median of the {tp.NEAREST} points nearest it on the axis drawn"
    ) in text
    assert "a fit that runs away leaves through the top" in text and "autoscale returns to this range" in text


def test_the_figures_docstring_quotes_the_numbers_of_the_rule():
    """The docstring says why the figure is drawn as it is, so it is held to the numbers it gives."""
    doc = " ".join(tp.figure.__doc__.split())
    ordinal = documentation._ordinal(tp.WHOLE_PERCENTILE)
    assert (
        f"{tp.MARGIN:g} times the larger of the {ordinal} percentile of the row's points measured to better than "
        f"{1 / tp.WELL_MEASURED:g}σ (all of them when fewer than {tp.MIN_WELL} are)"
    ) in doc
    assert f"within {tp.FIT_FACTOR:g} times the median of the {tp.NEAREST} points nearest it" in doc
