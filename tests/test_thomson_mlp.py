"""The Mirror Langmuir Probe's profiles on the Thomson scattering view (J13b), and the companions that bring them there.

What these tests hold the view and the page to, on the synthetic files of ``tests/thomson_fixtures.py`` and
``tests/mlp_fixtures.py`` for one shot, whose discharge window holds the probe's third plunge, in and out:

- the single-shot page opens a spec's companions, lazily and cut to the target's window, and hands them to ``render``,
  with ``None`` where the shot has no such file;
- the view offers the overlay only where the probe plunges in the window, judged from its track alone, with the discharge
  DB's ``mlp_mode`` beside the offer, and says why where it does not;
- **the probe's data are not read until the overlay is switched on**, and without it the view is what it was without the
  ASP file;
- switched on, the MLP's points are J13a's, placed at R_mid,sep + ρ and at the ρ_pol of that through the fit's own R_mid,
  on the n_e, T_e and p_e rows, unmoved unless a separatrix temperature of the file is chosen, with a caption that says
  how; and they never move a row's range, which the Thomson scattering points set.
"""

import json
import os

import numpy as np
import pytest
import xarray as xr
from streamlit.testing.v1 import AppTest

import fusion_ui.plots  # noqa: F401 - registers the specs the page offers
from fusion_ui.core import catalog, db, loader, mlp, registry
from fusion_ui.plots import thomson_profiles as tp
from tests import mlp_fixtures as mf
from tests import thomson_fixtures as tf
from tests.test_thomson_profiles import SINGLE_SHOT, charts, decode

SHOT = mf.SHOT
#: The shot's discharge window: the third plunge, in and out, as on 1160616011.
WINDOW = (1.15, 1.45)


class Target:
    """What the view reads of the page's target."""

    def __init__(self, window=WINDOW, mlp_mode="scan", entry=True):
        self.t_start, self.t_end = window
        self.key = f"cmod_{SHOT}_thomson_scattering_r"

        class Entry:
            pass

        self.discharge = None
        if entry:
            self.discharge = Entry()
            self.discharge.mlp_mode = mlp_mode


def ts_profiles(window=WINDOW, **options):
    """The Thomson scattering profiles of the fixture's file for the shot, cut to ``window``."""
    ds = tf.thomson_dataset(SHOT, window=window, **options)
    return tp.read(ds.sel(time=slice(*window)))


def overlay_of(source="none", window=WINDOW, ts=None, **asp_options):
    return tp.overlay(ts_profiles(window, **(ts or {})), mf.asp_dataset(**asp_options), *window, source)


# ---------------------------------------------------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------------------------------------------------


def test_the_view_names_the_asp_file_as_its_companion_and_keeps_its_one_parameter():
    spec = registry.get("thomson_profiles")
    assert spec.companions == ("asp",) == (tp.COMPANION,)
    assert [name for name in spec.params.__dataclass_fields__] == ["axis"]  # the overlay is view state, not a parameter
    assert [s.key for s in registry.REGISTRY.values() if s.companions] == ["thomson_profiles"]


# ---------------------------------------------------------------------------------------------------------------------
# The offer: judged from the probe's track alone, with the discharge DB's word beside it
# ---------------------------------------------------------------------------------------------------------------------


def test_the_overlay_is_offered_where_the_probe_plunges_in_the_window_with_the_dbs_mlp_mode_as_a_hint():
    strokes, said = tp.offer(mf.asp_dataset(), Target())
    assert strokes == [(3, "in"), (3, "out")]
    assert said == (
        "The probe plunges in the window: plunge 3, in and out, judged from its depth in the ASP file. "
        "The discharge DB's mlp_mode: “scan”."
    )
    strokes, said = tp.offer(mf.asp_dataset(), Target(window=(1.305, 1.45), mlp_mode="-"))
    assert strokes == [(3, "out")] and "plunge 3, out," in said and "mlp_mode: “-”" in said


def test_the_judgement_is_the_datas_and_not_the_dbs():
    """The hint is quoted, never believed: 1150618021's DB says 'none' and its probe scans."""
    strokes, said = tp.offer(mf.asp_dataset(), Target(mlp_mode="none"))
    assert strokes and said.endswith("The discharge DB's mlp_mode: “none”.")
    strokes, said = tp.offer(mf.asp_dataset(dwell=True), Target(mlp_mode="scan"))
    assert not strokes and said.endswith("The discharge DB's mlp_mode: “scan”.")


def test_where_the_probe_holds_still_the_offer_says_so_with_its_depth():
    strokes, said = tp.offer(mf.asp_dataset(dwell=True), Target(mlp_mode="dwell"))
    assert strokes == []
    assert said == (
        "No MLP profiles to draw over these: the probe holds still in the window at a depth of 10.9 cm. "
        "The discharge DB's mlp_mode: “dwell”."
    )


def test_a_file_without_a_track_and_a_window_past_the_plunges_say_why():
    strokes, said = tp.offer(mf.asp_dataset(track=False), Target())
    assert not strokes and "the ASP file has no plunge depth (`plunge_time`, `plunge_depth`)" in said
    time = np.linspace(0.0, 1.0, 10001)
    depth = mf.PARK + 0.1 * np.exp(-(((time - 0.5) / 0.05) ** 2))
    early = mf.asp_dataset().assign_attrs(plunge_time=time, plunge_depth=depth)  # a track that ends at 1 s
    strokes, said = tp.offer(early, Target(window=(1.2, 1.4)))
    assert not strokes and "its plunges are outside the window (their tips at 0.500 s)" in said
    flat = mf.asp_dataset().assign_attrs(plunge_time=time, plunge_depth=np.full(time.size, mf.PARK))
    strokes, said = tp.offer(flat, Target(window=(0.2, 0.4)))
    assert not strokes and "the probe does not plunge in this shot" in said


def test_a_probe_file_with_no_mlp_quantity_is_not_offered():
    other = mf.asp_dataset()
    other = other.drop_vars([name for name in other.data_vars if not name.startswith("rho_")])
    strokes, said = tp.offer(other, Target())
    assert not strokes and "the ASP file holds no MLP quantity" in said


def test_the_hint_says_when_the_db_has_no_word_on_the_probe_or_no_entry_for_the_shot():
    assert tp.mlp_mode_hint(Target(mlp_mode="")) == "The discharge DB gives no mlp_mode."
    assert tp.mlp_mode_hint(Target(entry=False)) == "The shot has no discharge-DB entry."
    assert tp.mlp_mode_hint(object()) == "The shot has no discharge-DB entry."


def test_the_offer_reads_no_variable_of_the_file(tmp_path, reads):
    """It is all the view does with the file while the overlay is off: it reads the names of the variables and the
    attributes, which come with the open, and nothing through the backend."""
    lazy = loader.lazily(str(mf.write(tmp_path, mf.asp_dataset())))
    assert tp.offer(lazy, Target())[0] == [(3, "in"), (3, "out")]
    assert reads == []
    lazy.close()


# ---------------------------------------------------------------------------------------------------------------------
# Where the MLP's points go
# ---------------------------------------------------------------------------------------------------------------------


def test_the_separatrix_temperatures_offered_are_the_files_and_not_moving_comes_first():
    names = lambda choices: [name for name, _, _ in choices]  # noqa: E731
    d6a6466 = ts_profiles()
    assert names(tp.separatrix_choices(d6a6466)) == ["none", "Te_sep_2pm"]
    assert tp.separatrix_choices(d6a6466)[1] == ("Te_sep_2pm", 36.0, "Te_sep_2pm = 36.0 eV, the two-point model's")
    moved = ts_profiles(commit="24e1499", shift=tf.SHIFT)
    assert names(tp.separatrix_choices(moved)) == ["none", "Te_sep_2pm", "Te_sep_fit"]
    assert names(tp.separatrix_choices(ts_profiles(commit="24e1499"))) == ["none", "Te_sep_2pm"]  # found no separatrix
    assert tp.separatrix_choices(ts_profiles(scalars=False)) == [("none", 0.0, "not moved: EFIT's mapping")]


def test_rho_pol_is_the_fits_own_r_mid_inverted_and_nothing_beyond_its_grid():
    fit = ts_profiles().fit
    rho = np.array([0.5, 0.95, 1.0, 1.05, 1.1, 1.11, np.nan])
    found = tp.rho_pol_of(fit, tf.r_mid(rho))
    assert found[:5] == pytest.approx(rho[:5], abs=1e-12)
    assert np.isnan(found[5:]).all()  # beyond the grid's end at 1.1, and no R_mid at all
    assert np.isnan(tp.rho_pol_of(None, [0.9])).all()
    assert np.isnan(tp.rho_pol_of(ts_profiles(fit_r_mid=False).fit, [0.9])).all()


def test_only_the_outermost_stretch_where_the_fits_r_mid_grows_is_inverted():
    """A fit moved inward has a few samples near the axis where its R_mid does not grow (35 of the 24e1499 files)."""
    fit = ts_profiles().fit
    bent = fit.R_mid.copy()
    bent[:5] = bent[5] + np.linspace(0.01, 0.002, 5)  # falls to the axis's R_mid, then grows outward
    fit.R_mid = bent
    found = tp.rho_pol_of(fit, tf.r_mid(np.array([0.5, 1.0, fit.rho_pol[2]])))
    assert found[:2] == pytest.approx([0.5, 1.0], abs=1e-12)
    assert np.isnan(found[2])  # inside the stretch that does not grow: no answer rather than a wrong one


def test_the_points_are_the_mlp_profiles_plots_own_placed_by_rho_on_both_axes():
    points = overlay_of()
    drawn = mlp.read(mf.asp_dataset(), *WINDOW).drawn
    assert points.size == drawn.sizes["window"] > 100
    for name, ours in (("time", points.time), ("rho", points.rho), ("ne", points.ne), ("Te", points.Te)):
        assert np.array_equal(ours, drawn[name].values), name  # J13a's reduction, reused and not copied
    assert points.source == "none" and points.te_sep == 0.0 and points.shift == 0.0
    assert points.separatrix_R_mid == pytest.approx(tf.r_mid(1.0))  # the file's R_mid_sep
    assert points.R_mid == pytest.approx(tf.r_mid(1.0) + drawn["rho"].values)  # R_mid = R_mid,sep + ρ
    inside = points.R_mid <= tf.r_mid(1.1)
    assert points.rho_pol[inside] == pytest.approx(1.0 + drawn["rho"].values[inside] / 0.21)  # through fit_R_mid
    assert inside.any() and (~inside).any() and np.isnan(points.rho_pol[~inside]).all()
    assert set(points.stroke) == {"in", "out"} and set(points.plunge) == {3}
    pe, errors = points.series("pe")
    assert pe == pytest.approx(tp.ELEMENTARY_CHARGE * points.ne * points.Te) and np.isnan(errors).all()


def test_moved_to_a_separatrix_temperature_the_points_move_as_the_mlp_profiles_plot_moves_them():
    points = overlay_of("Te_sep_2pm")
    found = mlp.read(mf.asp_dataset(), *WINDOW, te_sep=36.0)
    assert found.shift.ok and points.shift == found.shift.by == pytest.approx(-0.02 * np.log(60 / 36), abs=2e-4)
    assert points.source == "Te_sep_2pm" and points.te_sep == 36.0
    assert points.rho == pytest.approx(found.drawn["rho"].values + found.shift.by)
    assert points.R_mid == pytest.approx(tf.r_mid(1.0) + points.rho)
    # A temperature the file does not hold moves nothing.
    assert overlay_of("Te_sep_fit").shift == 0.0 and overlay_of("Te_sep_fit").source == "none"


def test_on_a_fit_moved_to_a_separatrix_rho_zero_is_the_separatrix_the_view_draws():
    profiles = ts_profiles(commit="24e1499", shift=tf.SHIFT)
    points = tp.overlay(profiles, mf.asp_dataset(), *WINDOW)
    assert (
        points.separatrix_R_mid == tp.separatrix(profiles, "R_mid") == pytest.approx(float(profiles.fit.R_mid_of(1.0)))
    )
    assert tp.rho_pol_of(profiles.fit, [points.separatrix_R_mid]) == pytest.approx([1.0], abs=1e-12)


def test_without_a_separatrix_on_r_mid_the_points_have_no_place():
    points = overlay_of(ts=dict(efit=False, fit=False))
    assert points.separatrix_R_mid is None and points.size
    assert np.isnan(points.R_mid).all() and np.isnan(points.rho_pol).all()


# ---------------------------------------------------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------------------------------------------------


def mlp_traces(fig):
    return [trace for trace in fig.data if (trace.name or "").startswith("MLP")]


@pytest.mark.parametrize("axis", ["rho_pol", "R_mid"])
def test_without_the_overlay_the_figure_is_the_one_drawn_before(axis):
    profiles = ts_profiles()
    assert tp.figure(profiles, axis, None).to_json() == tp.figure(profiles, axis).to_json()


@pytest.mark.parametrize("axis", ["rho_pol", "R_mid"])
def test_the_mlp_is_drawn_on_the_n_e_t_e_and_p_e_rows_by_stroke_once_in_the_legend(axis):
    profiles, points = ts_profiles(), overlay_of()
    fig = tp.figure(profiles, axis, points)
    traces = mlp_traces(fig)
    assert [(t.name, t.yaxis) for t in traces] == [
        (f"MLP, {way}", row) for row in ("y", "y2", "y3") for way in ("in", "out")
    ]
    assert [t.name for t in traces if t.showlegend] == ["MLP, in", "MLP, out"]
    assert {t.legendgroup for t in traces} == {"mlp"}
    assert [t.marker.symbol for t in traces[:2]] == ["triangle-left", "triangle-right-open"]
    assert {t.marker.color for t in traces} == {tp.MLP_LOOK["colour"]}
    on = points.on(axis)
    for trace in traces[:2]:
        mask = points.stroke == trace.name.split(", ")[1]
        assert np.array_equal(np.asarray(trace.x, dtype=float), on[mask], equal_nan=True)
        assert np.asarray(trace.y, dtype=float) == pytest.approx(points.ne[mask] * 1e-20)
    assert np.asarray(traces[2].y, dtype=float) == pytest.approx(points.Te[points.stroke == "in"])
    assert np.asarray(traces[4].y, dtype=float) == pytest.approx(
        tp.ELEMENTARY_CHARGE * (points.ne * points.Te)[points.stroke == "in"] * 1e-3
    )
    assert "MLP · plunge 3, in · t = " in traces[0].hovertext[0] and "ρ_pol = " in traces[0].hovertext[0]


@pytest.mark.parametrize("axis", ["rho_pol", "R_mid"])
def test_the_mlp_moves_no_rows_range_and_leaves_the_rest_of_the_figure_as_it_was(axis):
    profiles = ts_profiles()
    plain = json.loads(tp.figure(profiles, axis).to_json())
    drawn = json.loads(tp.figure(profiles, axis, overlay_of()).to_json())
    drawn["data"] = [trace for trace in drawn["data"] if not trace.get("name", "").startswith("MLP")]
    assert decode(drawn) == decode(plain)


def test_the_mlp_points_above_a_rows_top_are_counted_in_its_corner():
    profiles, points = ts_profiles(), overlay_of()
    points.ne = points.ne * 1e3  # a probe that read three decades too high: above any top the TS points set
    fig = tp.figure(profiles, "R_mid", points)
    placed = int(np.isfinite(points.R_mid).sum())
    corner = [a.text for a in fig.layout.annotations]
    assert any(text.endswith(f"▲ {placed} MLP points above") for text in corner)
    top = tp.y_limits(profiles, "ne", "R_mid", 1e-20)[1]
    assert list(fig.layout.yaxis.range) == pytest.approx([0.0, top])  # the range stays the TS points'


# ---------------------------------------------------------------------------------------------------------------------
# What is said of it
# ---------------------------------------------------------------------------------------------------------------------


def test_the_notes_say_what_the_points_are_and_how_they_were_placed():
    profiles, points = ts_profiles(), overlay_of()
    first, placed = tp.overlay_notes(points, profiles, "R_mid")
    assert first.startswith(
        f"MLP: {points.size} 1 ms means of the Mirror Langmuir Probe in the window (plunge 3, in and out)"
    )
    assert "each the mean of its 4 pins" in first and "e·n_e·T_e" in first
    assert (
        "on R_mid at R_mid,sep + ρ, with R_mid,sep = 0.8900 m (`R_mid_sep`, the separatrix drawn), and on ρ_pol at the "
        "ρ_pol of that R_mid through the fit's own R_mid (`fit_R_mid`, from EFIT at the window's middle time)."
    ) in placed
    assert placed.endswith(
        "Not moved: the MLP's ρ and the Thomson scattering points' ρ_pol both come from EFIT's mapping."
    )
    lost = int(np.isnan(points.rho_pol).sum())
    assert f"{lost} of the {points.size} MLP points lie where `fit_R_mid` does not reach (it ends at ρ_pol 1.1)" in (
        tp.overlay_notes(points, profiles, "rho_pol")[1]
    )


def test_the_notes_say_by_how_much_the_mlp_was_moved_or_that_it_was_not_with_a_moved_fit():
    profiles = ts_profiles()
    moved = tp.overlay_notes(overlay_of("Te_sep_2pm"), profiles, "rho_pol")[1]
    assert (
        "Moved by -1.02 cm, as the MLP profiles plot moves its profile, so that the MLP's T_e is 36.0 eV (Te_sep_2pm) at ρ = 0"
        in moved
    )
    head = ts_profiles(commit="24e1499", shift=tf.SHIFT)
    said = tp.overlay_notes(tp.overlay(head, mf.asp_dataset(), *WINDOW), head, "rho_pol")[1]
    assert "(the fit's R_mid at its ρ_pol 1, the separatrix drawn)" in said
    assert said.endswith(
        "Not moved: the MLP's ρ comes from EFIT's mapping, while the Thomson scattering points are drawn moved with the "
        "fit, by +0.0100 in ρ_pol. Moved to Te_sep_fit, the MLP is on the fit's separatrix too, where the probe's T_e "
        "reaches it."
    )


def test_a_move_that_cannot_be_made_and_a_plunge_with_no_mlp_point_are_warned_about():
    head = ts_profiles(commit="24e1499", shift=tf.SHIFT)
    points = tp.overlay(head, mf.asp_dataset(), *WINDOW, "Te_sep_fit")  # the fixture's fit is far hotter than the probe
    (said,) = tp.overlay_warnings(points)
    assert said.startswith(
        f"The MLP profile is not moved to T_e = {head.scalars['Te_sep_fit']:.1f} eV (Te_sep_fit): T_e never"
    )
    assert said.endswith("It is drawn on EFIT's mapping.")
    assert tp.overlay_warnings(overlay_of()) == []
    # Into the base depth from the parking position, the probe moves while the MLP does not yet record.
    (said,) = tp.overlay_warnings(overlay_of(window=(0.3, 0.45)))
    assert (
        said
        == "The probe plunges in the window, but there is no MLP point to draw: the MLP recorded nothing while the probe moved."
    )


# ---------------------------------------------------------------------------------------------------------------------
# The single-shot page
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def tree(monkeypatch, tmp_path):
    """``tree(asp=True, mlp_mode="scan", ts=None, **asp_options)``: a data tree holding the shot's Thomson scattering
    file, its ASP file (``asp=False`` leaves it out, ``"broken"`` writes one that cannot be opened), and a discharge DB
    with the shot's window and ``mlp_mode``. ``tree.root`` is the data folder."""
    import streamlit as st

    root = tmp_path / "alcator"
    database = tmp_path / "state" / "shot_explorer.sqlite"
    discharges = tmp_path / "plasma_discharges.json"
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(root))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(discharges))
    monkeypatch.setenv("FUSION_UI_DB", str(database))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")

    def install(asp=True, mlp_mode="scan", ts=None, **asp_options):
        tf.write_thomson(root, SHOT, window=WINDOW, **(ts or {}))
        path = root / "asp" / f"asp_{SHOT}.nc"
        if path.exists():
            path.unlink()
        if asp == "broken":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not a netCDF file")
        elif asp:
            mf.write(root, mf.asp_dataset(SHOT, **asp_options), SHOT)
        entry = {
            "shot_number": SHOT,
            "plasma_current": 0.54,
            "line_averaged_density": 1.07,
            "greenwald_fraction": 0.3,
            "t_start": WINDOW[0],
            "t_end": WINDOW[1],
            "mlp_mode": mlp_mode,
            "comment": "L",
        }
        discharges.write_text(json.dumps([entry]))
        conn = db.open_db(database)
        catalog.rescan(conn, str(root), "cmod", str(discharges))
        conn.close()
        st.cache_data.clear()
        st.cache_resource.clear()
        return SHOT

    install.root = root
    yield install
    st.cache_data.clear()
    st.cache_resource.clear()


def open_view(axis=None):
    """The single-shot page on the shot's Thomson scattering file."""
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60)
    app.session_state["selection"] = {
        "machine": "cmod",
        "shot": SHOT,
        "diagnostic": "thomson_scattering",
        "preprocessed": False,
    }
    if axis:
        app.session_state["params.thomson_profiles.axis"] = axis
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    return app


def toggle(app):
    found = [t for t in app.sidebar.toggle if t.label == "MLP profiles"]
    return found[0] if found else None


def switch_on(app):
    toggle(app).set_value(True).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def main_column(app):
    """Everything the main column shows, in order: each element's kind and its text, or a chart's figure."""
    shown = []

    def walk(node):
        kind = getattr(node, "type", type(node).__name__)
        if kind == "plotly_chart":
            shown.append((kind, decode(json.loads(node.proto.spec))))
        elif kind == "expander":
            shown.append((kind, node.label))
        elif hasattr(node, "value") and kind not in ("main",):
            shown.append((kind, node.value))
        for child in getattr(node, "children", {}).values():
            walk(child)

    walk(app.main)
    return shown


def sidebar_says(app):
    """The words of the offer in the sidebar: the plot's own description, under the picker, is not among them."""
    return [c.value for c in app.sidebar.caption if c.value.startswith(("The probe plunges", "No MLP profiles"))]


@pytest.mark.parametrize("axis", ["rho_pol", "R_mid"])
def test_the_page_offers_the_overlay_off_and_draws_what_it_draws_without_the_asp_file(tree, axis):
    tree(asp=False)
    without = open_view(axis)
    assert toggle(without) is None and not sidebar_says(without)
    tree()
    offered = open_view(axis)
    assert toggle(offered) is not None and toggle(offered).value is False
    assert sidebar_says(offered)[-1] == (
        "The probe plunges in the window: plunge 3, in and out, judged from its depth in the ASP file. "
        "The discharge DB's mlp_mode: “scan”."
    )
    assert main_column(offered) == main_column(without)


@pytest.fixture
def reads(monkeypatch):
    """``[(file name, variable), …]`` of every read of a variable's values through the netCDF4 backend: the probe's
    file is opened lazily, so its attributes come with the open and anything else is read here."""
    from xarray.backends import netCDF4_

    seen = []
    original = netCDF4_.NetCDF4ArrayWrapper._getitem

    def counting(self, key):
        seen.append((os.path.basename(self.datastore._filename), self.variable_name))
        return original(self, key)

    monkeypatch.setattr(netCDF4_.NetCDF4ArrayWrapper, "_getitem", counting)
    return seen


def test_the_probes_data_are_not_read_until_the_overlay_is_switched_on(tree, reads, monkeypatch):
    calls = []
    real = mlp.read
    monkeypatch.setattr(mlp, "read", lambda *args, **kwargs: calls.append(args) or real(*args, **kwargs))
    tree()
    asp = f"asp_{SHOT}.nc"
    app = open_view()
    app.sidebar.selectbox[0].set_value(SHOT).run()  # a rerun, as any widget makes one: still nothing read
    assert toggle(app).value is False
    assert [name for file, name in reads if file == asp] == [] and calls == []
    assert any(file == f"thomson_scattering_{SHOT}.nc" for file, _ in reads)  # the counter sees the page's own reads
    switch_on(app)
    read_now = {name for file, name in reads if file == asp}
    assert len(calls) == 1 and {"ne_0", "Te_0", "rho_ne_0", "time_ne_0"} <= read_now


@pytest.mark.parametrize("axis", ["rho_pol", "R_mid"])
def test_switched_on_the_mlp_is_drawn_on_both_axes_with_the_words_that_say_how(tree, axis):
    tree()
    app = switch_on(open_view(axis))
    (fig,) = charts(app)
    traces = mlp_traces(fig)
    assert [(t.name, t.yaxis) for t in traces] == [
        (f"MLP, {way}", row) for row in ("y", "y2", "y3") for way in ("in", "out")
    ]
    expected = overlay_of().on(axis)
    assert np.array_equal(
        np.concatenate([np.asarray(t.x, dtype=float) for t in traces[:2]]),
        np.concatenate([expected[overlay_of().stroke == way] for way in ("in", "out")]),
        equal_nan=True,
    )
    shown = [c.value for c in app.main.caption]
    (said,) = [c for c in shown if c.startswith("MLP: ")]
    (placed,) = [c for c in shown if c.startswith("Placed by ρ")]
    assert shown[-1] == tf.CAVEAT and shown.index(placed) == len(shown) - 2  # the caveat still ends the notes
    assert not app.warning
    box = [b for b in app.sidebar.selectbox if b.label == "Move the MLP profile to"]
    assert box and box[0].value == "none"
    assert box[0].options == ["not moved: EFIT's mapping", "Te_sep_2pm = 36.0 eV, the two-point model's"]


def test_the_mlp_is_moved_to_the_files_separatrix_temperature_when_it_is_chosen(tree):
    tree()
    app = switch_on(open_view())
    [box] = [b for b in app.sidebar.selectbox if b.label == "Move the MLP profile to"]
    box.set_value("Te_sep_2pm").run()
    shown = [c.value for c in app.main.caption]
    assert any("Moved by -1.02 cm" in c for c in shown)
    traces = mlp_traces(charts(app)[0])
    moved = overlay_of("Te_sep_2pm")
    assert np.array_equal(np.asarray(traces[0].x, dtype=float), moved.rho_pol[moved.stroke == "in"], equal_nan=True)


def test_a_dwell_is_not_offered_and_the_sidebar_says_why(tree):
    tree(mlp_mode="dwell", dwell=True)
    app = open_view()
    assert toggle(app) is None
    assert sidebar_says(app)[-1] == (
        "No MLP profiles to draw over these: the probe holds still in the window at a depth of 10.9 cm. "
        "The discharge DB's mlp_mode: “dwell”."
    )
    assert not mlp_traces(charts(app)[0])


def test_an_asp_file_that_cannot_be_opened_is_said_in_the_sidebar_and_the_view_is_drawn(tree):
    tree(asp="broken")
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60)
    app.session_state["selection"] = {
        "machine": "cmod",
        "shot": SHOT,
        "diagnostic": "thomson_scattering",
        "preprocessed": False,
    }
    app.run()
    assert not app.exception
    (said,) = [w.value for w in app.sidebar.warning]
    assert said.startswith(f"Could not open the asp file of this shot, `{tree.root / 'asp' / f'asp_{SHOT}.nc'}`")
    assert toggle(app) is None and len(charts(app)) == 1


# ---------------------------------------------------------------------------------------------------------------------
# The contract: what the page hands a spec that names companions
# ---------------------------------------------------------------------------------------------------------------------


def test_the_page_hands_render_each_companion_opened_lazily_and_cut_to_the_window_or_none(tree):
    """A spec of the test's own, on the Thomson scattering file, naming the ASP file, the APD file and the phantom file
    as companions: the shot has the first two, and the page cuts the APD file to the target's window."""
    from dataclasses import dataclass

    @dataclass
    class Params:
        note: str = ""

    handed = []

    def render(ds, params, target, companions):
        handed.append((target, companions))

    tree()
    apd = xr.Dataset(
        {"frames": (("y", "x", "time"), np.zeros((2, 3, 600)))},
        coords={"time": ("time", np.linspace(1.0, 1.6, 600))},
    )
    (tree.root / "apd").mkdir(parents=True, exist_ok=True)
    apd.to_netcdf(tree.root / "apd" / f"apd_{SHOT}.nc")
    tree()  # rescan, with the APD file in the tree
    spec = registry.register(
        registry.PlotSpec(
            key="companions_under_test",
            label="Companions under test",
            diagnostics=("thomson_scattering",),
            params=Params,
            render=render,
            companions=("asp", "apd", "phantom"),
        )
    )
    try:
        app = AppTest.from_file(SINGLE_SHOT, default_timeout=60)
        app.session_state["selection"] = {
            "machine": "cmod",
            "shot": SHOT,
            "diagnostic": "thomson_scattering",
            "preprocessed": False,
        }
        app.session_state["spec.thomson_scattering"] = spec
        app.run()
        assert not app.exception, [e.value for e in app.exception]
    finally:
        del registry.REGISTRY[spec.key]
    target, companions = handed[-1]
    assert list(companions) == ["asp", "apd", "phantom"]
    assert companions["phantom"] is None  # no such file for the shot
    probe, imaging = companions["asp"], companions["apd"]
    assert not probe.indexes and probe.attrs["probe_type"] == "Mirror Langmuir Probe (MLP)"  # opened lazily, whole
    assert list(imaging.indexes) == ["time"]
    assert (target.t_start, target.t_end) == WINDOW
    assert float(imaging.time.min()) >= WINDOW[0] and float(imaging.time.max()) <= WINDOW[1]
    assert 0 < imaging.sizes["time"] < 600
    assert target.discharge.mlp_mode == "scan"  # the shot's discharge-DB entry, for the view to quote
