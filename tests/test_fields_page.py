"""The Fields page: it reads what a batch job left, draws it, and never computes.

Products are seeded through ``store.result(batch=True)`` with stand-in specs (``tests/fields_fixtures``),
so the page, the store and the ledger run end to end on synthetic blobs. Nothing here touches
``~/Data`` or the server. AppTest cannot click a Plotly point or follow ``st.switch_page``, so
the click is covered by ``pixel_from_event`` (``test_views_geometry``) and by a real browser, and the
switch is replaced by a recorder.
"""

import dataclasses
import json
import os
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import fusion_ui.plots  # noqa: F401 - registers the specs the page reads
from fusion_ui.core import catalog, db, params_ui, registry, store, versions
from tests import fields_fixtures as ff
from tests.fields_fixtures import fields_world  # noqa: F401 - the fixture

REPO_ROOT = Path(__file__).resolve().parent.parent
FIELDS = str(REPO_ROOT / "fusion_ui" / "pages" / "5_fields.py")
SINGLE_SHOT = str(REPO_ROOT / "fusion_ui" / "pages" / "2_single_shot.py")
APP = str(REPO_ROOT / "fusion_ui" / "app.py")

#: Two run days, preprocessed shots on both; 1110201007 has a raw file only and must not be offered.
FILES = (
    "apd_1160616027.nc",
    "apd_1160616027_preprocessed.nc",
    "apd_1160616026_preprocessed.nc",
    "apd_1140827010_preprocessed.nc",
    "apd_1110201007.nc",
)
SHOT = 1160616027
SELECTION = {"machine": "cmod", "shot": SHOT, "diagnostic": "apd", "preprocessed": True}
LABELS = ("pixel_averages", "method_fields", "blob_parameters")


class Deployment:
    def __init__(self, database, root, world):
        self.database, self.root, self.world = database, root, world

    def conn(self):
        return db.connect(self.database)

    def seed(self, shot, *keys, **kwargs):
        conn = self.conn()
        try:
            return self.world.seed(conn, shot, *keys, **kwargs)
        finally:
            conn.close()

    def counts(self):
        conn = self.conn()
        try:
            return {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("runs", "param_sets", "scalars", "presets")
            }
        finally:
            conn.close()

    def hash_of(self, plot, shot, other_than=None):
        conn = self.conn()
        try:
            rows = conn.execute(
                "SELECT params_hash FROM runs WHERE plot = ? AND shot = ? ORDER BY id",
                (plot, shot),
            ).fetchall()
        finally:
            conn.close()
        return [r[0] for r in rows if r[0] != other_than][0]

    def touch(self, name, seconds):
        """The file was rewritten, and a rescan saw it."""
        os.utime(self.root / "alcator" / "apd" / name, (seconds, seconds))
        conn = self.conn()
        catalog.rescan(conn, str(self.root / "alcator"), "cmod", None)
        conn.close()


def environment(monkeypatch, tmp_path):
    import streamlit as st

    monkeypatch.setenv("FUSION_DATA_FOLDER", str(tmp_path / "alcator"))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(tmp_path / "no_such_discharges.json"))
    monkeypatch.setenv("FUSION_UI_DB", str(tmp_path / "state" / "shot_explorer.sqlite"))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")
    conn = db.open_db(tmp_path / "state" / "shot_explorer.sqlite")
    catalog.rescan(conn, str(tmp_path / "alcator"), "cmod", None)
    conn.close()
    st.cache_data.clear()
    st.cache_resource.clear()
    return tmp_path / "state" / "shot_explorer.sqlite"


@pytest.fixture
def deployment(monkeypatch, tmp_path, fields_world):
    """A tiny tree indexed into a fresh database, with the stand-in product specs registered."""
    import streamlit as st

    folder = tmp_path / "alcator" / "apd"
    folder.mkdir(parents=True)
    for index, name in enumerate(FILES):
        (folder / name).write_bytes(b"x" * (index + 1))
    yield Deployment(environment(monkeypatch, tmp_path), tmp_path, fields_world)
    st.cache_data.clear()
    st.cache_resource.clear()


def run(**state):
    """The page, with session state set before it runs (``fields__pixel`` is ``fields.pixel``)."""
    app = AppTest.from_file(FIELDS, default_timeout=60)
    for key, value in state.items():
        app.session_state[key.replace("__", ".")] = value
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def rerun(app):
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def figures(app):
    """The figures the page drew, as the JSON the browser was sent."""
    return [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]


def kinds(figure, kind):
    return [t for t in figure["data"] if t.get("meta", {}).get("kind") == kind]


def subheaders(app):
    return [s.value for s in app.subheader]


def commands(app):
    return [c.value for c in app.code if c.language == "bash"]


def widget(app, kind, label):
    matches = [w for w in getattr(app, kind) if w.label == label]
    assert (
        matches
    ), f"no {kind} labelled {label!r}: {[w.label for w in getattr(app, kind)]}"
    return matches[0]


def sidebar(app, label):
    matches = [w for w in app.sidebar.selectbox if w.label == label]
    assert matches, [w.label for w in app.sidebar.selectbox]
    return matches[0]


def message(figure):
    return " ".join(a["text"] for a in figure["layout"].get("annotations", []))


def product_lines(app):
    return [
        m.value
        for m in app.markdown
        if m.value.startswith("**") and any(k in m.value for k in LABELS)
    ]


# -- nothing is computed ----------------------------------------------------------------------


def test_with_no_products_the_page_lists_the_missing_ones_and_the_command_for_each(
    deployment,
):
    app = run(selection=SELECTION)
    assert commands(app) == [
        f"fusion-ui precompute pixel_averages --shot {SHOT}",
        f"fusion-ui precompute method_fields --shot {SHOT}",
        f"fusion-ui precompute blob_parameters --shot {SHOT}",
    ]
    assert len(app.warning) == 3 and all(
        "Nothing is computed here" in w.value for w in app.warning
    )
    assert "35–70 min" in app.warning[0].value  # what the person is about to start
    # The expander that holds them is open, since there is something to say.
    assert [(e.label, e.proto.expanded) for e in app.expander] == [
        ("Products for this shot and these settings", True)
    ]
    # There is nothing to draw, and the page says what it needs rather than failing.
    assert [i.value for i in app.info if "needs method_fields" in i.value]
    assert not app.get("plotly_chart") and not app.get("download_button")


def test_the_page_never_computes_or_records_whatever_it_is_asked_to_show(
    deployment, monkeypatch
):
    deployment.seed(SHOT)
    deployment.seed(
        1160616026, "method_fields", params={"method_fields": ff.with_window(40)}
    )
    before = deployment.counts()
    deployment.world.calls.clear()

    def refused(*args, **kwargs):
        raise AssertionError(
            "the Fields page must never compute or write to the ledger"
        )

    for name in (
        "result",
        "compute_and_store",
        "record_params",
        "record_run",
        "write_scalars",
        "delete_run",
    ):
        monkeypatch.setattr(store, name, refused)

    for state in (
        {"selection": SELECTION},
        {"selection": SELECTION, "fields__pixel": (5, 4)},
        {"selection": SELECTION, "fields__pixel": (2, 0)},  # dead
        {"selection": SELECTION, "fields__pixel": (6, 7)},
        {
            "selection": {**SELECTION, "shot": 1160616026}
        },  # not computed under the defaults
        {"selection": {**SELECTION, "shot": 1140827010}},  # nothing at all
    ):
        app = run(**state)
    # View state moves nothing either.
    app = run(selection=SELECTION, fields__pixel=(5, 4))
    widget(app, "number_input", "Minimum events").set_value(700)
    rerun(app)
    widget(app, "checkbox", "Interior pixels only").check()
    rerun(app)
    assert deployment.counts() == before
    assert deployment.world.calls == []


def test_nothing_on_the_page_or_in_the_builders_can_start_an_analysis():
    """A static guard beside the behavioural one."""
    sources = [Path(FIELDS)] + sorted((REPO_ROOT / "fusion_ui" / "views").glob("*.py"))
    for path in sources:
        text = path.read_text()
        for forbidden in (
            "store.result(",
            "compute_and_store",
            "record_params(",
            ".compute(",
            "write_scalars",
        ):
            assert forbidden not in text, f"{path.name} mentions {forbidden}"


def test_nothing_on_the_page_or_in_the_builders_depends_on_the_old_velocity_field():
    """J10 deletes ``plots/velocity_field.py``: what the quiver needed from it was copied, not imported."""
    sources = [Path(FIELDS)] + sorted((REPO_ROOT / "fusion_ui" / "views").glob("*.py"))
    for path in sources:
        text = path.read_text()
        assert "plots.velocity_field" not in text, path.name
        assert "import velocity_field" not in text, path.name


# -- the products ---------------------------------------------------------------------------------


def test_computed_products_are_listed_with_when_by_which_code_and_under_which_parameters(
    deployment,
):
    deployment.seed(SHOT)
    app = run(selection=SELECTION)
    lines = product_lines(app)
    assert [line.split("**")[1] for line in lines] == list(LABELS)
    conn = deployment.conn()
    runs = {r["plot"]: r for r in conn.execute("SELECT * FROM runs")}
    conn.close()
    for line, key in zip(lines, LABELS):
        assert runs[key]["params_hash"][:12] in line
        assert runs[key]["created_at"][:19] in line
        assert runs[key]["code_version"] in line
    assert not commands(app) and not app.warning
    # With nothing to say the expander is closed, and the three blobs can be downloaded for the laptop.
    assert [e.proto.expanded for e in app.expander] == [False]
    assert [d.proto.label for d in app.get("download_button")] == [
        "Download pixel_averages (.nc)",
        "Download method_fields (.nc)",
        "Download blob_parameters (.nc)",
    ]


def test_a_failed_product_shows_its_error_and_the_command_that_retries_it(deployment):
    spec = registry.get("method_fields")

    def boom(ds, params, upstream):
        raise ValueError("no events survived")

    conn = deployment.conn()
    store.result(
        conn,
        dataclasses.replace(spec, compute=boom),
        deployment.world.target(SHOT),
        spec.params(),
        None,
        batch=True,
    )
    conn.close()
    app = run(selection=SELECTION)
    assert [e.value for e in app.error if "failed in batch" in e.value] == [
        "**method_fields** failed in batch: ValueError: no events survived"
    ]
    assert commands(app) == [
        f"fusion-ui precompute method_fields --shot {SHOT} --retry-failed",
        f"fusion-ui precompute blob_parameters --shot {SHOT}",
    ]
    # Only the bank, which was computed on the way, can be downloaded.
    assert [d.proto.label for d in app.get("download_button")] == [
        "Download pixel_averages (.nc)"
    ]


def test_a_product_whose_input_changed_is_marked_stale_with_the_command_that_refreshes_it(
    deployment,
):
    deployment.seed(SHOT)
    assert "stale" not in " ".join(m.value for m in run(selection=SELECTION).markdown)
    deployment.touch(f"apd_{SHOT}_preprocessed.nc", 1_700_000_500)
    app = run(selection=SELECTION)
    assert (
        len(
            [
                m
                for m in app.markdown
                if ":orange-badge[stale: input changed]" in m.value
            ]
        )
        == 3
    )
    assert [e.proto.expanded for e in app.expander] == [True]
    refresh = " ".join(c.value for c in app.caption if "Refresh it with" in c.value)
    assert f"`fusion-ui precompute method_fields --shot {SHOT} --force`" in refresh
    assert figures(app)  # still drawn: stale is information, not an error


def test_a_product_computed_under_another_fusion_scripts_commit_is_marked(
    deployment, monkeypatch
):
    import streamlit as st

    monkeypatch.setattr(
        store, "_code_version", lambda: "fusion_scripts=1111111 fusion_ui=aaaaaaa"
    )
    deployment.seed(SHOT)
    monkeypatch.setattr(
        versions,
        "code_version",
        lambda: {"fusion_scripts": "2222222-dirty", "fusion_ui": "ccccccc"},
    )
    app = run(selection=SELECTION)
    note = ":orange-badge[computed under fusion_scripts 1111111, the checkout is now at 2222222]"
    assert sum(note in m.value for m in app.markdown) == 3
    assert [e.proto.expanded for e in app.expander] == [True]
    # The same commit, however it was read, is the same code.
    st.cache_data.clear()
    monkeypatch.setattr(
        versions,
        "code_version",
        lambda: {"fusion_scripts": "1111111", "fusion_ui": "ccccccc"},
    )
    assert "orange-badge" not in " ".join(
        m.value for m in run(selection=SELECTION).markdown
    )


def test_the_mask_is_shown_with_its_source(deployment):
    deployment.seed(SHOT)
    app = run(selection=SELECTION)
    assert any(
        "Dead-pixel mask: **hand-made, 1160616** · 22 of 90 pixels dead" in c.value
        for c in app.caption
    )
    other = ff.World(mask_source="estimated at preprocessing, run day 1140827")
    other.install()
    try:
        conn = deployment.conn()
        other.seed(conn, 1160616026)
        conn.close()
    finally:
        other.uninstall()
        deployment.world.install()
    app = run(selection={**SELECTION, "shot": 1160616026})
    assert any(
        "**estimated at preprocessing, run day 1140827**" in c.value
        for c in app.caption
    )


# -- the shots and the settings -----------------------------------------------------------------


def test_the_shots_are_run_day_first_with_the_computed_ones_before_the_rest(deployment):
    deployment.seed(SHOT)
    app = run()
    # The raw-only shot is not offered: the products are computed from the preprocessed record.
    assert sidebar(app, "Run day").options == [
        "1140827 · 0/1 computed",
        "1160616 · 1/2 computed",
    ]
    assert sidebar(app, "Run day").value == "1160616"
    assert sidebar(app, "Shot").options == [str(SHOT), "1160616026 · not computed"]
    assert (
        sidebar(app, "Shot").value == SHOT
    )  # the computed shot, not the first by number
    assert sidebar(app, "Settings").options == ["default · 1 shot"]
    # Choosing another day lists its shots.
    sidebar(app, "Run day").set_value("1140827")
    rerun(app)
    assert sidebar(app, "Shot").options == ["1140827010 · not computed"]


def test_a_new_selection_picks_the_shot_and_the_page_then_remembers_its_own_choice(
    deployment,
):
    deployment.seed(SHOT)
    deployment.seed(1160616026)
    app = run(selection={**SELECTION, "shot": 1160616026})
    assert sidebar(app, "Shot").value == 1160616026
    # The person picks another shot here; the unchanged selection must not pull it back.
    sidebar(app, "Shot").set_value(SHOT)
    rerun(app)
    assert sidebar(app, "Shot").value == SHOT
    # A different selection (the shot browser's) wins.
    app.session_state["selection"] = {**SELECTION, "shot": 1140827010}
    rerun(app)
    assert sidebar(app, "Shot").value == 1140827010


def test_a_request_to_open_the_page_wins_whatever_the_page_remembers(deployment):
    deployment.seed(SHOT)
    deployment.seed(1160616026)
    app = run(selection={**SELECTION, "shot": 1160616026})
    sidebar(app, "Shot").set_value(SHOT)
    rerun(app)
    assert sidebar(app, "Shot").value == SHOT
    # The same selection again, as a second click on the same multi-shot point leaves it: only the
    # request tells the page that someone asked.
    app.session_state["selection"] = {**SELECTION, "shot": 1160616026}
    app.session_state["fields.open"] = {"shot": 1160616026, "pixel": (3, 4)}
    rerun(app)
    assert sidebar(app, "Shot").value == 1160616026
    assert app.session_state["fields.pixel"] == (3, 4)
    assert "fields.open" not in app.session_state  # consumed: it asks once
    assert "Pixel (x=3, y=4)" in subheaders(app)
    # And the page is free again afterwards.
    sidebar(app, "Shot").set_value(SHOT)
    rerun(app)
    assert sidebar(app, "Shot").value == SHOT
    # Entries are optional, and a shot that is not offered is ignored.
    app.session_state["fields.open"] = {"shot": 1110201007}
    rerun(app)
    assert sidebar(app, "Shot").value == SHOT


def test_other_settings_are_offered_as_their_difference_from_the_default(deployment):
    deployment.seed(SHOT)
    short = dict(
        method_fields=ff.with_window(40),
        blob_parameters=dataclasses.replace(
            ff.BlobParametersParams(), averages=ff.Averages(window=40)
        ),
    )
    deployment.seed(1160616026, params=short)
    app = run(selection=SELECTION)
    picker = sidebar(app, "Settings")
    assert picker.options == ["default · 1 shot", "averages.window = 40 · 1 shot"]
    short_hash = deployment.hash_of("method_fields", 1160616026)

    picker.set_value(short_hash)
    rerun(app)
    # Under these settings 1160616026 is the computed shot, and it comes first.
    assert sidebar(app, "Shot").options == ["1160616026", f"{SHOT} · not computed"]
    assert (
        sidebar(app, "Shot").value == SHOT
    )  # the person's shot is kept, now marked not computed
    assert any(
        "averages.window" in c.value and "default 60" in c.value
        for c in app.sidebar.caption
    )
    # What is missing for 027 under these settings is named with its own parameters to save.
    assert (
        commands(app)[0]
        == f"fusion-ui precompute pixel_averages --shot {SHOT} --params-json params.json"
    )
    assert any('"window": 40' in c.value for c in app.code if c.language == "json")
    # On 026 the products are there, under the shorter window: the bank has 41 lags.
    sidebar(app, "Shot").set_value(1160616026)
    rerun(app)
    assert not commands(app)
    app.session_state["fields.pixel"] = (5, 4)
    rerun(app)
    slider = widget(app, "slider", "Lag [µs]")
    assert (slider.min, slider.max) == (
        -10.0,
        10.0,
    )  # the lags offered stop at this bank's window
    assert widget(app, "number_input", "Span ± [µs]").proto.placeholder.startswith(
        "auto: ±"
    )


def test_a_request_can_name_the_settings_and_unknown_ones_are_the_default(deployment):
    deployment.seed(SHOT)
    deployment.seed(
        SHOT,
        "pixel_averages",
        "method_fields",
        params={"method_fields": ff.with_window(40)},
    )
    default_hash = params_ui.hash_params(
        "method_fields", registry.get("method_fields").params()
    )[0]
    short_hash = deployment.hash_of("method_fields", SHOT, other_than=default_hash)
    app = run(selection=SELECTION, fields__open={"settings": short_hash})
    assert sidebar(app, "Settings").value == short_hash
    assert any(
        short_hash[:12] in line
        for line in product_lines(app)
        if line.startswith("**method_fields**")
    )
    unknown = run(selection=SELECTION, fields__open={"settings": "0" * 40})
    assert sidebar(unknown, "Settings").value == default_hash


# -- the shot level ---------------------------------------------------------------------------


def test_the_grid_draws_and_nothing_is_open_until_a_pixel_is_chosen(deployment):
    deployment.seed(SHOT)
    app = run(selection=SELECTION)
    assert subheaders(app) == [f"Shot {SHOT}", "Velocity fields"]
    (grid,) = figures(app)
    assert grid["layout"]["meta"]["mode"] == "arrows"
    assert all(
        kinds(grid, k) for k in ("shafts", "heads", "pixels", "dead", "key-shaft")
    )
    assert [i.value for i in app.info] == [
        "Click a pixel in any panel to see the average behind its numbers."
    ]
    # The toggle between arrows and maps on one diverging scale.
    assert widget(app, "radio", "Show").options == ["Arrows", "v_R map", "v_Z map"]
    widget(app, "radio", "Show").set_value("vr")
    rerun(app)
    (maps,) = figures(app)
    assert maps["layout"]["meta"]["mode"] == "vr"
    assert maps["layout"]["coloraxis"]["cmin"] == -maps["layout"]["coloraxis"]["cmax"]
    # Arrows can be made longer, all of them by the same factor, from the page.
    widget(app, "radio", "Show").set_value("arrows")
    rerun(app)
    scale = figures(app)[0]["layout"]["meta"]["arrow_scale"]
    widget(app, "slider", "Arrow length ×").set_value(2.0)
    rerun(app)
    assert figures(app)[0]["layout"]["meta"]["arrow_scale"] == pytest.approx(scale / 2)


def test_the_view_cuts_are_view_state_and_move_pixels_without_a_new_parameter_set(
    deployment,
):
    deployment.seed(SHOT)
    app = run(selection=SELECTION)
    before = deployment.counts()
    assert not kinds(figures(app)[0], "cut")
    widget(app, "number_input", "Minimum events").set_value(700)
    rerun(app)
    cut = kinds(figures(app)[0], "cut")
    assert cut and "events < 700" in cut[0]["text"][0]
    widget(app, "number_input", "Minimum lags").set_value(30)
    widget(app, "checkbox", "Interior pixels only").check()
    rerun(app)
    assert any("edge pixel" in t["text"][0] for t in kinds(figures(app)[0], "cut"))
    assert (
        deployment.counts() == before
    )  # moving a cut mints no param_sets row and recomputes nothing


@pytest.fixture
def both_versions(monkeypatch, tmp_path, apd_dataset_path, fields_world):
    """One tiny real APD file, raw and preprocessed, indexed: what the single-shot page can open."""
    import streamlit as st

    shutil.copy(
        apd_dataset_path, apd_dataset_path.with_name("apd_1234_preprocessed.nc")
    )
    database = environment(monkeypatch, tmp_path)
    yield Deployment(database, tmp_path, fields_world)
    st.cache_data.clear()
    st.cache_resource.clear()


TINY = {"machine": "cmod", "shot": 1234, "diagnostic": "apd", "preprocessed": True}


def open_fields(**state):
    """The real multipage app, on the Fields page as a person gets there from the landing page."""
    app = AppTest.from_file(APP, default_timeout=60)
    for key, value in state.items():
        app.session_state[key.replace("__", ".")] = value
    rerun(app)
    app.switch_page("pages/5_fields.py")
    return rerun(app)


def test_the_mask_links_to_the_dead_pixel_view_of_the_raw_file_and_back(both_versions):
    both_versions.seed(1234)
    app = open_fields(selection=TINY)
    assert [t.value for t in app.title] == ["Fields"]
    widget(app, "button", "Open the dead-pixel view of the raw file").click()
    rerun(app)
    # The click switched pages: the single-shot page, on the raw file of the same shot, on the
    # dead-pixel plot, which only the raw file offers.
    assert [t.value for t in app.title] == ["Single shot"]
    assert [r.value for r in app.sidebar.radio if r.label == "Version"] == ["Raw"]
    assert sidebar(app, "Shot").value == 1234
    plot = sidebar(app, "Plot")
    assert plot.value.key == "dead_pixels"
    assert "Dead pixels (PDF and spectrum of every pixel)" in plot.options
    # Back to the Fields page, where the person was.
    app.switch_page("pages/5_fields.py")
    rerun(app)
    assert [t.value for t in app.title] == ["Fields"]
    assert sidebar(app, "Shot").value == 1234


def test_the_single_shot_page_still_opens_on_the_preprocessed_file_otherwise(
    both_versions,
):
    """The browser lands on the preprocessed file whenever there is one, and the single-shot page used
    to ignore the selection's version altogether: only a selection that names the raw file of this
    shot changes what it opens, and the radio is still the person's to move."""
    raw = {**TINY, "preprocessed": False}
    for selection, version in (
        (None, "Preprocessed"),
        (TINY, "Preprocessed"),
        (raw, "Raw"),
        (
            {**raw, "shot": 99},
            "Preprocessed",
        ),  # another shot's selection is not this shot's
    ):
        app = AppTest.from_file(SINGLE_SHOT, default_timeout=60)
        app.session_state["selection"] = selection
        app.run()
        assert not app.exception, [e.value for e in app.exception]
        assert [r.value for r in app.sidebar.radio if r.label == "Version"] == [
            version
        ], selection
    app.session_state["selection"] = raw
    app.run()
    app.sidebar.radio[0].set_value("Preprocessed")
    app.run()
    assert [r.value for r in app.sidebar.radio if r.label == "Version"] == [
        "Preprocessed"
    ]


def test_no_raw_file_means_no_link(deployment):
    deployment.seed(1160616026)
    app = run(selection={**SELECTION, "shot": 1160616026})
    assert not [b for b in app.button if b.label.startswith("Open the dead-pixel")]
    assert any("No raw file is indexed" in c.value for c in app.caption)


def test_view_state_survives_a_visit_to_another_page(deployment):
    """Streamlit drops the state of every widget that is not drawn in a run. The person follows a
    link to another page and comes back: the shot, the cuts and the lags are where they left them.
    """
    deployment.seed(SHOT)
    deployment.seed(1160616026)
    app = open_fields(selection=SELECTION, fields__pixel=(5, 4))
    sidebar(app, "Shot").set_value(1160616026)
    widget(app, "number_input", "Minimum events").set_value(700)
    widget(app, "slider", "Panels").set_value(7)
    widget(app, "radio", "Show").set_value("vz")
    rerun(app)

    app.switch_page("pages/2_single_shot.py")
    rerun(app)
    assert [t.value for t in app.title] == ["Single shot"]
    # The widgets of the page that was left are gone, as Streamlit does; the page's own memory is not.
    assert "fields.min_events" not in app.session_state
    app.switch_page("pages/5_fields.py")
    rerun(app)
    assert sidebar(app, "Shot").value == 1160616026
    assert widget(app, "number_input", "Minimum events").value == 700
    assert widget(app, "slider", "Panels").value == 7
    assert widget(app, "radio", "Show").value == "vz"
    assert app.session_state["fields.pixel"] == (5, 4)


# -- the pixel level ----------------------------------------------------------------------------


def test_a_chosen_pixel_opens_the_pixel_level_below_the_grid(deployment):
    deployment.seed(SHOT)
    app = run(selection=SELECTION, fields__pixel=(5, 4))
    assert subheaders(app) == [
        f"Shot {SHOT}",
        "Velocity fields",
        "Pixel (x=5, y=4)",
        "The average at a few lags",
        "One frame",
        "Tracks",
        "Numbers",
    ]
    drawn = figures(app)
    assert len(drawn) == 5  # the grid, the strip, the frame and its trace, the tracks
    assert [
        b.label for b in app.button if b.label in ("◀ Previous", "Next ▶", "Close")
    ] == [
        "◀ Previous",
        "Next ▶",
        "Close",
    ]
    # The grid rings the pixel in view, in every panel.
    assert len(kinds(drawn[0], "selected")) == 7
    # The strip: two rows of five panels, at the lags the pixel's own fit asks for.
    strip = drawn[1]
    assert len([t for t in strip["data"] if t["type"] == "heatmap"]) == 10
    assert strip["layout"]["meta"]["source"] == "fit"
    # The numbers are the stored ones.
    methods_table, blobs_table = (d.value for d in app.dataframe)
    assert len(methods_table) == 7 and len(blobs_table) == 13
    conn = deployment.conn()
    digest = params_ui.hash_params(
        "method_fields", registry.get("method_fields").params()
    )[0]
    stored = store.load_result(
        conn,
        store.find_run(conn, deployment.world.target(SHOT), "method_fields", digest),
    )
    conn.close()
    assert methods_table.set_index("method").loc[
        "2DCA centroid", "v_R [m/s]"
    ] == pytest.approx(float(stored["vr_com"].values[4, 5]))


def test_the_lags_of_the_strip_and_the_frame_are_view_state(deployment):
    deployment.seed(SHOT)
    app = run(selection=SELECTION, fields__pixel=(5, 4))
    before = deployment.counts()
    span = widget(app, "number_input", "Span ± [µs]")
    assert span.value is None and span.proto.placeholder == "auto: ±7 µs"
    widget(app, "slider", "Panels").set_value(7)
    span.set_value(9.0)
    rerun(app)
    lags = figures(app)[1]["layout"]["meta"]["lags"]
    assert (
        len(lags) == 7
        and lags[0] == pytest.approx(-9e-6)
        and lags[-1] == pytest.approx(9e-6)
    )
    # Typed lags win.
    widget(app, "text_input", "Lags [µs]").set_value("-6, -3, 0, 3, 6")
    rerun(app)
    assert [round(t * 1e6, 3) for t in figures(app)[1]["layout"]["meta"]["lags"]] == [
        -6,
        -3,
        0,
        3,
        6,
    ]
    # The frame's lag and field.
    widget(app, "slider", "Lag [µs]").set_value(3.0)
    widget(app, "radio", "Field").set_value("cross_corr")
    rerun(app)
    frame_figure = figures(app)[2]
    assert frame_figure["layout"]["meta"]["lag"] == pytest.approx(3e-6)
    assert "cross-correlation" in frame_figure["layout"]["title"]["text"]
    assert deployment.counts() == before
    # The lag slider reaches the bank's window and no further.
    slider = widget(app, "slider", "Lag [µs]")
    assert (slider.min, slider.max) == (-15.0, 15.0)


def test_the_pixel_is_kept_when_the_shot_changes(deployment):
    deployment.seed(SHOT)
    deployment.seed(1160616026)
    app = run(selection=SELECTION, fields__pixel=(5, 4))
    sidebar(app, "Shot").set_value(1160616026)
    rerun(app)
    assert app.session_state["fields.pixel"] == (5, 4)
    assert "Pixel (x=5, y=4)" in subheaders(app)
    # Even through a shot with nothing computed, which has no array to say whether the pixel is on it.
    sidebar(app, "Run day").set_value("1140827")
    rerun(app)
    assert sidebar(app, "Shot").value == 1140827010
    assert app.session_state["fields.pixel"] == (5, 4)
    assert "Pixel (x=5, y=4)" not in subheaders(app)
    sidebar(app, "Run day").set_value("1160616")
    rerun(app)
    assert "Pixel (x=5, y=4)" in subheaders(app)


def test_a_dead_pixel_or_one_without_events_shows_a_message_and_not_an_exception(
    deployment,
):
    deployment.world.options.no_events = ((6, 7),)
    deployment.world.options.failed = ((7, 4),)
    deployment.seed(SHOT)

    dead = run(selection=SELECTION, fields__pixel=(2, 0))
    assert "Pixel (x=2, y=0)" in subheaders(dead)
    # The strip, the frame, its trace and the tracks each say so in the place of the figure ...
    assert all("is dead in this shot's mask" in message(f) for f in figures(dead)[1:])
    # ... and the numbers say it as a sentence.
    assert any("is dead in this shot's mask" in i.value for i in dead.info)

    empty = run(selection=SELECTION, fields__pixel=(6, 7))
    assert all(
        "No events at pixel (x=6, y=7)" in message(f) for f in figures(empty)[1:]
    )
    assert any("No events at pixel (x=6, y=7)" in i.value for i in empty.info)

    failed = run(
        selection=SELECTION, fields__pixel=(7, 4)
    )  # a live pixel whose fit failed still draws
    strip = figures(failed)[1]
    assert len([t for t in strip["data"] if t["type"] == "heatmap"]) == 10
    assert strip["layout"]["meta"]["source"] == "deck"


def test_a_pixel_that_is_not_on_the_array_is_dropped_not_an_error(deployment):
    deployment.seed(SHOT)
    app = run(selection=SELECTION, fields__pixel=(40, 40))
    assert app.session_state["fields.pixel"] is None
    assert subheaders(app) == [f"Shot {SHOT}", "Velocity fields"]


def test_previous_and_next_walk_the_live_pixels(deployment):
    deployment.seed(SHOT)
    app = run(selection=SELECTION, fields__pixel=(4, 9))
    widget(app, "button", "Next ▶").click()
    rerun(app)
    assert app.session_state["fields.pixel"] == (5, 9)
    widget(app, "button", "◀ Previous").click()
    rerun(app)
    assert app.session_state["fields.pixel"] == (4, 9)
    widget(app, "button", "◀ Previous").click()  # (3, 9) is dead: on to (2, 9)
    rerun(app)
    assert app.session_state["fields.pixel"] == (2, 9)
    widget(app, "button", "Close").click()
    rerun(app)
    assert app.session_state["fields.pixel"] is None
    assert subheaders(app) == [f"Shot {SHOT}", "Velocity fields"]


def test_products_missing_for_part_of_the_page_are_named_where_they_are_needed(
    deployment,
):
    # Only the bank: the strip and the frame can draw, what rests on method_fields cannot.
    deployment.seed(SHOT, "pixel_averages")
    app = run(selection=SELECTION, fields__pixel=(5, 4))
    needs = [i.value for i in app.info if i.value.startswith("This needs")]
    assert len(needs) == 3 and all(
        "method_fields" in n for n in needs
    )  # the grid, the tracks, the numbers
    assert commands(app) == [
        f"fusion-ui precompute method_fields --shot {SHOT}",
        f"fusion-ui precompute blob_parameters --shot {SHOT}",
    ]
    drawn = figures(app)
    assert len(drawn) == 3  # the strip, the frame and its trace
    assert len([t for t in drawn[0]["data"] if t["type"] == "heatmap"]) == 10
    assert not [
        t for t in drawn[0]["data"] if t.get("name") in ("2DCA max", "2DCC")
    ]  # no tracks to mark


# -- when there is nothing to show ---------------------------------------------------------------


def test_no_preprocessed_shot_is_a_message(monkeypatch, tmp_path, fields_world):
    import streamlit as st

    folder = tmp_path / "alcator" / "apd"
    folder.mkdir(parents=True)
    (folder / "apd_1110201007.nc").write_bytes(b"x")
    environment(monkeypatch, tmp_path)
    app = run()
    assert any("No preprocessed APD shot" in i.value for i in app.info)
    st.cache_data.clear()
    st.cache_resource.clear()


def test_an_unregistered_product_is_an_error_and_not_a_traceback(deployment):
    saved = registry.REGISTRY.pop("method_fields")
    try:
        app = run(selection=SELECTION)
    finally:
        registry.REGISTRY["method_fields"] = saved
    assert any("needs the `method_fields` product" in e.value for e in app.error)
