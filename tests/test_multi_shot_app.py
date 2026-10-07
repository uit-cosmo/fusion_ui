"""Smoke tests for the multi-shot page."""

import dataclasses
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import fusion_ui.plots  # noqa: F401
from fusion_ui.core import catalog, db, params_ui, registry, scalar_labels, store

REPO_ROOT = Path(__file__).resolve().parent.parent
MULTI_SHOT = str(REPO_ROOT / "fusion_ui" / "pages" / "3_multi_shot.py")


@dataclasses.dataclass
class _Params:
    source: str = "test"


@pytest.fixture
def deployment(monkeypatch, tmp_path, data_folder, discharge_db):
    import streamlit as st

    database = tmp_path / "state" / "shot_explorer.sqlite"
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(data_folder))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(discharge_db))
    monkeypatch.setenv("FUSION_UI_DB", str(database))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")

    conn = db.open_db(database)
    catalog.rescan(conn, str(data_folder), "cmod", str(discharge_db))
    _write_scalars(conn)
    conn.close()

    st.cache_data.clear()
    st.cache_resource.clear()
    yield database
    st.cache_data.clear()
    st.cache_resource.clear()


def _write_scalars(conn, machine="cmod", scalars=None):
    """Two shots with stored ``vx_c`` values, one source, per-pixel."""
    for shot, values in scalars or [
        (1160616027, {(6, 6): 400.0, (6, 5): 380.0}),
        (1110201007, {(6, 6): 100.0}),
    ]:
        target = registry.Target(machine, shot, "apd", True, "", 0.0, 0.0, "none")
        params_hash, _ = store.record_params(conn, "synthetic", _Params())
        run = store.record_run(
            conn,
            target,
            "synthetic",
            params_hash,
            blob_path=None,
            status="ok",
            error=None,
            seconds=None,
            code_version="test",
        )
        mapping = {(x, y, "vx_c"): v for (x, y), v in values.items()}
        store.write_scalars(conn, run["id"], mapping)


def run(path, default_timeout=60):
    app = AppTest.from_file(path, default_timeout=default_timeout).run()
    assert not app.exception, app.exception
    return app


def widget(app, kind, label):
    matches = [w for w in getattr(app, kind) if w.label == label]
    assert (
        matches
    ), f"no {kind} labelled {label!r}: {[w.label for w in getattr(app, kind)]}"
    return matches[0]


def test_an_empty_store_explains_itself(
    monkeypatch, tmp_path, data_folder, discharge_db
):
    import streamlit as st

    database = tmp_path / "state" / "shot_explorer.sqlite"
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(data_folder))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(discharge_db))
    monkeypatch.setenv("FUSION_UI_DB", str(database))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")

    conn = db.open_db(database)
    catalog.rescan(conn, str(data_folder), "cmod", str(discharge_db))
    conn.close()
    st.cache_data.clear()
    st.cache_resource.clear()

    app = run(MULTI_SHOT)
    assert app.warning
    st.cache_data.clear()
    st.cache_resource.clear()


def test_the_page_offers_the_stored_scalar_and_renders(deployment):
    app = run(MULTI_SHOT)
    assert widget(app, "selectbox", "Scalar").options == ["vx_c"]
    # Both shots have a stored value, so the scatter carries two points.
    assert any("2 shots" in c.value for c in app.caption)


def test_a_carried_over_selection_restricts_the_scatter(deployment):
    app = AppTest.from_file(MULTI_SHOT, default_timeout=60)
    app.session_state["shot_selection"] = [1110201007]
    app.run()
    assert not app.exception
    assert any("1 shot" in c.value for c in app.caption)


def test_the_aggregate_picker_offers_each_collapse(deployment):
    app = run(MULTI_SHOT)
    options = widget(app, "selectbox", "Aggregate").options
    assert options == [
        "mean over pixels",
        "median over pixels",
        "maximum over pixels",
        "fixed pixel",
    ]


def test_another_machines_scalars_are_left_out(deployment):
    """`rescan` only deletes its own machine's rows, so an index can hold two
    machines at once -- and shot numbers are not unique across them."""
    conn = db.open_db(deployment)
    _write_scalars(conn, machine="aug", scalars=[(1160616027, {(6, 6): -1.0})])
    conn.close()

    app = run(MULTI_SHOT)
    assert any("2 shots on `cmod`" in c.value for c in app.caption)


def test_a_shot_level_scalar_is_not_labelled_as_a_pixel_collapse(deployment):
    """`x = y = -1` is already one number per shot: there is no collapse to
    name, and the Aggregate picker is hidden rather than defaulting to mean."""
    conn = db.open_db(deployment)
    for shot in (1160616027, 1110201007):
        target = registry.Target("cmod", shot, "apd", True, "", 0.0, 0.0, "none")
        params_hash, _ = store.record_params(conn, "shotlevel", _Params())
        run = store.record_run(
            conn,
            target,
            "shotlevel",
            params_hash,
            blob_path=None,
            status="ok",
            error=None,
            seconds=None,
            code_version="test",
        )
        store.write_scalars(conn, run["id"], {"number_events": 42.0})
    conn.close()

    app = AppTest.from_file(MULTI_SHOT, default_timeout=60)
    app.session_state["ms.scalar"] = "number_events"
    app.run()
    assert not app.exception, app.exception
    assert not [w for w in app.selectbox if w.label == "Aggregate"]
    caption = next(c.value for c in app.caption if "shots on" in c.value)
    assert "mean over pixels" not in caption
    assert "one value per shot" in caption


# ---------------------------------------------------------------------------
# The two phase-06 products in the ledger, as a batch leaves them.
# ---------------------------------------------------------------------------

APP = str(REPO_ROOT / "fusion_ui" / "app.py")

#: The one shot of the conftest tree that has a preprocessed file, which is what the Fields page lists.
SHOT = 1160616027

#: Each seeded scalar's value at two pixels.
PIXELS = ((5, 4, 400.0), (6, 4, 380.0))


class Products:
    """``method_fields`` and ``blob_parameters`` runs on SHOT, seeded the way a batch leaves them.

    Under the real parameter classes, so the hashes are the ones the pages work with; there are no blobs,
    since the click only reads the ledger. Each run writes one scalar at two pixels.
    """

    def __init__(self, conn):
        from fusion_ui.views import products as prod

        found, _ = prod.specs(registry)
        method = found["method_fields"].params()
        blob = found["blob_parameters"].params()

        def window(params):
            """The same settings with a 2DCA window of 40 samples: a shot with faster dynamics."""
            return dataclasses.replace(
                params, averages=dataclasses.replace(params.averages, window=40)
            )

        self.params = {
            "method_fields": {
                "default": method,
                "short": window(method),
                # The TDE only: the blob parameters that go with it are the default's.
                "tde": dataclasses.replace(
                    method, tde=dataclasses.replace(method.tde, min_cc=0.4)
                ),
            },
            "blob_parameters": {
                "default": blob,
                "short": window(blob),
                # The ellipse fit's own settings: no velocity setting goes with them.
                "fit": dataclasses.replace(
                    blob,
                    blobs=dataclasses.replace(
                        blob.blobs,
                        gauss_fit=dataclasses.replace(
                            blob.blobs.gauss_fit, size_penalty=3.0
                        ),
                    ),
                ),
                # A 2DCA window that the velocity fields were never run with.
                "lonely": dataclasses.replace(
                    blob, averages=dataclasses.replace(blob.averages, window=30)
                ),
            },
        }
        self.hash = {}
        for plot, names in (
            ("method_fields", ("vr_com", "level_com")),
            ("blob_parameters", ("lr",)),
        ):
            for key, params in self.params[plot].items():
                self.hash[plot, key] = self.seed(conn, plot, params, *names)

    @staticmethod
    def seed(conn, plot, params, *names):
        target = registry.Target("cmod", SHOT, "apd", True, "", 0.0, 0.0, "none")
        digest, _ = store.record_params(conn, plot, params)
        run = store.record_run(
            conn,
            target,
            plot,
            digest,
            blob_path=None,
            status="ok",
            error=None,
            seconds=None,
            code_version="test",
        )
        store.write_scalars(
            conn,
            run["id"],
            {(x, y, name): value for name in names for x, y, value in PIXELS},
        )
        return digest

    def source(self, plot, key):
        return (plot, self.hash[plot, key], "apd", 1)


@pytest.fixture
def products(deployment):
    conn = db.open_db(deployment)
    try:
        yield Products(conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# What the y-axis calls a scalar: its label, or its stored name when it has none.
# ---------------------------------------------------------------------------


def y_axis_title(app):
    (chart,) = app.get("plotly_chart")
    return json.loads(chart.proto.spec)["layout"]["yaxis"]["title"]["text"]


def test_the_axis_title_names_a_labelled_scalar_by_what_it_is(deployment):
    conn = db.open_db(deployment)
    Products(conn)
    conn.close()
    app = AppTest.from_file(MULTI_SHOT, default_timeout=60)
    app.session_state["ms.scalar"] = "vr_com"
    app.run()
    assert not app.exception, app.exception
    assert y_axis_title(app) == "v_R, 2DCA centroid [m/s] (mean over pixels)"
    # The aggregate is still named, and the caption still says which stored scalar it is.
    app.sidebar.selectbox(key="ms.aggregate").set_value("median")
    app.run()
    # One line more than fits is broken at the aggregate, not inside it.
    assert y_axis_title(app) == "v_R, 2DCA centroid [m/s]<br>(median over pixels)"
    assert any(c.value.startswith("vr_com · median over pixels") for c in app.caption)


def test_a_long_title_is_broken_into_lines_the_axis_has_room_for(deployment):
    """Plotly draws a rotated title on one line, and cuts it off where the plot ends: this one is 104
    characters on a plot 520 pixels high. The label is whole, and so is the unit."""
    conn = db.open_db(deployment)
    Products(conn)
    conn.close()
    app = AppTest.from_file(MULTI_SHOT, default_timeout=60)
    app.session_state["ms.scalar"] = "level_com"
    app.run()
    assert not app.exception, app.exception
    plain = (
        "contour level of the 2DCA centroid track, fraction of the average's maximum "
        "[no unit] (mean over pixels)"
    )
    title = y_axis_title(app)
    assert title == scalar_labels.wrapped(plain)
    lines = title.split("<br>")
    assert 2 <= len(lines) <= 3
    assert all(len(line) <= scalar_labels.TITLE_WIDTH for line in lines)
    assert " ".join(lines) == plain
    # The unit and the aggregate are each on one line, never split across two.
    assert any("[no unit]" in line for line in lines)
    assert any("(mean over pixels)" in line for line in lines)


def test_the_axis_title_of_a_shot_level_labelled_scalar_has_no_collapse(deployment):
    """``number_events`` is shared with ``two_dca`` and the seed, and one number per shot here."""
    conn = db.open_db(deployment)
    for shot in (1160616027, 1110201007):
        target = registry.Target("cmod", shot, "apd", True, "", 0.0, 0.0, "none")
        params_hash, _ = store.record_params(conn, "shotlevel", _Params())
        run = store.record_run(
            conn,
            target,
            "shotlevel",
            params_hash,
            blob_path=None,
            status="ok",
            error=None,
            seconds=None,
            code_version="test",
        )
        store.write_scalars(conn, run["id"], {"number_events": 42.0})
    conn.close()
    app = AppTest.from_file(MULTI_SHOT, default_timeout=60)
    app.session_state["ms.scalar"] = "number_events"
    app.run()
    assert not app.exception, app.exception
    assert (
        y_axis_title(app) == "number of events in the conditional average<br>[no unit]"
    )


def test_a_scalar_without_a_label_keeps_its_stored_name_on_the_axis(deployment):
    app = run(MULTI_SHOT)
    assert widget(app, "selectbox", "Scalar").options == ["vx_c"]
    assert y_axis_title(app) == "vx_c (mean over pixels)"


def test_the_scalar_picker_shows_a_labelled_name_with_its_label_and_stores_the_name(
    deployment,
):
    """``vx_c`` is in the store with no label, and three of the 32 names are beside it."""
    conn = db.open_db(deployment)
    Products(conn)
    conn.close()
    app = run(MULTI_SHOT)
    picker = widget(app, "selectbox", "Scalar")
    assert picker.options == [
        f"level_com — {scalar_labels.LABELS['level_com']}",
        "lr — radial FWHM of the conditional average [m]",
        "vr_com — v_R, 2DCA centroid [m/s]",
        "vx_c",  # no label: as it is
    ]
    assert picker.value == "level_com"  # the stored name, whatever the picker shows

    picker.set_value("vr_com")
    app.run()
    assert not app.exception, app.exception
    assert app.session_state["ms.scalar"] == "vr_com"
    assert widget(app, "selectbox", "Scalar").value == "vr_com"
    # What is plotted is the stored scalar: the caption names it, and the axis its label.
    assert any(c.value.startswith("vr_com · mean over pixels") for c in app.caption)
    assert y_axis_title(app) == "v_R, 2DCA centroid [m/s] (mean over pixels)"

    picker = widget(app, "selectbox", "Scalar")
    picker.set_value("vx_c")
    app.run()
    assert app.session_state["ms.scalar"] == "vx_c"
    assert y_axis_title(app) == "vx_c (mean over pixels)"


def test_a_stored_name_set_before_the_page_runs_is_still_found_by_the_picker(
    deployment,
):
    """The pickers' state is the raw name, as the tests above and any other page that seeds it set it."""
    conn = db.open_db(deployment)
    Products(conn)
    conn.close()
    app = AppTest.from_file(MULTI_SHOT, default_timeout=60)
    app.session_state["ms.scalar"] = "lr"
    app.run()
    assert not app.exception, app.exception
    assert widget(app, "selectbox", "Scalar").value == "lr"
    assert any(c.value.startswith("lr · mean over pixels") for c in app.caption)


# ---------------------------------------------------------------------------
# A click on a point: the Fields page for the two phase-06 products, the
# single-shot page for every other source.
# ---------------------------------------------------------------------------


def click(monkeypatch, shot=SHOT):
    """AppTest cannot click a Plotly point: the page reads the chart's event through this function."""
    from fusion_ui.core import decimate

    monkeypatch.setattr(
        decimate,
        "selection_points",
        lambda event: [{"customdata": ["cmod", shot, 0.72, 1.42, 0.55]}],
    )


def open_multi_shot(**state):
    """The whole multipage app on the multi-shot page, so that a page switch has somewhere to go."""
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    for key, value in state.items():
        app.session_state[key.replace("__", ".")] = value
    app.switch_page("pages/3_multi_shot.py")
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def sidebar(app, label):
    matches = [w for w in app.sidebar.selectbox if w.label == label]
    assert matches, [w.label for w in app.sidebar.selectbox]
    return matches[0]


def titles(app):
    return [t.value for t in app.title]


def fixed_pixel(source, scalar, x=6, y=4):
    """The sidebar's state for a scatter of ``source`` collapsed to one pixel."""
    return {
        "ms__scalar": scalar,
        "ms__source": source,
        "ms__aggregate": "pixel",
        "ms__pixel__x": x,
        "ms__pixel__y": y,
    }


def ledger_counts(database):
    conn = db.connect(database)
    try:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("runs", "param_sets", "scalars", "presets")
        }
    finally:
        conn.close()


SELECTION = {
    "machine": "cmod",
    "shot": SHOT,
    "diagnostic": "apd",
    "preprocessed": True,
}


def test_a_point_nobody_clicked_stays_on_the_multi_shot_page(deployment, products):
    app = open_multi_shot(
        **fixed_pixel(products.source("method_fields", "short"), "vr_com")
    )
    assert titles(app) == ["Multi shot"]
    assert "fields.open" not in app.session_state


def test_a_method_fields_point_opens_the_fields_page_on_its_shot_settings_and_pixel(
    deployment, products, monkeypatch
):
    click(monkeypatch)
    app = open_multi_shot(
        **fixed_pixel(products.source("method_fields", "short"), "vr_com")
    )
    assert titles(app) == ["Fields"]
    assert sidebar(app, "Shot").value == SHOT
    # The point's own parameters, which are not the default's.
    assert sidebar(app, "Settings").value == products.hash["method_fields", "short"]
    assert (
        products.hash["method_fields", "short"]
        != products.hash["method_fields", "default"]
    )
    assert app.session_state["fields.pixel"] == (6, 4)
    # The other pages follow, and the request was read once: it is not left to be read again.
    assert app.session_state["selection"] == SELECTION
    assert "fields.open" not in app.session_state


def test_the_pixel_goes_only_with_a_fixed_pixel_aggregate(
    deployment, products, monkeypatch
):
    click(monkeypatch)
    app = open_multi_shot(
        ms__scalar="vr_com", ms__source=products.source("method_fields", "short")
    )
    assert titles(app) == ["Fields"]
    assert sidebar(app, "Settings").value == products.hash["method_fields", "short"]
    assert sidebar(app, "Shot").value == SHOT
    assert app.session_state["fields.pixel"] is None
    for how in ("median", "maximum"):
        app = open_multi_shot(
            ms__scalar="vr_com",
            ms__source=products.source("method_fields", "short"),
            ms__aggregate=how,
            ms__pixel__x=6,
            ms__pixel__y=4,
        )
        assert titles(app) == ["Fields"]
        assert app.session_state["fields.pixel"] is None, how


@pytest.mark.parametrize("key", ["default", "short"])
def test_a_blob_parameters_point_opens_on_the_method_fields_settings_that_go_with_it(
    deployment, products, monkeypatch, key
):
    click(monkeypatch)
    app = open_multi_shot(
        **fixed_pixel(products.source("blob_parameters", key), "lr", x=5)
    )
    assert titles(app) == ["Fields"]
    # The settings picker is a method_fields hash, and the one that maps to this blob run.
    assert sidebar(app, "Settings").value == products.hash["method_fields", key]
    assert sidebar(app, "Shot").value == SHOT
    assert app.session_state["fields.pixel"] == (5, 4)
    assert app.session_state["selection"] == SELECTION


def test_a_blob_point_that_several_settings_go_with_opens_on_the_one_with_fields_on_the_shot(
    deployment, products, monkeypatch
):
    """The default and the TDE variant give the same blob run; the default has no fields on this shot."""
    conn = db.open_db(deployment)
    target = registry.Target("cmod", SHOT, "apd", True, "", 0.0, 0.0, "none")
    default = store.find_run(
        conn, target, "method_fields", products.hash["method_fields", "default"]
    )
    store.delete_run(conn, default)
    conn.close()

    click(monkeypatch)
    app = open_multi_shot(
        **fixed_pixel(products.source("blob_parameters", "default"), "lr")
    )
    assert titles(app) == ["Fields"]
    assert sidebar(app, "Settings").value == products.hash["method_fields", "tde"]
    # With every method_fields run there, the default is the one: it comes first in the picker.
    conn = db.open_db(deployment)
    Products.seed(
        conn, "method_fields", products.params["method_fields"]["default"], "vr_com"
    )
    conn.close()
    app = open_multi_shot(
        **fixed_pixel(products.source("blob_parameters", "default"), "lr")
    )
    assert sidebar(app, "Settings").value == products.hash["method_fields", "default"]


#: The one leaf of the form each blob run that no velocity settings go with moves off the default.
NO_SETTINGS = {
    "fit": "params.blob_parameters.blobs.gauss_fit.size_penalty",
    "lonely": "params.blob_parameters.averages.window",
}


def form_of(params):
    """The single-shot page's widget state for ``blob_parameters`` parameters: what the jump seeds."""
    state = {}
    params_ui.seed_session_state(state, "params.blob_parameters", params)
    return state


@pytest.mark.parametrize("key", sorted(NO_SETTINGS))
def test_a_blob_point_no_settings_go_with_opens_the_single_shot_page_on_its_exact_run(
    deployment, products, monkeypatch, key
):
    """``fit`` has the ellipse fit's own settings, which are in no method_fields parameter set; ``lonely``
    a 2DCA window that nobody ran the velocity fields with. The Fields page shows the blob parameters of
    the settings it is on, which for either are other numbers than the one clicked. So the click goes where
    every other source goes, to the single-shot page, with the run's own parameters in its form and the
    run marked ready to show from the cache."""
    wanted = form_of(products.params["blob_parameters"][key])
    default = form_of(registry.get("blob_parameters").params())
    assert {k for k in wanted if wanted[k] != default[k]} == {NO_SETTINGS[key]}

    click(monkeypatch)
    app = open_multi_shot(
        # Whatever the Fields page was left on does not matter: it is not opened.
        fields__settings=products.hash["method_fields", "short"],
        **fixed_pixel(products.source("blob_parameters", key), "lr"),
    )
    assert titles(app) == ["Single shot"]
    assert "fields.open" not in app.session_state
    assert app.session_state["selection"] == SELECTION
    assert app.session_state["spec.apd"].key == "blob_parameters"
    # The run's own parameters, every leaf of them: the one it moved and the defaults it kept.
    assert {k: app.session_state[k] for k in wanted} == wanted
    assert app.session_state[f"ready.blob_parameters.cmod_{SHOT}_apd_p"] is True


@pytest.mark.parametrize("key", ["default", "short"])
def test_a_blob_point_with_settings_still_opens_the_fields_page(
    deployment, products, monkeypatch, key
):
    """The other branch: when ``method_fields`` settings go with the blob run, nothing changes. The
    single-shot page is not the destination, and the form of ``blob_parameters`` is not seeded.
    """
    click(monkeypatch)
    app = open_multi_shot(**fixed_pixel(products.source("blob_parameters", key), "lr"))
    assert titles(app) == ["Fields"]
    assert sidebar(app, "Settings").value == products.hash["method_fields", key]
    assert "spec.apd" not in app.session_state
    assert not [k for k in app.session_state.filtered_state if k.startswith("params.")]


def test_the_jump_reads_the_ledger_and_writes_nothing(
    deployment, products, monkeypatch
):
    before = ledger_counts(deployment)
    click(monkeypatch)
    for plot, key, name, destination in (
        ("method_fields", "short", "vr_com", "Fields"),
        ("blob_parameters", "short", "lr", "Fields"),
        ("blob_parameters", "fit", "lr", "Single shot"),
        ("blob_parameters", "lonely", "lr", "Single shot"),
    ):
        app = open_multi_shot(**fixed_pixel(products.source(plot, key), name))
        assert titles(app) == [destination], (plot, key)
    assert ledger_counts(deployment) == before


def test_a_point_from_an_unregistered_source_opens_the_single_shot_page(
    deployment, monkeypatch
):
    """The seed's rows, and any plot this deployment no longer has: the jump is the shot alone."""
    click(monkeypatch)
    app = open_multi_shot(ms__scalar="vx_c")
    assert titles(app) == ["Single shot"]
    assert app.session_state["selection"] == SELECTION
    assert "fields.open" not in app.session_state


def test_a_point_from_a_registered_source_opens_the_single_shot_page_on_its_plot(
    deployment, monkeypatch
):
    conn = db.open_db(deployment)
    spec = registry.get("taud_psd")
    params = spec.params(refx=3, refy=2)
    target = registry.Target("cmod", SHOT, "apd", True, "", 0.0, 0.0, "none")
    digest, _ = store.record_params(conn, "taud_psd", params)
    run = store.record_run(
        conn,
        target,
        "taud_psd",
        digest,
        blob_path=None,
        status="ok",
        error=None,
        seconds=None,
        code_version="test",
    )
    store.write_scalars(conn, run["id"], {(3, 2, "taud_psd"): 2e-5})
    conn.close()

    click(monkeypatch)
    app = open_multi_shot(ms__scalar="taud_psd")
    assert titles(app) == ["Single shot"]
    assert app.session_state["selection"] == SELECTION
    # The plot that made the point, with the parameters that made it, ready to show from cache.
    assert app.session_state["spec.apd"].key == "taud_psd"
    assert app.session_state["params.taud_psd.refx"] == 3
    assert app.session_state["params.taud_psd.refy"] == 2
    assert "fields.open" not in app.session_state


def test_a_product_point_falls_back_to_the_single_shot_page_without_a_fields_page(
    deployment, products, monkeypatch
):
    """A deployment that has not registered method_fields has nothing for the Fields page to read."""
    from fusion_ui.core import registry as registry_module

    click(monkeypatch)
    monkeypatch.delitem(registry_module.REGISTRY, "method_fields")
    app = open_multi_shot(
        **fixed_pixel(products.source("method_fields", "short"), "vr_com")
    )
    assert titles(app) == ["Single shot"]
    assert "fields.open" not in app.session_state
