"""The Fields page and its builders against the real specs, on the real blobs.

``tests/test_fields_page.py`` runs the page on stand-in specs and synthetic blobs. These tests run it on
what the real specs make: a synthetic record (3 x 3 pixels, with a dead pixel, a live pixel that never
sees an event, and one interior pixel, (1, 1)) goes through ``pixel_averages``, ``method_fields`` and
``blob_parameters`` by way of the store, once for the session (``tests/real_fixtures``), and the page,
the builders and ``views.products`` read the blobs back off disk.

What a stand-in cannot tell is what is checked: that the settings a batch job used are offered and
found, that ``related_params`` names the blob run of the velocity fields' settings, that every section of
the pixel level draws, and that the blobs have
what the builders read off them.
"""

import base64
import json
import shlex
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

import fusion_ui.plots  # noqa: F401 - registers the real specs
from fusion_ui import cli
from fusion_ui.core import params_ui, precompute, registry, store
from fusion_ui.views import numbers
from fusion_ui.views import products as prod
from fusion_ui.views.methods import METHODS, TRACKS
from tests import product_fixtures as fx
from tests import real_fixtures as real

REPO_ROOT = Path(__file__).resolve().parent.parent
FIELDS = str(REPO_ROOT / "fusion_ui" / "pages" / "5_fields.py")
SHOT = real.SHOT
KEYS = real.KEYS


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    return real.deployment(tmp_path_factory)


@pytest.fixture
def served(deployment, monkeypatch):
    """The deployment as the page sees it, for one test, with Streamlit's caches empty."""
    import streamlit as st

    deployment.use(monkeypatch)
    st.cache_data.clear()
    st.cache_resource.clear()
    yield deployment
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture(scope="module")
def products(deployment):
    return deployment.products()


def run(**state):
    """The page, with session state set before it runs (``fields__open`` is ``fields.open``)."""
    app = AppTest.from_file(FIELDS, default_timeout=120)
    for key, value in state.items():
        app.session_state[key.replace("__", ".")] = value
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def rerun(app):
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def settings_hash(deployment):
    """The ``method_fields`` parameter set the batch job used: what the picker offers beside the default."""
    return deployment.runs["method_fields"]["params_hash"]


def open_on(deployment, pixel=None):
    """The page opened the way the multi-shot page opens it: on this shot, these settings, this pixel."""
    request = {"shot": SHOT, "settings": settings_hash(deployment)}
    if pixel is not None:
        request["pixel"] = pixel
    return run(fields__open=request)


def figures(app):
    return [go.Figure(json.loads(c.proto.spec)) for c in app.get("plotly_chart")]


def decode(array):
    """A trace's numbers, whether Plotly sent a list or a typed array (``{"dtype": ..., "bdata": ...}``)."""
    if isinstance(array, dict) and "bdata" in array:
        raw = base64.b64decode(array["bdata"])
        return np.frombuffer(raw, dtype=np.dtype(array["dtype"]))
    return np.asarray(array)


def kinds(figure, kind, panel=None):
    return [
        t
        for t in figure.data
        if t.meta
        and t.meta.get("kind") == kind
        and panel in (None, t.meta.get("panel"))
    ]


def sidebar(app, label):
    matches = [w for w in app.sidebar.selectbox if w.label == label]
    assert matches, [w.label for w in app.sidebar.selectbox]
    return matches[0]


def commands(app):
    return [c.value for c in app.code if c.language == "bash"]


def counts(deployment):
    conn = deployment.conn()
    try:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("runs", "param_sets", "scalars", "presets")
        }
    finally:
        conn.close()


def target():
    return registry.Target(
        "cmod", SHOT, "apd", True, "unused", float("nan"), float("nan"), "none"
    )


# -- the shot opens, and the products are found ------------------------------------------------------


def test_the_shot_opens_at_the_default_settings_with_the_commands_the_real_specs_give(
    served,
):
    before = counts(served)
    app = run()
    assert sidebar(app, "Run day").options == ["1990101 · 0/1 computed"]
    assert sidebar(app, "Shot").options == [f"{SHOT} · not computed"]
    default, other = sidebar(app, "Settings").options
    assert default == "default · 0 shots"
    # Only the parameter set the batch job used is beside it, shown as its difference from the default.
    defaults = params_ui.hash_params(
        "method_fields", registry.get("method_fields").params()
    )
    used = params_ui.hash_params("method_fields", real.settings()["method_fields"])
    diff = prod.params_diff(defaults[1], used[1])
    assert ("averages.window", 60, 10) in diff and len(diff) > 3
    assert other == f"{prod.describe(diff)} · 1 shot"
    assert commands(app) == [
        f"fusion-ui precompute pixel_averages --shot {SHOT}",
        f"fusion-ui precompute method_fields --shot {SHOT}",
        f"fusion-ui precompute blob_parameters --shot {SHOT}",
    ]
    assert len(app.warning) == 3 and "35–70 min" in app.warning[0].value
    assert not app.get("plotly_chart")
    assert counts(served) == before  # looking records nothing


def test_the_settings_the_batch_job_used_find_all_three_products_and_draw_the_shot_level(
    served,
):
    before = counts(served)
    app = run()
    sidebar(app, "Settings").set_value(settings_hash(served))
    rerun(app)
    assert sidebar(app, "Run day").options == ["1990101 · 1/1 computed"]
    assert sidebar(app, "Shot").options == [str(SHOT)]
    assert not app.warning and not app.error and not commands(app)
    lines = [
        m.value
        for m in app.markdown
        if m.value.startswith("**") and "computed" in m.value
    ]
    assert [line.split("**")[1] for line in lines] == list(KEYS)
    for line, key in zip(lines, KEYS):
        # The run the page found is the run the batch job left, under the hash it was stored with.
        assert f"params `{served.runs[key]['params_hash'][:12]}`" in line, key
    assert [b.proto.label for b in app.get("download_button")] == [
        f"Download {key} (.nc)" for key in KEYS
    ]
    assert any(
        "Dead-pixel mask: **synthetic, the full mask** · 1 of 9 pixels dead" in c.value
        for c in app.caption
    )
    (shot_level,) = figures(app)
    assert shot_level.layout.meta["mode"] == "arrows"
    assert {t.meta["panel"] for t in kinds(shot_level, "pixels")} <= {
        m.key for m in METHODS
    }
    assert any("Click a pixel" in i.value for i in app.info)
    assert counts(served) == before


def test_related_params_name_the_stored_run_of_every_product_of_the_velocity_fields_settings(
    served,
):
    found, missing = prod.specs(registry)
    assert missing == []
    method_params = real.settings()["method_fields"]
    related = prod.related_params(found, method_params)
    conn = served.conn()
    try:
        for key in KEYS:
            digest = params_ui.hash_params(key, related[key])[0]
            assert digest == served.runs[key]["params_hash"], key
            found_run = store.find_run(conn, target(), key, digest)
            assert found_run["id"] == served.runs[key]["id"], key
        # ``collect`` reads all three back as the page does, loaded and current.
        collected = prod.collect(
            conn,
            target(),
            found,
            method_params,
            lambda run: store.load_result(conn, run),
            stale=prod.stale_reasons(conn),
        )
    finally:
        conn.close()
    assert [p.state for p in collected.values()] == [prod.OK] * 3
    assert all(p.ok and p.stale is None for p in collected.values())
    assert (
        collected["blob_parameters"].run["upstream_run_id"]
        == served.runs["pixel_averages"]["id"]
    )
    assert (
        collected["method_fields"].run["upstream_run_id"]
        == served.runs["pixel_averages"]["id"]
    )


# -- the pixel level --------------------------------------------------------------------------------


def number_tables(app):
    method, blob = (frame.value for frame in app.dataframe)
    return method, blob


def test_a_pixel_opens_every_section_of_the_pixel_level_from_the_real_blobs(
    served, products
):
    bank, fields, blobs = (products[key] for key in KEYS)
    before = counts(served)
    app = open_on(served, pixel=(1, 1))
    assert [s.value for s in app.subheader] == [
        f"Shot {SHOT}",
        "Velocity fields",
        "Pixel (x=1, y=1)",
        "The average at a few lags",
        "One frame",
        "Tracks",
        "Numbers",
    ]
    shot_level, strip, frame, trace, tracks = figures(app)
    # The strip draws the pixel's own lags: the fit's, widened, off the real ``fit_com``.
    assert strip.layout.meta["source"] == "fit" and len(strip.layout.meta["lags"]) >= 2
    assert set(np.round(strip.layout.meta["lags"], 12)) <= set(
        np.round(bank["time"].values, 12)
    )
    assert frame.layout.meta["lag"] == pytest.approx(0.0)
    assert "tracks at reference pixel (x=1, y=1)" in tracks.layout.title.text
    # What it shows of each track is the stored positions, as millimetres from the reference pixel.
    x, y = 1, 1
    for track in TRACKS:
        stored = np.asarray(fields[f"pos_r_{track.key}"].values)[y, x]
        drawn_r = [
            t for t in tracks.data if t.name == track.label and t.mode == "markers"
        ][0]
        assert len(decode(drawn_r.x)) == int(np.isfinite(stored).sum()), track.key

    # The numbers are the blobs' own, to the digit.
    method, blob = number_tables(app)
    for row, m in zip(method.to_dict("records"), METHODS):
        assert row["method"] == m.label
        assert row["v_R [m/s]"] == pytest.approx(
            float(fields[m.vr].values[y, x]), nan_ok=True
        )
        assert row["v_Z [m/s]"] == pytest.approx(
            float(fields[m.vz].values[y, x]), nan_ok=True
        )
        if m.nlags:
            assert row["lags"] == float(fields[m.nlags].values[y, x])
        assert np.isnan(row["events"]) == (not m.uses_events)
    assert list(blob["parameter"]) == [n for n, _, _ in numbers.BLOB_PARAMETERS]
    for name, value in zip(blob["parameter"], blob["value"]):
        assert value == pytest.approx(
            float(blobs[name].values[y, x]), nan_ok=True
        ), name
    assert counts(served) == before


@pytest.mark.parametrize(
    "pixel, said",
    [
        ((0, 1), "dead in this shot's mask"),  # the mask's one dead pixel
        ((2, 0), "No events at pixel"),  # live, and its average is empty
    ],
)
def test_a_pixel_with_nothing_to_show_says_why_in_every_section(served, pixel, said):
    app = open_on(served, pixel=pixel)
    assert [i.value for i in app.info if said in i.value]  # the numbers' sentence
    for figure in figures(app)[1:]:
        assert said in " ".join(a.text for a in figure.layout.annotations)
    assert not app.exception


# -- what the builders assume of the blobs ----------------------------------------------------------------


def test_the_real_blobs_are_what_the_builders_read_off_them(products):
    bank, fields, blobs = (products[key] for key in KEYS)
    ny, nx = fields["dead"].shape
    # One lag axis for the bank and the fields: the strip indexes the fields' lags by the bank's.
    assert "time" in fields.coords and fields.sizes["time"] == bank.sizes["time"]
    np.testing.assert_array_equal(fields["time"].values, bank["time"].values)
    # The array, and where it is, in metres, the same in all three.
    for blob in (bank, fields, blobs):
        assert blob["R"].dims == blob["Z"].dims == ("y", "x")
        assert blob["dead"].dtype == bool and blob["dead"].shape == (ny, nx)
        assert 0.8 < float(blob["R"].max()) < 1.0
    np.testing.assert_array_equal(bank["dead"].values, fields["dead"].values)
    # A fit is a bool mask over the lags; a position is a number or NaN, R and Z together.
    for track in TRACKS:
        fit = fields[f"fit_{track.key}"]
        assert fit.dtype == bool and fit.dims == ("y", "x", "time")
        r, z = (np.asarray(fields[f"pos_{c}_{track.key}"].values) for c in "rz")
        np.testing.assert_array_equal(np.isfinite(r), np.isfinite(z))
        # ``nlags`` counts the lags of the fit, and a pixel never computed has none.
        nlags = np.asarray(fields[f"nlags_{track.key}"].values)
        live = ~fields["dead"].values
        np.testing.assert_array_equal(nlags[live], fit.values.sum(-1)[live])
        assert np.isnan(nlags[~live]).all()
    # The contour's level is the centroid's alone: a maximum track has none.
    assert np.isnan(fields["level_max"].values).all()
    assert np.isnan(fields["level_2dcc"].values).all()
    assert np.isfinite(fields["level_com"].values[~fields["dead"].values]).any()
    # Nothing was tracked where no live average was: no positions, and no events at the reference.
    empty = np.asarray(bank["nevents"].values) == 0
    for track in TRACKS:
        assert not np.isfinite(
            np.asarray(fields[f"pos_r_{track.key}"].values)[empty]
        ).any()
    # The count of events a pixel carries is one number in the bank, the fields and the blobs.
    live = ~fields["dead"].values
    for other in (fields["nevents"].values, blobs["nevents"].values):
        np.testing.assert_array_equal(
            np.asarray(other)[live], np.asarray(bank["nevents"].values)[live]
        )
    # A mask source and the TDE's minimum correlation are stamped as the page reads them.
    assert str(fields.attrs["dead_mask_source"]) == "synthetic, the full mask"
    assert float(fields.attrs["min_cc"]) == fx.TDE.min_cc


# -- the command for what is missing, with the real specs ------------------------------------------------


def test_the_command_for_a_missing_run_saves_the_parameters_the_page_looks_for(
    served, tmp_path, monkeypatch
):
    """The page shows the command and the parameter set to save as ``params.json``. Run as written, through
    the real parameter classes and the CLI's own loader, it fills the run the page looks up: the same
    parameter set, so the same key."""
    import streamlit as st

    copy = served.copy(tmp_path / "copy").use(monkeypatch)
    conn = copy.conn()
    try:
        # The ledger's row only: the blob stays where the other tests read it.
        store.delete_run(conn, {**copy.runs["blob_parameters"], "blob_path": None})
    finally:
        conn.close()
    st.cache_data.clear()
    app = run(fields__open={"shot": SHOT, "settings": settings_hash(copy)})
    (command,) = commands(app)
    assert (
        command
        == f"fusion-ui precompute blob_parameters --shot {SHOT} --params-json params.json"
    )
    (shown,) = [c.value for c in app.code if c.language == "json"]
    wanted = params_ui.hash_params("blob_parameters", real.blob_params())
    assert shown == wanted[1]

    file = tmp_path / "params.json"
    file.write_text(shown)
    loaded = precompute.params_from_file(file, [registry.get("blob_parameters")])
    assert (
        params_ui.hash_params("blob_parameters", loaded["blob_parameters"])[0]
        == wanted[0]
        == served.runs["blob_parameters"]["params_hash"]
    )
    args = cli.build_parser().parse_args(shlex.split(command)[1:])
    assert (args.plots, args.shot, args.params_json) == (
        ["blob_parameters"],
        [SHOT],
        "params.json",
    )
