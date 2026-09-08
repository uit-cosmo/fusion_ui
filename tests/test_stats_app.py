"""Smoke tests for the statistics page.

Uses a ``statistics_deployment`` fixture holding **an APD file and an ASP file
under the same shot number**. The real tree has no such shot (45 APD shots;
one ASP shot, 1150618021, with no APD), so the fixture is the only place the
cross-diagnostic path gets exercised.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from fusion_ui.core import catalog, db

REPO_ROOT = Path(__file__).resolve().parent.parent
STATISTICS = str(REPO_ROOT / "fusion_ui" / "pages" / "4_statistics.py")
SHOT = 9999


@pytest.fixture
def statistics_deployment(monkeypatch, tmp_path):
    """Point the app at a throwaway tree with one imaging and one probe file
    for the same shot, index it, clear the caches."""
    import numpy as np
    import xarray as xr
    import streamlit as st

    root = tmp_path / "alcator"
    apd_folder = root / "apd"
    asp_folder = root / "asp"
    apd_folder.mkdir(parents=True)
    asp_folder.mkdir(parents=True)

    rng = np.random.default_rng(0)
    n_time, n_y, n_x = 200, 4, 5
    time = np.linspace(1.0, 1.02, n_time)
    apd = xr.Dataset(
        {"frames": (["y", "x", "time"], rng.normal(size=(n_y, n_x, n_time)))},
        coords={
            "R": (["y", "x"], np.tile(np.linspace(88.0, 91.0, n_x), (n_y, 1))),
            "Z": (["y", "x"], np.tile(np.linspace(-4.0, -1.0, n_y), (n_x, 1)).T),
            "time": ("time", time),
        },
        attrs={"shot_number": SHOT},
    )
    apd.to_netcdf(apd_folder / f"apd_{SHOT}.nc")

    data_vars, coords = {}, {}
    for quantity in ("ne", "Vf"):
        for position in (0, 1):
            n = 60 + position * 10  # ragged, on the APD's own window
            data_vars[f"{quantity}_{position}"] = (
                f"time_{quantity}_{position}",
                rng.normal(size=n),
            )
            data_vars[f"rho_{quantity}_{position}"] = (
                f"rho_time_{quantity}_{position}",
                np.linspace(0.1, 0.9, 20),
            )
            coords[f"time_{quantity}_{position}"] = np.linspace(1.0, 1.02, n)
            coords[f"rho_time_{quantity}_{position}"] = np.linspace(1.0, 1.02, 20)
    asp = xr.Dataset(data_vars, coords=coords, attrs={"shot_number": SHOT})
    asp.to_netcdf(asp_folder / f"asp_{SHOT}.nc")

    database = tmp_path / "state" / "shot_explorer.sqlite"
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(root))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(tmp_path / "no_such.json"))
    monkeypatch.setenv("FUSION_UI_DB", str(database))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")

    conn = db.open_db(database)
    catalog.rescan(conn, str(root), "cmod", None)
    conn.close()

    st.cache_data.clear()
    st.cache_resource.clear()
    yield database
    st.cache_data.clear()
    st.cache_resource.clear()


def run(path, **kwargs):
    app = AppTest.from_file(path, default_timeout=120).run()
    assert not app.exception, app.exception
    return app


def widget(app, kind, label):
    matches = [w for w in getattr(app, kind) if w.label == label]
    assert matches, (
        f"no {kind} labelled {label!r}: {[w.label for w in getattr(app, kind)]}"
    )
    return matches[0]


def add_button(app):
    matches = [b for b in app.button if b.label.startswith("Add ")]
    assert matches, f"no Add button: {[b.label for b in app.button]}"
    return matches[0]


def basket_rows(app):
    assert app.dataframe, "expected the basket table to render"
    return app.dataframe[0].value


def test_the_page_renders_with_an_empty_basket(statistics_deployment):
    app = run(STATISTICS)
    assert any("basket is empty" in i.value for i in app.info)
    assert not app.error


def test_adding_two_pixels_draws_a_figure(statistics_deployment):
    app = AppTest.from_file(STATISTICS, default_timeout=120)
    app.session_state["pixels.cmod_9999_apd_r"] = [(0, 0), (1, 1)]
    app.run()
    assert not app.exception, app.exception

    add_button(app).click().run()
    assert not app.exception, app.exception
    assert len(basket_rows(app)) == 2
    # The pixel-map selector plus the statistics figure.
    assert len(app.get("plotly_chart")) >= 2


def test_switching_the_statistic_redraws_without_clearing_the_basket(
    statistics_deployment,
):
    app = AppTest.from_file(STATISTICS, default_timeout=120)
    app.session_state["pixels.cmod_9999_apd_r"] = [(0, 0), (1, 1)]
    app.run()
    assert not app.exception, app.exception
    add_button(app).click().run()
    assert not app.exception, app.exception
    assert len(basket_rows(app)) == 2

    widget(app, "selectbox", "Statistic").set_value("acf").run()
    assert not app.exception, app.exception
    assert len(basket_rows(app)) == 2
    assert len(app.get("plotly_chart")) >= 2


def test_ccf_asks_for_a_reference_and_draws_against_it(statistics_deployment):
    app = AppTest.from_file(STATISTICS, default_timeout=120)
    app.session_state["pixels.cmod_9999_apd_r"] = [(0, 0), (1, 1)]
    app.run()
    assert not app.exception, app.exception
    add_button(app).click().run()
    assert not app.exception, app.exception

    widget(app, "selectbox", "Statistic").set_value("ccf").run()
    assert not app.exception, app.exception
    assert widget(app, "selectbox", "Reference")
    assert len(app.get("plotly_chart")) >= 2


def test_a_mixed_basket_draws_and_reports_the_resampling(statistics_deployment):
    app = AppTest.from_file(STATISTICS, default_timeout=120)
    app.session_state["pixels.cmod_9999_apd_r"] = [(0, 0), (1, 1)]
    app.run()
    assert not app.exception, app.exception
    add_button(app).click().run()
    assert not app.exception, app.exception

    widget(app, "selectbox", "Statistic").set_value("pdf").run()
    assert not app.exception, app.exception

    radios = [w for w in app.radio if w.label == "Diagnostic"]
    assert radios, "expected the diagnostic picker"
    radios[0].set_value("asp").run()
    assert not app.exception, app.exception

    picks = [w for w in app.multiselect if w.label == "Channels"]
    assert picks, "expected the probe channel picker"
    picks[0].set_value(["ne_0", "ne_1"]).run()
    assert not app.exception, app.exception
    add_button(app).click().run()
    assert not app.exception, app.exception

    # Two pixels plus two probe channels on one PDF axis. The picker is on a
    # probe diagnostic, so there is no pixel-map chart -- just the figure.
    assert len(basket_rows(app)) == 4
    assert len(app.get("plotly_chart")) == 1
    assert any("4 traces" in c.value for c in app.caption)

    # CCF puts the ragged probe traces on the reference's base and says so.
    widget(app, "selectbox", "Statistic").set_value("ccf").run()
    assert not app.exception, app.exception
    assert any("resampled" in c.value for c in app.caption)

    conn = db.connect(statistics_deployment)
    (runs,) = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
    conn.close()
    assert runs == 0, "a live statistics view must not write ledger rows"
