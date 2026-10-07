"""A cached batch-only result on the single-shot page, computed under parameters that are not the defaults.

The caption under it says how to compute it again: ``fusion-ui precompute … --params-json params.json``.
The page cannot write that file for the person, so it shows what goes in it, as it does for a result that is
missing (``show_batch_missing``). Nothing is computed here: a lookup, and a command to copy.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from streamlit.testing.v1 import AppTest

import fusion_ui.plots  # noqa: F401 - registers the specs the page offers
from fusion_ui.core import catalog, db, precompute, registry, store

REPO_ROOT = Path(__file__).resolve().parent.parent
SINGLE_SHOT = str(REPO_ROOT / "fusion_ui" / "pages" / "2_single_shot.py")

CALLS = []


@dataclasses.dataclass
class BankParams:
    window: int = 60
    threshold: float = 2.5


def compute(ds, params):
    CALLS.append(params)
    return xr.Dataset({"y": ("t", np.arange(4.0))})


def render(result, params, target):
    return None


@pytest.fixture
def bank():
    """A batch-only spec, registered for one test."""
    CALLS.clear()
    spec = registry.register(
        registry.PlotSpec(
            key="toy_batch_bank",
            label="Toy batch bank",
            diagnostics=("apd",),
            params=BankParams,
            render=render,
            compute=compute,
            batch_only=True,
        )
    )
    yield spec
    registry.REGISTRY.pop("toy_batch_bank", None)
    CALLS.clear()


@pytest.fixture
def database(monkeypatch, tmp_path, apd_dataset_path):
    """One indexed tiny APD file, shot 1234."""
    import streamlit as st

    database = tmp_path / "state" / "shot_explorer.sqlite"
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(apd_dataset_path.parent.parent))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(tmp_path / "no_such_discharges.json"))
    monkeypatch.setenv("FUSION_UI_DB", str(database))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")
    conn = db.open_db(database)
    catalog.rescan(conn, str(apd_dataset_path.parent.parent), "cmod", None)
    conn.close()
    st.cache_data.clear()
    st.cache_resource.clear()
    yield database
    st.cache_data.clear()
    st.cache_resource.clear()


def fill(database, spec, **changes):
    """What ``fusion-ui precompute toy_batch_bank --shot 1234`` leaves behind."""
    conn = db.open_db(database)
    target = registry.Target("cmod", 1234, "apd", False, "unused", 1.0, 1.02, "none")
    store.result(conn, spec, target, BankParams(**changes), ds=None, batch=True)
    conn.close()
    CALLS.clear()


def page(bank, **state):
    app = AppTest.from_file(SINGLE_SHOT, default_timeout=60)
    app.session_state["spec.apd"] = bank
    for name, value in state.items():
        app.session_state[f"params.toy_batch_bank.{name}"] = value
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    return app


def captions(app):
    return [c.value for c in app.caption]


def saved_parameters(app):
    """The JSON blocks the page shows: what to save as the command's ``--params-json`` file."""
    return [c.value for c in app.code if c.language == "json"]


def test_a_cached_result_under_other_parameters_shows_what_to_save(database, bank):
    fill(database, bank, window=30)
    app = page(bank, window=30)

    assert CALLS == []  # a lookup: nothing was computed
    command = "fusion-ui precompute toy_batch_bank --shot 1234 --params-json params.json --force"
    # The command is in the caption, and what it needs the file to hold follows it, as it does for a
    # result that is missing: one sentence, not a command that names a file nobody can see.
    assert any(
        c.startswith("Computed in batch only. To compute it again, run ")
        and command in c
        and not c.endswith(".")
        for c in captions(app)
    ), captions(app)
    assert "with these parameters saved as `params.json`:" in captions(app)
    (shown,) = saved_parameters(app)
    assert '"window": 30' in shown and '"plot": "toy_batch_bank"' in shown
    # Under the provenance line, in the column it is drawn in: the sentence with the command, what the file
    # holds, the block itself -- in that order.
    left = [(c.type, c.value) for c in app.columns[0].children.values()]
    assert [kind for kind, _ in left[1:]] == ["caption", "caption", "code"]
    assert left[2][1] == "with these parameters saved as `params.json`:"
    assert left[3][1] == shown


def test_what_the_page_shows_is_what_the_command_reads(database, bank, tmp_path):
    """Saved as ``params.json``, the block is the parameter set the cached result was computed under:
    ``--params-json`` accepts it and gives these parameters, so the command computes the same result.
    """
    fill(database, bank, window=30, threshold=3.0)
    app = page(bank, window=30, threshold=3.0)
    (shown,) = saved_parameters(app)

    path = tmp_path / precompute.PARAMS_FILE
    path.write_text(shown)
    assert precompute.params_from_file(path, [bank]) == {
        "toy_batch_bank": BankParams(window=30, threshold=3.0)
    }


def test_it_is_the_block_a_missing_result_shows_for_the_same_parameters(database, bank):
    """One helper writes both, so a result that is cached and one that is missing say the same thing."""
    missing = page(bank, window=30)
    (before,) = saved_parameters(missing)
    assert "with these parameters saved as `params.json`:" in captions(missing)

    fill(database, bank, window=30)
    cached = page(bank, window=30)
    assert saved_parameters(cached) == [before]


def test_the_defaults_need_no_file_and_show_none(database, bank):
    fill(database, bank)
    app = page(bank)
    assert saved_parameters(app) == []
    assert not [c for c in captions(app) if "saved as" in c]
    # The sentence ends where it always did, with a full stop.
    assert any(
        c == "Computed in batch only. To compute it again, run "
        "`fusion-ui precompute toy_batch_bank --shot 1234 --force`."
        for c in captions(app)
    ), captions(app)
