"""The .env conventions: a dotenv file fills the environment, the shell wins."""

import os
from pathlib import Path

import pytest

from fusion_ui import config


def dotenv(tmp_path, text):
    """Write a .env and load it the way importing fusion_ui.config would."""
    path = tmp_path / ".env"
    path.write_text(text)
    config._load_dotenv(path)


def test_reads_dotenv(tmp_path):
    dotenv(
        tmp_path,
        "# comment\n\nnot a pair\nFUSION_DATA_FOLDER=/data/alcator\n"
        'FUSION_UI_DB="/state/app.sqlite"\n',
    )
    assert config.DATA_FOLDER == "/data/alcator"
    assert config.UI_DB_PATH == "/state/app.sqlite"


def test_shell_environment_wins_over_dotenv(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSION_DATA_FOLDER", "/from/shell")
    dotenv(tmp_path, "FUSION_DATA_FOLDER=/from/dotenv\n")
    assert config.DATA_FOLDER == "/from/shell"


def test_missing_dotenv_is_not_an_error(tmp_path):
    config._load_dotenv(tmp_path / "nothing-here")


def test_dotenv_is_looked_for_where_asked_then_upwards_then_beside_the_checkout(
    monkeypatch, tmp_path
):
    work = tmp_path / "a" / "b"
    work.mkdir(parents=True)
    monkeypatch.chdir(work)
    monkeypatch.setenv("FUSION_UI_DOTENV", str(tmp_path / "explicit.env"))
    paths = config._dotenv_paths()
    assert paths[:3] == [tmp_path / "explicit.env", work / ".env", work.parent / ".env"]
    assert tmp_path / ".env" in paths
    assert paths[-1] == Path(config.__file__).resolve().parent.parent / ".env"


def test_expands_user(monkeypatch):
    monkeypatch.setenv("FUSION_DATA_FOLDER", "~/Data/alcator")
    assert config.DATA_FOLDER == os.path.expanduser("~/Data/alcator")


def test_missing_variable_names_the_fix():
    with pytest.raises(RuntimeError) as error:
        _ = config.DISCHARGE_DB_PATH
    assert "FUSION_DISCHARGE_DB" in str(error.value)
    assert ".env" in str(error.value)


def test_machine_defaults_to_cmod(monkeypatch):
    assert config.MACHINE == "cmod"
    monkeypatch.setenv("FUSION_MACHINE", "w7x")
    assert config.MACHINE == "w7x"


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = config.NOT_A_SETTING
