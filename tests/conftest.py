"""Fixtures: a throwaway data tree and discharge descriptor.

Everything here is empty files and a handful of JSON entries. No test touches
the real data tree — a single APD file is ~425 MB, and the collection is tens
of terabytes.
"""

import json

import pytest

from fusion_ui.core import db

# Mirrors the real descriptor's shapes: a fully curated shot, one where f_GW has
# to be derived from I_p and n̄_e, one with nothing but a shot number, and an
# exact duplicate entry (the real file has nine of them).
DISCHARGES = [
    {
        "shot_number": 1160616027,
        "plasma_current": 0.55,
        "line_averaged_density": 1.42,
        "greenwald_fraction": 0.72,
        "t_start": 1.3,
        "t_end": 1.6,
        "mlp_mode": "scan",
        "comment": "IWL",
    },
    {
        "shot_number": 1110201007,
        "plasma_current": 0.93,
        "line_averaged_density": 1.10,
        "greenwald_fraction": float("nan"),
        "t_start": 1.1,
        "t_end": 1.4,
        "mlp_mode": "none",
        "comment": "EDA-H",
    },
    {
        "shot_number": 1110201007,  # exact duplicate
        "plasma_current": 0.93,
        "line_averaged_density": 1.10,
        "greenwald_fraction": float("nan"),
        "t_start": 1.1,
        "t_end": 1.4,
        "mlp_mode": "none",
        "comment": "EDA-H",
    },
    {
        "shot_number": 1090813019,
        "plasma_current": 0.0,
        "line_averaged_density": 0.0,
        "greenwald_fraction": float("nan"),
        "t_start": float("nan"),
        "t_end": float("nan"),
        "mlp_mode": "",
        "comment": "",
    },
]

# 1150618021 is deliberately absent from DISCHARGES: files on disk, nobody has
# curated it yet. It must still show up in the browser, flagged.
FILES = {
    "apd": [
        "apd_1160616027.nc",
        "apd_1160616027_preprocessed.nc",
        "apd_1110201007.nc",
        "apd_1150618021.nc",
        "notes.txt",  # skipped
        "apd_broken.nc",  # skipped
    ],
    "asp": ["asp_1150618021.nc"],
    "fsp": [],
    # "phantom" folder is missing entirely.
}


ENV_VARS = (
    "FUSION_DISCHARGE_DB",
    "FUSION_DATA_FOLDER",
    "FUSION_UI_DB",
    "FUSION_UI_CACHE",
    "FUSION_MACHINE",
)


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    """Keep the developer's own ``.env`` out of the suite.

    ``fusion_ui.config`` loads the repository's ``.env`` into ``os.environ`` at
    import time, which would otherwise point these tests at the real data tree.
    """
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def data_folder(tmp_path):
    root = tmp_path / "alcator"
    for subfolder, names in FILES.items():
        folder = root / subfolder
        folder.mkdir(parents=True)
        for index, name in enumerate(names):
            (folder / name).write_bytes(b"x" * (index + 1))
    return root


@pytest.fixture
def discharge_db(tmp_path):
    path = tmp_path / "plasma_discharges.json"
    path.write_text(json.dumps(DISCHARGES))
    return path


@pytest.fixture
def conn(tmp_path):
    connection = db.open_db(tmp_path / "state" / "shot_explorer.sqlite")
    yield connection
    connection.close()
