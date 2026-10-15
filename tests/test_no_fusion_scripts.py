"""The app imports nothing from fusion_scripts (L14, step 3).

Every method the app runs lives in ``imaging_methods`` (and the hand-made 1160616 mask in
``experimental_database``). This walks every module of ``fusion_ui/`` and of ``tests/`` and fails on an
import of fusion_scripts' own top-level packages, or of a top-level ``config``. fusion_scripts stays in
``pyproject.toml`` until the user takes it out (step 6), so this, and not the dependency list, is what
keeps it from coming back.
"""

from pathlib import Path

import pytest

from tests.import_guard import FUSION_SCRIPTS, reaches_fusion_scripts

REPO = Path(__file__).resolve().parents[1]
MODULES = sorted(
    p
    for folder in ("fusion_ui", "tests")
    for p in (REPO / folder).rglob("*.py")
    if ".venv" not in p.parts
)


def test_the_walk_covers_the_app_and_its_tests():
    names = {p.relative_to(REPO).as_posix() for p in MODULES}
    assert "fusion_ui/plots/_pipeline.py" in names
    assert "tests/test_no_fusion_scripts.py" in names
    assert len(names) > 100


def test_no_module_of_the_app_or_its_tests_imports_fusion_scripts():
    offenders = {
        p.relative_to(REPO).as_posix(): found
        for p in MODULES
        if (found := reaches_fusion_scripts(p.read_text()))
    }
    assert not offenders, offenders


@pytest.mark.parametrize(
    "line",
    [
        "import fusion_scripts",
        "import density_scan.dead_pixels",
        "from decorrelation import pipeline",
        "from decorrelation.pipeline import Averages",
        "from twodca_manuscript.tracking import get_positions_and_mask",
        "from plotting_scripts.figure_plots import calculate_splinted_LCFS",
        "import waveform_analysis as wf",
        "from individual_events.parameters import SEParameters",
        "import figure_scripts",
        "import phantom_events",
        "import config",
        "from config import DATA_FOLDER",
        "from fusion_ui.core import fusion_scripts, loader",  # the retired door
        "from fusion_ui.core.fusion_scripts import import_config",
        "from ..core import fusion_scripts",
    ],
)
def test_the_guard_sees_each_way_of_reaching_fusion_scripts(line):
    assert reaches_fusion_scripts(line), line


@pytest.mark.parametrize(
    "line",
    [
        "from imaging_methods import decorrelation",
        "from imaging_methods.decorrelation import Averages, dead_mask",
        "import imaging_methods.decorrelation as pipeline",
        "from imaging_methods import dead_pixels, puff",
        "from fusion_ui import config",
        "from . import config",
        "from fusion_ui.core import loader, registry\nimport numpy as np",
        "x = 'import density_scan'  # import decorrelation",
    ],
)
def test_the_guard_lets_the_allowed_ones_through(line):
    assert reaches_fusion_scripts(line) == [], line


def test_the_guard_names_the_packages_the_note_lists():
    assert {
        "density_scan",
        "decorrelation",
        "twodca_manuscript",
        "plotting_scripts",
        "waveform_analysis",
        "individual_events",
        "figure_scripts",
        "phantom_events",
        "config",
    } <= FUSION_SCRIPTS
