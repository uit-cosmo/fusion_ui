"""The app and its tests import nothing from the ``fusion_scripts`` repository.

That repository's modules are top-level packages, so an absolute import whose
first part is one of them is an import from it. ``imaging_methods.decorrelation``
and ``fusion_ui.config`` are other things than the top-level ``decorrelation``
and ``config``, and are allowed. Imports are read from the syntax tree, so a
string or a comment that names one is not one.
"""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: The top-level packages and modules of fusion_scripts.
FORBIDDEN = frozenset(
    {
        "fusion_scripts",
        "density_scan",
        "decorrelation",
        "twodca_manuscript",
        "plotting_scripts",
        "waveform_analysis",
        "individual_events",
        "figure_scripts",
        "phantom_events",
        "tools",
        "config",
    }
)


def forbidden_imports(source):
    """The imports of ``source`` that reach fusion_scripts: an absolute import
    of one of its packages, or any module named ``fusion_scripts``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [(0, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [(node.level, module)] + [
                (node.level, f"{module}.{alias.name}" if module else alias.name)
                for alias in node.names
            ]
        else:
            continue
        for level, name in names:
            parts = name.split(".")
            if (level == 0 and parts[0] in FORBIDDEN) or "fusion_scripts" in parts:
                found.append(name)
    return found


def test_no_module_of_the_app_or_its_tests_imports_fusion_scripts():
    modules = [
        p for folder in ("fusion_ui", "tests") for p in (REPO / folder).rglob("*.py")
    ]
    names = {p.relative_to(REPO).as_posix() for p in modules}
    assert {"fusion_ui/plots/_pipeline.py", "tests/test_no_fusion_scripts.py"} <= names
    offenders = {
        p.relative_to(REPO).as_posix(): found
        for p in modules
        if (found := forbidden_imports(p.read_text()))
    }
    assert not offenders, offenders


@pytest.mark.parametrize(
    "line",
    [
        "import fusion_scripts",
        "import density_scan.dead_pixels",
        "from decorrelation import pipeline",
        "from decorrelation.pipeline import Averages",
        "import waveform_analysis as wf",
        "import config",
        "from config import DATA_FOLDER",
        "from fusion_ui.core import fusion_scripts, loader",
        "from ..core import fusion_scripts",
    ],
)
def test_the_guard_sees_each_way_of_reaching_fusion_scripts(line):
    assert forbidden_imports(line), line


@pytest.mark.parametrize(
    "line",
    [
        "from imaging_methods import decorrelation",
        "from imaging_methods.decorrelation import Averages, dead_mask",
        "import imaging_methods.decorrelation as pipeline",
        "from fusion_ui import config",
        "from . import config",
        "x = 'import density_scan'  # import decorrelation",
    ],
)
def test_the_guard_lets_the_allowed_ones_through(line):
    assert forbidden_imports(line) == [], line
