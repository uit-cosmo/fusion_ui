"""fusion_scripts' ``config`` under the bare name, whatever ``sys.path`` says.

``streamlit run fusion_ui/app.py`` puts the package directory first on ``sys.path``, where
``fusion_ui/config.py`` answers to the bare name ``config`` that fusion_scripts reads its own settings
by. Deployed, every page that imports the specs failed with ``module 'config' has no attribute
'W7X_PRESENTATION_DIR'``.

Each test runs in a fresh interpreter. In this one ``fusion_ui.plots`` is long imported, and with it
fusion_scripts' ``config``, which is what hid the failure from the suite. A child runs from the
repository root, so it imports this checkout, with every ``FUSION_*`` variable pointed at a tmp tree:
no ``.env`` can choose one.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fusion_ui.core import catalog, db

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "fusion_ui"
APP = PACKAGE / "app.py"
PAGES = (
    "pages/2_single_shot.py",
    "pages/3_multi_shot.py",
    "pages/5_fields.py",
    "pages/6_documentation.py",
)

#: The last thing every child does: report, as one line of JSON, what the bare name ``config`` is, where
#: fusion_scripts' own is (beside the ``decorrelation`` the specs imported), and what fusion_ui reads.
REPORT = """
import json, sys
import decorrelation, fusion_ui
from fusion_ui import config as ui_config
bare = sys.modules.get("config")
print(json.dumps({
    "bare": getattr(bare, "__file__", None),
    "fusion_scripts": decorrelation.__file__,
    "w7x": hasattr(bare, "W7X_PRESENTATION_DIR"),
    "fusion_ui": fusion_ui.__file__,
    "paths": [ui_config.DISCHARGE_DB_PATH, ui_config.DATA_FOLDER,
              ui_config.UI_DB_PATH, ui_config.CACHE_DIR],
    **extra,
}))
"""


def environment(tmp_path):
    """The child's environment, with every ``FUSION_*`` variable set to the tmp tree."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("FUSION_")}
    env.update(
        FUSION_DISCHARGE_DB=str(tmp_path / "plasma_discharges.json"),
        FUSION_DATA_FOLDER=str(tmp_path / "alcator"),
        FUSION_UI_DB=str(tmp_path / "state" / "shot_explorer.sqlite"),
        FUSION_UI_CACHE=str(tmp_path / "cache"),
        FUSION_MACHINE="cmod",
    )
    return env


def child(code, env):
    """Run ``code`` in a fresh interpreter from the repository root; its report."""
    done = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def assert_the_bare_name_is_fusion_scripts(seen, env):
    bare = Path(seen["bare"]).resolve()
    assert bare.parent == Path(seen["fusion_scripts"]).resolve().parent.parent
    assert PACKAGE.resolve() not in bare.parents
    assert seen["w7x"]  # the setting whose absence was the failure
    # This checkout, reading the paths the environment gives it, not either .env's.
    assert Path(seen["fusion_ui"]).resolve().parent == PACKAGE.resolve()
    assert seen["paths"] == [
        env[k]
        for k in (
            "FUSION_DISCHARGE_DB",
            "FUSION_DATA_FOLDER",
            "FUSION_UI_DB",
            "FUSION_UI_CACHE",
        )
    ]


def test_the_specs_import_with_the_package_directory_first_on_sys_path(tmp_path):
    """As ``_fix_sys_path`` leaves it, which the service never undoes. The path is put back as it was."""
    env = environment(tmp_path)
    seen = child(
        "import sys\n"
        f"sys.path.insert(0, {str(PACKAGE)!r})\n"
        "before = list(sys.path)\n"
        "import fusion_ui.plots\n"
        "extra = {'kept': sys.path == before}\n" + REPORT,
        env,
    )
    assert_the_bare_name_is_fusion_scripts(seen, env)
    assert seen["kept"]


def test_fusion_ui_config_already_imported_as_the_bare_config_is_replaced(tmp_path):
    """What a page gets when fusion_scripts code it reaches first imports ``config`` with the package
    directory on the path, before anything has made the name fusion_scripts': fusion_ui's own
    ``config.py``, under the bare name."""
    env = environment(tmp_path)
    seen = child(
        "import sys\n"
        f"sys.path.insert(0, {str(PACKAGE)!r})\n"
        "import config\n"
        "extra = {'first': config.__file__}\n"
        "import fusion_ui.plots\n" + REPORT,
        env,
    )
    assert Path(seen["first"]).resolve() == (PACKAGE / "config.py").resolve()
    assert_the_bare_name_is_fusion_scripts(seen, env)


@pytest.fixture
def deployment(tmp_path, apd_dataset_path, discharge_db):
    """The tiny APD file, raw and preprocessed, indexed into a fresh ledger: what the pages open."""
    shutil.copy(
        apd_dataset_path, apd_dataset_path.with_name("apd_1234_preprocessed.nc")
    )
    env = environment(tmp_path)
    assert env["FUSION_DATA_FOLDER"] == str(apd_dataset_path.parent.parent)
    assert env["FUSION_DISCHARGE_DB"] == str(discharge_db)
    conn = db.open_db(Path(env["FUSION_UI_DB"]))
    catalog.rescan(
        conn, env["FUSION_DATA_FOLDER"], "cmod", env["FUSION_DISCHARGE_DB"]
    )
    conn.close()
    return env


def test_the_app_opens_every_page_that_imports_the_specs(deployment):
    """The real multipage app, as the service runs it: Streamlit puts the main script's directory, the
    package's, on ``sys.path`` around every run. Nothing the pages need is imported beforehand."""
    seen = child(
        "import sys\n"
        "from streamlit.testing.v1 import AppTest\n"
        "early = [m for m in ('fusion_ui.plots', 'decorrelation', 'config')"
        " if m in sys.modules]\n"
        f"app = AppTest.from_file({str(APP)!r}, default_timeout=120)\n"
        "runs = []\n"
        f"for page in (None, *{PAGES!r}):\n"
        "    if page:\n"
        "        app.switch_page(page)\n"
        "    app.run()\n"
        "    runs.append({'titles': [t.value for t in app.title],\n"
        "                 'exceptions': [e.value for e in app.exception]})\n"
        "extra = {'early': early, 'runs': runs}\n" + REPORT,
        deployment,
    )
    assert seen["early"] == []
    assert [run["exceptions"] for run in seen["runs"]] == [[]] * (1 + len(PAGES))
    titles = [run["titles"] for run in seen["runs"]]
    assert "Shot Explorer" in titles[0][0]
    assert titles[1:] == [
        ["Single shot"],
        ["Multi shot"],
        ["Fields"],
        ["Documentation"],
    ]
    assert_the_bare_name_is_fusion_scripts(seen, deployment)
