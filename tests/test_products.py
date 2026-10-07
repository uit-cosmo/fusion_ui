"""The three decorrelation products: ``pixel_averages``, ``method_fields`` and ``blob_parameters``.

They are thin adapters over ``decorrelation.pipeline``, so what is tested here is the adapting: that each
spec hands the API what it takes (the record in metres, the bank's own mask, the settings as they are),
returns its Dataset unchanged, stores it and gets it back bit for bit, writes the scalars the plan names, and
hashes the way the plan says. The physics itself is the API's, tested in fusion_scripts against the paper's.

The record is ``tests/product_fixtures``: 3 x 3 pixels in centimetres, like the real files. A product
computed through the store is compared with the API called directly on the record in metres. The tests that
need a computed product share two module-scoped ones, ``full`` (eight live pixels, for the velocity fields)
and ``sparse`` (three, for the blob parameters: the duration-time fit costs 0.7 s a live pixel). ``slow``
marks what spawns worker processes.
"""

import ast
import dataclasses
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import pytest
import xarray as xr

import fusion_ui.plots  # noqa: F401 - registers the real specs
from fusion_ui.core import catalog, db, params_ui, precompute, registry, store
from fusion_ui.plots import _pipeline, blob_parameters, method_fields, pixel_averages
from fusion_ui.plots._pipeline import pipeline
from fusion_ui.views import products
from tests import product_fixtures as fx

REPO = Path(__file__).resolve().parent.parent
KEYS = ("pixel_averages", "method_fields", "blob_parameters")
PARAMS = {
    "pixel_averages": fx.pixel_averages_params,
    "method_fields": fx.method_fields_params,
    "blob_parameters": fx.blob_parameters_params,
}

#: The plan's scalar names (docs/PHASE_06_DECORRELATION.md, "Scalars"), written out here as the user
#: confirms them at G2. The test is an independent statement of them.
METHOD_FIELDS_NAMES = (
    "vr_max vz_max nlags_max vr_com vz_com nlags_com level_com vr_2dcc vz_2dcc nlags_2dcc nevents"
    " vr3_tde vz3_tde vr2_tde vz2_tde cc_tde vr3_catde vz3_catde vr2_catde vz2_catde"
).split()
BLOB_PARAMETERS_NAMES = (
    "level area lx_c ly_c theta_c lr lz lx_f ly_f theta_f taud lam".split()
)


def target(shot=fx.SHOT, preprocessed=True):
    return registry.Target(
        "cmod", shot, "apd", preprocessed, "unused", float("nan"), float("nan"), "none"
    )


# ---------------------------------------------------------------------------
# Products computed through the store, once per module
# ---------------------------------------------------------------------------


def _stored(case, keys, root):
    """``keys`` computed on the ``case`` record through the store, the way a batch job does it.

    The bank first, with ``batch=True``, so that what comes back is the 2DCA straight out of memory; the
    products built on it after, without: once the bank is cached a page may compute them, and they are
    computed from the bank *as stored*. ``fresh`` is what each compute returned, ``loaded`` the blob read
    back from disk.
    """
    record = fx.make_record(case)
    out = SimpleNamespace(
        case=case, record=record, root=root, fresh={}, loaded={}, runs={}
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("FUSION_UI_CACHE", str(root / "cache"))
        out.conn = db.open_db(root / "state" / "shot_explorer.sqlite")
        for key in keys:
            result, run = store.result(
                out.conn,
                registry.get(key),
                target(),
                PARAMS[key](),
                record,
                batch=key == "pixel_averages",
            )
            assert run["status"] == "ok", run["error"]
            out.fresh[key], out.runs[key] = result, run
            out.loaded[key] = store.load_result(out.conn, run)
    return out


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    stored = _stored(
        "full", ("pixel_averages", "method_fields"), tmp_path_factory.mktemp("full")
    )
    yield stored
    stored.conn.close()


@pytest.fixture(scope="module")
def sparse(tmp_path_factory):
    stored = _stored(
        "sparse",
        ("pixel_averages", "blob_parameters"),
        tmp_path_factory.mktemp("sparse"),
    )
    yield stored
    stored.conn.close()


def live_pixels(case):
    return fx.CASES[case]["live"]


# ---------------------------------------------------------------------------
# What is registered, and how
# ---------------------------------------------------------------------------


def test_the_three_specs_are_imported_bank_first_and_after_the_older_ones():
    """``plots/__init__.py`` imports them in this order, which ``register`` needs (an upstream before what
    is built on it) and the plot picker lists. The order of the live registry is not a contract: a test
    that pops a spec and puts it back moves it to the end."""
    tree = ast.parse((Path(fusion_ui.plots.__file__)).read_text())
    (imported,) = [
        [alias.name for alias in node.names]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "fusion_ui.plots"
    ]
    positions = [imported.index(k) for k in KEYS]
    assert positions == sorted(positions)
    assert imported.index("velocity_field") < positions[0]
    assert imported[-3:] == list(KEYS), "after every older spec"
    assert set(KEYS) <= set(registry.REGISTRY)


def test_only_the_bank_is_batch_only_and_the_other_two_are_built_on_it():
    bank, fields, blobs = (registry.get(k) for k in KEYS)
    assert [s.batch_only for s in (bank, fields, blobs)] == [True, False, False]
    assert bank.requires is None and bank.upstream_params is None
    assert fields.requires == blobs.requires == "pixel_averages"
    assert [registry.chain(s)[-1].key for s in (fields, blobs)] == [
        "pixel_averages"
    ] * 2


def test_the_products_run_on_preprocessed_apd_files_only():
    """The 2DCA's threshold is in standard deviations of the normalised record and the mask is the one
    preprocessing stored: without this, ``precompute --run-day`` would try every raw file of the day.
    """
    for key in KEYS:
        spec = registry.get(key)
        assert spec.diagnostics == ("apd",)
        assert spec.accepts(True) and not spec.accepts(False)
    offered_on_raw = [s.key for s in registry.for_diagnostic("apd", False)]
    assert not set(KEYS) & set(offered_on_raw)
    assert set(KEYS) <= {s.key for s in registry.for_diagnostic("apd", True)}


def test_only_the_derived_products_write_scalars():
    assert registry.get("pixel_averages").scalars is None
    assert registry.get("method_fields").scalars is not None
    assert registry.get("blob_parameters").scalars is not None


# ---------------------------------------------------------------------------
# Parameters: the plan's names, one shared ``averages``, and what moves which key
# ---------------------------------------------------------------------------


def test_the_params_classes_are_the_plans_and_module_level():
    expected = {
        "pixel_averages": (pixel_averages.PixelAveragesParams, ["averages"]),
        "method_fields": (
            method_fields.MethodFieldsParams,
            ["averages", "tracking", "tde"],
        ),
        "blob_parameters": (
            blob_parameters.BlobParametersParams,
            ["averages", "neighbour_step", "blobs"],
        ),
    }
    for key, (cls, names) in expected.items():
        assert registry.get(key).params is cls
        assert [f.name for f in dataclasses.fields(cls)] == names
        # Importable by qualified name: a precompute pool worker receives them pickled.
        module = sys.modules[cls.__module__]
        assert getattr(module, cls.__qualname__) is cls
        assert cls.__module__ == f"fusion_ui.plots.{key}"


def test_every_product_shares_one_averages_of_the_apis_class():
    for key in KEYS:
        defaults = registry.get(key).params()
        assert type(defaults.averages) is pipeline.Averages
        assert defaults.averages == pipeline.Averages()
    assert method_fields.MethodFieldsParams().tracking == pipeline.Tracking()
    assert method_fields.MethodFieldsParams().tde == pipeline.Tde()
    assert blob_parameters.BlobParametersParams().blobs == pipeline.Blobs()
    assert (
        blob_parameters.BlobParametersParams().neighbour_step
        == pipeline.Tracking().neighbour_step
    )


def test_the_defaults_do_not_share_state():
    a, b = method_fields.MethodFieldsParams(), method_fields.MethodFieldsParams()
    a.tracking.position_filter.mask_signal_factor = 0.1
    a.averages.window = 7
    assert b.tracking.position_filter.mask_signal_factor != 0.1
    assert b.averages.window != 7


def _flatten(tree, prefix=""):
    out = {}
    for name, value in tree.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out


def _leaves(params):
    return _flatten(params_ui.canonical(params)["values"])


def _changed(params, path):
    """``params`` with the leaf at ``path`` set to something else of its own type."""
    old = _leaves(params)[path]
    if isinstance(old, bool):
        new = not old
    elif isinstance(old, int):
        new = old + 1
    elif isinstance(old, float):
        new = old + 0.5
    elif old is None:
        new = 1.0
    else:
        new = old + "_changed"
    change = new
    for name in reversed(path.split(".")):
        change = {name: change}
    return params_ui.with_values(params, change)


def _digest(key, params):
    return params_ui.hash_params(key, params)[0]


def test_averages_threshold_moves_all_three_keys():
    for key in KEYS:
        params = registry.get(key).params()
        moved = _changed(params, "averages.threshold")
        assert _digest(key, moved) != _digest(key, params), key
    # ... and it is one bank: the derived products ask for the very bank the threshold selects.
    for key in ("method_fields", "blob_parameters"):
        spec = registry.get(key)
        moved = _changed(spec.params(), "averages.threshold")
        assert _digest("pixel_averages", spec.upstream_params(moved)) == _digest(
            "pixel_averages",
            _changed(registry.get("pixel_averages").params(), "averages.threshold"),
        )


def test_tracking_moves_only_method_fields_and_blobs_only_blob_parameters():
    """Every leaf outside ``averages`` belongs to exactly one product, moves its key, and leaves the bank's."""
    bank_key = _digest("pixel_averages", registry.get("pixel_averages").params())
    owners = {}
    for key in KEYS:
        for path in _leaves(registry.get(key).params()):
            owners.setdefault(path, set()).add(key)
    assert owners  # not vacuous
    for path, who in owners.items():
        if path.startswith("averages."):
            assert who == set(KEYS), path
        elif path.startswith(("tracking.", "tde.")):
            assert who == {"method_fields"}, path
        elif path.startswith("blobs.") or path == "neighbour_step":
            assert who == {"blob_parameters"}, path
        else:
            pytest.fail(f"{path} is a leaf of {who} that the plan does not place")

    for key in ("method_fields", "blob_parameters"):
        spec = registry.get(key)
        base = spec.params()
        for path in _leaves(base):
            if path.startswith("averages."):
                continue
            moved = _changed(base, path)
            assert _digest(key, moved) != _digest(key, base), f"{key} {path}"
            # The bank is untouched: a tracking, TDE or blob change recomputes minutes, not an hour.
            assert (
                _digest("pixel_averages", spec.upstream_params(moved)) == bank_key
            ), f"{key} {path}"


def test_the_neighbour_step_is_a_separate_knob_in_each_product():
    """``tracking.neighbour_step`` and ``blob_parameters``'s ``neighbour_step`` are one setting named
    twice: moving the first leaves the blob key alone, as moving the cond_av mask must.
    """
    fields = method_fields.MethodFieldsParams()
    blobs = blob_parameters.BlobParametersParams()
    moved = _changed(fields, "tracking.neighbour_step")
    assert _digest("method_fields", moved) != _digest("method_fields", fields)
    assert _digest("blob_parameters", blobs) == _digest(
        "blob_parameters", blob_parameters.BlobParametersParams()
    )
    assert _changed(blobs, "neighbour_step").neighbour_step == blobs.neighbour_step + 1


def test_a_derived_spec_lifts_its_upstream_out_of_its_own_params():
    for key in ("method_fields", "blob_parameters"):
        spec = registry.get(key)
        params = spec.params()
        params.averages.window = 30
        lifted = spec.upstream_params(params)
        assert isinstance(lifted, pixel_averages.PixelAveragesParams)
        assert lifted.averages.window == 30 and lifted.averages == params.averages
        assert lifted.averages is not params.averages  # a copy: nothing is shared


def test_the_fields_the_fields_page_reads_are_there():
    """``views.products.related_params`` follows ``averages``, ``tracking.neighbour_step`` and
    ``neighbour_step``: a rename here would leave the page looking for the wrong blob parameters.
    """
    found = {key: registry.get(key) for key in KEYS}
    fields = method_fields.MethodFieldsParams()
    assert (
        products.related_params(found, fields)["blob_parameters"]
        == found["blob_parameters"].params()
    )

    fields.averages.window = 30
    fields.tracking.neighbour_step = 2
    related = products.related_params(found, fields)
    assert related["pixel_averages"].averages.window == 30
    assert related["blob_parameters"].averages.window == 30
    assert related["blob_parameters"].neighbour_step == 2
    assert products.neighbour_step(fields) == 2


def _form_leaves(cls, prefix=""):
    """``(path, owning class, annotation)`` of every leaf the parameter form would draw a widget for."""
    hints = params_ui._hints(cls)
    for f in params_ui._dataclass_fields(cls):
        annotation, _ = params_ui._unwrap_optional(hints.get(f.name, f.type))
        path = f"{prefix}.{f.name}" if prefix else f.name
        if params_ui._dataclass_fields(annotation) is None:
            yield path, cls, f.name
        else:
            yield from _form_leaves(annotation, path)


@pytest.mark.parametrize("key", KEYS)
def test_every_leaf_of_every_products_form_is_explained(key):
    """A leaf with no help is an unlabelled number in the sidebar. The nested imaging_methods classes carry
    little docstring, and ``params_ui.HELP`` is keyed by the path from the top of the tree walked: the
    products nest those classes under ``tracking`` and ``blobs``, one level lower than the older specs.
    """
    leaves = list(_form_leaves(registry.get(key).params))
    assert leaves
    unexplained = [
        path for path, cls, name in leaves if not params_ui._help_for(cls, name, path)
    ]
    assert not unexplained


def test_the_default_keys_are_stable():
    """The defaults hash to these, and so does a fresh interpreter (see the subprocess test). Moving one
    means the deck's settings changed, a params field was added or renamed, or a params class moved: a new
    cache entry appears beside the old, which is the intended behaviour, but it is worth knowing.
    """
    expected = {
        "pixel_averages": "302e4217c137fa0d510a11f49500371362dfd4ae",
        "method_fields": "d190bcb9d4f6d4b56debefcbdd8fc1a98e16a7ba",
        "blob_parameters": "42728c93d43db7547a097156dd5c6d5edec0bccb",
    }
    assert {k: _digest(k, registry.get(k).params()) for k in KEYS} == expected


@pytest.mark.parametrize("key", KEYS)
def test_a_params_json_file_changing_the_window_fits_all_three(key, tmp_path):
    """J2b's ``--params-json {"averages": {"window": 30}}`` names every plot it is applied to."""
    path = tmp_path / "short.json"
    path.write_text(json.dumps({"averages": {"window": 30}}))
    specs = [registry.get(k) for k in KEYS]
    everything = precompute.params_from_file(str(path), specs)
    assert {k: p.averages.window for k, p in everything.items()} == dict.fromkeys(
        KEYS, 30
    )
    # One bank for all three: the derived ones ask for the bank the bank's own params select.
    bank = _digest("pixel_averages", everything["pixel_averages"])
    for derived in ("method_fields", "blob_parameters"):
        up = registry.get(derived).upstream_params(everything[derived])
        assert _digest("pixel_averages", up) == bank
    assert everything[key].averages.window == 30


def test_params_survive_pickling_as_a_pool_worker_receives_them():
    for key in KEYS:
        for params in (registry.get(key).params(), PARAMS[key]()):
            again = pickle.loads(pickle.dumps(params))
            assert again == params and type(again) is type(params)
            assert _digest(key, again) == _digest(key, params)


def test_a_fresh_interpreter_hashes_the_same_and_leaves_the_data_folder_to_fusion_ui(
    tmp_path,
):
    """The import-order trap, for real: ``decorrelation.pipeline`` imports fusion_scripts' ``config``, which
    fills every unset ``FUSION_*`` variable from its own ``.env`` -- on the server a folder that does not
    exist. ``fusion_ui.config`` must have filled them first. The process is fresh, with a dotenv of its own
    whose data folder is not fusion_scripts', and the params arrive pickled, as a pool worker gets them.
    """
    folder = tmp_path / "the_data_folder_of_the_ui_dotenv"
    dotenv = tmp_path / "ui.env"
    dotenv.write_text(f"FUSION_DATA_FOLDER={folder}\n")
    environment = {k: v for k, v in os.environ.items() if not k.startswith("FUSION_")}
    environment["FUSION_UI_DOTENV"] = str(dotenv)

    child = (
        "import json, pickle, sys\n"
        "import fusion_ui.plots\n"
        "from fusion_ui import config\n"
        "from fusion_ui.core import params_ui, registry\n"
        "params = pickle.loads(sys.stdin.buffer.read())\n"
        "print(json.dumps({'folder': config.DATA_FOLDER,\n"
        "    'types': [type(p).__module__ + '.' + type(p).__qualname__ for p in params],\n"
        "    'hashes': [params_ui.hash_params(k, p)[0] for k, p in zip(%r, params)],\n"
        "    'registered': [k in registry.REGISTRY for k in %r]}))\n" % (KEYS, KEYS)
    )
    sent = [registry.get(key).params() for key in KEYS]
    done = subprocess.run(
        [sys.executable, "-c", child],
        input=pickle.dumps(sent),
        env=environment,
        cwd=REPO,
        capture_output=True,
        check=True,
    )
    seen = json.loads(done.stdout.decode().strip().splitlines()[-1])
    assert seen["folder"] == str(folder)
    assert seen["types"] == [
        f"fusion_ui.plots.{k}.{cls}"
        for k, cls in zip(
            KEYS, ("PixelAveragesParams", "MethodFieldsParams", "BlobParametersParams")
        )
    ]
    assert seen["hashes"] == [_digest(k, p) for k, p in zip(KEYS, sent)]
    assert seen["registered"] == [True] * 3


def test_importing_the_api_first_would_lose_the_data_folder(tmp_path):
    """The trap the import order avoids, shown: with ``decorrelation.pipeline`` imported before
    ``fusion_ui.config``, fusion_scripts' ``.env`` fills the data folder first and the UI's dotenv loses.
    It needs a fusion_scripts ``.env`` that names a data folder, as every machine that runs the app has.
    """
    folder = tmp_path / "the_data_folder_of_the_ui_dotenv"
    dotenv = tmp_path / "ui.env"
    dotenv.write_text(f"FUSION_DATA_FOLDER={folder}\n")
    environment = {k: v for k, v in os.environ.items() if not k.startswith("FUSION_")}
    environment["FUSION_UI_DOTENV"] = str(dotenv)
    child = (
        "import json, os\n"
        "import decorrelation.pipeline\n"
        "scripts = os.environ.get('FUSION_DATA_FOLDER')\n"
        "from fusion_ui import config\n"
        "print(json.dumps([scripts, config.DATA_FOLDER if scripts else None]))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", child],
        env=environment,
        cwd=REPO,
        capture_output=True,
        check=True,
    )
    scripts, seen = json.loads(done.stdout.decode().strip().splitlines()[-1])
    if scripts is None:
        pytest.skip("fusion_scripts has no .env naming a data folder here")
    assert seen == os.path.expanduser(scripts) != str(folder)


def _imports(path):
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            yield node.lineno, node.module or ""


def test_fusion_ui_config_is_imported_above_the_api_and_the_api_only_there():
    lines = list(_imports(_pipeline.__file__))
    config_line = min(n for n, m in lines if m == "fusion_ui")
    api_line = min(n for n, m in lines if m.split(".")[0] == "decorrelation")
    assert config_line < api_line

    # Nothing else in fusion_ui imports the API: a second import site is a second place to get the order
    # wrong. The one exception is a test, which may need the API to say what it expects of it.
    for path in (REPO / "fusion_ui").rglob("*.py"):
        if path == Path(_pipeline.__file__):
            continue
        offenders = [m for _, m in _imports(path) if m.split(".")[0] == "decorrelation"]
        assert not offenders, f"{path.relative_to(REPO)} imports {offenders}"


@pytest.mark.parametrize(
    "module", [pixel_averages, method_fields, blob_parameters, _pipeline]
)
def test_the_specs_never_touch_streamlit(module):
    """``compute``, ``render`` and ``scalars`` stay pure: the page draws what a render returns."""
    assert not [
        m for _, m in _imports(module.__file__) if m.split(".")[0] == "streamlit"
    ]


# ---------------------------------------------------------------------------
# The record the API takes
# ---------------------------------------------------------------------------


def test_the_record_is_converted_from_centimetres_to_metres_as_the_papers_loader_does():
    ds = fx.make_record("full")
    assert 88 < float(ds.R.max()) < 91  # centimetres, like the files
    prepared = _pipeline.record(ds)
    assert 0.88 < float(prepared.R.max()) < 0.91
    assert -0.05 < float(prepared.Z.min()) and float(prepared.Z.max()) < 0.0
    # The same expression as apd_check.fields.load, so the dtype R has on disk survives: float32.
    expected = ds.assign_coords(R=ds.R / 100, Z=ds.Z / 100)
    assert prepared.R.dtype == np.float32 == expected.R.dtype
    assert np.array_equal(prepared.R.values, expected.R.values)
    assert np.array_equal(prepared.Z.values, expected.Z.values)
    assert prepared.frames.dtype == ds.frames.dtype == np.float64
    assert prepared.attrs["shot_number"] == fx.SHOT  # dead_mask reads it
    assert float(ds.R.max()) > 10  # the record it was given is as it was


def test_a_record_that_is_not_in_centimetres_is_refused_not_divided_again():
    ds = fx.make_record("full")
    in_metres = ds.assign_coords(R=ds.R / 100, Z=ds.Z / 100)
    in_millimetres = ds.assign_coords(R=ds.R * 10, Z=ds.Z * 10)
    for wrong in (in_metres, in_millimetres):
        with pytest.raises(ValueError, match="not a major radius in centimetres"):
            _pipeline.record(wrong)
    with pytest.raises(ValueError, match="no R coordinate"):
        _pipeline.record(ds.drop_vars("R"))
    with pytest.raises(ValueError, match="no Z coordinate"):
        _pipeline.record(ds.drop_vars("Z"))


def test_a_lazily_opened_record_is_loaded_into_memory_and_the_original_left_lazy(
    tmp_path,
):
    path = tmp_path / "record.nc"
    fx.make_record("full").to_netcdf(path)
    with xr.open_dataset(path) as lazy:
        assert not lazy.frames.variable._in_memory
        prepared = _pipeline.record(lazy)
        assert prepared.frames.variable._in_memory
        assert not lazy.frames.variable._in_memory
        # Through the file the stored mask comes back as a bool, which dead_mask needs.
        assert prepared.dead.dtype == bool
        assert pipeline.dead_mask(prepared).dtype == bool


# ---------------------------------------------------------------------------
# What each adapter hands the API, and what it returns
# ---------------------------------------------------------------------------


def test_pixel_averages_asks_for_every_live_reference_and_never_a_dead_one(
    full, monkeypatch
):
    asked = []
    real = pipeline.average

    def average(ds, averages, x, y):
        asked.append((x, y))
        assert averages is params.averages
        assert float(ds.R.max()) < 2.0  # metres
        return real(ds, averages, x, y)

    params = fx.pixel_averages_params()
    monkeypatch.setattr(pipeline, "average", average)
    bank = pixel_averages.compute(full.record, params)
    assert asked == [
        (x, y) for y in range(fx.NY) for x in range(fx.NX) if (x, y) != (0, 1)
    ]
    assert bank.attrs["dead_mask_source"] == full.record.attrs["dead_mask_source"]


def test_the_derived_products_use_the_banks_mask_and_never_compute_one(
    full, monkeypatch
):
    """The mask the bank was computed with, with its source, and ``dead_mask`` is not asked again: the
    file may have changed since, and the bank, the fields and the blobs must agree on what was never
    computed."""
    bank = full.loaded["pixel_averages"]
    seen = {}

    def spy(name):
        def call(*args, **kwargs):
            seen[name] = (args, kwargs)
            return xr.Dataset({"returned": 1})

        return call

    def refuse(*args, **kwargs):
        raise AssertionError("dead_mask was computed again downstream")

    monkeypatch.setattr(pipeline, "dead_mask", refuse)
    monkeypatch.setattr(pipeline, "fields", spy("fields"))
    monkeypatch.setattr(pipeline, "blobs", spy("blobs"))

    fields_params, blobs_params = fx.method_fields_params(), fx.blob_parameters_params()
    out_fields = method_fields.compute(full.record, fields_params, bank)
    out_blobs = blob_parameters.compute(full.record, blobs_params, bank)

    # What the API returns comes back as it is: no renaming, no casting, nothing dropped.
    assert list(out_fields.data_vars) == list(out_blobs.data_vars) == ["returned"]

    args, kwargs = seen["fields"]
    assert not kwargs, "no `pixels`: the TDE's result depends on the pixel order"
    ds, average_at, averages, tracking, tde, dead = args
    assert float(ds.R.max()) < 2.0 and ds.frames.variable._in_memory
    assert (averages, tracking, tde) == (
        fields_params.averages,
        fields_params.tracking,
        fields_params.tde,
    )
    assert dead.dims == ("y", "x") and dead.dtype == bool
    assert np.array_equal(dead.values, bank.dead.values)
    assert dead.attrs["dead_mask_source"] == bank.attrs["dead_mask_source"]
    for x, y in live_pixels("full"):
        assert average_at(x, y).identical(pipeline.at(bank, x, y))
    assert average_at(0, 1) is None  # the dead reference was never computed

    args, kwargs = seen["blobs"]
    assert not kwargs
    ds, average_at, averages, neighbour_step, blobs, dead = args
    assert float(ds.R.max()) < 2.0
    assert (averages, neighbour_step, blobs) == (
        blobs_params.averages,
        blobs_params.neighbour_step,
        blobs_params.blobs,
    )
    assert dead.attrs["dead_mask_source"] == bank.attrs["dead_mask_source"]
    assert np.array_equal(dead.values, bank.dead.values)


def test_the_neighbour_step_reaches_the_api(full, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        pipeline, "blobs", lambda *args: seen.update(step=args[3]) or xr.Dataset()
    )
    params = fx.blob_parameters_params()
    params.neighbour_step = 3
    blob_parameters.compute(full.record, params, full.loaded["pixel_averages"])
    assert seen["step"] == 3


# ---------------------------------------------------------------------------
# The products themselves: equal to the API called directly, through the store and back
# ---------------------------------------------------------------------------


def test_pixel_averages_equals_a_direct_api_call_and_survives_the_store(full):
    expected = fx.direct_bank("full")
    fx.assert_bit_equal(full.fresh["pixel_averages"], expected, "computed")
    fx.assert_bit_equal(full.loaded["pixel_averages"], expected, "stored")


def test_method_fields_equals_a_direct_api_call_and_survives_the_store(full):
    """Computed from the bank *as stored*, against the API called on the bank straight out of memory: a
    derived product computed after the 2DCA is bit-identical to one computed later from its blob.
    """
    expected = fx.direct_fields("full")
    fx.assert_bit_equal(full.fresh["method_fields"], expected, "computed")
    fx.assert_bit_equal(full.loaded["method_fields"], expected, "stored")


def test_blob_parameters_equals_a_direct_api_call_and_survives_the_store(sparse):
    expected = fx.direct_blobs("sparse")
    fx.assert_bit_equal(sparse.fresh["blob_parameters"], expected, "computed")
    fx.assert_bit_equal(sparse.loaded["blob_parameters"], expected, "stored")


def test_the_bank_comes_back_out_of_the_store_reference_by_reference(full, sparse):
    for stored in (full, sparse):
        fresh, loaded = stored.fresh["pixel_averages"], stored.loaded["pixel_averages"]
        for y in range(fx.NY):
            for x in range(fx.NX):
                a, b = pipeline.at(fresh, x, y), pipeline.at(loaded, x, y)
                assert (a is None) == (b is None)
                if a is not None:
                    assert a.identical(b), (stored.case, x, y)


def test_the_fixtures_settings_differ_from_the_decks_in_every_group():
    """So that a setting an adapter failed to pass on shows up as a mismatch in the equalities above,
    where the deck's own values would hide it."""
    assert fx.AVERAGES != pipeline.Averages()
    assert fx.TRACKING != pipeline.Tracking()
    assert fx.TRACKING.position_filter != pipeline.Tracking().position_filter
    assert fx.TDE != pipeline.Tde()
    assert fx.BLOBS != pipeline.Blobs()
    assert fx.BLOBS.gauss_fit != pipeline.Blobs().gauss_fit
    assert fx.BLOBS.taud_estimation != pipeline.Blobs().taud_estimation


# ---------------------------------------------------------------------------
# The schemas, as the plan states them: names, dims, dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["fresh", "loaded"])
def test_the_bank_has_the_planned_schema(full, which):
    bank = getattr(full, which)["pixel_averages"]
    assert dict(bank.sizes) == {
        "ref_y": fx.NY,
        "ref_x": fx.NX,
        "y": fx.NY,
        "x": fx.NX,
        "time": fx.LAGS,
    }
    reference = ("ref_y", "ref_x")
    for name in ("cond_av", "cond_repr", "cross_corr"):
        assert bank[name].dims == reference + ("y", "x", "time")
        assert bank[name].dtype == np.float64
    assert bank["nevents"].dims == bank["computed"].dims == reference
    assert bank["nevents"].dtype == np.int64 and bank["computed"].dtype == bool
    assert bank["dead"].dims == ("y", "x") and bank["dead"].dtype == bool
    assert set(bank.data_vars) == {
        "cond_av",
        "cond_repr",
        "cross_corr",
        "nevents",
        "computed",
        "dead",
    }
    assert bank.attrs["averages"] == (
        "threshold=2.0; window=10; check_max=1; single_counting=True"
    )
    assert bank.attrs["dead_mask_source"] == "synthetic, the full mask"

    # NaN at the skipped reference, 0 events where none or skipped, computed False where dead.
    dead = (0, 1)
    assert not bank["computed"].values[dead[1], dead[0]]
    assert bank["nevents"].values[dead[1], dead[0]] == 0
    assert np.isnan(bank["cond_av"].isel(ref_y=dead[1], ref_x=dead[0]).values).all()
    flat_x, flat_y = fx.FLAT
    assert bank["computed"].values[flat_y, flat_x]
    assert bank["nevents"].values[flat_y, flat_x] == 0, "live, no events"
    assert (bank["nevents"].values > 0).sum() == len(live_pixels("full")) - 1
    # Metres, and the lags in seconds around zero.
    assert 0.88 < float(bank["R"].max()) < 0.91
    assert np.isclose(float(bank["time"].values[fx.LAGS // 2]), 0.0)
    assert np.isclose(np.diff(bank["time"].values), fx.DT).all()


@pytest.mark.parametrize("which", ["fresh", "loaded"])
def test_method_fields_has_the_planned_schema(full, which):
    fields = getattr(full, which)["method_fields"]
    assert dict(fields.sizes) == {"y": fx.NY, "x": fx.NX, "time": fx.LAGS}
    per_pixel = set(fx.FIELD_VARIABLES)
    per_lag = {
        f"{kind}_{track}" for track in fx.TRACKS for kind in ("pos_r", "pos_z", "fit")
    }
    assert set(fields.data_vars) == per_pixel | per_lag | {"dead"}
    for name in per_pixel:
        assert fields[name].dims == ("y", "x"), name
        # where(~dead) makes the counts float64 too, with NaN at dead pixels, as velocities.nc has them.
        assert fields[name].dtype == np.float64, name
    for track in fx.TRACKS:
        for kind in ("pos_r", "pos_z"):
            assert fields[f"{kind}_{track}"].dims == ("y", "x", "time")
            assert fields[f"{kind}_{track}"].dtype == np.float64
        assert fields[f"fit_{track}"].dims == ("y", "x", "time")
        assert fields[f"fit_{track}"].dtype == bool
    assert fields["dead"].dims == ("y", "x") and fields["dead"].dtype == bool
    assert set(fields.attrs) - {
        k for k in fields.attrs if k.startswith("fusion_ui_")
    } == {
        "min_cc",
        "averages",
        "tracking",
        "tde",
        "dead_mask_source",
    }
    assert fields.attrs["min_cc"] == fx.TDE.min_cc
    assert fields.attrs["dead_mask_source"] == "synthetic, the full mask"
    assert 0.88 < float(fields["R"].max()) < 0.91  # metres

    dead = fields["dead"].values
    for name in per_pixel:
        assert np.isnan(fields[name].values[dead]).all(), name
    # The level exists for the centroid track alone.
    assert np.isnan(fields["level_max"].values).all()
    assert np.isnan(fields["level_2dcc"].values).all()
    assert np.isfinite(fields["level_com"].values[~dead]).any()
    # What is tracked is where it can be: no position without an average, and the fitted lags are lags.
    assert fields["fit_com"].values.any()


@pytest.mark.parametrize("which", ["fresh", "loaded"])
def test_blob_parameters_has_the_planned_schema(sparse, which):
    blobs = getattr(sparse, which)["blob_parameters"]
    assert dict(blobs.sizes) == {"y": fx.NY, "x": fx.NX}
    assert set(blobs.data_vars) == set(fx.BLOB_VARIABLES) | {"dead"}
    for name in fx.BLOB_VARIABLES:
        assert blobs[name].dims == ("y", "x") and blobs[name].dtype == np.float64, name
    assert blobs["dead"].dtype == bool
    assert blobs.attrs["units"] == "m, m^2, s, rad"
    assert blobs.attrs["neighbour_step"] == fx.TRACKING.neighbour_step
    assert blobs.attrs["averages"] == (
        "threshold=2.0; window=10; check_max=1; single_counting=True"
    )
    assert blobs.attrs["dead_mask_source"] == "synthetic, the sparse mask"
    assert 0.88 < float(blobs["R"].max()) < 0.91


# ---------------------------------------------------------------------------
# Units: the numbers are in metres and seconds, not centimetres
# ---------------------------------------------------------------------------


def test_a_velocity_is_in_metres_per_second_and_a_size_in_metres(full, sparse):
    """The pulses move at (500, 150) m/s and are 3.5 mm wide. In cm/s the velocity would read 5e4, in m/s
    off an unconverted R and Z too. A hundred times too small if R and Z were converted twice.
    """
    fields, blobs = full.loaded["method_fields"], sparse.loaded["blob_parameters"]
    x, y = 1, 1
    for track in ("com", "max"):
        vr = float(fields[f"vr_{track}"].values[y, x])
        assert 250 < vr < 1000, (track, vr)
    speed = np.hypot(fields["vr_com"].values[y, x], fields["vz_com"].values[y, x])
    assert 250 < speed < 1500
    assert 0.001 < float(blobs["lx_f"].values[y, x]) < 0.02
    assert 0.001 < float(blobs["ly_f"].values[y, x]) < 0.02
    assert 1e-6 < float(blobs["area"].values[y, x]) < 1e-3
    assert 1e-6 < float(blobs["taud"].values[y, x]) < 1e-3  # seconds, not microseconds
    assert 0.0 <= float(blobs["lam"].values[y, x]) <= 1.0
    assert 0.0 < float(blobs["level"].values[y, x]) < 1.0


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------


def stored_rows(conn, run):
    return {
        (r["x"], r["y"], r["name"]): r["value"]
        for r in conn.execute(
            "SELECT x, y, name, value FROM scalars WHERE run_id = ?", (run["id"],)
        )
    }


def test_there_are_twenty_and_twelve_names_as_the_plan_lists_them():
    assert (
        list(method_fields.SCALARS) == METHOD_FIELDS_NAMES
        and len(METHOD_FIELDS_NAMES) == 20
    )
    assert (
        list(blob_parameters.SCALARS) == BLOB_PARAMETERS_NAMES
        and len(BLOB_PARAMETERS_NAMES) == 12
    )
    assert "nevents" not in blob_parameters.SCALARS  # method_fields writes it
    assert not set(method_fields.SCALARS) & set(blob_parameters.SCALARS)
    # Every name is a variable the API makes; level_max and level_2dcc are all NaN and not names.
    assert set(METHOD_FIELDS_NAMES) <= set(fx.FIELD_VARIABLES)
    assert set(BLOB_PARAMETERS_NAMES) | {"nevents"} == set(fx.BLOB_VARIABLES)


def test_method_fields_writes_twenty_names_at_every_live_pixel_and_nothing_at_a_dead_one(
    full,
):
    result = full.loaded["method_fields"]
    mapping = method_fields.scalars(result)
    live = live_pixels("full")
    assert len(mapping) == len(live) * 20 == 160
    assert {name for _, _, name in mapping} == set(METHOD_FIELDS_NAMES)
    assert {(x, y) for x, y, _ in mapping} == set(live)
    assert all(isinstance(k, tuple) and len(k) == 3 for k in mapping)
    assert not [
        k for k in mapping if k[:2] == (0, 1)
    ], "a dead pixel was never computed"
    for (x, y, name), value in mapping.items():
        expected = float(result[name].values[y, x])
        assert (np.isnan(value) and np.isnan(expected)) or value == expected


def test_the_scalar_rows_in_the_ledger_are_the_mapping_with_failures_as_null(full):
    run = full.runs["method_fields"]
    rows = stored_rows(full.conn, run)
    result = full.loaded["method_fields"]
    assert len(rows) == len(live_pixels("full")) * 20
    assert not [k for k in rows if k[:2] == (0, 1)]
    nulls = 0
    for (x, y, name), value in rows.items():
        expected = float(result[name].values[y, x])
        if np.isnan(expected):
            assert value is None, (x, y, name)
            nulls += 1
        else:
            assert value == expected, (x, y, name)
    assert (
        nulls
    ), "some estimates fail on a record this small, and are NULL, not missing"
    # Tried and failed is not never tried: the live pixel without events has its rows.
    flat = {n: v for (x, y, n), v in rows.items() if (x, y) == fx.FLAT}
    assert len(flat) == 20 and flat["nevents"] == 0.0 and flat["vr_com"] is None


def test_blob_parameters_writes_twelve_names_and_never_nevents(sparse):
    result = sparse.loaded["blob_parameters"]
    mapping = blob_parameters.scalars(result)
    live = live_pixels("sparse")
    assert len(mapping) == len(live) * 12 == 36
    assert {name for _, _, name in mapping} == set(BLOB_PARAMETERS_NAMES)
    assert {(x, y) for x, y, _ in mapping} == set(live)

    rows = stored_rows(sparse.conn, sparse.runs["blob_parameters"])
    assert len(rows) == 36
    assert {n for _, _, n in rows} == set(BLOB_PARAMETERS_NAMES)
    for (x, y, name), value in rows.items():
        expected = float(result[name].values[y, x])
        assert (value is None and np.isnan(expected)) or value == expected
    flat_x, flat_y = fx.FLAT
    assert rows[(flat_x, flat_y, "lx_f")] is None  # no average to fit: tried and failed
    assert rows[(1, 1, "lx_f")] is not None


def test_pixel_averages_writes_no_scalars(full):
    assert not stored_rows(full.conn, full.runs["pixel_averages"])


# ---------------------------------------------------------------------------
# The store: batch only, chained, linked
# ---------------------------------------------------------------------------


def test_the_bank_is_never_computed_unless_the_caller_is_a_batch_job(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    conn = db.open_db(tmp_path / "s.sqlite")
    record = fx.make_record("sparse")

    def refuse(*args, **kwargs):
        raise AssertionError("computed")

    monkeypatch.setattr(pipeline, "average", refuse)
    for key in KEYS:
        with pytest.raises(store.BatchOnlyError) as error:
            store.result(conn, registry.get(key), target(), PARAMS[key](), record)
        assert error.value.spec.key == "pixel_averages"
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0

    missing = store.missing_batch_upstreams(
        conn, registry.get("method_fields"), target(), PARAMS["method_fields"]()
    )
    assert [link.key for link, _, _ in missing] == ["pixel_averages"]
    assert store.lookup(
        conn, registry.get("pixel_averages"), target(), PARAMS["pixel_averages"]()
    ) == (None, None)
    conn.close()


def test_derived_products_are_computed_once_the_bank_is_cached_and_link_to_it(full):
    runs = full.runs
    assert runs["method_fields"]["upstream_run_id"] == runs["pixel_averages"]["id"]
    assert runs["pixel_averages"]["upstream_run_id"] is None
    assert not store.missing_batch_upstreams(
        full.conn, registry.get("method_fields"), target(), fx.method_fields_params()
    )
    result, run = store.lookup(
        full.conn, registry.get("method_fields"), target(), fx.method_fields_params()
    )
    assert run["id"] == runs["method_fields"]["id"] and result is not None


def test_the_blobs_ask_for_the_same_bank_as_the_fields_at_the_same_averages():
    a = registry.get("method_fields").upstream_params(fx.method_fields_params())
    b = registry.get("blob_parameters").upstream_params(fx.blob_parameters_params())
    assert _digest("pixel_averages", a) == _digest("pixel_averages", b)


# ---------------------------------------------------------------------------
# The views: pure, and they draw
# ---------------------------------------------------------------------------


def assert_a_figure(figure):
    assert isinstance(figure, go.Figure)
    json.loads(figure.to_json())  # what Streamlit sends the browser


def test_pixel_averages_draws_the_pixel_with_the_most_events(full):
    bank = full.loaded["pixel_averages"]
    events = bank["nevents"].values
    x, y = pixel_averages.best_pixel(bank)
    assert events[y, x] == events.max()
    figure = pixel_averages.SPEC.render(bank, PARAMS["pixel_averages"](), target())
    assert_a_figure(figure)
    assert f"x={x}, y={y}" in figure.layout.title.text
    maps = [t for t in figure.data if isinstance(t, go.Heatmap)]
    # Two rows, the average and the cross-correlation, at the strip's lags.
    assert len(maps) >= 4 and len(maps) % 2 == 0
    assert figure.layout.meta["source"] in ("deck", "fit", "span")


def test_pixel_averages_says_so_when_no_reference_has_events(full):
    empty = full.loaded["pixel_averages"].copy(deep=True)
    empty["nevents"][:] = 0
    assert pixel_averages.best_pixel(empty) is None
    figure = pixel_averages.SPEC.render(empty, PARAMS["pixel_averages"](), target())
    assert_a_figure(figure)
    assert "No reference pixel has events" in figure.layout.annotations[0].text


def test_method_fields_draws_one_v_r_map_per_method_on_one_scale(full):
    figure = method_fields.SPEC.render(
        full.loaded["method_fields"], PARAMS["method_fields"](), target()
    )
    assert_a_figure(figure)
    assert figure.layout.meta["mode"] == "vr"
    maps = [t for t in figure.data if isinstance(t, go.Heatmap)]
    assert len(maps) == 7
    from fusion_ui.views.bundle import Cuts

    title = figure.layout.title.text
    assert "Fields page" in title
    # It says which cuts it drew at, whatever the page's defaults are.
    assert f"{Cuts().min_lags} lags or {Cuts().min_events} events" in title


def test_blob_parameters_draws_one_map_per_parameter_marking_dead_and_failed(sparse):
    figure = blob_parameters.SPEC.render(
        sparse.loaded["blob_parameters"], PARAMS["blob_parameters"](), target()
    )
    assert_a_figure(figure)
    maps = [t for t in figure.data if isinstance(t, go.Heatmap)]
    assert [t.meta["parameter"] for t in maps] == fx.BLOB_VARIABLES
    titles = [a.text for a in figure.layout.annotations]
    assert "area [m²]" in titles and "taud [s]" in titles and "nevents" in titles
    kinds = {t.name for t in figure.data if isinstance(t, go.Scatter)}
    assert kinds == {"dead pixel (mask)", "no estimate"}
    # Dead pixels are blank in the map, so a number is only ever a pixel that was computed.
    dead = sparse.loaded["blob_parameters"]["dead"].values
    for trace in maps:
        assert np.isnan(np.asarray(trace.z, dtype=float)[dead]).all()
    assert [t.showlegend for t in figure.data if isinstance(t, go.Scatter)].count(
        True
    ) == 2


def test_the_renders_survive_a_blob_without_some_parameters(sparse):
    result = sparse.loaded["blob_parameters"].drop_vars(["taud", "lam"])
    figure = blob_parameters.SPEC.render(result, PARAMS["blob_parameters"](), target())
    assert_a_figure(figure)
    assert len([t for t in figure.data if isinstance(t, go.Heatmap)]) == 11


def test_the_fields_pages_builders_draw_the_real_products(full, sparse):
    """The page was written against synthetic blobs from the frozen schemas; these are what the API makes,
    read back from netCDF. Every builder must draw them, with nothing but real pixels where they are.
    """
    from fusion_ui.views import frame, lag_strip, numbers, panels, tracks
    from fusion_ui.views.bundle import Bundle

    bundle = Bundle(
        shot=fx.SHOT,
        bank=full.loaded["pixel_averages"],
        fields=full.loaded["method_fields"],
        pixel=(1, 1),
    )
    assert bundle.shape == (fx.NY, fx.NX) and bundle.has_coordinates
    assert bundle.mask_source == "synthetic, the full mask"
    assert bundle.min_cc == fx.TDE.min_cc
    assert bundle.dead.sum() == 1 and bundle.dead[1, 0]

    for mode in ("arrows", "vr", "vz"):
        figure = panels.velocity_panels(bundle, mode=mode)
        assert_a_figure(figure)
        assert figure.layout.meta["mode"] == mode and len(figure.data) > 7
    figures = [
        lag_strip.lag_strip(bundle),
        frame.frame_figure(bundle, "cond_av", 0.0),
        frame.frame_figure(bundle, "cross_corr", 3.0),
        frame.trace_figure(bundle, "cond_av", 0.0),
        tracks.tracks_figure(bundle),
    ]
    for figure in figures:
        assert_a_figure(figure)
        assert figure.data, "a pixel with events draws, it is not a message"

    blobs = Bundle(
        shot=fx.SHOT,
        bank=sparse.loaded["pixel_averages"],
        blobs=sparse.loaded["blob_parameters"],
        pixel=(1, 1),
    )
    tables = numbers.numbers(bundle)
    assert len(tables) == 1, "no blob table until blob_parameters is computed"
    methods = tables[0]
    assert len(methods) == 7
    # In metres per second, for the 2DCA centroid at the interior pixel: the planted 500 m/s.
    centroid = methods.loc[methods["method"] == "2DCA centroid", "v_R [m/s]"]
    assert 250 < float(centroid.iloc[0]) < 1000
    with_blobs = numbers.numbers(blobs)
    assert len(with_blobs) == 2 and len(with_blobs[1]) == 13
    assert set(with_blobs[1]["parameter"]) == set(fx.BLOB_VARIABLES)
    # A live pixel without events is a sentence, not an exception; a dead one too.
    for pixel in (fx.FLAT, (0, 1)):
        silent = dataclasses.replace(bundle, pixel=pixel)
        assert isinstance(numbers.numbers(silent), str)
        assert lag_strip.lag_strip(silent).layout.annotations


# ---------------------------------------------------------------------------
# End to end: the precompute engine, on a real file in a real (tiny) data tree
# ---------------------------------------------------------------------------


@pytest.fixture
def tree(monkeypatch, tmp_path):
    """Two preprocessed APD files holding the sparse record (so a fill has something to be parallel over)
    in a data tree, an empty descriptor, the index and the environment."""
    folder = tmp_path / "alcator" / "apd"
    folder.mkdir(parents=True)
    shots = (fx.SHOT, fx.SHOT + 1)
    for shot in shots:
        record = fx.make_record("sparse")
        record.attrs["shot_number"] = shot
        record.to_netcdf(folder / f"apd_{shot}_preprocessed.nc")
    descriptor = tmp_path / "plasma_discharges.json"
    descriptor.write_text("[]")
    database = tmp_path / "state" / "shot_explorer.sqlite"
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(tmp_path / "alcator"))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(descriptor))
    monkeypatch.setenv("FUSION_UI_DB", str(database))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")
    conn = db.open_db(database)
    catalog.rescan(conn, str(tmp_path / "alcator"), "cmod", str(descriptor))
    yield SimpleNamespace(conn=conn, shots=shots, database=database, root=tmp_path)
    conn.close()


def fill(conn, plots=KEYS, workers=1, **selection):
    specs = precompute.in_dependency_order([registry.get(k) for k in plots])
    targets = precompute.select_targets(conn, specs, "cmod", **selection)
    plan = precompute.plan_fill(conn, [(s, PARAMS[s.key]()) for s in specs], targets)
    lines = []
    report = precompute.execute(conn, plan, log=lines.append, workers=workers)
    return report, lines


def runs_of(conn):
    return {(r["shot"], r["plot"]): dict(r) for r in conn.execute("SELECT * FROM runs")}


def test_a_run_day_fill_never_selects_a_raw_file(tree):
    """A day's raw files sit beside its preprocessed ones. A raw record is not normalised and has no mask:
    ``precompute --run-day`` would otherwise try a 35 to 70 minute bank on each."""
    shot = tree.shots[0]
    raw = Path(tree.root) / "alcator" / "apd" / f"apd_{shot}.nc"
    fx.make_record("sparse").to_netcdf(raw)
    catalog.rescan(tree.conn, str(tree.root / "alcator"), "cmod", None)
    indexed = {
        (r["shot"], bool(r["preprocessed"]))
        for r in tree.conn.execute("SELECT shot, preprocessed FROM shots")
    }
    assert {(shot, True), (shot, False)} <= indexed
    specs = [registry.get(k) for k in KEYS]
    targets = precompute.select_targets(
        tree.conn, specs, "cmod", run_days={precompute.run_day(shot)}
    )
    assert targets and all(t.preprocessed for t in targets)
    assert {t.shot for t in targets} == set(tree.shots)


def test_precompute_of_the_two_derived_products_computes_the_bank_once_and_a_second_fill_skips_all(
    tree,
):
    """The plan's first fill names ``method_fields blob_parameters`` and leaves the bank to the chain: a
    batch job computes it with the first of them, and the second finds it cached."""
    shot = tree.shots[0]
    report, lines = fill(tree.conn, ("method_fields", "blob_parameters"), shots={shot})
    assert [s.computed for s in report.stats] == [1, 1], lines
    runs = runs_of(tree.conn)
    assert {plot: runs[(shot, plot)]["status"] for plot in KEYS} == dict.fromkeys(
        KEYS, "ok"
    )
    assert all(os.path.exists(runs[(shot, plot)]["blob_path"]) for plot in KEYS)
    bank = runs[(shot, "pixel_averages")]
    assert runs[(shot, "method_fields")]["upstream_run_id"] == bank["id"]
    assert runs[(shot, "blob_parameters")]["upstream_run_id"] == bank["id"]
    assert (
        bank["created_at"]
        <= runs[(shot, "method_fields")]["created_at"]
        <= runs[(shot, "blob_parameters")]["created_at"]
    )
    # The record's mtime as the index has it, which is what staleness compares.
    assert all(runs[(shot, plot)]["input_mtime"] for plot in KEYS)
    assert not store.stale_runs(tree.conn)

    # Through a real file: the record came off disk, sliced to its window and converted from centimetres,
    # with the mask stored in it as a bool; the products are what the API gives on the record in memory.
    fx.assert_bit_equal(
        store.load_result(tree.conn, runs[(shot, "pixel_averages")]),
        fx.direct_bank("sparse"),
        "bank",
    )
    fx.assert_bit_equal(
        store.load_result(tree.conn, runs[(shot, "blob_parameters")]),
        fx.direct_blobs("sparse"),
        "blobs",
    )

    # Now all three named: nothing is computed again.
    again, lines = fill(tree.conn, shots={shot})
    assert [s.cached for s in again.stats] == [1, 1, 1], lines
    assert [s.computed for s in again.stats] == [0, 0, 0]


@pytest.mark.slow
def test_precompute_on_a_pool_of_workers_gives_the_same_products(tree):
    """Spawned workers receive the parameters pickled and import ``fusion_ui.plots`` to find the specs."""
    report, lines = fill(tree.conn, workers=2)
    assert [s.computed for s in report.stats] == [2, 2, 2], lines
    runs = runs_of(tree.conn)
    first, second = tree.shots
    for plot in KEYS:
        a = store.load_result(tree.conn, runs[(first, plot)])
        b = store.load_result(tree.conn, runs[(second, plot)])
        fx.assert_bit_equal(a, b, plot)  # the two files hold the same record
    fx.assert_bit_equal(
        store.load_result(tree.conn, runs[(second, "pixel_averages")]),
        fx.direct_bank("sparse"),
        "bank",
    )
    fx.assert_bit_equal(
        store.load_result(tree.conn, runs[(second, "blob_parameters")]),
        fx.direct_blobs("sparse"),
        "blobs",
    )


# ---------------------------------------------------------------------------
# The single-shot page: the bank is shown from cache, the derived products compute off it
# ---------------------------------------------------------------------------

SINGLE_SHOT = str(REPO / "fusion_ui" / "pages" / "2_single_shot.py")


def widget(app, kind, label):
    matches = [w for w in getattr(app, kind) if w.label == label]
    assert (
        matches
    ), f"no {kind} labelled {label!r}: {[w.label for w in getattr(app, kind)]}"
    return matches[0]


def charts(app):
    return app.get("plotly_chart")


def page(shot, key):
    """The single-shot page on ``shot`` with ``key`` picked and its form holding the fixture's settings,
    which is what the multi-shot jump seeds it with: the page looks for the parameters it is shown.
    """
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(SINGLE_SHOT, default_timeout=120)
    app.session_state["selection"] = {
        "machine": "cmod",
        "shot": shot,
        "diagnostic": "apd",
        "preprocessed": True,
    }
    app.session_state["spec.apd"] = registry.get(key)
    params_ui.seed_session_state(app.session_state, f"params.{key}", PARAMS[key]())
    return app


def test_the_single_shot_page_shows_the_bank_from_cache_and_computes_the_rest_off_it(
    tree,
):
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()
    shot = tree.shots[0]
    fill(tree.conn, ("pixel_averages",), shots={shot})  # what the batch leaves behind
    before = runs_of(tree.conn)
    assert set(before) == {(shot, "pixel_averages")}

    # The bank: looked up and drawn, never computed, with the command that would compute it again.
    app = page(shot, "pixel_averages").run()
    assert not app.exception, app.exception
    assert not app.error, [e.value for e in app.error]
    assert len(charts(app)) == 1
    buttons = [b.label for b in app.button]
    assert "Show" in buttons and "Compute" not in buttons and "Recompute" not in buttons
    assert any(
        "fusion-ui precompute pixel_averages --shot" in c.value and "--force" in c.value
        for c in app.caption
    )
    # The form is the whole parameter tree, labelled: the nested imaging_methods classes included.
    assert widget(app, "number_input", "threshold").value == fx.AVERAGES.threshold
    assert widget(app, "number_input", "window").value == fx.AVERAGES.window

    # The derived ones wait for Compute, which computes them off the cached bank and nothing more.
    for key in ("method_fields", "blob_parameters"):
        app = page(shot, key).run()
        assert not app.exception, app.exception
        assert app.info and not charts(app), "a derived spec waits for Compute"
        if key == "method_fields":
            assert widget(app, "number_input", "min cc").value == fx.TDE.min_cc
            assert widget(app, "selectbox", "estimator").options == [
                "central_diff",
                "lsq",
            ]
        widget(app, "button", "Compute").click().run()
        assert not app.exception, app.exception
        assert not app.error, [e.value for e in app.error]
        assert len(charts(app)) == 1, key
    after = runs_of(tree.conn)
    assert set(after) == {(shot, plot) for plot in KEYS}
    assert (
        after[(shot, "pixel_averages")] == before[(shot, "pixel_averages")]
    ), "the bank was not touched"
    for plot in ("method_fields", "blob_parameters"):
        assert (
            after[(shot, plot)]["upstream_run_id"]
            == before[(shot, "pixel_averages")]["id"]
        )
    assert len(stored_rows(tree.conn, after[(shot, "method_fields")])) == 3 * 20
    assert len(stored_rows(tree.conn, after[(shot, "blob_parameters")])) == 3 * 12
