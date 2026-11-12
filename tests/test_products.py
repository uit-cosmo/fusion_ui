"""The three decorrelation products: ``pixel_averages``, ``method_fields`` and ``blob_parameters``.

They are thin adapters over ``imaging_methods.decorrelation``, so what is tested here is the adapting: that
each spec hands the API what it takes (the record in metres, the bank's own mask, the settings as they
are), returns its Dataset unchanged, stores it and gets it back bit for bit, writes its scalars, and hashes
as it should. The physics is the library's, tested there.

The record is ``tests/product_fixtures``: 3 x 3 pixels in centimetres, like the real files. A product
computed through the store is compared with the API called directly on the record in metres. The tests
that need a computed product share two module-scoped ones, ``full`` (eight live pixels, for the velocity
fields) and ``sparse`` (three, for the blob parameters: the duration-time fit costs 0.7 s a live pixel).
``slow`` marks what spawns worker processes.
"""

import ast
import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

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
DERIVED = KEYS[1:]
PARAMS = {
    "pixel_averages": fx.pixel_averages_params,
    "method_fields": fx.method_fields_params,
    "blob_parameters": fx.blob_parameters_params,
}

#: The scalar names, written out here as an independent statement of them: a stored row is found by its
#: name. Three are not their blob variable's name (``RENAMED``).
METHOD_FIELDS_NAMES = (
    "vr_max vz_max nlags_max vr_com vz_com nlags_com level_com vr_2dcc vz_2dcc nlags_2dcc number_events"
    " vr3_tde vz3_tde vr2_tde vz2_tde cc_tde vr3_catde vz3_catde vr2_catde vz2_catde"
).split()
BLOB_PARAMETERS_NAMES = "area lr lz taud_psd lambda_psd".split()
RENAMED = {"number_events": "nevents", "taud_psd": "taud", "lambda_psd": "lam"}

#: The blob variables the app shows, in the order of its table.
SHOWN = ["nevents", "area", "lr", "lz", "taud", "lam"]


def variable_of(name):
    """The blob variable the scalar ``name`` is read from."""
    return RENAMED.get(name, name)


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


def _imported(path):
    """The dotted names ``path`` imports: ``a.b`` for ``import a.b`` and for ``from a import b``."""
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""
            yield from (f"{node.module}.{alias.name}" for alias in node.names)


def test_only_pipeline_imports_the_decorrelation_api_and_the_specs_import_no_streamlit():
    importing = {
        path.relative_to(REPO).as_posix()
        for path in (REPO / "fusion_ui").rglob("*.py")
        if any(n.startswith("imaging_methods.decorrelation") for n in _imported(path))
    }
    assert importing == {"fusion_ui/plots/_pipeline.py"}
    for module in (pixel_averages, method_fields, blob_parameters, _pipeline):
        assert not [
            n for n in _imported(module.__file__) if n.split(".")[0] == "streamlit"
        ]


# ---------------------------------------------------------------------------
# Parameters: one shared ``averages``, and what moves which key
# ---------------------------------------------------------------------------


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


def test_the_averages_move_all_three_keys_and_select_one_bank():
    moved_bank = _changed(registry.get("pixel_averages").params(), "averages.threshold")
    for key in KEYS:
        params = registry.get(key).params()
        assert _digest(key, _changed(params, "averages.threshold")) != _digest(
            key, params
        )
    for key in DERIVED:
        spec = registry.get(key)
        lifted = spec.upstream_params(_changed(spec.params(), "averages.threshold"))
        assert isinstance(lifted, pixel_averages.PixelAveragesParams)
        assert _digest("pixel_averages", lifted) == _digest(
            "pixel_averages", moved_bank
        )
    # A copy: the bank's parameters share nothing with the product's.
    params = method_fields.MethodFieldsParams()
    assert pixel_averages.lifted_from(params).averages is not params.averages


def test_every_other_setting_moves_its_own_product_and_leaves_the_bank():
    """A tracking, TDE or blob change recomputes minutes, not the bank's hour."""
    bank_key = _digest("pixel_averages", registry.get("pixel_averages").params())
    for key in DERIVED:
        spec = registry.get(key)
        base = spec.params()
        own = [path for path in _leaves(base) if not path.startswith("averages.")]
        assert own
        for path in own:
            moved = _changed(base, path)
            assert _digest(key, moved) != _digest(key, base), f"{key} {path}"
            assert (
                _digest("pixel_averages", spec.upstream_params(moved)) == bank_key
            ), f"{key} {path}"


def _form_leaves(cls, prefix=""):
    """``(path, owning class, field name)`` of every leaf the parameter form draws a widget for."""
    hints = get_type_hints(cls)
    for f in dataclasses.fields(cls):
        path = f"{prefix}.{f.name}" if prefix else f.name
        if dataclasses.is_dataclass(hints[f.name]):
            yield from _form_leaves(hints[f.name], path)
        else:
            yield path, cls, f.name


@pytest.mark.parametrize("key", KEYS)
def test_every_leaf_of_every_products_form_is_explained(key):
    """A leaf with no help is an unlabelled number in the sidebar: ``params_ui.HELP`` is keyed by the
    path from the top of the tree walked, and the products nest the library's classes.
    """
    leaves = list(_form_leaves(registry.get(key).params))
    assert leaves
    unexplained = [
        path for path, cls, name in leaves if not params_ui.help_for(cls, name)
    ]
    assert not unexplained


def test_the_default_keys_are_stable():
    """The defaults hash to these. Moving one means a default setting changed, a params field was added
    or renamed, or a params class moved: a new cache entry appears beside the old, and every result
    stored under the old one is no longer found.
    """
    expected = {
        "pixel_averages": "302e4217c137fa0d510a11f49500371362dfd4ae",
        "method_fields": "d190bcb9d4f6d4b56debefcbdd8fc1a98e16a7ba",
        "blob_parameters": "42728c93d43db7547a097156dd5c6d5edec0bccb",
    }
    assert {k: _digest(k, registry.get(k).params()) for k in KEYS} == expected


#: Every registered spec's default hash. A stored result is found by it, so a moved one leaves every
#: result of that spec unreachable.
DEFAULT_HASHES = {
    "raw_frames": "f29f91e6ac616c2ee2980872c988504bb9de8c51",
    "dead_pixels": "c87bebadae3c0c04ac973bf608a4d6ea0571133b",
    "probe_trace": "1cb53206d761fb519becfafebedce0c54e020a7a",
    "thomson_profiles": "482534bb11949e3bff6fca56ff653e37328c9f89",
    "mlp_profiles": "ba6d655e1fa2aab4c5319fb35080e79b66e96928",
    "taud_psd": "7c56c709e582ced8ea0a30c3527804ba293b049c",
    "velocity_tde": "02e50d35120403814b240a9276ed7787032caa9a",
    "pixel_averages": "302e4217c137fa0d510a11f49500371362dfd4ae",
    "method_fields": "d190bcb9d4f6d4b56debefcbdd8fc1a98e16a7ba",
    "blob_parameters": "42728c93d43db7547a097156dd5c6d5edec0bccb",
}


def test_every_registered_specs_default_hash_is_stable():
    """Every spec, not only the three products: a stored result is found by its hash."""
    assert set(registry.REGISTRY) == set(DEFAULT_HASHES)
    assert len(DEFAULT_HASHES) == 10
    found = {
        key: params_ui.hash_params(key, spec.params())[0]
        for key, spec in registry.REGISTRY.items()
    }
    assert found == DEFAULT_HASHES


def test_a_params_json_file_changing_the_window_fits_all_three(tmp_path):
    """``--params-json`` with ``{"averages": {"window": 30}}`` applies to every plot named, as one bank."""
    path = tmp_path / "short.json"
    path.write_text(json.dumps({"averages": {"window": 30}}))
    everything = precompute.params_from_file(str(path), [registry.get(k) for k in KEYS])
    assert {k: p.averages.window for k, p in everything.items()} == dict.fromkeys(
        KEYS, 30
    )
    bank = _digest("pixel_averages", everything["pixel_averages"])
    for key in DERIVED:
        up = registry.get(key).upstream_params(everything[key])
        assert _digest("pixel_averages", up) == bank


# ---------------------------------------------------------------------------
# The record the API takes
# ---------------------------------------------------------------------------


def test_the_record_is_converted_from_centimetres_to_metres():
    ds = fx.make_record("full")
    assert 88 < float(ds.R.max()) < 91  # centimetres, like the files
    prepared = _pipeline.record(ds)
    expected = fx.in_metres(ds)
    assert 0.88 < float(prepared.R.max()) < 0.91
    # R and Z keep the dtype they have on disk.
    assert prepared.R.dtype == np.float32
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
    blobs_params.neighbour_step = 3  # not the fixture's, which is also the default
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


def test_the_fixtures_settings_differ_from_the_defaults_in_every_group():
    """So that a setting an adapter failed to pass on shows up as a mismatch in the equalities above,
    where the defaults would hide it."""
    assert fx.AVERAGES != pipeline.Averages()
    assert fx.TRACKING != pipeline.Tracking()
    assert fx.TRACKING.position_filter != pipeline.Tracking().position_filter
    assert fx.TDE != pipeline.Tde()
    assert fx.BLOBS != pipeline.Blobs()
    assert fx.BLOBS.gauss_fit != pipeline.Blobs().gauss_fit
    assert fx.BLOBS.taud_estimation != pipeline.Blobs().taud_estimation


def test_the_stored_blobs_are_laid_out_as_the_modules_say(full, sparse):
    bank = full.loaded["pixel_averages"]
    assert dict(bank.sizes) == {
        "ref_y": fx.NY,
        "ref_x": fx.NX,
        "y": fx.NY,
        "x": fx.NX,
        "time": fx.LAGS,
    }
    averages = {"cond_av", "cond_repr", "cross_corr"}
    assert set(bank.data_vars) == averages | {"nevents", "computed", "dead"}
    for name in averages:
        assert bank[name].dims == ("ref_y", "ref_x", "y", "x", "time")
    # Metres, and the lags in seconds around zero.
    assert 0.88 < float(bank["R"].max()) < 0.91
    assert np.isclose(float(bank["time"].values[fx.LAGS // 2]), 0.0)
    assert np.isclose(np.diff(bank["time"].values), fx.DT).all()
    # The dead reference was never computed; the flat one was, and found no events.
    assert not bank["computed"].values[1, 0]
    assert bank["nevents"].values[1, 0] == 0
    assert np.isnan(bank["cond_av"].isel(ref_y=1, ref_x=0).values).all()
    flat_x, flat_y = fx.FLAT
    assert bank["computed"].values[flat_y, flat_x]
    assert bank["nevents"].values[flat_y, flat_x] == 0
    assert (bank["nevents"].values > 0).sum() == len(live_pixels("full")) - 1

    fields = full.loaded["method_fields"]
    assert dict(fields.sizes) == {"y": fx.NY, "x": fx.NX, "time": fx.LAGS}
    per_lag = {
        f"{kind}_{track}" for track in fx.TRACKS for kind in ("pos_r", "pos_z", "fit")
    }
    assert set(fields.data_vars) == set(fx.FIELD_VARIABLES) | per_lag | {"dead"}
    dead = fields["dead"].values
    for name in fx.FIELD_VARIABLES:
        assert fields[name].dims == ("y", "x"), name
        assert np.isnan(fields[name].values[dead]).all(), name
    for name in per_lag:
        assert fields[name].dims == ("y", "x", "time"), name
    # The level exists for the centroid track alone.
    assert np.isnan(fields["level_max"].values).all()
    assert np.isnan(fields["level_2dcc"].values).all()
    assert np.isfinite(fields["level_com"].values[~dead]).any()

    blobs = sparse.loaded["blob_parameters"]
    assert dict(blobs.sizes) == {"y": fx.NY, "x": fx.NX}
    assert set(blobs.data_vars) == set(fx.BLOB_VARIABLES) | {"dead"}
    assert blobs.attrs["neighbour_step"] == fx.TRACKING.neighbour_step


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


def test_the_duration_time_scalars_are_the_names_the_taud_psd_spec_writes():
    """A number that two plots write under one name lines up on one multi-shot axis."""
    from fusion_ui.plots import spectra

    stand_in = xr.Dataset({"refx": 1, "refy": 2, "taud": 1e-5, "lam": 0.5})
    written = {name for _, _, name in spectra.scalars(stand_in)}
    assert written == {"taud_psd", "lambda_psd"} <= set(blob_parameters.SCALARS)


def test_method_fields_writes_its_names_at_every_live_pixel_and_failures_as_null(full):
    result = full.loaded["method_fields"]
    live = live_pixels("full")
    mapping = method_fields.scalars(result)
    assert {name for _, _, name in mapping} == set(METHOD_FIELDS_NAMES)
    assert {(x, y) for x, y, _ in mapping} == set(live), "a dead pixel gets no row"
    assert len(mapping) == len(live) * len(METHOD_FIELDS_NAMES)

    rows = stored_rows(full.conn, full.runs["method_fields"])
    assert set(rows) == set(mapping)
    nulls = 0
    for (x, y, name), value in rows.items():
        expected = float(result[variable_of(name)].values[y, x])
        if np.isnan(expected):
            assert value is None and np.isnan(mapping[(x, y, name)]), (x, y, name)
            nulls += 1
        else:
            assert value == expected == mapping[(x, y, name)], (x, y, name)
    assert (
        nulls
    ), "some estimates fail on a record this small, and are NULL, not missing"
    # Tried and failed is not never tried: the live pixel without events has its rows.
    flat = {n: v for (x, y, n), v in rows.items() if (x, y) == fx.FLAT}
    assert len(flat) == 20 and flat["number_events"] == 0.0 and flat["vr_com"] is None


def test_blob_parameters_writes_its_names_and_not_the_event_count(sparse):
    result = sparse.loaded["blob_parameters"]
    mapping = blob_parameters.scalars(result)
    live = live_pixels("sparse")
    assert len(mapping) == len(live) * 5 == 15
    assert {name for _, _, name in mapping} == set(BLOB_PARAMETERS_NAMES)
    assert {(x, y) for x, y, _ in mapping} == set(live)
    assert "number_events" not in {name for _, _, name in mapping}

    rows = stored_rows(sparse.conn, sparse.runs["blob_parameters"])
    assert len(rows) == 15
    assert {n for _, _, n in rows} == set(BLOB_PARAMETERS_NAMES)
    for (x, y, name), value in rows.items():
        expected = float(result[variable_of(name)].values[y, x])
        assert (value is None and np.isnan(expected)) or value == expected
    flat_x, flat_y = fx.FLAT
    assert rows[(flat_x, flat_y, "lr")] is None  # no average to fit: tried and failed
    assert rows[(1, 1, "area")] is not None


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
    from fusion_ui.views.methods import describe_cut

    title = figure.layout.title.text
    assert title.startswith("v_R of every method, on one scale")
    # Which cuts it was drawn at is the line across the top of the figure, per method.
    assert describe_cut(Cuts()) in [a.text for a in figure.layout.annotations]


def test_blob_parameters_draws_one_map_per_parameter_marking_dead_and_failed(sparse):
    figure = blob_parameters.SPEC.render(
        sparse.loaded["blob_parameters"], PARAMS["blob_parameters"](), target()
    )
    assert_a_figure(figure)
    maps = [t for t in figure.data if isinstance(t, go.Heatmap)]
    assert [t.meta["parameter"] for t in maps] == SHOWN
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
    assert len([t for t in figure.data if isinstance(t, go.Heatmap)]) == 4


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
    """A fill that names only the derived products leaves the bank to the chain: a batch job computes it
    with the first of them, and the second finds it cached."""
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
    assert len(stored_rows(tree.conn, after[(shot, "blob_parameters")])) == 3 * 5


MULTI_SHOT_APP = str(REPO / "fusion_ui" / "app.py")


@pytest.mark.slow
def test_a_multi_shot_click_on_blob_parameters_that_no_settings_go_with_shows_that_exact_run(
    tree, monkeypatch
):
    """The fixture's blob parameters carry the fits' own settings (``fx.BLOBS``), which are in no
    ``method_fields`` parameter set, so the Fields page would show other numbers than the point's. The
    click opens the single-shot page instead, which restores the run's parameters and finds the result
    in the cache: nothing is computed."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from fusion_ui import jump
    from fusion_ui.core import decimate

    st.cache_data.clear()
    st.cache_resource.clear()
    shot = tree.shots[0]
    fill(tree.conn, KEYS, shots={shot})  # what the batch leaves behind
    before = runs_of(tree.conn)
    assert set(before) == {(shot, plot) for plot in KEYS}
    blob, fields = before[(shot, "blob_parameters")], before[(shot, "method_fields")]

    # Nothing goes with the blob run, though the velocity fields were computed at the same 2DCA settings.
    found, _ = products.specs(registry)
    options = products.settings(tree.conn, found["method_fields"], "cmod")
    assert fields["params_hash"] in {o.hash for o in options}
    assert (
        jump.settings_for(
            "blob_parameters",
            blob["params_hash"],
            found,
            options,
            {fields["params_hash"]},
        )
        is None
    )

    monkeypatch.setattr(
        decimate,
        "selection_points",
        lambda event: [{"customdata": ["cmod", shot, 0.7, 1.4, 0.5]}],
    )
    app = AppTest.from_file(MULTI_SHOT_APP, default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    app.session_state["ms.scalar"] = "lr"
    app.session_state["ms.source"] = ("blob_parameters", blob["params_hash"], "apd", 1)
    app.switch_page("pages/3_multi_shot.py")
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]

    assert [t.value for t in app.title] == ["Single shot"]
    assert "fields.open" not in app.session_state
    # The run's own settings are in the form, down to the fit's...
    assert (
        widget(app, "number_input", "aspect penalty").value
        == fx.BLOBS.gauss_fit.aspect_penalty
    )
    assert widget(app, "number_input", "window").value == fx.AVERAGES.window
    # ... and the page found its result under them: it is drawn, with the provenance line of a stored
    # result, and not the "press Compute" a form that found nothing says. The jump marks the run ready,
    # so a miss would have been computed on the spot: the ledger says it was not.
    assert len(charts(app)) == 1
    assert not app.info, [i.value for i in app.info]
    assert "Recompute" in [b.label for b in app.button]
    assert runs_of(tree.conn) == before, "the page computed or recorded something"
