"""``fusion_ui.api`` on the real products of J3's synthetic record, as a script would read them.

The deployment (``tests/real_fixtures``) holds ``blob_parameters`` and ``method_fields`` for one 3 x 3 shot,
under settings that are not the defaults, with a dead pixel at (0, 0) of the record. What is checked is that
the API finds them under the settings a person names, hands back the stored values for the pixels asked for,
says what to run when something is not stored, and writes nothing.
"""

import hashlib

import numpy as np
import pytest

from fusion_ui import api
from fusion_ui.core import db
from tests import product_fixtures as fx
from tests import real_fixtures as real

SHOT = real.SHOT


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    return real.deployment(tmp_path_factory)


@pytest.fixture
def served(deployment, monkeypatch):
    return deployment.use(monkeypatch)


@pytest.fixture(scope="module")
def stored(deployment):
    return deployment.products()


BLOB = real.blob_params()
FIELDS = fx.method_fields_params()


def test_shots_lists_the_stored_shot_under_its_settings(served):
    assert api.shots(settings=BLOB) == [SHOT]
    assert api.shots("velocity_fields", settings=FIELDS) == [SHOT]


def test_a_dict_of_changes_finds_the_same_settings(served):
    changes = {"averages": {"window": fx.AVERAGES.window, "threshold": fx.AVERAGES.threshold}}
    assert api.shots(settings=changes) == api.shots(settings=BLOB) == [SHOT]


def test_the_default_settings_have_nothing_stored(served):
    assert api.shots() == []
    with pytest.raises(api.NotStored) as raised:
        api.blob_parameters(SHOT)
    assert raised.value.command == f"fusion-ui precompute blob_parameters --shot {SHOT}"


def test_the_whole_shot_is_the_stored_blob(served, stored):
    ds = api.blob_parameters(SHOT, settings=BLOB)
    blob = stored["blob_parameters"]
    assert ds.sizes == blob.sizes
    for name in blob.data_vars:
        np.testing.assert_array_equal(ds[name].values, blob[name].values)
    assert int(ds.shot) == SHOT
    assert ds["area"].attrs["units"] == "m²"
    assert ds["nevents"].attrs["units"] == "1"


def test_pixels_are_x_then_y(served, stored):
    blob = stored["blob_parameters"]
    ds = api.blob_parameters(SHOT, pixels=[(2, 1), (1, 1)], settings=BLOB)
    assert ds.sizes["pixel"] == 2
    assert list(ds.x.values) == [2, 1] and list(ds.y.values) == [1, 1]
    np.testing.assert_array_equal(ds["R"].values, blob["R"].isel(y=1, x=[2, 1]).values)
    np.testing.assert_array_equal(ds["lr"].values, blob["lr"].isel(y=1, x=[2, 1]).values)


def test_a_pixel_off_the_array_raises(served):
    with pytest.raises(ValueError, match="outside"):
        api.blob_parameters(SHOT, pixels=[(3, 0)], settings=BLOB)


def test_velocity_fields_on_pixels_keep_their_tracks(served, stored):
    ds = api.velocity_fields(SHOT, pixels=[(1, 1)], settings=FIELDS)
    fields = stored["method_fields"]
    np.testing.assert_array_equal(ds["vr_max"].values, fields["vr_max"].isel(y=[1], x=[1]).values.ravel())
    assert ds["pos_r_com"].dims == ("pixel", "time")


def test_table_has_a_row_per_live_pixel(served, stored):
    blob = stored["blob_parameters"]
    df = api.table(settings=BLOB)
    live = int((~blob["dead"].values.astype(bool)).sum())
    assert len(df) == live
    assert list(df.columns[:6]) == ["shot", "x", "y", "R", "Z", "dead"]
    row = df[(df.x == 1) & (df.y == 1)].iloc[0]
    assert row["lr"] == pytest.approx(float(blob["lr"].isel(y=1, x=1)), nan_ok=True)
    assert len(api.table(settings=BLOB, dead=True)) == blob.sizes["x"] * blob.sizes["y"]


def test_table_of_velocity_fields_leaves_out_the_tracks(served):
    df = api.table("velocity_fields", settings=FIELDS)
    assert "vr_max" in df and "pos_r_com" not in df


def test_it_writes_nothing(served, deployment):
    before = hashlib.sha1(deployment.database.read_bytes()).hexdigest()
    api.shots()
    api.table(settings=BLOB)
    with pytest.raises(api.NotStored):
        api.blob_parameters(SHOT)
    assert hashlib.sha1(deployment.database.read_bytes()).hexdigest() == before


def test_an_older_schema_is_refused(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = db.connect(path)
    conn.execute("PRAGMA user_version = 1")
    conn.close()
    with pytest.raises(RuntimeError, match="init-db"):
        db.open_readonly(path)


def test_unknown_settings_fields_raise(served):
    with pytest.raises((TypeError, ValueError)):
        api.shots(settings={"no_such_knob": 1})
