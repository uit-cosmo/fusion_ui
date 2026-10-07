"""The three real products on J3's synthetic record, computed once for the whole test session.

``tests/product_fixtures`` makes the record (3 x 3 pixels, centimetres, with its dead-pixel mask stored) and
the settings. This runs the real specs on it through the store, the way a batch job does: the bank with
``batch=True``, the products built on it without. What comes back is a *deployment* a page can read: a data
tree with the record's file indexed, a state database and a result cache, holding

- ``pixel_averages`` under ``fx.pixel_averages_params()``;
- ``method_fields`` under ``fx.method_fields_params()``, which moves a knob in every group of settings, so
  that a setting the page fails to carry shows up as another hash;
- ``blob_parameters`` under the settings a person gets from ``precompute method_fields blob_parameters
  --params-json short.json``: the 2DCA's ``averages`` shared with the velocity fields and everything else
  at its default. That is stated here, independently of ``views.products.related_params``, which a test
  then holds against it. (``fx.blob_parameters_params()`` also moves the ellipse fit's and the duration
  time's own knobs, which no setting of the velocity fields can name.)

It takes about a quarter of a minute, so it is done once, on the first use, and tests only read: a test
that changes the ledger works on a copy (:meth:`Deployment.copy`).
"""

import dataclasses
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

import fusion_ui.plots  # noqa: F401 - registers the real specs
from fusion_ui.core import catalog, db, registry, store
from fusion_ui.plots.blob_parameters import BlobParametersParams
from tests import product_fixtures as fx

SHOT = fx.SHOT
KEYS = ("pixel_averages", "method_fields", "blob_parameters")


def blob_params():
    """The blob parameters of the velocity fields' settings, as a person's ``--params-json`` makes them."""
    return BlobParametersParams(
        averages=fx.AVERAGES, neighbour_step=fx.TRACKING.neighbour_step
    )


def settings():
    """``{key: params}`` the deployment's three products were computed under."""
    return {
        "pixel_averages": fx.pixel_averages_params(),
        "method_fields": fx.method_fields_params(),
        "blob_parameters": blob_params(),
    }


@dataclass
class Deployment:
    root: Path
    cache: Path  # where the blobs are; the ledger holds their absolute paths
    record: object  # the record in memory, in centimetres, as the batch job had it
    runs: dict  # {key: the ledger row as a dict}

    @property
    def data(self):
        return self.root / "alcator"

    @property
    def path(self):
        """The record's file, indexed under the shot."""
        return self.data / "apd" / f"apd_{SHOT}_preprocessed.nc"

    @property
    def database(self):
        return self.root / "state" / "shot_explorer.sqlite"

    def use(self, monkeypatch):
        """Point the app's configuration at this deployment, for one test."""
        monkeypatch.setenv("FUSION_DATA_FOLDER", str(self.data))
        monkeypatch.setenv(
            "FUSION_DISCHARGE_DB", str(self.root / "no_such_discharges.json")
        )
        monkeypatch.setenv("FUSION_UI_DB", str(self.database))
        monkeypatch.setenv("FUSION_UI_CACHE", str(self.cache))
        monkeypatch.setenv("FUSION_MACHINE", "cmod")
        return self

    def conn(self):
        return db.connect(self.database)

    def products(self):
        """``{key: dataset}`` as the page reads them: out of the store, from disk."""
        conn = self.conn()
        try:
            return {key: store.load_result(conn, self.runs[key]) for key in KEYS}
        finally:
            conn.close()

    def copy(self, where):
        """The ledger and the data tree in ``where``, to change; the blobs stay where they are, read only.

        The database holds the blobs' absolute paths, so a copy reads the same files and cannot write to
        them. Only the index is rebuilt, from the copy's own data tree.
        """
        root = Path(where)
        shutil.copytree(self.data, root / "alcator")
        shutil.copytree(self.database.parent, root / "state")
        return dataclasses.replace(self, root=root)


def _compute(root, target_path):
    """The three products through the store, with the record in memory."""
    record = fx.make_record("full")
    record.to_netcdf(target_path)
    runs = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("FUSION_UI_CACHE", str(root / "cache"))
        patch.setenv("FUSION_MACHINE", "cmod")
        conn = db.open_db(root / "state" / "shot_explorer.sqlite")
        try:
            catalog.rescan(conn, str(root / "alcator"), "cmod", None)
            target = registry.Target(
                "cmod",
                SHOT,
                "apd",
                True,
                str(target_path),
                float("nan"),
                float("nan"),
                "none",
            )
            for key, params in settings().items():
                result, run = store.result(
                    conn,
                    registry.get(key),
                    target,
                    params,
                    record,
                    batch=key == "pixel_averages",
                )
                assert run["status"] == "ok", run["error"]
                runs[key] = dict(run)
        finally:
            conn.close()
    return record, runs


_DEPLOYMENT = None


def deployment(tmp_path_factory):
    """The session's deployment, computed on first use. Read it; copy it to change it."""
    global _DEPLOYMENT
    if _DEPLOYMENT is None:
        root = tmp_path_factory.mktemp("real_products")
        path = root / "alcator" / "apd" / f"apd_{SHOT}_preprocessed.nc"
        path.parent.mkdir(parents=True)
        record, runs = _compute(root, path)
        _DEPLOYMENT = Deployment(
            root=root, cache=root / "cache", record=record, runs=runs
        )
    return _DEPLOYMENT
