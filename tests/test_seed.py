"""Importing density_scan's results.json into the scalar store.

Against a hand-built file, not the real 3 MB one: the shapes that matter are a
missing pixel, a NaN, and a second run over the same content.
"""

import json

import pytest

from fusion_ui.core import seed, store

# The real file's shape: {shot: {plasma_discharge: {...},
#                                blob_params: {refx: {refy: {...} | null}}}}
# and it is written with bare NaN literals, which Python's json reads and a
# strict parser does not -- the fixture keeps that.
RESULTS = {
    "1160616027": {
        "plasma_discharge": {
            "shot_number": 1160616027,
            "plasma_current": 0.51,
            "line_averaged_density": 2.82,
            "greenwald_fraction": 0.8,
            "t_start": 1.15,
            "t_end": 1.45,
            "mlp_mode": "-",
            "comment": "L",
        },
        "blob_params": {
            "6": {
                "6": {"vx_c": 566.6, "taud_psd": 1.98e-05, "number_events": 801},
                "5": {"vx_c": float("nan"), "taud_psd": 2.0e-05, "number_events": 12},
            },
            "4": {"6": None},  # a pixel that was never analysed
        },
    },
    "1110201007": {
        "plasma_discharge": {
            "shot_number": 1110201007,
            "plasma_current": 0.93,
            "line_averaged_density": 1.10,
            "greenwald_fraction": float("nan"),
            "t_start": 1.1,
            "t_end": 1.4,
            "mlp_mode": "none",
            "comment": "EDA-H",
        },
        "blob_params": {"6": {"6": {"vx_c": 100.0, "taud_psd": 3.0e-05}}},
    },
}


@pytest.fixture
def results_json(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(RESULTS, indent=4))
    return str(path)


def test_every_analysed_pixel_becomes_scalar_rows(conn, results_json):
    stats = seed.import_results(conn, results_json, machine="cmod")

    assert (stats.shots, stats.pixels) == (2, 3)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
    rows = {
        (r["x"], r["y"], r["name"]): r["value"]
        for r in conn.execute(
            "SELECT s.* FROM scalars s JOIN runs r ON r.id = s.run_id"
            " WHERE r.shot = 1160616027"
        )
    }
    assert rows[(6, 6, "vx_c")] == 566.6
    assert rows[(6, 6, "number_events")] == 801
    assert (4, 6, "vx_c") not in rows, "an unanalysed pixel must not become a row"


def test_a_nan_is_stored_as_null_not_as_a_number(conn, results_json):
    seed.import_results(conn, results_json, machine="cmod")
    value = conn.execute(
        "SELECT value FROM scalars WHERE x = 6 AND y = 5 AND name = 'vx_c'"
    ).fetchone()[0]
    assert value is None


def test_the_rows_are_attributed_to_the_import_not_to_an_analysis(conn, results_json):
    seed.import_results(conn, results_json, machine="cmod")
    run = conn.execute("SELECT * FROM runs LIMIT 1").fetchone()

    assert run["plot"] == seed.IMPORT_PLOT
    assert run["code_version"] == "imported"
    assert run["blob_path"] is None
    assert run["diagnostic"] == "apd"
    assert run["preprocessed"] == 1, "density_scan ran on the preprocessed file"

    # The parameter set records provenance, not settings that were never kept.
    params = json.loads(store.params_json(conn, run["params_hash"]))
    assert params["plot"] == seed.IMPORT_PLOT
    assert params["params"]["values"]["source"] == "results.json"


def test_re_importing_the_same_file_is_a_no_op(conn, results_json):
    seed.import_results(conn, results_json, machine="cmod")
    before = conn.execute("SELECT COUNT(*) FROM scalars").fetchone()[0]

    again = seed.import_results(conn, results_json, machine="cmod")
    assert (again.shots, again.skipped) == (0, 2)
    assert conn.execute("SELECT COUNT(*) FROM scalars").fetchone()[0] == before
    assert conn.execute("SELECT COUNT(*) FROM param_sets").fetchone()[0] == 1


def test_a_changed_file_imports_beside_the_old_rows(conn, results_json, tmp_path):
    """The parameter set is the file's digest, so two versions of results.json
    are two distinguishable imports rather than a silent overwrite."""
    seed.import_results(conn, results_json, machine="cmod")

    changed = dict(RESULTS)
    changed["1110201007"]["blob_params"]["6"]["6"]["vx_c"] = 999.0
    other = tmp_path / "results2.json"
    other.write_text(json.dumps(changed, indent=4))

    seed.import_results(conn, str(other), machine="cmod")
    hashes = {r[0] for r in conn.execute("SELECT DISTINCT params_hash FROM runs")}
    assert len(hashes) == 2
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 4


def test_the_seeded_scalars_are_visible_to_the_multi_shot_query(conn, results_json):
    seed.import_results(conn, results_json, machine="cmod")
    frame = store.scalar_frame(conn, names=["taud_psd"], plot=seed.IMPORT_PLOT)
    assert sorted(frame["shot"]) == [1110201007, 1160616027, 1160616027]
    assert set(frame["name"]) == {"taud_psd"}


def test_a_missing_file_is_reported_not_traced_back(conn, tmp_path):
    with pytest.raises(FileNotFoundError):
        seed.import_results(conn, str(tmp_path / "nope.json"), machine="cmod")
