"""One trace from any diagnostic: naming it, and materialising it.

Pure logic over the tiny real fixtures -- ``extract`` branches on the channel
kind the way the statistics page needs it to, and ``common_grid`` is what makes
a CCF between two different files possible at all.
"""

import numpy as np
import pytest

from fusion_ui.core import loader, traces


def _ref(**overrides):
    base = dict(
        machine="cmod", shot=1234, diagnostic="apd", preprocessed=False,
        channel=("pixel", 2, 1),
    )
    base.update(overrides)
    return traces.TraceRef(**base)


def test_channels_lists_every_pixel_of_an_imaging_file(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path))
    found = traces.channels(ds, "apd")
    assert len(found) == ds.sizes["x"] * ds.sizes["y"]
    assert ("pixel", 0, 0) in found
    assert ("pixel", ds.sizes["x"] - 1, ds.sizes["y"] - 1) in found


def test_channels_lists_every_probe_combination(asp_dataset_path):
    ds = loader.open_dataset(str(asp_dataset_path))
    assert traces.channels(ds, "asp") == [
        ("probe", "Vf", 0),
        ("probe", "Vf", 1),
        ("probe", "ne", 0),
        ("probe", "ne", 1),
    ]


def test_extract_imaging_returns_the_windowed_samples(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path))
    trace = traces.extract(ds, _ref(), 1.005, 1.01)
    time = np.asarray(ds["time"].values)
    assert trace.time.shape == trace.value.shape == ((time >= 1.005) & (time <= 1.01)).sum()
    assert trace.time[0] >= 1.005 and trace.time[-1] <= 1.01
    assert np.isfinite(trace.dt) and trace.dt > 0
    assert trace.coords["R"] == pytest.approx(85.0)
    assert trace.coords["rho"] is None


def test_extract_probe_masks_on_that_channels_own_axis(asp_dataset_path):
    ds = loader.open_dataset(str(asp_dataset_path))
    ref = _ref(shot=5678, diagnostic="asp", channel=("probe", "ne", 0))
    trace = traces.extract(ds, ref, 0.52, 0.55)
    assert trace.time.shape == trace.value.shape
    assert trace.time.min() >= 0.52 and trace.time.max() <= 0.55
    assert np.isfinite(trace.dt) and trace.dt > 0
    # The ragged axis is respected: position 1 has more samples than 0.
    other = traces.extract(
        ds, _ref(shot=5678, diagnostic="asp", channel=("probe", "ne", 1)),
        0.0, 1.0,
    )
    whole = traces.extract(ds, ref, 0.0, 1.0)
    assert other.time.shape != whole.time.shape


def test_extract_returns_none_when_the_window_misses_the_record(
    apd_dataset_path, asp_dataset_path
):
    imaging = loader.open_dataset(str(apd_dataset_path))
    assert traces.extract(imaging, _ref(), 5.0, 6.0) is None
    probe = loader.open_dataset(str(asp_dataset_path))
    ref = _ref(shot=5678, diagnostic="asp", channel=("probe", "ne", 0))
    assert traces.extract(probe, ref, 5.0, 6.0) is None


def test_extract_rejects_an_unknown_channel_kind(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path))
    with pytest.raises(KeyError):
        traces.extract(ds, _ref(channel=("mystery", 0)), 1.0, 1.01)


def test_common_grid_is_a_noop_when_the_bases_match(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path))
    first = traces.extract(ds, _ref(), 1.0, 1.02)
    second = traces.extract(ds, _ref(channel=("pixel", 0, 0)), 1.0, 1.02)
    gridded = traces.common_grid([first, second], first)
    assert gridded[0] is first and gridded[1] is second


def test_common_grid_interpolates_onto_the_reference(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path))
    reference = traces.extract(ds, _ref(), 1.0, 1.02)
    shifted = traces.extract(ds, _ref(channel=("pixel", 0, 0)), 1.0, 1.02)
    import dataclasses

    shifted = dataclasses.replace(shifted, time=shifted.time + 1e-6)
    (gridded,) = traces.common_grid([shifted], reference)
    assert gridded is not shifted
    np.testing.assert_allclose(gridded.time, reference.time)


def _synthetic_trace(t_start, t_end, n=5, channel=("pixel", 2, 1)):
    time = np.linspace(t_start, t_end, n)
    dt = float((t_end - t_start) / (n - 1)) if n > 1 else float("nan")
    return traces.Trace(
        ref=_ref(channel=channel), time=time, value=time.copy(),
        dt=dt, coords={}, label="",
    )


def test_common_grid_masks_non_overlapping_regions_to_nan():
    """Points outside a trace's own range are NaN, not flat extrapolation.

    ``np.interp`` clamps outside its range; two barely-overlapping windows
    must read as missing data rather than a bogus flat line with a spurious
    CCF peak.
    """
    reference = _synthetic_trace(0.0, 2.0)
    other = _synthetic_trace(1.0, 3.0)
    (gridded,) = traces.common_grid([other], reference)
    np.testing.assert_allclose(gridded.time, reference.time)
    # Reference base is [0, 0.5, 1, 1.5, 2]: below the trace's range -> NaN.
    assert np.isnan(gridded.value[:2]).all()
    np.testing.assert_allclose(gridded.value[2:], [1.0, 1.5, 2.0])

    disjoint = _synthetic_trace(10.0, 12.0)
    (gridded_disjoint,) = traces.common_grid([disjoint], reference)
    assert np.isnan(gridded_disjoint.value).all()


def test_common_grid_handles_a_degenerate_time_base():
    reference = _synthetic_trace(0.0, 2.0)
    single = _synthetic_trace(1.0, 1.0, n=1)
    (gridded,) = traces.common_grid([single], reference)
    # Only the exactly-matching sample is filled; the rest stays missing.
    assert gridded.value[2] == 1.0
    assert np.isnan(np.delete(gridded.value, 2)).all()


def _trace_with(coords, channel=("pixel", 2, 1)):
    return traces.Trace(
        ref=_ref(channel=channel), time=np.arange(4.0),
        value=np.arange(4.0), dt=0.25, coords=coords, label="",
    )


def test_label_styles():
    coords = {"R": 89.12, "Z": -2.34, "rho": None, "dr_sep": 1.234}
    trace = _trace_with(coords)
    assert traces.label(trace, "channel") == "1234 apd pixel (2, 1)"
    assert traces.label(trace, "rz") == "1234 apd R=89.12, Z=-2.34"
    assert traces.label(trace, "magnetic") == "1234 apd R-R_sep=+1.23 cm"


def test_label_probe_styles():
    coords = {"R": None, "Z": None, "rho": 0.424, "dr_sep": None}
    trace = _trace_with(coords, channel=("probe", "ne", 0))
    assert traces.label(trace, "channel") == "1234 apd ne_0"
    assert traces.label(trace, "magnetic") == "1234 apd rho=0.42"
    with pytest.raises(ValueError):
        traces.label(trace, "flux")


def test_label_falls_back_when_a_coordinate_is_none():
    assert traces.label(_trace_with({}), "magnetic") == "1234 apd pixel (2, 1)"
    assert traces.label(_trace_with({}), "rz") == "1234 apd pixel (2, 1)"
    probe = _trace_with({}, channel=("probe", "ne", 0))
    assert traces.label(probe, "magnetic") == "1234 apd ne_0"


def test_ref_keys_match_the_target_convention():
    ref = _ref()
    assert ref.target_key == "cmod_1234_apd_r"
    assert ref.key == "cmod_1234_apd_r_pixel_2_1"
    assert _ref(preprocessed=True).target_key == "cmod_1234_apd_p"
