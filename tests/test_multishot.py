"""The pure multi-shot helpers: collapsing per-pixel scalars to one per shot."""

import pandas as pd
import pytest

from fusion_ui.core import multishot

COLUMNS = [
    "machine",
    "shot",
    "diagnostic",
    "preprocessed",
    "plot",
    "params_hash",
    "x",
    "y",
    "name",
    "value",
]


def frame(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def two_shot_frame():
    """Two shots, two pixels each, under one source; plus a NaN to be dropped."""
    return frame(
        [
            ("cmod", 1, "apd", 1, "velocity_contour", "h", 0, 0, "vx_c", 10.0),
            ("cmod", 1, "apd", 1, "velocity_contour", "h", 0, 1, "vx_c", 20.0),
            ("cmod", 2, "apd", 1, "velocity_contour", "h", 0, 0, "vx_c", 30.0),
            ("cmod", 2, "apd", 1, "velocity_contour", "h", 0, 1, "vx_c", float("nan")),
            ("cmod", 1, "apd", 1, "velocity_contour", "h", 0, 0, "area_c", 1.0),
        ]
    )


def test_distinct_names_are_sorted():
    assert multishot.distinct_names(two_shot_frame()) == ["area_c", "vx_c"]


def test_distinct_sources_group_by_plot_hash_diagnostic_and_preprocessed():
    rows = two_shot_frame()
    extra = frame(
        [("cmod", 3, "apd", 1, "velocity_contour", "other", 0, 0, "vx_c", 5.0)]
    )
    sources = multishot.distinct_sources(pd.concat([rows, extra]), "vx_c")
    assert sources == [
        ("velocity_contour", "h", "apd", 1),
        ("velocity_contour", "other", "apd", 1),
    ]


def test_mean_aggregates_over_pixels():
    result = multishot.aggregate(
        two_shot_frame(), ("velocity_contour", "h", "apd", 1), "vx_c", "mean"
    )
    values = dict(zip(result["shot"], result["value"]))
    assert values == {1: 15.0, 2: 30.0}


def test_median_and_maximum():
    source = ("velocity_contour", "h", "apd", 1)
    median = multishot.aggregate(two_shot_frame(), source, "vx_c", "median")
    maximum = multishot.aggregate(two_shot_frame(), source, "vx_c", "maximum")
    assert dict(zip(median["shot"], median["value"])) == {1: 15.0, 2: 30.0}
    assert dict(zip(maximum["shot"], maximum["value"])) == {1: 20.0, 2: 30.0}


def test_a_nan_is_dropped_not_averaged():
    """Shot 2 has one NaN pixel and one 30.0 pixel; the mean must be 30, not 15."""
    result = multishot.aggregate(
        two_shot_frame(), ("velocity_contour", "h", "apd", 1), "vx_c", "mean"
    )
    assert result.set_index("shot").loc[2, "value"] == 30.0


def test_fixed_pixel_selects_one_pixel():
    result = multishot.aggregate(
        two_shot_frame(),
        ("velocity_contour", "h", "apd", 1),
        "vx_c",
        "pixel",
        pixel=(0, 1),
    )
    assert dict(zip(result["shot"], result["value"])) == {1: 20.0}


def test_fixed_pixel_requires_a_pixel():
    with pytest.raises(TypeError, match="pixel"):
        multishot.aggregate(
            two_shot_frame(), ("velocity_contour", "h", "apd", 1), "vx_c", "pixel"
        )


def test_shot_level_scalars_pass_through_unchanged():
    rows = frame(
        [
            ("cmod", 1, "apd", 1, "plot", "h", -1, -1, "total", 6.0),
            ("cmod", 2, "apd", 1, "plot", "h", -1, -1, "total", 9.0),
        ]
    )
    result = multishot.aggregate(
        rows, ("plot", "h", "apd", 1), "total", "pixel", pixel=(0, 0)
    )
    assert dict(zip(result["shot"], result["value"])) == {1: 6.0, 2: 9.0}


def test_pixel_choices_ignore_the_shot_level_sentinel():
    rows = frame(
        [
            ("cmod", 1, "apd", 1, "plot", "h", 3, 4, "vx_c", 1.0),
            ("cmod", 2, "apd", 1, "plot", "h", -1, -1, "vx_c", 2.0),
        ]
    )
    assert multishot.pixel_choices(rows, ("plot", "h", "apd", 1), "vx_c") == [(3, 4)]


def test_pixel_choices_return_only_pairs_that_occur():
    """xs=[0, 1] and ys=[0, 1] with only (0, 1) and (1, 0) present must not let a
    picker offer the impossible (0, 0) / (1, 1)."""
    rows = frame(
        [
            ("cmod", 1, "apd", 1, "plot", "h", 0, 1, "vx_c", 1.0),
            ("cmod", 2, "apd", 1, "plot", "h", 1, 0, "vx_c", 2.0),
        ]
    )
    assert multishot.pixel_choices(rows, ("plot", "h", "apd", 1), "vx_c") == [
        (0, 1),
        (1, 0),
    ]


def test_is_shot_level():
    rows = frame([("cmod", 1, "apd", 1, "plot", "h", -1, -1, "total", 6.0)])
    assert multishot.is_shot_level(rows, ("plot", "h", "apd", 1), "total")


def test_with_metadata_joins_and_leaves_the_missing_ones_blank():
    aggregated = pd.DataFrame(
        {"machine": ["cmod", "cmod"], "shot": [1, 3], "value": [10.0, 20.0]}
    )
    table = pd.DataFrame(
        {
            "machine": ["cmod", "cmod"],
            "shot": [1, 2],
            "f_GW": [0.5, 0.8],
            "n_e_bar": [1.0, 2.0],
            "I_p": [0.5, 0.9],
            "mode": ["L", "H"],
            "has_metadata": [True, True],
        }
    )
    merged = multishot.with_metadata(aggregated, table)
    assert merged.loc[merged["shot"] == 1, "mode"].iloc[0] == "L"
    assert merged.loc[merged["shot"] == 3, "f_GW"].isna().all()
    assert list(merged["shot"]) == [1, 3]
