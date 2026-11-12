"""The labels the multi-shot axis shows for the products' scalars, and how a long title is wrapped."""

import re

from fusion_ui.core import multishot, scalar_labels
from fusion_ui.core.scalar_labels import LABELS, TITLE_WIDTH
from fusion_ui.plots import blob_parameters, method_fields
from fusion_ui.views import numbers

UNLABELLED = ("vx_tde", "dead", "not_a_scalar")


def test_the_labelled_names_are_those_the_two_products_write():
    """So that a name a product gains shows up here, not as a raw name on an axis."""
    assert set(LABELS) == set(method_fields.SCALARS) | set(blob_parameters.SCALARS)


def test_every_label_ends_in_a_unit_or_says_it_has_none():
    for name, label in LABELS.items():
        assert label.unit in scalar_labels.UNITS, name
        assert label.quantity.strip() and not re.search(r"[\[\]]", label.quantity), name
        assert str(label) == f"{label.quantity} [{label.unit}]"
    assert str(LABELS["lambda_psd"]).endswith("[no unit]")


def test_no_two_names_share_a_label():
    texts = [str(label) for label in LABELS.values()]
    assert len(set(texts)) == len(texts)


def test_a_velocity_says_its_component_and_is_in_metres_per_second():
    for name in (n for n in LABELS if re.match(r"v[rz]", n)):
        component = "v_R" if name.startswith("vr") else "v_Z"
        assert LABELS[name].quantity.startswith(f"{component}, "), name
        assert LABELS[name].unit == "m/s", name


def test_the_units_of_the_blob_parameters_are_the_fields_pages():
    """``views.numbers`` is keyed by blob variable and says "" for a number with no unit."""
    page = {name: unit for name, _, unit in numbers.BLOB_PARAMETERS}
    for name, variable in blob_parameters.SCALARS.items():
        assert LABELS[name].unit == (page[variable] or scalar_labels.NO_UNIT), name


# -- what the picker and the axis show ---------------------------------------------------------------------


def test_a_labelled_name_shows_its_label_after_the_name_in_the_picker_and_alone_on_the_axis():
    assert scalar_labels.label("vr_com") == "v_R, 2DCA centroid [m/s]"
    assert scalar_labels.option("vr_com") == "vr_com — v_R, 2DCA centroid [m/s]"
    assert scalar_labels.axis_title("vr_com") == "v_R, 2DCA centroid [m/s]"
    for text in multishot.AGGREGATES.values():
        assert (
            scalar_labels.axis_title("vr_com", text)
            == f"v_R, 2DCA centroid [m/s] ({text})"
        )


def test_a_name_without_a_label_shows_as_its_raw_name():
    for name in UNLABELLED:
        assert scalar_labels.label(name) == name
        assert scalar_labels.option(name) == name
        assert scalar_labels.axis_title(name) == name
        assert scalar_labels.axis_title(name, "mean over pixels") == (
            f"{name} (mean over pixels)"
        )


# -- a title that fits the axis ----------------------------------------------------------------------------


def titles():
    """Every label under every aggregate the page offers, and under none (a shot-level scalar)."""
    for name in LABELS:
        yield scalar_labels.axis_title(name)
        for text in multishot.AGGREGATES.values():
            yield scalar_labels.axis_title(name, text)


def test_a_title_that_fits_is_left_as_it_is():
    short = "v_R, 2DCA centroid [m/s] (mean over pixels)"
    assert len(short) <= TITLE_WIDTH
    assert scalar_labels.wrapped(short) == short
    assert scalar_labels.wrapped("") == ""


def test_a_long_title_is_broken_at_spaces_into_at_most_three_lines_and_loses_nothing():
    for title in titles():
        lines = scalar_labels.wrapped(title).split("<br>")
        assert " ".join(lines) == title
        assert all(len(line) <= TITLE_WIDTH for line in lines), title
        assert len(lines) <= 3, title


def test_the_unit_and_the_aggregate_are_never_split_across_two_lines():
    for title in titles():
        lines = scalar_labels.wrapped(title).split("<br>")
        for whole in re.findall(r"\[[^\]]*\]|\([^)]*\)", title):
            assert sum(whole in line for line in lines) == 1, (whole, lines)


def test_a_word_wider_than_the_line_is_left_whole():
    word = "x" * (TITLE_WIDTH + 10)
    assert scalar_labels.wrapped(f"a {word} b").split("<br>") == ["a", word, "b"]
