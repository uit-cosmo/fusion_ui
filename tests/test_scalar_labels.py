"""The label table the multi-shot axis reads: the 32 names the phase-06 products write."""

import dataclasses
import re

import pytest
import xarray as xr

import fusion_ui.plots  # noqa: F401 - registers the specs whose names are checked
from fusion_ui.core import multishot, registry, scalar_labels, seed
from fusion_ui.plots import blob_parameters, method_fields
from fusion_ui.views import numbers

#: G2's list of names, as the plan has it (docs/PHASE_06_DECORRELATION.md, "Products"): an independent
#: statement of what the table must cover, in case the specs' own ``SCALARS`` move.
METHOD_FIELDS_NAMES = (
    "vr_max vz_max nlags_max vr_com vz_com nlags_com level_com vr_2dcc vz_2dcc nlags_2dcc "
    "number_events vr3_tde vz3_tde vr2_tde vz2_tde cc_tde vr3_catde vz3_catde vr2_catde vz2_catde"
).split()
BLOB_PARAMETERS_NAMES = (
    "level area lx_c ly_c theta_c lr lz lx_f ly_f theta_f taud_psd lambda_psd"
).split()
NAMES = METHOD_FIELDS_NAMES + BLOB_PARAMETERS_NAMES

#: What has no unit: counts, fractions of a maximum, a correlation coefficient, an asymmetry.
UNITLESS = {
    "nlags_max",
    "nlags_com",
    "nlags_2dcc",
    "level_com",
    "number_events",
    "cc_tde",
    "level",
    "lambda_psd",
}


def unit_of(name):
    """The unit a label states: the last thing in it, in square brackets."""
    match = re.fullmatch(r"(.+) \[([^\[\]]+)\]", str(scalar_labels.LABELS[name]))
    assert match, f"{name}: {scalar_labels.LABELS[name]!s} does not end in [unit]"
    return match.group(2)


# -- the table covers the names, exactly ------------------------------------------------------------


def test_the_table_covers_exactly_the_32_names():
    assert len(NAMES) == 32 and len(set(NAMES)) == 32
    assert set(scalar_labels.LABELS) == set(NAMES)
    assert len(scalar_labels.LABELS) == 32


def test_the_names_are_those_the_two_specs_write():
    """So that a name a spec gains later, or loses, shows up here rather than as a raw name on an axis."""
    assert set(method_fields.SCALARS) == set(METHOD_FIELDS_NAMES)
    assert set(blob_parameters.SCALARS) == set(BLOB_PARAMETERS_NAMES)
    assert set(scalar_labels.LABELS) == set(method_fields.SCALARS) | set(
        blob_parameters.SCALARS
    )


# -- every label carries a unit, or says it has none ---------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_every_label_carries_a_unit_or_says_it_has_none(name):
    label = scalar_labels.LABELS[name]
    assert unit_of(name) in scalar_labels.UNITS
    assert label.unit in scalar_labels.UNITS
    # What is in front of the unit says what the number is, and holds no brackets of its own.
    assert label.quantity.strip() == label.quantity and len(label.quantity) > 3
    assert not re.search(r"[\[\]]", label.quantity)
    assert str(label) == f"{label.quantity} [{label.unit}]"


def test_the_numbers_with_no_unit_say_so_and_the_others_do_not():
    said = {
        name
        for name in NAMES
        if scalar_labels.LABELS[name].unit == scalar_labels.NO_UNIT
    }
    assert said == UNITLESS
    assert scalar_labels.NO_UNIT == "no unit"
    assert str(scalar_labels.LABELS["lambda_psd"]).endswith("[no unit]")


def test_the_velocities_say_their_component_and_are_in_metres_per_second():
    velocities = [n for n in NAMES if re.match(r"v[rz]", n)]
    assert len(velocities) == 14
    for name in velocities:
        label = scalar_labels.LABELS[name]
        assert label.unit == "m/s", name
        component = "v_R" if name.startswith("vr") else "v_Z"
        assert label.quantity.startswith(f"{component}, "), name


def test_the_orchestrators_example_label():
    assert scalar_labels.label("vr_com") == "v_R, 2DCA centroid [m/s]"


def test_the_units_of_the_blob_parameters_are_the_fields_pages():
    """``views.numbers`` is keyed by blob variable and says "" for a number with no unit; the two tables
    describe the same twelve quantities and must not drift apart."""
    page = {name: unit for name, _, unit in numbers.BLOB_PARAMETERS}
    for name, variable in blob_parameters.SCALARS.items():
        expected = page[variable] or scalar_labels.NO_UNIT
        assert scalar_labels.LABELS[name].unit == expected, name


def test_no_two_names_share_a_label():
    texts = [str(label) for label in scalar_labels.LABELS.values()]
    assert len(set(texts)) == len(texts)


# -- a label is per name, not per source -------------------------------------------------------------

#: What a label must not name, since it is shown for every source that writes the name.
SOURCES = (
    "pixel_averages",
    "method_fields",
    "blob_parameters",
    "two_dca",
    "fwhm_sizes",
    "gaussian_sizes",
    "taud_psd",
    "density_scan",
    "batch",
)


def test_the_shared_names_are_the_five_groups_g2_listed():
    assert set(scalar_labels.SHARED) == {
        "number_events",
        "taud_psd",
        "lambda_psd",
        "lr",
        "lz",
        "lx_f",
        "ly_f",
        "theta_f",
    }
    assert set(scalar_labels.SHARED) <= set(scalar_labels.LABELS)


def test_the_sources_of_the_shared_names_exist():
    plots = set(registry.REGISTRY) | {seed.IMPORT_PLOT}
    for name, sources in scalar_labels.SHARED.items():
        assert len(sources) >= 2, f"{name} is not shared with anything"
        assert set(sources) <= plots, name


def written_by(source):
    """The scalar names ``source`` writes: a spec's ``scalars`` on a result that has every variable any
    of the four older specs reads, or the fields of the seed's per-pixel record."""
    if source == seed.IMPORT_PLOT:
        discharge = pytest.importorskip("density_scan.discharge")
        return {f.name for f in dataclasses.fields(discharge.BlobParameters)}
    result = xr.Dataset(
        {
            "refx": 3,
            "refy": 2,
            "number_events": 5,
            "lr": 0.01,
            "lz": 0.02,
            "lx": 0.01,
            "ly": 0.02,
            "theta": 0.1,
            "taud": 2e-5,
            "lam": 0.4,
        }
    )
    written = registry.get(source).scalars(result)
    return {key if isinstance(key, str) else key[2] for key in written}


def test_every_source_listed_for_a_shared_name_writes_it():
    """So that the list this table is held to is true: a name that an older spec stops writing, or a
    source added without being listed here, is what a reader of the axis is misled by.
    """
    for name, sources in scalar_labels.SHARED.items():
        for source in sources:
            assert name in written_by(source), f"{source} does not write {name}"


def test_a_shared_names_label_names_no_source():
    for name in scalar_labels.SHARED:
        quantity = scalar_labels.LABELS[name].quantity.lower()
        for source in SOURCES:
            assert source not in quantity, f"{name}'s label names {source}"


def test_the_shared_names_keep_the_unit_every_source_gives_them():
    """Sizes in metres (``fwhm_sizes`` and ``gaussian_sizes`` divide by 100 for it, as the seed's rows
    are), the tilt in radians, the duration time in seconds, and counts and asymmetries without.
    """
    expected = {
        "number_events": scalar_labels.NO_UNIT,
        "taud_psd": "s",
        "lambda_psd": scalar_labels.NO_UNIT,
        "lr": "m",
        "lz": "m",
        "lx_f": "m",
        "ly_f": "m",
        "theta_f": "rad",
    }
    assert {name: unit_of(name) for name in expected} == expected


# -- what an axis shows --------------------------------------------------------------------------------


def test_a_name_without_a_label_keeps_its_raw_name():
    for name in ("vx_c", "area_c", "number_events_field", "dead", "not_a_scalar"):
        assert not scalar_labels.has_label(name)
        assert scalar_labels.label(name) == name
        assert (
            scalar_labels.axis_title(name, "mean over pixels")
            == f"{name} (mean over pixels)"
        )
        assert scalar_labels.axis_title(name) == name


def test_the_axis_title_is_the_label_then_how_the_pixels_were_collapsed():
    for text in multishot.AGGREGATES.values():
        assert (
            scalar_labels.axis_title("vr_com", text)
            == f"v_R, 2DCA centroid [m/s] ({text})"
        )
    assert scalar_labels.axis_title("vr_com") == "v_R, 2DCA centroid [m/s]"


# -- a title that fits the axis ------------------------------------------------------------------------


def titles():
    """Every label under every aggregate the page offers, and under none (a shot-level scalar)."""
    for name in scalar_labels.LABELS:
        yield scalar_labels.axis_title(name)
        for text in multishot.AGGREGATES.values():
            yield scalar_labels.axis_title(name, text)


def test_a_title_that_fits_is_left_as_it_is():
    short = "v_R, 2DCA centroid [m/s] (mean over pixels)"
    assert len(short) <= scalar_labels.TITLE_WIDTH
    assert scalar_labels.wrapped(short) == short
    assert scalar_labels.wrapped("vx_c (mean over pixels)") == "vx_c (mean over pixels)"
    assert scalar_labels.wrapped("") == ""


def test_a_long_title_is_broken_at_spaces_and_loses_nothing():
    for title in titles():
        lines = scalar_labels.wrapped(title).split("<br>")
        assert " ".join(lines) == title
        # Within the width, which no word of any label is wider than.
        assert all(len(line) <= scalar_labels.TITLE_WIDTH for line in lines), title
        # So that a rotated title keeps to the margin the axis makes for it.
        assert len(lines) <= 3, title


def test_the_unit_and_the_aggregate_are_never_split_across_two_lines():
    for title in titles():
        lines = scalar_labels.wrapped(title).split("<br>")
        for whole in re.findall(r"\[[^\]]*\]|\([^)]*\)", title):
            assert sum(whole in line for line in lines) == 1, (whole, lines)


def test_a_word_wider_than_the_line_is_left_whole():
    word = "x" * (scalar_labels.TITLE_WIDTH + 10)
    assert scalar_labels.wrapped(f"a {word} b").split("<br>") == ["a", word, "b"]
