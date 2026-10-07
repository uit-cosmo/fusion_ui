"""Where a click on a multi-shot point goes: which settings the Fields page opens on, and what it is asked.

Pure, on the real specs: the question is how a ``blob_parameters`` run maps back to ``method_fields``
settings through ``views.products.related_params``, so the specs' own parameter classes and hashing are
what is tested, not stand-ins.
"""

import dataclasses

import pytest

import fusion_ui.plots  # noqa: F401 - registers the three products
from fusion_ui import jump
from fusion_ui.core import params_ui, registry
from fusion_ui.views import products as prod


@pytest.fixture
def found():
    specs, missing = prod.specs(registry)
    assert not missing
    return specs


def method_params(found, **changes):
    """The default ``method_fields`` parameters with ``section.field = value`` changes."""
    params = found["method_fields"].params()
    return with_changes(params, changes)


def blob_params(found, **changes):
    return with_changes(found["blob_parameters"].params(), changes)


def with_changes(params, changes):
    for path, value in changes.items():
        *parents, leaf = path.split("__")
        if not parents:
            params = dataclasses.replace(params, **{leaf: value})
            continue
        chain = [params]
        for name in parents:
            chain.append(getattr(chain[-1], name))
        replaced = dataclasses.replace(chain[-1], **{leaf: value})
        for parent, name in zip(reversed(chain[:-1]), reversed(parents)):
            replaced = dataclasses.replace(parent, **{name: replaced})
        params = replaced
    return params


def setting(params, default=False):
    """What the settings picker offers for a ``method_fields`` parameter set."""
    digest, text = params_ui.hash_params("method_fields", params)
    return prod.Setting(
        hash=digest,
        params_json=text,
        is_default=default,
        diff=(),
        shots=1,
        label="default" if default else digest[:8],
    )


def blob_hash(params):
    return params_ui.hash_params("blob_parameters", params)[0]


# -- the two sources the Fields page opens on, and every other -----------------------------------------


def test_only_the_two_products_open_the_fields_page():
    assert jump.FIELDS_PLOTS == ("method_fields", "blob_parameters")
    assert jump.opens_fields("method_fields") and jump.opens_fields("blob_parameters")
    for plot in (
        "pixel_averages",
        "velocity_contour",
        "velocity_field",
        "taud_psd",
        "density_scan_import",
        "dead_pixels",
    ):
        assert not jump.opens_fields(plot), plot


def test_the_selection_is_the_shared_contract():
    source = ("method_fields", "h" * 40, "apd", 1)
    assert jump.selection("cmod", 1160616027, source) == {
        "machine": "cmod",
        "shot": 1160616027,
        "diagnostic": "apd",
        "preprocessed": True,
    }
    raw = ("taud_psd", "h", "apd", 0)
    assert jump.selection("cmod", "1160616027", raw)["preprocessed"] is False
    assert jump.selection("cmod", "1160616027", raw)["shot"] == 1160616027


# -- a method_fields point: its own settings --------------------------------------------------------


def test_a_method_fields_point_opens_on_its_own_parameters(found):
    default = setting(method_params(found), default=True)
    short = setting(method_params(found, averages__window=40))
    options = [default, short]
    for option in options:
        assert (
            jump.settings_for("method_fields", option.hash, found, options)
            == option.hash
        )
    # It is the point's own hash whether or not the picker has it, or the shot has a run under it:
    # the page checks the hash against what it offers and falls back to the default.
    assert jump.settings_for("method_fields", "0" * 40, found, options) == "0" * 40


# -- a blob_parameters point: the settings the blob parameters go with -----------------------------------


def test_a_default_blob_point_opens_on_the_default_settings(found):
    default = setting(method_params(found), default=True)
    short = setting(method_params(found, averages__window=40))
    wanted = blob_hash(blob_params(found))
    assert (
        jump.settings_for("blob_parameters", wanted, found, [default, short])
        == default.hash
    )


def test_a_blob_point_computed_under_other_2dca_settings_opens_on_those(found):
    default = setting(method_params(found), default=True)
    short = setting(method_params(found, averages__window=40))
    wanted = blob_hash(blob_params(found, averages__window=40))
    assert wanted != blob_hash(blob_params(found))
    assert (
        jump.settings_for("blob_parameters", wanted, found, [default, short])
        == short.hash
    )


def test_a_blob_point_computed_at_another_neighbour_step_opens_on_that_step(found):
    """The contour's neighbour step is the one tracking setting the blob parameters read."""
    default = setting(method_params(found), default=True)
    stepped = setting(method_params(found, tracking__neighbour_step=2))
    wanted = blob_hash(blob_params(found, neighbour_step=2))
    assert (
        jump.settings_for("blob_parameters", wanted, found, [default, stepped])
        == stepped.hash
    )


def test_the_hash_is_asked_of_the_pages_own_mapping(found):
    """Not restated here: whatever ``related_params`` maps a setting to is what a point is matched on."""
    for changes in (
        {},
        {"averages__threshold": 3.0},
        {"tracking__neighbour_step": 2},
        {"tde__min_cc": 0.4},
        {"tracking__cross_corr_mask_signal_factor": 0.8},
    ):
        params = method_params(found, **changes)
        option = setting(params)
        related = prod.related_params(found, params)["blob_parameters"]
        assert jump.related_blob_hash(found, option) == blob_hash(related), changes


# -- several settings go with one blob run ---------------------------------------------------------------


@pytest.fixture
def several(found):
    """Three settings that give the same blob run, and one that gives another.

    The blob parameters read only the 2DCA settings and the neighbour step, so a ``method_fields`` set
    that differs from the default in its TDE or in the filters of a track maps to the default's blob run.
    """
    options = [
        setting(method_params(found), default=True),
        setting(method_params(found, tde__min_cc=0.4)),
        setting(method_params(found, tracking__cross_corr_mask_signal_factor=0.8)),
        setting(method_params(found, averages__window=40)),
    ]
    wanted = blob_hash(blob_params(found))
    assert [jump.related_blob_hash(found, o) for o in options[:3]] == [wanted] * 3
    assert jump.related_blob_hash(found, options[3]) != wanted
    return options, wanted


def test_the_default_wins_among_several_when_the_shot_has_fields_under_it(
    found, several
):
    options, wanted = several
    default, tde, filters, short = options
    good = {default.hash, tde.hash, filters.hash, short.hash}
    assert (
        jump.settings_for("blob_parameters", wanted, found, options, good)
        == default.hash
    )


def test_a_setting_with_fields_on_the_shot_beats_one_without(found, several):
    """The page has velocity fields to draw beside the blob parameters, instead of "not computed"."""
    options, wanted = several
    default, tde, filters, short = options
    assert (
        jump.settings_for("blob_parameters", wanted, found, options, {tde.hash})
        == tde.hash
    )
    assert (
        jump.settings_for("blob_parameters", wanted, found, options, {filters.hash})
        == filters.hash
    )
    # The first of those that has fields, in the picker's order, and never a setting that does not match
    # even though the shot has fields under it.
    assert (
        jump.settings_for(
            "blob_parameters", wanted, found, options, {filters.hash, tde.hash}
        )
        == tde.hash
    )
    assert (
        jump.settings_for("blob_parameters", wanted, found, options, {short.hash})
        == default.hash
    )


def test_with_fields_nowhere_the_first_match_is_still_chosen(found, several):
    """The blob parameters are what the person clicked: they are the ones shown, the fields are missing."""
    options, wanted = several
    assert (
        jump.settings_for("blob_parameters", wanted, found, options, ())
        == options[0].hash
    )
    assert (
        jump.settings_for("blob_parameters", wanted, found, options[1:], ())
        == options[1].hash
    )


# -- none goes with it: the click goes to the single-shot page, on that exact run ----------------------------


def test_a_blob_point_nobody_has_velocity_fields_for_has_no_settings(found):
    """Its 2DCA settings were run for the blob parameters alone: no method_fields set maps to them."""
    options = [setting(method_params(found), default=True)]
    wanted = blob_hash(blob_params(found, averages__window=30))
    assert jump.settings_for("blob_parameters", wanted, found, options) is None


def test_a_blob_point_with_its_own_fit_settings_has_no_settings(found):
    """The ellipse fit and the duration time fit are the blob product's own: no velocity setting
    carries them, so none can be said to go with a run that changed them. The Fields page would show
    other blob numbers than the one clicked, so the page sends that click to the single-shot page.
    """
    options = [
        setting(method_params(found), default=True),
        setting(method_params(found, averages__window=40)),
    ]
    changed = blob_params(found, blobs__gauss_fit__size_penalty=3.0)
    assert blob_hash(changed) != blob_hash(blob_params(found))
    assert (
        jump.settings_for("blob_parameters", blob_hash(changed), found, options) is None
    )
    # Moving the 2DCA window as well does not make it match either.
    both = blob_params(found, averages__window=40, blobs__gauss_fit__size_penalty=3.0)
    assert jump.settings_for("blob_parameters", blob_hash(both), found, options) is None


def test_no_settings_at_all_is_none(found):
    wanted = blob_hash(blob_params(found))
    assert jump.settings_for("blob_parameters", wanted, found, []) is None


def test_another_plot_has_no_fields_settings(found):
    options = [setting(method_params(found), default=True)]
    assert jump.settings_for("velocity_contour", "h" * 40, found, options) is None


def test_a_setting_that_cannot_be_read_goes_with_nothing(found):
    broken = prod.Setting(
        hash="b" * 40,
        params_json="this is not json",
        is_default=False,
        diff=(),
        shots=0,
        label="broken",
    )
    assert jump.related_blob_hash(found, broken) is None
    wrong = prod.Setting(
        hash="c" * 40,
        params_json='{"params": {"values": {"no_such_field": 1}}}',
        is_default=False,
        diff=(),
        shots=0,
        label="wrong",
    )
    assert jump.related_blob_hash(found, wrong) is None
    default = setting(method_params(found), default=True)
    wanted = blob_hash(blob_params(found))
    assert (
        jump.settings_for("blob_parameters", wanted, found, [broken, wrong, default])
        == default.hash
    )


def test_without_a_blob_parameters_spec_nothing_goes_with_a_blob_point(found):
    without = {key: spec for key, spec in found.items() if key != "blob_parameters"}
    option = setting(method_params(found), default=True)
    assert jump.related_blob_hash(without, option) is None
    assert (
        jump.settings_for(
            "blob_parameters", blob_hash(blob_params(found)), without, [option]
        )
        is None
    )


# -- the request the Fields page reads once ---------------------------------------------------------


def test_the_request_names_the_shot_the_settings_and_the_pixel():
    assert jump.fields_request(1160616027, "d" * 40, "pixel", (5, 4)) == {
        "shot": 1160616027,
        "settings": "d" * 40,
        "pixel": (5, 4),
    }


def test_the_pixel_goes_only_with_a_fixed_pixel_aggregate():
    from fusion_ui.core import multishot

    for how in multishot.AGGREGATES:
        request = jump.fields_request(1, "d" * 40, how, (5, 4))
        assert ("pixel" in request) == (how == "pixel"), how
    # A fixed pixel with none chosen has nothing to send.
    assert "pixel" not in jump.fields_request(1, "d" * 40, "pixel", None)


def test_the_request_is_plain_python():
    """It goes into session state and is read back by ``apply_request``: ints and a tuple."""
    import numpy as np

    request = jump.fields_request(
        np.int64(7), "d" * 40, "pixel", (np.int64(5), np.int64(4))
    )
    assert request == {"shot": 7, "settings": "d" * 40, "pixel": (5, 4)}
    assert type(request["shot"]) is int
    assert all(type(v) is int for v in request["pixel"])
