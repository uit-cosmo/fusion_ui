"""The canonical params form and the cache key it produces.

Nothing else catches a mistake here: an unsorted key or an unhandled type gives
a cache that quietly misses or collides, and the wrong figure appears under the
right label.
"""

import dataclasses
import json

import pytest

import imaging_methods as im
from fusion_ui.core import params_ui


@dataclasses.dataclass
class Simple:
    count: int = 3
    scale: float = 1.0
    flag: bool = False
    name: str = "a"


@dataclasses.dataclass
class Empty:
    pass


@dataclasses.dataclass
class Nests:
    inner: Simple = dataclasses.field(default_factory=Simple)


def test_an_int_and_a_float_of_the_same_value_hash_alike():
    """``GaussFitParams.size_penalty`` is annotated float but defaults to the
    int 5, so the widget's 5.0 and the default's 5 are the same parameter set
    and must not produce two cache entries."""
    assert params_ui.hash_params("p", Simple(scale=5))[0] == (
        params_ui.hash_params("p", Simple(scale=5.0))[0]
    )
    assert params_ui.canonical(Simple(scale=5))["values"]["scale"] == 5.0


def test_an_int_field_survives_a_json_round_trip_as_an_int():
    """``run_norm_ds`` does a strict ``isinstance(radius, int)``; JSON has one
    number type, so a naive rebuild hands it 1000.0 and it hard-fails."""
    values = json.loads(json.dumps(params_ui.canonical(Simple())["values"]))
    rebuilt = params_ui.from_dict(Simple, values)
    assert isinstance(rebuilt.count, int)
    assert rebuilt == Simple()


def test_none_is_preserved_and_is_not_zero():
    """``size_max: float = None`` means "use the full grid extent". Coercing it
    to 0.0 would silently change the ellipse fit."""
    none = im.GaussFitParams(size_max=None)
    zero = im.GaussFitParams(size_max=0.0)
    assert params_ui.canonical(none)["values"]["size_max"] is None
    assert params_ui.hash_params("p", none)[0] != params_ui.hash_params("p", zero)[0]
    assert params_ui.from_dict(im.GaussFitParams, {"size_max": None}).size_max is None


def test_the_plot_key_is_part_of_the_hash():
    """param_sets.hash is the primary key and plot is an ordinary column, so
    two plots sharing a default parameter set would collide on it."""
    assert (
        params_ui.hash_params("taud_psd", Simple())[0]
        != params_ui.hash_params("velocity_tde", Simple())[0]
    )


def test_a_subclass_does_not_hash_like_its_base():
    """Their values are the same, so values alone do not identify the set."""

    @dataclasses.dataclass
    class Subclass(Simple):
        pass

    assert (
        params_ui.hash_params("p", Simple())[0]
        != params_ui.hash_params("p", Subclass())[0]
    )


def test_a_dataclass_with_no_fields_raises():
    """velocity_estimation's option classes carry @dataclass but declare their
    own __init__, so fields() is empty -- an empty, colliding hash with no
    error at all if this guard is missing."""
    with pytest.raises(TypeError, match="no fields"):
        params_ui.canonical(Empty())
    with pytest.raises(TypeError, match="no fields"):
        params_ui.canonical(Nests(inner=Empty()))


def test_a_non_finite_value_raises_rather_than_hashing():
    with pytest.raises(ValueError, match="not finite"):
        params_ui.hash_params("p", Simple(scale=float("nan")))
    with pytest.raises(ValueError, match="not finite"):
        params_ui.hash_params("p", Simple(scale=float("inf")))


def test_a_bool_is_not_an_int_in_either_direction():
    with pytest.raises(TypeError):
        params_ui.canonical(Simple(count=True))
    with pytest.raises(TypeError):
        params_ui.canonical(Simple(flag=1))


def test_a_non_integral_value_for_an_int_field_raises():
    with pytest.raises(TypeError, match="not an exact integer"):
        params_ui.canonical(Simple(count=2.5))


def test_an_unknown_type_raises_naming_the_field():
    @dataclasses.dataclass
    class Odd:
        thing: complex = 1j

    with pytest.raises(TypeError, match="thing"):
        params_ui.canonical(Odd())


def test_the_json_is_key_sorted_and_the_hash_is_a_literal():
    """A literal, not a recomputation: a change to the canonical form, which
    would silently invalidate every cached result, fails here first."""
    digest, text = params_ui.hash_params("p", Nests())
    body = json.loads(text)
    assert list(body) == ["params", "plot"]
    assert list(body["params"]["values"]["inner"]) == ["count", "flag", "name", "scale"]
    assert digest == "a2c8f2dfdf10f426fa87338ceeb2cd9149026d17"


def test_canonical_rejects_a_class():
    with pytest.raises(TypeError):
        params_ui.canonical(Simple)


def test_seed_session_state_writes_the_widget_keys():
    """The multi-shot view jumps to the single-shot view by seeding these keys;
    they must match the scheme ``_fields`` reads back."""
    state = {}
    params_ui.seed_session_state(
        state, "params.p", Nests(inner=Simple(count=7, scale=2.5))
    )
    assert state["params.p.inner.count"] == 7
    assert state["params.p.inner.scale"] == 2.5
    assert state["params.p.inner.flag"] is False


def test_seed_session_state_marks_optional_fields():
    """``size_max=None`` means "work it out"; the form represents that with an
    ``.__auto__`` toggle plus a placeholder number."""
    state = {}
    params_ui.seed_session_state(state, "p", im.GaussFitParams(size_max=None))
    assert state["p.size_max.__auto__"] is True
    assert state["p.size_max"] == 0.0


def test_seed_session_state_keeps_a_present_zero_value():
    """``size_max=0.0`` is a real instruction, not the None placeholder: it must
    not flip the ``.__auto__`` toggle or be swapped for a default."""
    state = {}
    params_ui.seed_session_state(state, "p", im.GaussFitParams(size_max=0.0))
    assert state["p.size_max.__auto__"] is False
    assert state["p.size_max"] == 0.0


def test_help_is_the_table_entry_or_else_the_class_docstring():
    @dataclasses.dataclass
    class Documented:
        """Settings.

        - first, second (int): what both of them do,
          on two lines.

        - third: on its own.
        """

        first: int = 1
        second: int = 2
        third: int = 3
        fourth: int = 4

    assert params_ui.help_for(Documented, "first") == "what both of them do, on two lines."
    assert params_ui.help_for(Documented, "second") == params_ui.help_for(Documented, "first")
    assert params_ui.help_for(Documented, "third") == "on its own."
    assert params_ui.help_for(Documented, "fourth") is None
    entry = params_ui.HELP[("TaudEstimationParams", "cutoff")]
    assert params_ui.help_for(im.TaudEstimationParams, "cutoff", "blobs.taud_estimation.cutoff") == entry


# ---------------------------------------------------------------------------
# The strict inverse: parameters a person wrote down (--params-json)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Optionals:
    inner: Simple = None
    size_max: float = None


def test_every_registered_spec_round_trips_through_its_stored_form():
    """What `param_sets.params_json` holds rebuilds, strictly and leniently, to
    the parameters that hash to it."""
    import fusion_ui.plots  # noqa: F401 - registers every spec
    from fusion_ui.core import registry

    for key, spec in registry.REGISTRY.items():
        digest, text = params_ui.hash_params(key, spec.params())
        for form in (text, json.loads(text), json.loads(text)["params"]):
            back = params_ui.from_canonical(spec.params, form, plot_key=key)
            assert params_ui.hash_params(key, back)[0] == digest, key
        values = json.loads(text)["params"]["values"]
        assert params_ui.from_dict(spec.params, values) == spec.params(), key


def test_from_dict_reads_a_set_stored_by_another_version_of_the_class():
    """A field left out keeps its default and a name the class lacks is ignored."""
    assert params_ui.from_dict(Nests, {"inner": {"count": 7, "gone": 1}}) == Nests(
        inner=Simple(count=7)
    )


def test_from_canonical_rebuilds_types_and_none():
    form = params_ui.canonical(Optionals(size_max=None))
    assert params_ui.from_canonical(Optionals, form) == Optionals()
    back = params_ui.from_canonical(Simple, json.dumps(params_ui.canonical(Simple())))
    assert back == Simple() and isinstance(back.count, int)


@pytest.mark.parametrize(
    "values, message",
    [
        ({"flag": "false"}, "expected true or false"),
        ({"flag": 1}, "expected true or false"),
        ({"count": 2.5}, "whole number"),
        ({"count": True}, "for a int field"),
        ({"scale": "2.5"}, "finite number"),
        ({"name": 7}, "expected a string"),
        ({"name": None, "extra": 1}, "no field 'extra'"),
    ],
)
def test_from_canonical_refuses_what_from_dict_would_coerce(values, message):
    whole = {"count": 3, "scale": 1.0, "flag": False, "name": "a", **values}
    form = {"__type__": params_ui._type_name(Simple), "values": whole}
    with pytest.raises(ValueError, match=message):
        params_ui.from_canonical(Simple, form)


def test_from_canonical_wants_every_field():
    values = {"count": 3, "scale": 1.0, "flag": False}
    form = {"__type__": params_ui._type_name(Simple), "values": values}
    with pytest.raises(ValueError, match="missing 'name'"):
        params_ui.from_canonical(Simple, form)


def test_from_canonical_checks_the_class_and_the_plot():
    text = params_ui.hash_params("p", Simple())[1]
    with pytest.raises(ValueError, match="parameter set of plot 'p', not of 'q'"):
        params_ui.from_canonical(Simple, text, plot_key="q")
    with pytest.raises(ValueError, match="but the plot's parameters are"):
        params_ui.from_canonical(Nests, text)


def test_with_values_changes_only_what_it_names():
    base = Nests(inner=Simple(count=4))
    changed = params_ui.with_values(base, {"inner": {"scale": 2}})
    assert changed == Nests(inner=Simple(count=4, scale=2.0))
    assert isinstance(changed.inner.scale, float)
    assert base == Nests(inner=Simple(count=4)), "the input is not modified"
    assert changed.inner is not base.inner
    assert params_ui.with_values(base, {}) == base


def test_with_values_refuses_unknown_names_and_bad_leaves():
    with pytest.raises(ValueError, match="inner: Simple has no field 'scael'"):
        params_ui.with_values(Nests(), {"inner": {"scael": 2.0}})
    with pytest.raises(ValueError, match="inner.flag: expected true or false"):
        params_ui.with_values(Nests(), {"inner": {"flag": "yes"}})
    with pytest.raises(ValueError, match="expected an object of fields"):
        params_ui.with_values(Nests(), {"inner": 3})


def test_with_values_fills_a_none_subtree_only_completely():
    """Nothing to keep under a None, so a change there names every field."""
    with pytest.raises(ValueError, match="missing"):
        params_ui.with_values(Optionals(), {"inner": {"count": 1}})
    full = {"count": 1, "scale": 1.0, "flag": False, "name": "a"}
    assert params_ui.with_values(Optionals(), {"inner": full}).inner == Simple(count=1)
