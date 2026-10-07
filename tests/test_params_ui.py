"""The canonical params form and the cache key it produces.

This is the code whose mistakes tests have to catch, because nothing else
will: an unsorted key, a float that reprs differently, or an unhandled type
produces a cache that quietly misses or -- worse -- collides, and the wrong
figure appears under the right label.
"""

import dataclasses
import json
from enum import Enum

import pytest

import imaging_methods as im
from fusion_ui.core import params_ui


@dataclasses.dataclass
class Simple:
    count: int = 3
    scale: float = 1.0
    flag: bool = False
    name: str = "a"


class Colour(Enum):
    red = 1
    blue = 2


@dataclasses.dataclass
class WithEnum:
    colour: Colour = Colour.red


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


def test_a_real_method_parameters_tree_round_trips():
    params = im.MethodParameters()
    values = json.loads(json.dumps(params_ui.canonical(params)["values"]))
    assert params_ui.from_dict(im.MethodParameters, values) == params
    assert isinstance(
        params_ui.from_dict(im.MethodParameters, values).preprocessing.radius, int
    )


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
        != params_ui.hash_params("velocity_2dca", Simple())[0]
    )


def test_a_subclass_does_not_hash_like_its_base():
    """SEParameters extends MethodParameters, so their defaults share a prefix
    and values alone do not identify the parameter set."""
    from individual_events.parameters import SEParameters

    assert (
        params_ui.hash_params("p", im.MethodParameters())[0]
        != params_ui.hash_params("p", SEParameters())[0]
    )


def test_a_dataclass_with_no_fields_raises():
    """velocity_estimation's option classes carry @dataclass but declare their
    own __init__, so fields() is empty -- an empty, colliding hash with no
    error at all if this guard is missing."""
    with pytest.raises(TypeError, match="no fields"):
        params_ui.canonical(Empty())
    with pytest.raises(TypeError, match="no fields"):
        params_ui.canonical(Nests(inner=Empty()))


def test_the_real_velocity_estimation_options_are_caught():
    import velocity_estimation as ve

    assert dataclasses.fields(ve.EstimationOptions()) == ()
    with pytest.raises(TypeError, match="no fields"):
        params_ui.canonical(ve.EstimationOptions())


def test_a_non_finite_value_raises_rather_than_hashing():
    with pytest.raises(ValueError, match="not finite"):
        params_ui.hash_params("p", Simple(scale=float("nan")))
    with pytest.raises(ValueError, match="not finite"):
        params_ui.hash_params("p", Simple(scale=float("inf")))


def test_an_enum_serialises_as_its_name_and_comes_back():
    assert params_ui.canonical(WithEnum())["values"]["colour"] == "red"
    assert params_ui.from_dict(WithEnum, {"colour": "blue"}).colour is Colour.blue


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


def test_the_json_is_key_sorted_so_the_hash_is_stable():
    _, text = params_ui.hash_params("p", im.MethodParameters())
    body = json.loads(text)
    assert list(body) == sorted(body)
    assert list(body["params"]["values"]) == sorted(body["params"]["values"])


def test_the_hash_is_stable_across_processes():
    """A literal, not a recomputation: a change to the canonical form that
    silently invalidates every cached result should fail here first."""
    digest, _ = params_ui.hash_params("taud_psd", im.MethodParameters())
    assert digest == "7238ed04e1d86f3b2c27756a6635f2c9e474b8d4"


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


# ---------------------------------------------------------------------------
# The strict inverse: parameters a person wrote down (--params-json)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Optionals:
    inner: Simple = None
    size_max: float = None


def test_every_registered_spec_round_trips_through_its_stored_form():
    """What `param_sets.params_json` holds, and what the single-shot page tells
    people to save, rebuilds to the parameters that hash to it."""
    import fusion_ui.plots  # noqa: F401 - registers every spec
    from fusion_ui.core import registry

    for key, spec in registry.REGISTRY.items():
        digest, text = params_ui.hash_params(key, spec.params())
        for form in (text, json.loads(text), json.loads(text)["params"]):
            back = params_ui.from_canonical(spec.params, form, plot_key=key)
            assert params_ui.hash_params(key, back)[0] == digest, key


def test_from_canonical_rebuilds_types_and_none():
    form = params_ui.canonical(Optionals(size_max=None))
    assert params_ui.from_canonical(Optionals, form) == Optionals()
    form = params_ui.canonical(WithEnum(Colour.blue))
    assert params_ui.from_canonical(WithEnum, form).colour is Colour.blue


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
        params_ui.from_canonical(WithEnum, text)
    form = {"__type__": params_ui._type_name(WithEnum), "values": {"colour": "green"}}
    with pytest.raises(ValueError, match="expected one of"):
        params_ui.from_canonical(WithEnum, form)


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
