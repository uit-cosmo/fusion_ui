"""The registry's contract: unique keys, diagnostic filtering, target identity."""

import dataclasses

import pytest

from fusion_ui.core import registry


@dataclasses.dataclass
class NoParams:
    pass


def spec(key, diagnostics=("apd",), **kwargs):
    return registry.PlotSpec(
        key=key,
        label=key,
        diagnostics=diagnostics,
        params=NoParams,
        render=lambda result, params, target: None,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def clean_registry():
    """The registry is module-level global state; phase 03 will add to it from
    imports, so a test must not leave entries behind."""
    saved = dict(registry.REGISTRY)
    registry.REGISTRY.clear()
    yield
    registry.REGISTRY.clear()
    registry.REGISTRY.update(saved)


def test_a_duplicate_key_is_rejected():
    registry.register(spec("raw_frames"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec("raw_frames"))


def test_a_spec_must_accept_at_least_one_diagnostic():
    with pytest.raises(ValueError, match="no diagnostics"):
        registry.register(spec("nowhere", diagnostics=()))


def test_for_diagnostic_filters_and_keeps_registration_order():
    registry.register(spec("frames", ("apd", "phantom")))
    registry.register(spec("probe", ("asp", "fsp")))
    registry.register(spec("psd", ("apd",)))
    assert [s.key for s in registry.for_diagnostic("apd")] == ["frames", "psd"]
    assert [s.key for s in registry.for_diagnostic("asp")] == ["probe"]
    assert registry.for_diagnostic("nothing") == []


def test_compute_decides_whether_a_spec_is_cached():
    assert not spec("live").cached
    assert spec("heavy", compute=lambda ds, p: ds).cached


def target(**kwargs):
    defaults = dict(
        machine="cmod",
        shot=1160616027,
        diagnostic="apd",
        preprocessed=True,
        path="/data/apd_1160616027_preprocessed.nc",
        t_start=1.15,
        t_end=1.45,
    )
    return registry.Target(**{**defaults, **kwargs})


def test_the_raw_and_preprocessed_variants_never_share_a_key():
    """They are different data; sharing a session-state or blob key would show
    one shot's frame under the other's label."""
    assert target(preprocessed=True).key != target(preprocessed=False).key
    assert target().key == "cmod_1160616027_apd_p"


def test_the_target_key_separates_machines():
    """Shot numbers are unique only within a machine."""
    assert target(machine="w7x").key != target(machine="cmod").key


def test_a_target_is_hashable_and_frozen():
    assert {target(), target()} == {target()}
    with pytest.raises(dataclasses.FrozenInstanceError):
        target().shot = 1
