"""``sys.modules['__main__']`` after a Streamlit test: put back, so that a later spawned pool still works.

The mechanism is in the docstring of ``keep_the_main_module`` (``conftest.py``): Streamlit installs the script it runs
as ``__main__`` and never puts the old one back, and a process started with ``spawn`` runs its parent's ``__main__``
again from the file it came from. ``AppTest.from_function`` generates a script that ends in
``function(*__args, **__kwargs)``, which only AppTest, which puts those names in the running module, can run.

Two tests, and their order matters. The first leaves ``__main__`` as a test can; the second is the test that follows it,
and is the regression test: it gets the ``__main__`` the session started with and a pool that works. Without the
fixture it fails twice over -- ``__main__`` is still the generated script, and the pool's children die with
``NameError: name '__args' is not defined``, as every pool test did when it ran straight after
``tests/test_plots_dead_pixel_words.py``, whose tests use ``from_function`` too.
"""

import concurrent.futures
import multiprocessing
import multiprocessing.spawn
import runpy
import sys

import pytest
from streamlit.testing.v1 import AppTest

#: ``__main__`` as the session started: collection imports this module before any test has run.
ORIGINAL_MAIN = sys.modules["__main__"]

#: Filled by the first test, so that the second knows it follows it.
LEFT_REPLACED = []


def say(word):
    import streamlit as st

    st.write(word)


def test_an_apptest_leaves_main_replaced_by_a_script_only_apptest_can_run():
    """The hazard, reproduced: what a spawned child would run is not a script that runs on its own."""
    app = AppTest.from_function(say, args=("hello",)).run()
    LEFT_REPLACED.append(True)
    assert not app.exception, [e.value for e in app.exception]
    assert sys.modules["__main__"] is not ORIGINAL_MAIN

    path = multiprocessing.spawn.get_preparation_data("probe").get("init_main_from_path")
    assert path and path != getattr(ORIGINAL_MAIN, "__file__", None)
    with pytest.raises(NameError, match="__args"):
        runpy.run_path(path, run_name="__mp_main__")  # what ``multiprocessing.spawn`` does in each child


def test_the_next_test_has_its_main_back_so_a_spawned_pool_works():
    if not LEFT_REPLACED:
        pytest.skip("it follows the test above, which leaves __main__ replaced; run them together")
    # The symptom: a pool whose children cannot start, "A process in the process pool was terminated abruptly", with
    # the NameError of the child in the captured stderr.
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(1, mp_context=context) as pool:
        assert pool.submit(len, "ab").result(timeout=120) == 2
    # The cause.
    assert sys.modules["__main__"] is ORIGINAL_MAIN
