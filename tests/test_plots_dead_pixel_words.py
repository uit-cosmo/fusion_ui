"""The words both dead-pixel views show: a plain summary, and the technical method under it.

``dead_pixels.explain`` draws the plain summary, which is hand-written Markdown in ``fusion_ui/data/`` beside
``run_days.md``, and below it ``METHOD`` in a collapsed expander. The dead-pixel view (raw files) and the stored-mask
view (preprocessed files) both call it, so the two read alike and rewording is an edit to that one file. What each
view does with it is tested with the view: ``test_app.py`` for the raw one, ``test_plots_stored_mask.py`` for the
preprocessed one.
"""

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

import fusion_ui
import fusion_ui.plots  # noqa: F401 - registers every spec
from fusion_ui.plots import dead_pixels


def explained():
    from fusion_ui.plots import dead_pixels

    dead_pixels.explain()


def test_explain_puts_the_plain_summary_above_the_technical_method():
    app = AppTest.from_function(explained).run()
    assert not app.exception
    assert [element.type for element in app.main.children.values()] == ["markdown", "expander"]
    assert app.markdown[0].value == dead_pixels.plain_summary()
    (expander,) = app.expander
    assert expander.label == dead_pixels.METHOD_TITLE == "How dead pixels are found"
    method = [m.value for m in expander.markdown]
    assert method and "Red spectrum" in method[0]
    assert "**150**" in method[0] and "**0.1**" in method[0]  # the thresholds in force, filled in
    # Both views show this text, so it must be true of both: the window the spectra are judged over, and what each
    # view draws of the mask (J6d's note: it used to say "This view shows the single shot").
    assert "Welch PSD of each pixel over the analysis window" in method[0]
    assert "single shot" not in method[0]
    assert "the mask the file was made with, which is the day's" in method[0]


def test_explain_quotes_the_thresholds_it_is_given():
    def with_thresholds():
        from fusion_ui.plots import dead_pixels

        dead_pixels.explain(red=300.0, gain=0.25)

    app = AppTest.from_function(with_thresholds).run()
    (method,) = [m.value for m in app.expander[0].markdown]
    assert "**300**" in method and "**0.25**" in method


def test_the_summary_is_a_markdown_file_beside_the_run_days():
    """Where the user finds it to reword it, and where the package data (``data/*.md``) ships it from."""
    path = Path(dead_pixels.PLAIN_SUMMARY_PATH).resolve()
    assert path.parent == Path(fusion_ui.__file__).resolve().parent / "data"
    assert path.name == "dead_pixels_in_plain_words.md" and path.is_file()
    assert (path.parent / "run_days.md").is_file()
    assert os.path.getsize(path) > 0


def test_the_summary_is_read_from_its_file_every_time_it_is_asked_for(tmp_path, monkeypatch):
    """No copy is kept: an edit shows on the next rerun, without a restart."""
    edited = tmp_path / "words.md"
    monkeypatch.setattr(dead_pixels, "PLAIN_SUMMARY_PATH", str(edited))
    edited.write_text("\n\nFirst wording, with an en dash – and a µ.\n\n", encoding="utf-8")
    assert dead_pixels.plain_summary() == "First wording, with an en dash – and a µ."
    edited.write_text("Second wording.\n", encoding="utf-8")
    assert dead_pixels.plain_summary() == "Second wording."

    app = AppTest.from_function(explained).run()
    assert app.markdown[0].value == "Second wording."


def test_a_missing_summary_file_is_a_warning_and_the_method_still_shows(tmp_path, monkeypatch):
    monkeypatch.setattr(dead_pixels, "PLAIN_SUMMARY_PATH", str(tmp_path / "gone.md"))
    app = AppTest.from_function(explained).run()
    assert not app.exception
    (warning,) = [w.value for w in app.warning]
    assert "could not be read" in warning and "gone.md" in warning
    assert [e.label for e in app.expander] == [dead_pixels.METHOD_TITLE]


def test_the_plain_summary_says_what_live_means_the_two_tests_and_the_run_day_rule():
    text = dead_pixels.plain_summary()
    assert "*live*" in text and "*dead*" in text
    assert "1. **" in text and "2. **" in text  # the two tests
    assert "Run-day rule" in text and "a third" in text
    assert "1160616" in text and "hand-made" in text
    assert len(text.split()) < 220  # a summary, not a second METHOD
    assert "{" not in text  # plain Markdown: nothing is filled in
    for line in text.splitlines():
        assert len(line) <= 120  # hard-wrapped, as run_days.md is, so an edit stays a small diff


def test_the_technical_method_is_kept_as_it_was():
    """The user asked for the plain summary on top of it, not in place of it."""
    assert "Red spectrum" in dead_pixels.METHOD and "Neighbours" in dead_pixels.METHOD
    assert "{red:g}" in dead_pixels.METHOD and "{gain:g}" in dead_pixels.METHOD
    assert "Checked on the 111 raw APD shots" in dead_pixels.METHOD
