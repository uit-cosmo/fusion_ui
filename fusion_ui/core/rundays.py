"""Run days: what each day of shots was actually for.

A shot number is ``1YYMMDDnnn`` -- the first seven digits are the run day, and a
run day is the unit the group thinks in: one miniproposal (occasionally two or
three) owns the day, the session leader writes a plan, and every shot taken that
day serves it. The discharge database carries none of that, so the prose lives
in ``fusion_ui/data/run_days.md``, **hand-written** from the C-Mod run pages.

There is deliberately no scraper. The run pages sit behind
``www-internal.psfc.mit.edu`` with a certificate that does not verify, they are
written once and never again, and a landing page that reaches the network is a
landing page that hangs when the network is not there. Adding a run day is a
section in the markdown file, which a physicist can write without touching
Python.
"""

import os
import re
from dataclasses import dataclass

import pandas as pd

from fusion_ui.core import catalog

RUN_DAYS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "run_days.md",
)

# The run page links each miniproposal to a PDF under this path; the number in
# the markdown file is the whole of it, so the URL is derived rather than
# written down twice. Served over http on the run page, but https works and is
# what we link to.
MINIPROPOSAL_URL = (
    "https://www-internal.psfc.mit.edu/research/alcator/miniproposals/{}.pdf"
)

_HEADING = re.compile(r"^##\s+(\d{7})\s*$")
# The one bold line under a heading: "**MP800 -- <verbatim title>**".
_MINIPROPOSAL = re.compile(r"^\*\*MP\s*(\d+)\s*[—–-]+\s*(.+?)\*\*\s*$")


@dataclass(frozen=True)
class RunDay:
    """One day's entry. ``mp`` and ``title`` are quoted from the run page;
    ``summary`` is prose written by hand and may paraphrase badly."""

    day: str
    mp: str
    title: str
    summary: str

    @property
    def date(self):
        return day_to_date(self.day)

    @property
    def url(self):
        """The miniproposal PDF on the C-Mod internal web.

        Reachable from the group server and from MIT; from anywhere else it
        simply will not load. Nothing here fetches it -- it is a link for the
        person reading the page.
        """
        return MINIPROPOSAL_URL.format(self.mp.removeprefix("MP"))


def day_of(shot):
    """The run day a shot belongs to: the first seven digits of its number."""
    return str(shot)[:7]


def day_to_date(day):
    """``"1160616"`` -> ``"2016-06-16"``.

    The leading 1 is C-Mod's machine digit, not part of the date; the two after
    it are the year within 2000..2099, which every C-Mod run day satisfies --
    the machine ran from 1993 under a different numbering and shut down in 2016.
    """
    return f"20{day[1:3]}-{day[3:5]}-{day[5:7]}"


def parse(text):
    """``{day: RunDay}`` from the markdown. Sections with no bold MP line are
    skipped -- that is the file's own preamble, and a half-written entry."""
    days, day, mp, title, body = {}, None, None, None, []

    def flush():
        if day is not None and mp is not None:
            days[day] = RunDay(day, mp, title, "\n".join(body).strip())

    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            flush()
            day, mp, title, body = heading.group(1), None, None, []
            continue
        if day is None:
            continue
        miniproposal = _MINIPROPOSAL.match(line.strip())
        if miniproposal and mp is None:
            mp, title = f"MP{miniproposal.group(1)}", miniproposal.group(2).strip()
            continue
        body.append(line)
    flush()
    return days


def load(path=None):
    """The run-day file, parsed. Missing file means no summaries, not a crash:
    the overview still lists the days, just without their purpose."""
    path = path or RUN_DAYS_PATH
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return parse(handle.read())


# ---------------------------------------------------------------------------
# The overview table
# ---------------------------------------------------------------------------

TABLE_COLUMNS = [
    "date",
    "day",
    "shots",
    "on_disk",
    "diagnostics",
    "bytes",
    "mp",
    "mp_url",
    "title",
]


def run_day_table(conn, discharge_db_path, run_days=None):
    """One row per run day: how many shots, how many are on disk, what for.

    Every curated day appears whether or not its files are here -- the point of
    the overview is the campaign, not this machine's copy of it -- and a day
    with files but no curated shots appears too, the same way the shot browser
    lists an uncurated shot rather than hiding it.
    """
    run_days = load() if run_days is None else run_days

    curated = {}
    for shot in catalog.load_discharges(discharge_db_path) if discharge_db_path else {}:
        curated.setdefault(day_of(shot), set()).add(shot)

    on_disk, diagnostics, sizes = {}, {}, {}
    for row in conn.execute(
        "SELECT shot, diagnostic, COALESCE(SUM(bytes), 0) AS bytes"
        "  FROM shots GROUP BY shot, diagnostic"
    ):
        day = day_of(row["shot"])
        on_disk.setdefault(day, set()).add(row["shot"])
        diagnostics.setdefault(day, set()).add(row["diagnostic"])
        sizes[day] = sizes.get(day, 0) + (row["bytes"] or 0)

    records = []
    for day in sorted(set(curated) | set(on_disk)):
        entry = run_days.get(day)
        records.append(
            {
                "date": day_to_date(day),
                "day": day,
                "shots": len(curated.get(day, ())),
                "on_disk": len(on_disk.get(day, ())),
                "diagnostics": " ".join(sorted(diagnostics.get(day, ()))),
                "bytes": sizes.get(day, 0),
                "mp": entry.mp if entry else "",
                "mp_url": entry.url if entry else "",
                "title": entry.title if entry else "",
            }
        )
    return pd.DataFrame(records, columns=TABLE_COLUMNS)
