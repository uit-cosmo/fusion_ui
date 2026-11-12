"""Run days: what each day of shots was for.

A shot number is ``1YYMMDDnnn``: its first seven digits are the run day, which
one miniproposal usually owns. The discharge DB says nothing of that, so the
prose is hand-written in ``fusion_ui/data/run_days.md`` from the C-Mod run
pages.

There is no scraper, on purpose: the run pages are on MIT's internal web and
never change once written, and a landing page that reaches the network hangs
when the network is not there.
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

# A miniproposal's PDF, from its number. Reachable from the group server and
# from MIT only; nothing here fetches it.
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
        """The miniproposal PDF on the C-Mod internal web."""
        return MINIPROPOSAL_URL.format(self.mp.removeprefix("MP"))


def day_of(shot):
    """The run day a shot belongs to: the first seven digits of its number."""
    return str(shot)[:7]


def day_to_date(day):
    """``"1160616"`` -> ``"2016-06-16"``. The leading 1 is not part of the
    date, and the two digits after it are a year of this century."""
    return f"20{day[1:3]}-{day[3:5]}-{day[5:7]}"


def parse(text):
    """``{day: RunDay}`` from the markdown: a ``## <seven digits>`` heading,
    one bold ``**MP<n> — <title>**`` line, then the summary. A section with no
    MP line is skipped."""
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
    """The run-day file, parsed; ``{}`` when it is missing."""
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


def run_day_table(conn, discharge_db_path):
    """One row per run day: how many shots, how many are on disk, what for.

    Every day in the discharge DB is listed whether or not its files are here,
    and so is a day with files but no shot in the discharge DB.
    """
    run_days = load()

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
