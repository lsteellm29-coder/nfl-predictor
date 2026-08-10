# conditional alt-line ladder display: real multi-book alt spreads/totals, shown only once they exist
"""Master Honing Plan, Section B: data/fetch_props.py's parse_alt_lines()
builds real, multi-book-aggregated alternate-spread/total/team-total
ladders every week, but the module's own docstring flags why it was
never wired into the UI -- most rungs run single-book (n_books=1) this
far before kickoff, and "a ladder" built out of one book's own number
isn't really a ladder yet. That's a timing problem, not a permanent one:
books fill in alternate markets as kickoff approaches. This module gates
on both axes rather than picking one: only within DISPLAY_WINDOW_HOURS of
kickoff (when coverage has actually had a chance to fill in), and only
the individual rungs that already have real multi-book agreement
(MIN_BOOKS_FOR_REAL_LADDER) -- a single-book rung inside an otherwise-
real ladder is still skipped, not just the ladder as a whole gated on/off.
"""

import datetime

import pandas as pd

DISPLAY_WINDOW_HOURS = 48
MIN_BOOKS_FOR_REAL_LADDER = 2


def within_display_window(kickoff, now: datetime.datetime | None = None) -> bool:
    """True only for a game whose kickoff is upcoming (not already
    started/passed) and within the next DISPLAY_WINDOW_HOURS -- games
    further out don't have real multi-book alt-line coverage yet (see
    this module's docstring), and a game that's already kicked off has
    no live line left to show. `kickoff` is a naive datetime (this
    project's gameday/gametime fields carry no explicit timezone --
    same precision level the rest of the report already operates at, see
    data/fetch_weather.py's ET-assumption comment)."""
    if kickoff is None or pd.isna(kickoff):
        return False
    now = now or datetime.datetime.now()
    delta = kickoff - now
    return datetime.timedelta(0) <= delta <= datetime.timedelta(hours=DISPLAY_WINDOW_HOURS)


def _real_rungs(ladder: list[dict]) -> list[dict]:
    return [r for r in ladder if r.get("n_books", 0) >= MIN_BOOKS_FOR_REAL_LADDER]


def _fmt_price(price: float) -> str:
    return f"{price:+.0f}"


RUNG_ROW = '<div class="ladder-rung"><span class="point">{point:+.1f}</span><span class="prices">{over_label} {over_price} &middot; {under_label} {under_price}</span></div>'

LADDER_BLOCK = """<div class="ladder">
  <div class="ladder-label">{label}</div>
  {rungs}
</div>"""

ALT_LINES_SECTION = """<div class="alt-lines">
  <div class="alt-lines-head">Alternate Lines <span class="alt-lines-note">real multi-book lines, updates as kickoff nears</span></div>
  {ladders}
</div>"""

ALT_LINES_STYLE = """
.alt-lines { margin-top: 16px; padding-top: 14px; border-top: 1px dashed var(--border); }
.alt-lines-head { font-size: 12.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: var(--accent); margin-bottom: 10px; }
.alt-lines-note { font-weight: 400; text-transform: none; letter-spacing: normal; color: var(--muted); margin-left: 6px; font-size: 11.5px; }
.ladder { margin-bottom: 10px; }
.ladder:last-child { margin-bottom: 0; }
.ladder-label { font-size: 11.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; }
.ladder-rung { display: flex; justify-content: space-between; gap: 10px; font-size: 13px; padding: 4px 0;
  border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
.ladder-rung:last-child { border-bottom: none; }
.ladder-rung .point { font-weight: 700; color: var(--ink); }
.ladder-rung .prices { color: var(--muted); }
"""


def alt_lines_html(alt_lines_for_game: dict | None, kickoff, now: datetime.datetime | None = None,
                    home_team: str = "Home", away_team: str = "Away") -> str:
    """Empty string (no section at all -- not an empty state; this is a
    bonus feature that's simply absent most of the week, which is
    expected and not worth a "check back later" message the way a core
    section like props would need) unless the game is within the display
    window AND at least one market actually has real multi-book rungs."""
    if not alt_lines_for_game or not within_display_window(alt_lines_for_game.get("kickoff", kickoff), now):
        return ""

    ladders = []
    spreads = _real_rungs(alt_lines_for_game.get("alternate_spreads", []))
    if spreads:
        rungs = "\n".join(
            RUNG_ROW.format(point=r["point"], over_label=home_team, over_price=_fmt_price(r["price_a"]),
                             under_label=away_team, under_price=_fmt_price(r["price_b"]))
            for r in sorted(spreads, key=lambda r: r["point"])
        )
        ladders.append(LADDER_BLOCK.format(label="Alternate Spread", rungs=rungs))

    totals = _real_rungs(alt_lines_for_game.get("alternate_totals", []))
    if totals:
        rungs = "\n".join(
            RUNG_ROW.format(point=r["point"], over_label="Over", over_price=_fmt_price(r["price_a"]),
                             under_label="Under", under_price=_fmt_price(r["price_b"]))
            for r in sorted(totals, key=lambda r: r["point"])
        )
        ladders.append(LADDER_BLOCK.format(label="Alternate Total", rungs=rungs))

    team_totals = alt_lines_for_game.get("team_totals", {})
    for side, team_name in (("home", home_team), ("away", away_team)):
        rungs_data = _real_rungs(team_totals.get(side, []))
        if not rungs_data:
            continue
        rungs = "\n".join(
            RUNG_ROW.format(point=r["point"], over_label="Over", over_price=_fmt_price(r["price_a"]),
                             under_label="Under", under_price=_fmt_price(r["price_b"]))
            for r in sorted(rungs_data, key=lambda r: r["point"])
        )
        ladders.append(LADDER_BLOCK.format(label=f"{team_name} Team Total", rungs=rungs))

    if not ladders:
        return ""
    return ALT_LINES_SECTION.format(ladders="\n".join(ladders))
