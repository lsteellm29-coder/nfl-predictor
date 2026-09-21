# explicit temporal-leakage tripwire (Week 1 Audit & Tuning Plan Phase 2)
"""Every rolling-stat/opponent-adjusted/Elo feature in this pipeline
already carries its own no-leakage guarantee by CONSTRUCTION --
data/team_stats.py's and data/opponent_adjust.py's rolling averages are
built with `.expanding().mean().shift(1)` (or `.rolling(n).mean().
shift(1)`), so a game's own row is never included in its own rolling
average; model/elo.py's compute_elo_ratings() appends each game's
"_pre" snapshot BEFORE that game's result updates the rating. Verified
by reading the actual construction, not assumed -- see AUDIT.md's
Phase 2 section for the full per-feature trace.

This module is the second, explicit layer on top of that: a named,
reusable tripwire applied right at the week-boundary filters that are
the one place a future refactor could quietly weaken the guarantee
above (e.g. changing `week < target` to `week <= target` without
noticing the consequence). It doesn't re-derive whether any individual
number is correct -- that's what the construction-time shift already
guarantees -- it catches the case where a wrong SLICE of otherwise-
correct data gets used in the first place.
"""

import pandas as pd


def assert_no_leakage(df: pd.DataFrame, week: int, week_col: str = "week", context: str = "") -> None:
    """Raises loudly if `df` contains any row at or after `week` -- the
    boundary a pre-game feature table must never cross. Fails with the
    actual offending rows printed, not just a count, so a real
    regression is immediately diagnosable rather than a silently-too-
    good backtest number down the line."""
    bad = df[df[week_col] >= week]
    if not bad.empty:
        label = f" ({context})" if context else ""
        raise AssertionError(
            f"LEAKAGE{label}: {len(bad)} row(s) at or after week {week} found in a pre-game "
            f"feature table. Offending rows:\n{bad.head(5).to_string()}"
        )


def assert_pregame_selection(rows: pd.DataFrame, schedule: pd.DataFrame, season: int, week: int,
                             context: str = "") -> None:
    """The counterpart of assert_no_leakage for data/live_stats.py's selection,
    which deliberately takes rows AT/AFTER `week` (a rolling row for week W is
    already pre-game -- see that module's docstring), so a plain "no row >=
    week" check can't apply to it.

    What it checks instead: each chosen row's `games_played` (the number of that
    team's scheduled games before the row) must equal the number of that team's
    scheduled regular-season games strictly before `week`. Fewer means the
    selection is stale (the bug this replaced: W-1's row, one game behind, or
    the empty Week 1 row for W=2); more means a game at/after `week` got in."""
    reg = schedule[(schedule["game_type"] == "REG") & (schedule["season"] == season)
                   & (schedule["week"] < week)]
    expected = reg.groupby("home_team").size().add(reg.groupby("away_team").size(), fill_value=0)
    got = rows.set_index("team")["games_played"]
    bad = {t: (int(got[t]), int(expected.get(t, 0))) for t in got.index if got[t] != expected.get(t, 0)}
    # every team that still has a game at/after `week` must have been selected -- a team
    # silently missing from `rows` would quietly fall back to last season's stats
    upcoming = schedule[(schedule["game_type"] == "REG") & (schedule["season"] == season)
                        & (schedule["week"] >= week)]
    absent = sorted((set(upcoming["home_team"]) | set(upcoming["away_team"])) - set(got.index))
    if absent:
        label = f" ({context})" if context else ""
        raise AssertionError(f"PRE-GAME SELECTION{label}: no pre-game row selected for {absent} at week {week}")
    if bad:
        label = f" ({context})" if context else ""
        raise AssertionError(
            f"PRE-GAME SELECTION{label}: games_played != regular-season games scheduled before week "
            f"{week} for {len(bad)} team(s) (team: (got, expected)): {bad}"
        )
