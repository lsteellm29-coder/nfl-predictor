# shared seasonal-roster fetch with current-season-wins deduplication + team-code canonicalization
"""One place to pull nfl_data_py's seasonal rosters across multiple
seasons, collapse them to one row per player_id (most recent season
wins), and canonicalize team abbreviations to the codes every OTHER
dataset this codebase reads actually uses.

Two real, related bugs this fixes (Combined Build Plan Part 1 roster
sweep):

1. Several call sites pull `[season, season - 1]` together so a player
   not yet in this season's cut-down roster can still resolve via last
   season's row, then built a lookup dict straight off the concatenated
   frame -- `dict(zip(rosters["player_name"], rosters["team"]))` and
   similar -- which silently keeps whichever season's row happens to
   come LAST when a player_id appears in both, not whichever season is
   actually current.

2. Found chasing bug 1 live: nfl_data_py's CURRENT-season
   (import_seasonal_rosters) roster pull uses "AZ" for Arizona, while
   every other nfl_data_py dataset this codebase reads -- schedules,
   import_snap_counts, prior-season rosters, and this codebase's own
   cached team_stats.parquet/td_backtest.parquet -- still uses "ARI".
   Fixing bug 1 alone (season-preference) made this WORSE, not better:
   once a returning Cardinal's row correctly resolved to the current
   season, their team read "AZ", and every downstream `team ==
   game["home_team"]`-style filter (model/player_stats.py's
   required_lineup, model/predict.py's team_roster_turnover) compares
   against the schedule's "ARI" -- so "AZ" matched nothing, and nearly
   all 91 real players on Arizona's current roster silently vanished
   from anything that filters by team, not just the two who happened to
   still carry a stale "ARI" fallback row. data/team_codes.py's
   TEAM_CODE_ALIASES (this module's AZ->ARI entry moved there, Week 1
   Audit & Tuning Plan Phase 1.1 -- consolidated into one canonical map
   shared with the historical OAK/SD/STL relocations, rather than two
   separate alias dicts that could drift apart) closes that gap by
   normalizing to the code every OTHER dataset already uses, so a
   roster-sourced team code is always safe to compare against a
   schedule-sourced one.
"""

import nfl_data_py as nfl
import pandas as pd

from data.team_codes import normalize_team_codes


def fetch_rosters(seasons: list[int]) -> pd.DataFrame:
    """`seasons` in any order -- the most recent one always wins a
    duplicate player_id, not whichever happens to be listed/returned
    first. Drops the small number of rows nfl_data_py returns with no
    player_id at all (practice-squad/tryout entries this codebase can't
    key anything to). `team` is canonicalized via data/team_codes.py's
    normalize_team_codes() before returning -- always safe to compare
    against a schedule's home_team/away_team without a separate
    translation step."""
    rosters = nfl.import_seasonal_rosters(seasons)
    rosters = rosters[rosters["player_id"] != ""]
    rosters = normalize_team_codes(rosters, columns=["team"])
    return rosters.sort_values("season", ascending=False).drop_duplicates("player_id", keep="first")
