# shared seasonal-roster fetch with current-season-wins deduplication
"""One place to pull nfl_data_py's seasonal rosters across multiple
seasons and collapse them to one row per player_id -- the most recent
season wins when a player appears in more than one.

Real bug this fixes (Combined Build Plan Part 1 roster sweep): several
call sites across this codebase pull `[season, season - 1]` together so
a player not yet in this season's cut-down roster can still resolve via
last season's row. But every one of them then built a lookup dict
straight off the concatenated frame -- `dict(zip(rosters["player_name"],
rosters["team"]))` and similar -- which silently keeps whichever
season's row happens to come LAST when a player_id appears in both,
not whichever season is actually current. Found live: nfl_data_py's
current-season roster pull uses "AZ" for Arizona while every other
dataset this codebase reads (schedules, team_stats.parquet,
td_backtest.parquet, prior-season rosters) still uses "ARI" -- every
returning Cardinal's team was resolving to last season's "ARI" row by
accident of dict-insertion order, not by any actual current-season-
preference logic. That specific case is cosmetically harmless (still
the same team, just an abbreviation change), but the underlying pattern
is a real bug: a genuine trade between the two seasons would have shown
identical staleness with an actually wrong team.
"""

import nfl_data_py as nfl
import pandas as pd


def fetch_rosters(seasons: list[int]) -> pd.DataFrame:
    """`seasons` in any order -- the most recent one always wins a
    duplicate player_id, not whichever happens to be listed/returned
    first. Drops the small number of rows nfl_data_py returns with no
    player_id at all (practice-squad/tryout entries this codebase can't
    key anything to)."""
    rosters = nfl.import_seasonal_rosters(seasons)
    rosters = rosters[rosters["player_id"] != ""]
    return rosters.sort_values("season", ascending=False).drop_duplicates("player_id", keep="first")
