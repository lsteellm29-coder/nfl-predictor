# pre-game rolling team stats for live scoring
"""The rolling team stats the model should see when scoring `week` of `season`.

build_rolling_team_stats() is already pre-game by construction: every stat is
`expanding().mean().shift(1)`, so the row for week W holds stats from games
BEFORE week W, and training joins each game to the row for its own week.

model/predict.py used to filter the live table to `week < W` and take each
team's last row. That is the row for week W-1 -- stats through W-2 -- so:
  * W=2: the Week 1 row, which is empty by construction (every stat NaN for all
    32 teams) and would have fed the model nothing but NaN;
  * W>=3: one game stale.
Week 1 never hit this because it scores entirely off the prior-season fallback.

The right selection is each team's FIRST row with week >= W (>= rather than ==
so a team on a bye in W uses its next game's row, which is still built only
from games before W).

Two more things the live path has to do to match training:
  * ats_win_pct_last5's rolling window runs across the season boundary in
    training (the whole history is built at once), so a week-2 row's "last 5"
    includes late prior-season games. The live table therefore has to be built
    from prior-season + current-season data together.
  * The state is cut to "start of week W" before building (scores blanked and
    pbp dropped for week >= W), so a game already played this week -- e.g. a
    Thursday game -- can't reach any stat.
"""

import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from data.leakage import assert_pregame_selection
from data.team_stats import build_rolling_team_stats, build_team_game_stats

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT_DIR, "data", "cache")
SCHEDULES_PATH = os.path.join(CACHE_DIR, "schedules.parquet")
PBP_PATH = os.path.join(CACHE_DIR, "pbp.parquet")


def load_prior_season_inputs(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(schedule, pbp) for season-1 from the historical cache -- the same data
    the training-time team_stats.parquet was built from."""
    schedule = pd.read_parquet(SCHEDULES_PATH)
    schedule = schedule[schedule["season"] == season - 1]
    pbp = pq.read_table(PBP_PATH, filters=[("season", "=", season - 1)]).to_pandas()
    return schedule, pbp


def select_pregame_rows(rolling: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Each team's first rolling row at/after `week` of `season` (see module docstring)."""
    rows = rolling[(rolling["season"] == season) & (rolling["week"] >= week)]
    return rows.sort_values("week").groupby("team").head(1)


def assert_pbp_covers_played_games(schedule: pd.DataFrame, pbp: pd.DataFrame, season: int, week: int) -> None:
    """Every regular-season game of `season` before `week` that the schedule shows as
    played must be in the play-by-play. If one isn't (nflverse publishes the schedule
    score and the pbp file separately, so one can lag the other), the EPA/yardage
    stats would silently be built without that game while the schedule-derived ones
    include it -- a stale, internally inconsistent input. Refuse loudly instead."""
    played = schedule[(schedule["season"] == season) & (schedule["game_type"] == "REG")
                      & (schedule["week"] < week) & schedule["home_score"].notna()]
    have = set(pbp.loc[pbp["season"] == season, "game_id"].unique()) if "game_id" in pbp.columns else set()
    missing = sorted(set(played["game_id"]) - have)
    if missing:
        raise AssertionError(f"play-by-play is missing {len(missing)} game(s) the schedule shows as played "
                             f"before week {week}: {missing[:6]} -- refusing to build stats without them "
                             f"(nflverse's pbp file may just be lagging; re-run later)")


def _rolling_as_of(schedule: pd.DataFrame, pbp: pd.DataFrame, season: int, week: int,
                   prior_schedule: pd.DataFrame | None, prior_pbp: pd.DataFrame | None):
    """(rolling rows for `season`, the cut schedule) with the state cut to the
    start of `week` -- see the module docstring for why each step is needed."""
    if prior_schedule is None or prior_pbp is None:
        prior_schedule, prior_pbp = load_prior_season_inputs(season)

    unknown = set(schedule["home_team"]) - set(prior_schedule["home_team"])
    if len(prior_schedule) and unknown:
        raise ValueError(f"team codes in the {season} schedule not found in the cached {season - 1} "
                         f"schedule (renamed/relocated franchise?): {sorted(unknown)}")

    schedule = pd.concat([prior_schedule, schedule], ignore_index=True)
    pbp = pd.concat([prior_pbp, pbp], ignore_index=True)
    assert_pbp_covers_played_games(schedule, pbp, season, week)

    # state at the start of `week`
    schedule = schedule.copy()
    schedule.loc[(schedule["season"] == season) & (schedule["week"] >= week),
                 ["home_score", "away_score"]] = np.nan
    pbp = pbp[(pbp["season"] != season) | (pbp["week"] < week)]

    rolling = build_rolling_team_stats(build_team_game_stats(schedule, pbp))
    return rolling[rolling["season"] == season], schedule


def pregame_team_stats(schedule: pd.DataFrame, pbp: pd.DataFrame, season: int, week: int,
                       prior_schedule: pd.DataFrame | None = None,
                       prior_pbp: pd.DataFrame | None = None,
                       required_cols: list[str] | None = None) -> pd.DataFrame:
    """One row per team holding rolling stats from games before `week`.

    `schedule`/`pbp` are the current season's (whole schedule incl. unplayed
    games, pbp for whatever has been played). The prior season defaults to the
    historical cache."""
    rolling, cut_schedule = _rolling_as_of(schedule, pbp, season, week, prior_schedule, prior_pbp)
    rows = select_pregame_rows(rolling, season, week)
    assert_pregame_selection(rows, cut_schedule, season, week, context="pregame_team_stats")

    if required_cols:
        # every team has played by week 2, so an undefined play-by-play-derived stat means the
        # data didn't come through -- not something to paper over with a fallback
        missing = rows[rows[required_cols].isna().any(axis=1)]
        if not missing.empty:
            raise AssertionError(f"pre-game {required_cols} are undefined for {sorted(missing['team'])} "
                                 f"at week {week} -- refusing to score off missing play-by-play stats")
    return rows


def season_to_date_frames(season: int, team_stats_df: pd.DataFrame, schedules_df: pd.DataFrame,
                          schedule: pd.DataFrame | None = None, pbp: pd.DataFrame | None = None,
                          prior_schedule: pd.DataFrame | None = None,
                          prior_pbp: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The cached historical team-stats and schedule frames the team hubs read
    (which end at last season), extended with `season`'s results so far:
      * schedules_df gains the season's schedule with scores -> W-L record;
      * team_stats_df gains the season's rolling rows through the week after the
        latest played one (that row is the stats through the latest played week;
        rows beyond it would just repeat the same numbers, so they're left out).
    Returns the inputs unchanged if the season hasn't produced a played game."""
    if schedule is None:
        import nfl_data_py as nfl
        schedule = nfl.import_schedules([season])
    reg = schedule[schedule["game_type"] == "REG"]
    played = reg[reg["home_score"].notna()]
    if played.empty:
        return team_stats_df, schedules_df
    last_week = int(played["week"].max())

    if pbp is None:
        from data.pbp_loader import load_season_pbp
        pbp = load_season_pbp(season)
    rolling, _ = _rolling_as_of(schedule, pbp, season, last_week + 1, prior_schedule, prior_pbp)
    rolling = rolling[rolling["week"] <= last_week + 1]

    team_stats = pd.concat([team_stats_df[team_stats_df["season"] != season], rolling], ignore_index=True)
    schedules = pd.concat([schedules_df[schedules_df["season"] != season], schedule], ignore_index=True)
    return team_stats, schedules
