# builds rolling stat table per team
"""Turns the raw schedules/play-by-play cache into a per-team, per-week rolling
stat table (Section 3 of the spec). Every rolling value at (season, week, team)
is computed strictly from that team's games *before* that week, so the table
is safe to use as pre-game features -- no leakage from the game being predicted.
"""

import os

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SCHEDULES_PATH = os.path.join(CACHE_DIR, "schedules.parquet")
PBP_PATH = os.path.join(CACHE_DIR, "pbp.parquet")
TEAM_STATS_PATH = os.path.join(CACHE_DIR, "team_stats.parquet")


def _offense_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team) offensive production for that single game."""
    snaps = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna()]

    epa_ypp = snaps.groupby(["game_id", "posteam"]).agg(
        off_epa_per_play=("epa", "mean"),
        off_ypp=("yards_gained", "mean"),
    )

    third = pbp[(pbp["down"] == 3) & pbp["posteam"].notna()]
    third_down = third.groupby(["game_id", "posteam"]).agg(
        _conv=("third_down_converted", "sum"),
        _fail=("third_down_failed", "sum"),
    )
    third_down["off_third_down_pct"] = third_down["_conv"] / (
        third_down["_conv"] + third_down["_fail"]
    )

    turnovers = pbp[pbp["posteam"].notna()].groupby(["game_id", "posteam"]).agg(
        turnovers_committed=("interception", "sum"),
    )
    fumbles = (
        pbp[pbp["posteam"].notna()]
        .groupby(["game_id", "posteam"])["fumble_lost"]
        .sum()
    )
    turnovers["turnovers_committed"] += fumbles

    drives = (
        pbp[pbp["posteam"].notna()]
        .groupby(["game_id", "posteam", "fixed_drive"])
        .agg(
            reached_rz=("drive_inside20", "max"),
            result=("fixed_drive_result", "first"),
        )
    )
    drives["scored_td"] = drives["result"] == "Touchdown"
    rz = drives[drives["reached_rz"] == 1].groupby(["game_id", "posteam"]).agg(
        rz_trips=("scored_td", "size"),
        rz_tds=("scored_td", "sum"),
    )
    rz["red_zone_td_pct"] = rz["rz_tds"] / rz["rz_trips"]

    out = epa_ypp.join(third_down[["off_third_down_pct"]], how="outer")
    out = out.join(turnovers, how="outer")
    out = out.join(rz[["red_zone_td_pct"]], how="outer")
    return out.reset_index().rename(columns={"posteam": "team"})


def _defense_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team) production *allowed* while on defense."""
    snaps = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["defteam"].notna()]

    epa_ypp = snaps.groupby(["game_id", "defteam"]).agg(
        def_epa_per_play=("epa", "mean"),
        def_ypp=("yards_gained", "mean"),
    )

    third = pbp[(pbp["down"] == 3) & pbp["defteam"].notna()]
    third_down = third.groupby(["game_id", "defteam"]).agg(
        _conv=("third_down_converted", "sum"),
        _fail=("third_down_failed", "sum"),
    )
    third_down["def_third_down_pct"] = third_down["_conv"] / (
        third_down["_conv"] + third_down["_fail"]
    )

    turnovers = pbp[pbp["defteam"].notna()].groupby(["game_id", "defteam"]).agg(
        turnovers_forced=("interception", "sum"),
    )
    fumbles = (
        pbp[pbp["defteam"].notna()]
        .groupby(["game_id", "defteam"])["fumble_lost"]
        .sum()
    )
    turnovers["turnovers_forced"] += fumbles

    out = epa_ypp.join(third_down[["def_third_down_pct"]], how="outer")
    out = out.join(turnovers, how="outer")
    return out.reset_index().rename(columns={"defteam": "team"})


def build_team_game_stats(schedules: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team) with that single game's raw stats --
    not rolling yet. Both the team's own game and its opponent are represented
    (one row for each side of every game)."""

    reg = schedules[schedules["game_type"] == "REG"].copy()
    reg["result"] = reg["home_score"] - reg["away_score"]

    home = reg.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_score": "points_for", "away_score": "points_against",
        "home_rest": "rest_days", "spread_line": "spread_line",
    }).copy()
    home["is_home"] = True
    # spread_line is the home team's market-implied margin of victory
    # (positive = home favored); home covers when it beats that margin.
    home["ats_margin"] = home["result"] - home["spread_line"]

    away = reg.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_score": "points_for", "home_score": "points_against",
        "away_rest": "rest_days", "spread_line": "spread_line",
    }).copy()
    away["is_home"] = False
    away["ats_margin"] = away["spread_line"] - away["result"]

    keep = ["game_id", "season", "week", "team", "opponent", "points_for",
            "points_against", "rest_days", "is_home", "ats_margin", "div_game"]
    games = pd.concat([home[keep], away[keep]], ignore_index=True)

    games["point_diff"] = games["points_for"] - games["points_against"]
    games["ats_win"] = games["ats_margin"].apply(
        lambda m: 1.0 if m > 0 else (0.0 if m < 0 else np.nan)
    )

    off = _offense_game_stats(pbp)
    dfn = _defense_game_stats(pbp)
    games = games.merge(off, on=["game_id", "team"], how="left")
    games = games.merge(dfn, on=["game_id", "team"], how="left")

    games["turnover_diff"] = games["turnovers_forced"] - games["turnovers_committed"]

    return games.sort_values(["team", "season", "week"]).reset_index(drop=True)


ROLLING_SEASON_COLS = [
    "points_for", "points_against", "off_epa_per_play", "def_epa_per_play",
    "off_ypp", "def_ypp", "off_third_down_pct", "def_third_down_pct",
    "turnover_diff", "red_zone_td_pct",
]


def _rolling_for_team_season(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("week").copy()

    for col in ROLLING_SEASON_COLS:
        g[f"{col}_avg"] = g[col].expanding().mean().shift(1)

    home_vals = g["point_diff"].where(g["is_home"])
    away_vals = g["point_diff"].where(~g["is_home"])
    g["home_point_diff_avg"] = home_vals.expanding().mean().shift(1)
    g["away_point_diff_avg"] = away_vals.expanding().mean().shift(1)

    g["ats_win_pct_season"] = g["ats_win"].expanding().mean().shift(1)
    g["games_played"] = range(len(g))

    return g


def build_rolling_team_stats(team_game_stats: pd.DataFrame) -> pd.DataFrame:
    rolled = (
        team_game_stats.groupby(["team", "season"])
        .apply(_rolling_for_team_season)
        .reset_index(level=["team", "season"])
    )

    rolled = rolled.sort_values(["team", "season", "week"]).reset_index(drop=True)
    rolled["ats_win_pct_last5"] = (
        rolled.groupby("team")["ats_win"]
        .apply(lambda s: s.rolling(5, min_periods=1).mean().shift(1))
        .reset_index(drop=True)
    )

    rolling_cols = [f"{c}_avg" for c in ROLLING_SEASON_COLS] + [
        "home_point_diff_avg", "away_point_diff_avg",
        "ats_win_pct_season", "ats_win_pct_last5", "games_played",
    ]
    keep = ["season", "week", "team", "opponent", "is_home", "rest_days",
            "div_game"] + rolling_cols
    return rolled[keep]


def main():
    schedules = pd.read_parquet(SCHEDULES_PATH)
    pbp = pd.read_parquet(PBP_PATH)

    print("Building per-game team stats...")
    team_game_stats = build_team_game_stats(schedules, pbp)

    print("Building rolling stat table...")
    rolling = build_rolling_team_stats(team_game_stats)

    rolling.to_parquet(TEAM_STATS_PATH)
    print(f"  saved {len(rolling)} team-weeks -> {TEAM_STATS_PATH}")


if __name__ == "__main__":
    main()
