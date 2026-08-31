# builds rolling stat table per team
"""Turns the raw schedules/play-by-play cache into a per-team, per-week rolling
stat table (Section 3 of the spec). Every rolling value at (season, week, team)
is computed strictly from that team's games *before* that week, so the table
is safe to use as pre-game features -- no leakage from the game being predicted.
"""

import os

import numpy as np
import pandas as pd

from data.opponent_adjust import OPPONENT_STAT, iterative_opponent_adjust

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
        # nflfastR's own success flag (>=40%/60%/100% of yards-to-go
        # gained on 1st/2nd/3rd-4th down) -- % of plays that moved the
        # chains "enough," which a single EPA average can mask (a few
        # explosive plays can carry a good EPA average on an otherwise
        # inconsistent, series-killing offense).
        off_success_rate=("success", "mean"),
    )

    catches = snaps[(snaps["complete_pass"] == 1) & snaps["xyac_mean_yardage"].notna()].copy()
    catches["yac_oe"] = catches["yards_after_catch"] - catches["xyac_mean_yardage"]
    yac_oe = catches.groupby(["game_id", "posteam"])["yac_oe"].mean().rename("off_yac_oe")

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

    # Master Honing Plan, Section A (ablation candidate): rate at which
    # this offense's own dropbacks result in a pressure (nflfastR's own
    # was_pressure flag, which already folds sacks in as a pressure
    # outcome) -- the offense's pass-block/pocket-protection signal, as
    # distinct from off_epa_per_play which blends this in with everything
    # else that happens on a dropback. was_pressure has ~93-100% coverage
    # across the 2016-2025 historical window (100% since 2023); restricted
    # to rows where it's actually populated rather than treating a missing
    # flag as "no pressure," same as how success/other nflfastR-provided
    # flags are used elsewhere in this file without a special NaN case.
    dropbacks = pbp[(pbp["play_type"] == "pass") & pbp["was_pressure"].notna()]
    pressure_off = dropbacks.groupby(["game_id", "posteam"])["was_pressure"].mean().rename("off_pressure_rate_allowed")

    out = epa_ypp.join(third_down[["off_third_down_pct"]], how="outer")
    out = out.join(turnovers, how="outer")
    out = out.join(rz[["red_zone_td_pct"]], how="outer")
    out = out.join(yac_oe, how="outer")
    out = out.join(pressure_off, how="outer")
    return out.reset_index().rename(columns={"posteam": "team"})


def _defense_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team) production *allowed* while on defense."""
    snaps = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["defteam"].notna()]

    epa_ypp = snaps.groupby(["game_id", "defteam"]).agg(
        def_epa_per_play=("epa", "mean"),
        def_ypp=("yards_gained", "mean"),
        def_success_rate=("success", "mean"),
    )

    catches = snaps[(snaps["complete_pass"] == 1) & snaps["xyac_mean_yardage"].notna()].copy()
    catches["yac_oe"] = catches["yards_after_catch"] - catches["xyac_mean_yardage"]
    def_yac_oe = catches.groupby(["game_id", "defteam"])["yac_oe"].mean().rename("def_yac_oe")

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

    # Mirror of off_pressure_rate_allowed above, from the defense's side:
    # rate at which this defense pressures the opponent's dropbacks --
    # the pass-rush signal, distinct from def_epa_per_play.
    dropbacks = pbp[(pbp["play_type"] == "pass") & pbp["was_pressure"].notna()]
    pressure_def = dropbacks.groupby(["game_id", "defteam"])["was_pressure"].mean().rename("def_pressure_rate")

    out = epa_ypp.join(third_down[["def_third_down_pct"]], how="outer")
    out = out.join(turnovers, how="outer")
    out = out.join(def_yac_oe, how="outer")
    out = out.join(pressure_def, how="outer")
    return out.reset_index().rename(columns={"defteam": "team"})


def _special_teams_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team) special-teams EPA -- field goals, punts, and
    kickoffs, from the kicking/punting team's own perspective (pbp's
    posteam convention for these play types). EPA already distance- and
    situation-adjusts field goals (a made 50-yarder is worth more than a
    made 30-yarder, a missed 50-yarder costs less than a missed 30-yarder)
    and already prices in field position on punts/kickoffs, so this
    doesn't need a hand-built kicker-distance model or a separate
    return-value calculation -- coverage quality is already baked into a
    below-average (for the kicking team) or above-average punt/kickoff
    EPA."""
    st = pbp[pbp["play_type"].isin(["field_goal", "punt", "kickoff"]) & pbp["posteam"].notna()]
    return (
        st.groupby(["game_id", "posteam"])["epa"].mean()
        .rename("special_teams_epa").reset_index().rename(columns={"posteam": "team"})
    )


def _qb_game_stats(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team) EPA/dropback and CPOE for *that game's starting
    QB only* (per the schedule's home_qb_id/away_qb_id), not the team's
    blended offensive average -- isolates the passer's own efficiency from
    rushing production, other skill players, and any garbage-time
    backup-QB relief snaps, none of which this feature is meant to
    capture."""
    reg = schedules[schedules["game_type"] == "REG"]
    home_qb = reg.rename(columns={"home_team": "team", "home_qb_id": "starter_id"})[
        ["game_id", "team", "starter_id"]]
    away_qb = reg.rename(columns={"away_team": "team", "away_qb_id": "starter_id"})[
        ["game_id", "team", "starter_id"]]
    starters = pd.concat([home_qb, away_qb], ignore_index=True)

    dropbacks = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna()]
    dropbacks = dropbacks.merge(starters, left_on=["game_id", "posteam"], right_on=["game_id", "team"])
    starter_plays = dropbacks[dropbacks["passer_player_id"] == dropbacks["starter_id"]]

    return starter_plays.groupby(["game_id", "team"]).agg(
        qb_epa_per_play=("epa", "mean"), qb_cpoe=("cpoe", "mean"),
    ).reset_index()


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
    qb = _qb_game_stats(pbp, schedules)
    st = _special_teams_game_stats(pbp)
    games = games.merge(off, on=["game_id", "team"], how="left")
    games = games.merge(dfn, on=["game_id", "team"], how="left")
    games = games.merge(qb, on=["game_id", "team"], how="left")
    games = games.merge(st, on=["game_id", "team"], how="left")

    games["turnover_diff"] = games["turnovers_forced"] - games["turnovers_committed"]

    return games.sort_values(["team", "season", "week"]).reset_index(drop=True)


ROLLING_SEASON_COLS = [
    "points_for", "points_against", "off_epa_per_play", "def_epa_per_play",
    "off_ypp", "def_ypp", "off_third_down_pct", "def_third_down_pct",
    "turnover_diff", "red_zone_td_pct",
    # Phase 4 (v3 spec): success rate, receiving-YAC-over-expected, and
    # starting-QB-specific EPA/CPOE. True rush-yards-over-expected (RYOE)
    # needs a NextGenStats-style expected-rushing-yards model that isn't
    # part of nflfastR's standard play-by-play, so it's left out here
    # rather than faked with a weaker proxy -- the spec's own "if
    # available" already anticipated this gap.
    "off_success_rate", "def_success_rate", "off_yac_oe", "def_yac_oe",
    "qb_epa_per_play", "qb_cpoe",
    # Phase 10 (v3 spec): special-teams EPA (field goals, punts, kickoffs).
    "special_teams_epa",
    # Master Honing Plan, Section A (ablation candidate): pass-rush
    # pressure rate, both sides (see _offense_game_stats/_defense_game_stats
    # above for the exact was_pressure-based computation).
    "off_pressure_rate_allowed", "def_pressure_rate",
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
    result = rolled[keep]

    # Section 1: opponent-adjusted EPA/play and yards/play, kept as
    # separate *_adj_avg columns alongside the raw ones above. Lives in
    # this function (not just main()'s CLI path) so live current-season
    # scoring in model/predict.py gets these too, not just the cached
    # historical parquet. Iterative (3-pass) as of Audit Fix Plan Step 4 --
    # see data/opponent_adjust.py's docstring.
    adj_rolling = iterative_opponent_adjust(team_game_stats, result)
    result = result.merge(adj_rolling, on=["season", "week", "team"], how="left")
    result = _fill_adjusted_warmup_gap(result)

    return result


def _fill_adjusted_warmup_gap(result: pd.DataFrame) -> pd.DataFrame:
    """Week 1 Audit & Tuning Plan Phase 2 (found doing the leakage exit
    test, not leakage itself -- the opposite problem): each of the 3
    iterative opponent-adjustment passes adds its OWN
    expanding().shift(1) "warmup" requirement on top of the raw stat's
    own (data/opponent_adjust.py's roll_opponent_adjusted(), called once
    per pass) -- pass 1 needs 1 prior week, pass 2 needs pass 1's own
    rolled value, pass 3 needs pass 2's, so with n_passes=3 the
    *_adj_avg columns don't produce a real number until roughly week 5
    of a season. Verified empirically: every week 1-4 game across all 10
    cached seasons (about 24% of all training rows) was silently dropped
    by model/train.py's `.dropna(subset=FEATURE_COLS)`, meaning the
    model has never trained on a single real Week 1 game -- for a
    project whose entire deployment target is Week 1, that's the single
    most consequential finding of this audit.

    Fix: when the adjusted value isn't warmed up yet, fall back to that
    same team's own RAW rolling average for that stat/week rather than
    leaving it null -- not a fabricated number, the team's own already-
    computed, already-leak-free raw figure, just not yet opponent-
    refined. This is strictly better than dropping the row: an early-
    season game trains on a slightly less-refined version of a real
    signal instead of not existing in the training set at all."""
    result = result.copy()
    for stat_col in OPPONENT_STAT:
        adj_col = f"{stat_col}_adj_avg"
        raw_col = f"{stat_col}_avg"
        result[adj_col] = result[adj_col].fillna(result[raw_col])
    return result


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
