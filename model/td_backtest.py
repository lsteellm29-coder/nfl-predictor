# walk-forward backtest for the anytime-TD model: real predicted-prob vs actual outcome, no leakage
"""Builds the (predicted_prob, actual_outcome) pairs model/td_calibration.py
grades -- for every week in a season (after a minimum warm-up), every
player who actually had a real offensive role that week (>=1 rush attempt
or target) gets a pre-game TD probability computed using ONLY strictly
earlier weeks' data (model/td_model.py's own upto_week filtering already
enforces this), then checked against whether they actually scored. This
is the same no-leakage contract model/train.py's rolling stats and
model/predict.py's live scoring already use, just applied to the TD model
instead of the win-probability one.

The evaluation population is deliberately "who actually touched the ball
that week" rather than model/player_stats.py's required_lineup coverage
population -- required_lineup exists to decide who gets a live UI card
(a display/completeness concern), not who belongs in a backtest's
ground truth set. Someone who didn't touch the ball at all that game
was never a real "did the model call this right" test case either way.
"""

import nfl_data_py as nfl
import pandas as pd

from data.positional_matchups import position_map
from model.td_model import (
    season_to_date, player_red_zone_touches, positional_baseline_conversion,
    project_td_probability, recency_weighted_touch_share, team_red_zone_defense, team_red_zone_touches_per_game,
)

MIN_WARMUP_WEEKS = 6  # need real prior-week history before a projection means anything


def _week_participants(pbp: pd.DataFrame, week: int) -> pd.DataFrame:
    """Every (team, player_id) with >=1 rush attempt or target in this
    specific week -- the real population with an actual chance to score,
    used as this backtest's evaluation set."""
    week_pbp = pbp[pbp["week"] == week]
    rush = week_pbp[(week_pbp["play_type"] == "run") & week_pbp["rusher_player_id"].notna()]
    recv = week_pbp[(week_pbp["play_type"] == "pass") & week_pbp["receiver_player_id"].notna()]
    participants = pd.concat([
        rush[["posteam", "rusher_player_id"]].rename(columns={"posteam": "team", "rusher_player_id": "player_id"}),
        recv[["posteam", "receiver_player_id"]].rename(columns={"posteam": "team", "receiver_player_id": "player_id"}),
    ], ignore_index=True).drop_duplicates()
    return participants


def _week_scorers(pbp: pd.DataFrame, week: int) -> set:
    """{(team, player_id)} for every rush/receiving TD scored this
    specific week -- the ground truth actual_outcome=1 set; everyone else
    in _week_participants is actual_outcome=0."""
    week_pbp = pbp[pbp["week"] == week]
    rush_td = week_pbp[(week_pbp["rush_touchdown"] == 1) & week_pbp["rusher_player_id"].notna()]
    rec_td = week_pbp[(week_pbp["pass_touchdown"] == 1) & week_pbp["receiver_player_id"].notna()]
    return (set(zip(rush_td["posteam"], rush_td["rusher_player_id"]))
            | set(zip(rec_td["posteam"], rec_td["receiver_player_id"])))


def _week_opponent_map(schedule: pd.DataFrame, season: int, week: int) -> dict:
    """team -> opponent for this specific week, from the schedule."""
    games = schedule[(schedule["season"] == season) & (schedule["week"] == week) & (schedule["game_type"] == "REG")]
    out = {}
    for _, g in games.iterrows():
        out[g["home_team"]] = g["away_team"]
        out[g["away_team"]] = g["home_team"]
    return out


def run_season_backtest(pbp: pd.DataFrame, schedule: pd.DataFrame, pos_map: dict, season: int,
                         min_week: int = MIN_WARMUP_WEEKS) -> pd.DataFrame:
    """One row per (week, team, player_id) evaluated this season: the
    model's pre-game probability (built from strictly earlier weeks only)
    and whether they actually scored. Weeks before `min_week` are skipped
    -- too little season-to-date history for a projection to mean
    anything, same MIN_GAMES_FOR_TREND-style warm-up gate
    data/player_trends.py already uses elsewhere in this codebase."""
    weeks = sorted(w for w in pbp["week"].dropna().unique() if w >= min_week)
    touch_log = player_red_zone_touches(pbp)
    rz_def_log = team_red_zone_defense(pbp)

    rows = []
    for week in weeks:
        opponent_map = _week_opponent_map(schedule, season, week)
        if not opponent_map:
            continue

        touch_share = recency_weighted_touch_share(touch_log, week)
        if touch_share.empty:
            continue
        team_touches = team_red_zone_touches_per_game(touch_log, week)
        team_def_std = season_to_date(rz_def_log, ["team"], ["rz_trips_allowed", "rz_tds_allowed"], week)
        baseline_rates = positional_baseline_conversion(touch_log, pos_map, week)
        league_avg_rz_td_rate = (
            (team_def_std["rz_tds_allowed"] / team_def_std["rz_trips_allowed"]).mean()
            if not team_def_std.empty else None
        )
        if not league_avg_rz_td_rate:
            continue

        participants = _week_participants(pbp, week)
        scorers = _week_scorers(pbp, week)

        for _, p in participants.iterrows():
            team, player_id = p["team"], p["player_id"]
            opponent = opponent_map.get(team)
            if opponent is None:
                continue
            position = pos_map.get(player_id)
            result = project_td_probability(
                team, opponent, player_id, position, touch_share, team_touches, team_def_std,
                baseline_rates, league_avg_rz_td_rate)
            if result is None:
                continue
            rows.append({
                "season": season, "week": week, "team": team, "player_id": player_id, "position": position,
                "predicted_prob": result["prob"], "actual_td": int((team, player_id) in scorers),
                "n_games": result["n_games"],
                # Underlying components, not just the final Poisson
                # probability -- lets a downstream consumer (e.g. an
                # ensembled classifier, model/td_calibration.py's own
                # ablations) train on the same real signal without
                # duplicating this module's projection math.
                "expected_tds": result["expected_tds"], "player_share": result["player_share"],
                "player_conversion": result["player_conversion"], "def_factor": result["def_factor"],
                "team_rz_touches_per_game": result["team_rz_touches_per_game"],
            })

    return pd.DataFrame(rows)


def run_walk_forward_backtest(seasons: list[int]) -> pd.DataFrame:
    """Concatenates run_season_backtest across every season in `seasons`
    -- each season's projections only ever use that same season's prior
    weeks (touch_log/team logs are built fresh per season, matching
    model/player_stats.py's own season-boundary reset -- Elo is the only
    rating in this codebase that intentionally carries across seasons;
    red-zone role/usage genuinely does reset with rosters and schemes
    each offseason, so re-starting each season fresh here is correct, not
    a limitation)."""
    schedules = pd.read_parquet("data/cache/schedules.parquet")
    pbp_all = pd.read_parquet("data/cache/pbp.parquet")
    pos_map = position_map(seasons)

    frames = []
    for season in seasons:
        pbp = pbp_all[pbp_all["season"] == season]
        print(f"Backtesting {season}...")
        frames.append(run_season_backtest(pbp, schedules, pos_map, season))
    return pd.concat(frames, ignore_index=True)


def main():
    import argparse

    from config import HISTORICAL_SEASONS

    parser = argparse.ArgumentParser()
    # Last 4 of the cached historical seasons by default -- derived from
    # config.py's HISTORICAL_SEASONS (itself relative to CURRENT_SEASON)
    # rather than a hardcoded [2022, 2023, 2024, 2025], so this doesn't
    # quietly go stale as CURRENT_SEASON advances in future years.
    parser.add_argument("--seasons", type=int, nargs="+", default=HISTORICAL_SEASONS[-4:])
    args = parser.parse_args()

    results = run_walk_forward_backtest(args.seasons)
    out_path = "data/cache/td_backtest.parquet"
    results.to_parquet(out_path)
    print(f"\n{len(results)} (player, week) predictions across {results['season'].nunique()} seasons -> {out_path}")
    print(f"Actual TD rate in eval population: {results['actual_td'].mean():.1%}")


if __name__ == "__main__":
    main()
