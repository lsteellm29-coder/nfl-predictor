# loads model, scores this week's games
"""Loads the trained model and scores one week's games: win probability,
implied spread, and the edge against the current Vegas line.

Pre-game team stats come from the current season's games played so far. Early
in a season a team may not have played yet, in which case its stats carry
forward from the end of the previous season (there's no in-season signal yet,
so last season's final read on the team is the best available estimate).
"""

import math
import os

import joblib
import nfl_data_py as nfl
import pandas as pd
import requests

from config import CURRENT_SEASON
from data.fetch_injuries import fetch_current_injury_impact
from data.fetch_week import fetch_week
from data.team_stats import build_rolling_team_stats, build_team_game_stats
from model.elo import compute_elo_ratings
from model.train import FEATURE_COLS, STAT_COLS

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAM_STATS_PATH = os.path.join(ROOT_DIR, "data", "cache", "team_stats.parquet")
PBP_PATH = os.path.join(ROOT_DIR, "data", "cache", "pbp.parquet")
SCHEDULES_PATH = os.path.join(ROOT_DIR, "data", "cache", "schedules.parquet")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

SPLIT_COLS = ["home_point_diff_avg", "away_point_diff_avg"]
STATS_COLS = STAT_COLS + SPLIT_COLS


def _current_season_stats(season: int, week: int) -> pd.DataFrame:
    """Rolling stats built from this season's games played before `week`.
    Empty if the season hasn't started yet or no one has played."""
    schedule = nfl.import_schedules([season])
    played = schedule[
        (schedule["game_type"] == "REG")
        & schedule["home_score"].notna()
        & (schedule["week"] < week)
    ]
    if played.empty:
        return pd.DataFrame(columns=["team", *STATS_COLS])

    pbp = nfl.import_pbp_data([season], downcast=True)
    team_game_stats = build_team_game_stats(schedule, pbp)
    rolling = build_rolling_team_stats(team_game_stats)
    rolling = rolling[rolling["week"] < week]
    return rolling.sort_values("week").groupby("team").tail(1)


def _fallback_stats(season: int) -> pd.DataFrame:
    """Each team's final rolling-stat row from the prior cached season, for
    teams with no games played yet this season."""
    historical = pd.read_parquet(TEAM_STATS_PATH)
    prior = historical[historical["season"] == season - 1]
    return prior.sort_values("week").groupby("team").tail(1)


def get_current_elo_ratings(season: int) -> dict:
    """Each team's Elo rating as of right now, computed by replaying every
    game from the start of the cached historical window through whatever's
    been played of the current season so far -- Elo has to be run
    continuously to mean anything, unlike the rolling stats which reset
    each season.

    The full current-season schedule (including future, unplayed games) is
    passed in, not filtered to a target week -- unplayed games never update
    a rating (compute_elo_ratings skips games with no score), but the
    season needs at least one row present so the season-boundary
    regression-to-the-mean actually fires even before the season's first
    game has been played."""
    historical = pd.read_parquet(SCHEDULES_PATH)
    current = nfl.import_schedules([season])
    combined = pd.concat([historical, current], ignore_index=True)
    _, ratings = compute_elo_ratings(combined)
    return ratings


def get_pregame_stats(season: int, week: int) -> pd.DataFrame:
    current = _current_season_stats(season, week)
    fallback = _fallback_stats(season)
    combined = pd.concat([
        current,
        fallback[~fallback["team"].isin(current["team"])],
    ])
    return combined.set_index("team")


def _build_features(
    game: pd.Series, stats: pd.DataFrame, injury_impact: dict, elo_ratings: dict,
) -> dict | None:
    home, away = game["home_team"], game["away_team"]
    if home not in stats.index or away not in stats.index:
        return None
    h, a = stats.loc[home], stats.loc[away]

    feat = {f"{col}_diff": h[col] - a[col] for col in STAT_COLS}
    feat["home_field_context_diff"] = h["home_point_diff_avg"] - a["away_point_diff_avg"]
    feat["rest_diff"] = game["home_rest"] - game["away_rest"]
    # A team with no key absent from the live injury feed just had nothing
    # worth listing -- 0 impact, not missing data.
    feat["injury_impact_diff"] = injury_impact.get(home, 0.0) - injury_impact.get(away, 0.0)
    feat["elo_diff"] = elo_ratings.get(home, 1500.0) - elo_ratings.get(away, 1500.0)
    return feat


def _offensive_tds(pbp: pd.DataFrame) -> pd.DataFrame:
    """Rush + receiving TD plays -- the standard "anytime TD scorer"
    definition. Excludes return TDs (special teams/defense, not an
    offensive skill-position score)."""
    return pbp[
        (pbp["touchdown"] == 1)
        & ((pbp["pass_touchdown"] == 1) | (pbp["rush_touchdown"] == 1))
    ]


def _player_td_counts(pbp: pd.DataFrame) -> pd.DataFrame:
    """Rush + receiving TDs per (team, player)."""
    tds = _offensive_tds(pbp)
    tds = tds[tds["td_player_name"].notna()]
    return (
        tds.groupby(["posteam", "td_player_name"])
        .size()
        .reset_index(name="td_count")
    )


def _defense_td_rate_allowed(pbp: pd.DataFrame, games_played: dict) -> dict:
    """Rush + receiving TDs allowed per game, by defense -- how often a
    team's defense lets *any* offensive player score, independent of who."""
    tds = _offensive_tds(pbp)
    allowed = tds.groupby("defteam").size()
    return {
        team: allowed.get(team, 0) / games
        for team, games in games_played.items() if games > 0
    }


def _team_games_played(schedule: pd.DataFrame) -> dict:
    """team -> number of completed REG games in the given schedule slice."""
    reg = schedule[(schedule["game_type"] == "REG") & schedule["home_score"].notna()]
    home = reg.groupby("home_team").size()
    away = reg.groupby("away_team").size()
    return home.add(away, fill_value=0).to_dict()


def _current_season_td_data(season: int, week: int):
    schedule = nfl.import_schedules([season])
    played = schedule[
        (schedule["game_type"] == "REG")
        & schedule["home_score"].notna()
        & (schedule["week"] < week)
    ]
    if played.empty:
        return pd.DataFrame(columns=["posteam", "td_player_name", "td_count"]), {}, {}

    pbp = nfl.import_pbp_data([season], downcast=True)
    pbp = pbp[pbp["week"] < week]
    games_played = _team_games_played(played)
    counts = _player_td_counts(pbp)
    def_rates = _defense_td_rate_allowed(pbp, games_played)
    return counts, games_played, def_rates


def _fallback_td_data(season: int):
    """Prior season's full-season TD counts + games played, for teams/players
    with no current-season data yet. Note: attributes players to whatever
    team they played for *last* season -- if someone changed teams in the
    offseason, this won't reflect that until they've scored for their new
    team."""
    pbp = pd.read_parquet(PBP_PATH)
    pbp = pbp[pbp["season"] == season - 1]
    schedule = pd.read_parquet(SCHEDULES_PATH)
    games_played = _team_games_played(schedule[schedule["season"] == season - 1])
    counts = _player_td_counts(pbp)
    def_rates = _defense_td_rate_allowed(pbp, games_played)
    return counts, games_played, def_rates


def get_td_scorer_prediction(
    team: str, opponent: str, current, fallback,
    def_rates: dict, league_avg_def_rate: float,
    implied_totals: dict, league_avg_implied_total: float,
) -> dict | None:
    (current_counts, current_games), (fallback_counts, fallback_games) = current, fallback

    pool, games, source = current_counts[current_counts["posteam"] == team], current_games.get(team, 0), "this season"
    if pool.empty or pool["td_count"].sum() == 0 or games == 0:
        pool = fallback_counts[fallback_counts["posteam"] == team]
        games, source = fallback_games.get(team, 0), "last season"
    if pool.empty or games == 0:
        return None

    top = pool.sort_values("td_count", ascending=False).iloc[0]
    td_count = int(top["td_count"])
    base_rate = td_count / games

    # Two matchup adjustments to the player's own scoring rate, each capped
    # at 0.6x-1.6x so a small sample (e.g. 2 games of defensive data) can't
    # swing the estimate wildly:
    #  - opposing defense: how many TDs it allows per game vs. league average
    #  - game environment: this game's Vegas-implied team total vs. the
    #    average implied total across this week's games (a shootout raises
    #    everyone's odds; a low-scoring line lowers them)
    def_factor = 1.0
    if league_avg_def_rate and opponent in def_rates:
        def_factor = _clip(def_rates[opponent] / league_avg_def_rate, 0.6, 1.6)

    total_factor = 1.0
    if league_avg_implied_total and team in implied_totals:
        total_factor = _clip(implied_totals[team] / league_avg_implied_total, 0.6, 1.6)

    adjusted_rate = base_rate * def_factor * total_factor
    prob = 1 - math.exp(-adjusted_rate)
    return {
        "player": top["td_player_name"], "td_count": td_count, "games": int(games),
        "source": source, "base_prob": 1 - math.exp(-base_rate),
        "def_factor": def_factor, "total_factor": total_factor, "prob": prob,
    }


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def get_implied_team_totals(games: pd.DataFrame) -> dict:
    """Vegas-implied points for each team in this week's games, derived from
    total_line and spread_line (home team's favored margin): home = (total +
    spread) / 2, away = (total - spread) / 2."""
    totals = {}
    for _, g in games.iterrows():
        if pd.isna(g.get("total_line")) or pd.isna(g.get("spread_line")):
            continue
        totals[g["home_team"]] = (g["total_line"] + g["spread_line"]) / 2
        totals[g["away_team"]] = (g["total_line"] - g["spread_line"]) / 2
    return totals


def _logistic_contributions(model, feat_df: pd.DataFrame) -> dict:
    """Each feature's signed contribution to the home-win logit: the
    standardized value the logistic regression actually saw, times its
    learned coefficient. This is what the model actually did, not an
    assumption about which raw stat direction is 'good' (see the
    def_epa_per_play_avg_diff sign flip caught during training -- collinear
    features can have counter-intuitive individual coefficients)."""
    scaler = model.named_steps["standardscaler"]
    logreg = model.named_steps["logisticregression"]
    scaled = scaler.transform(feat_df)[0]
    return dict(zip(feat_df.columns, scaled * logreg.coef_[0]))


def _xgboost_contributions(model, feat_df: pd.DataFrame) -> dict:
    """Exact per-feature SHAP contributions to the predicted margin, via
    XGBoost's own TreeExplainer-equivalent (pred_contribs=True) -- this is
    the real decomposition of what the trees did, not an approximation.
    Coefficients don't exist for a tree ensemble, so this replaces
    _logistic_contributions for that model type."""
    import xgboost

    booster = model.get_booster()
    contribs = booster.predict(xgboost.DMatrix(feat_df), pred_contribs=True)[0]
    return dict(zip(feat_df.columns, contribs[:-1]))  # last column is the bias term


def _feature_contributions(model, model_type: str, feat_df: pd.DataFrame) -> dict:
    if model_type == "xgboost":
        return _xgboost_contributions(model, feat_df)
    if model_type == "ensemble":
        # The ensemble's actual prediction blends both models, but for a
        # readable explanation we use the logistic half's contributions --
        # blending SHAP values with linear coefficients into one coherent
        # story isn't worth the complexity for prose purposes.
        return _logistic_contributions(model.logistic_model, feat_df)
    return _logistic_contributions(model, feat_df)


def score_week(week: int, season: int = CURRENT_SEASON) -> pd.DataFrame:
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    model_type = saved.get("model_type", "logistic")
    spread_calibration = saved["spread_calibration"]

    games = fetch_week(week, season)
    stats = get_pregame_stats(season, week)
    elo_ratings = get_current_elo_ratings(season)

    try:
        injury_impact = fetch_current_injury_impact()
    except requests.RequestException as e:
        print(f"Warning: couldn't fetch live injury data ({e}); scoring without it.")
        injury_impact = {}

    current_counts, current_games, current_def_rates = _current_season_td_data(season, week)
    fallback_counts, fallback_games, fallback_def_rates = _fallback_td_data(season)
    current_td_data = (current_counts, current_games)
    fallback_td_data = (fallback_counts, fallback_games)

    def_rates = {**fallback_def_rates, **current_def_rates}
    league_avg_def_rate = sum(def_rates.values()) / len(def_rates) if def_rates else 0.0

    implied_totals = get_implied_team_totals(games)
    league_avg_implied_total = (
        sum(implied_totals.values()) / len(implied_totals) if implied_totals else 0.0
    )

    rows = []
    for _, game in games.iterrows():
        feat = _build_features(game, stats, injury_impact, elo_ratings)
        row = game.to_dict()
        row["home_td_scorer"] = get_td_scorer_prediction(
            game["home_team"], game["away_team"], current_td_data, fallback_td_data,
            def_rates, league_avg_def_rate, implied_totals, league_avg_implied_total)
        row["away_td_scorer"] = get_td_scorer_prediction(
            game["away_team"], game["home_team"], current_td_data, fallback_td_data,
            def_rates, league_avg_def_rate, implied_totals, league_avg_implied_total)
        if game["home_team"] in stats.index:
            row["home_stats"] = stats.loc[game["home_team"]].to_dict()
            row["home_stats"]["injury_impact"] = injury_impact.get(game["home_team"], 0.0)
            row["home_stats"]["elo"] = elo_ratings.get(game["home_team"], 1500.0)
        if game["away_team"] in stats.index:
            row["away_stats"] = stats.loc[game["away_team"]].to_dict()
            row["away_stats"]["injury_impact"] = injury_impact.get(game["away_team"], 0.0)
            row["away_stats"]["elo"] = elo_ratings.get(game["away_team"], 1500.0)
        if feat is None:
            row["home_win_prob"] = None
            row["implied_spread"] = None
            row["edge"] = None
            row["top_factors"] = None
            rows.append(row)
            continue

        feat_df = pd.DataFrame([feat])[FEATURE_COLS]
        home_win_prob = model.predict_proba(feat_df)[0, 1]
        implied_spread = spread_calibration.predict([[home_win_prob]])[0]

        row["home_win_prob"] = home_win_prob
        row["implied_spread"] = implied_spread
        # positive edge = model favors the home team more than the market does
        row["edge"] = implied_spread - game["spread_line"] if pd.notna(game["spread_line"]) else None

        contributions = _feature_contributions(model, model_type, feat_df)
        row["top_factors"] = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:3]
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    args = parser.parse_args()

    preds = score_week(args.week, args.season)
    cols = ["away_team", "home_team", "home_win_prob", "spread_line",
            "implied_spread", "edge"]
    with pd.option_context("display.float_format", "{:.2f}".format):
        print(preds[cols].to_string(index=False))


if __name__ == "__main__":
    main()
