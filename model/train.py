# trains + saves logistic regression model
"""Logistic regression: input = home team's rolling stats minus away team's
rolling stats, output = probability the home team wins. Trains on the first 9
of the 10 cached seasons, tests on the 10th, and prints accuracy so we know
whether to trust it before running it on live games.
"""

import os

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import HISTORICAL_SEASONS
from data.fetch_injuries import historical_injury_impact
from data.situational import (
    away_travel_penalty, blowout_loss_flags, is_short_week, lookahead_flags,
)
from model.elo import compute_elo_ratings

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT_DIR, "data", "cache")
SCHEDULES_PATH = os.path.join(CACHE_DIR, "schedules.parquet")
TEAM_STATS_PATH = os.path.join(CACHE_DIR, "team_stats.parquet")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

TRAIN_SEASONS = HISTORICAL_SEASONS[:-1]
TEST_SEASON = HISTORICAL_SEASONS[-1]

# Rolling stats from Section 3 -- diffed as (home - away) to build each game's
# feature row.
STAT_COLS = [
    "points_for_avg", "points_against_avg",
    "off_epa_per_play_avg", "def_epa_per_play_avg",
    "off_ypp_avg", "def_ypp_avg",
    "off_third_down_pct_avg", "def_third_down_pct_avg",
    "turnover_diff_avg", "red_zone_td_pct_avg",
    "ats_win_pct_season", "ats_win_pct_last5",
    # Section 1: opponent-adjusted versions of the same EPA/play and
    # yards/play stats (data/opponent_adjust.py), kept alongside the raw
    # ones above rather than replacing them -- correlated but not
    # redundant, and worth letting a regularized model weigh separately.
    "off_epa_per_play_adj_avg", "def_epa_per_play_adj_avg",
    "off_ypp_adj_avg", "def_ypp_adj_avg",
    # Phase 4 (v3 spec): success rate and receiving-YAC-over-expected.
    # data/team_stats.py also computes starting-QB-specific EPA/CPOE
    # (qb_epa_per_play_avg, qb_cpoe_avg) but an ablation test showed it
    # actively hurting 2025 holdout accuracy (0.625 -> 0.612 ensemble,
    # cancelling out the real gains from the two features below) --
    # likely collinear with off_epa_per_play_avg (QB dropbacks dominate
    # team offensive EPA) plus real early-season CPOE noise (rolling
    # averages swung as wide as -37 to +28 in the first couple of games).
    # Left out of training rather than kept on faith; still computed and
    # available in team_stats.parquet for narrative/props use.
    "off_success_rate_avg", "def_success_rate_avg",
    "off_yac_oe_avg", "def_yac_oe_avg",
]
FEATURE_COLS = [f"{c}_diff" for c in STAT_COLS] + [
    "home_field_context_diff", "rest_diff", "injury_impact_diff",
    "off_elo_diff", "def_elo_diff",
    "wind_speed", "short_week_diff", "away_travel_penalty", "div_game",
    "blowout_loss_diff", "lookahead_diff",
]


def build_feature_frame(
    schedules: pd.DataFrame, team_stats: pd.DataFrame, injuries: pd.DataFrame,
    elo_per_game: pd.DataFrame, blowouts: pd.DataFrame, lookaheads: pd.DataFrame,
) -> pd.DataFrame:
    """One row per game: home-minus-away rolling stat diffs + home_win label."""
    reg = schedules[
        (schedules["game_type"] == "REG") & schedules["home_score"].notna()
    ].copy()
    reg = reg[reg["home_score"] != reg["away_score"]]  # drop ties (ambiguous label)

    home = team_stats[team_stats["is_home"]]
    away = team_stats[~team_stats["is_home"]]

    games = reg.merge(
        home, left_on=["season", "week", "home_team"],
        right_on=["season", "week", "team"], how="inner",
    ).merge(
        away, left_on=["season", "week", "away_team"],
        right_on=["season", "week", "team"], how="inner",
        suffixes=("_home", "_away"),
    )

    for col in STAT_COLS:
        games[f"{col}_diff"] = games[f"{col}_home"] - games[f"{col}_away"]

    games["home_field_context_diff"] = (
        games["home_point_diff_avg_home"] - games["away_point_diff_avg_away"]
    )
    games["rest_diff"] = games["rest_days_home"] - games["rest_days_away"]

    # Left merge + fillna(0): a team with no rows in `injuries` that week
    # simply had nothing worth listing, i.e. zero injury impact -- not
    # missing data.
    games = games.merge(
        injuries.rename(columns={"team": "home_team", "injury_impact": "home_injury_impact"}),
        on=["season", "week", "home_team"], how="left",
    ).merge(
        injuries.rename(columns={"team": "away_team", "injury_impact": "away_injury_impact"}),
        on=["season", "week", "away_team"], how="left",
    )
    games["home_injury_impact"] = games["home_injury_impact"].fillna(0.0)
    games["away_injury_impact"] = games["away_injury_impact"].fillna(0.0)
    games["injury_impact_diff"] = games["home_injury_impact"] - games["away_injury_impact"]

    games = games.merge(
        elo_per_game[["game_id", "home_off_elo_pre", "home_def_elo_pre",
                      "away_off_elo_pre", "away_def_elo_pre"]],
        on="game_id", how="left",
    )
    # Home offense vs. away defense, and away offense vs. home defense,
    # kept as two separate features rather than one blended rating -- see
    # model/elo.py's docstring for why the split matters.
    games["off_elo_diff"] = games["home_off_elo_pre"] - games["away_def_elo_pre"]
    games["def_elo_diff"] = games["home_def_elo_pre"] - games["away_off_elo_pre"]

    # Wind is game-level, not team-relative -- both sides play in the same
    # conditions, so no home/away diff makes sense here. NaN (dome/closed
    # roof, or a handful of older outdoor games missing the field) means no
    # wind impact, i.e. 0.
    games["wind_speed"] = games["wind"].fillna(0.0)

    # Section 5 situational spots.
    games["short_week_diff"] = (
        games["rest_days_home"].apply(is_short_week).astype(int)
        - games["rest_days_away"].apply(is_short_week).astype(int)
    )
    games["away_travel_penalty"] = games.apply(
        lambda g: -1 if away_travel_penalty(g["home_team"], g["away_team"], g["gametime"]) else 0,
        axis=1,
    )
    games["div_game"] = games["div_game_x"]  # identical across div_game_x/_y, from the schedule directly

    games = games.merge(
        blowouts.rename(columns={"team": "home_team", "blowout_loss_last_game": "home_blowout_loss"}),
        on=["season", "week", "home_team"], how="left",
    ).merge(
        blowouts.rename(columns={"team": "away_team", "blowout_loss_last_game": "away_blowout_loss"}),
        on=["season", "week", "away_team"], how="left",
    )
    games["blowout_loss_diff"] = games["home_blowout_loss"].fillna(0) - games["away_blowout_loss"].fillna(0)

    games = games.merge(
        lookaheads.rename(columns={"team": "home_team", "lookahead_spot": "home_lookahead"}),
        on=["season", "week", "home_team"], how="left",
    ).merge(
        lookaheads.rename(columns={"team": "away_team", "lookahead_spot": "away_lookahead"}),
        on=["season", "week", "away_team"], how="left",
    )
    games["lookahead_diff"] = games["home_lookahead"].fillna(0) - games["away_lookahead"].fillna(0)

    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    return games.dropna(subset=FEATURE_COLS)


def train_logistic(train_df: pd.DataFrame):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train_df[FEATURE_COLS], train_df["home_win"])
    return model


def train_xgboost(train_df: pd.DataFrame) -> XGBClassifier:
    # Small dataset (~2000 rows) -- kept shallow and regularized so it
    # doesn't just memorize train. Tree models don't need feature scaling.
    model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="logloss", random_state=42,
    )
    model.fit(train_df[FEATURE_COLS], train_df["home_win"])
    return model


def predict_proba(model_type: str, logistic_model, xgb_model, X):
    """Win-probability array for whichever model type won, given both
    sub-models. Deliberately a plain function rather than a custom
    picklable class -- a class instance's pickled identity depends on
    which module __name__ happens to be "__main__" in at save time (e.g.
    `python -m model.train` vs `python run_week.py`), which breaks
    unpickling from a different entry point. Plain sklearn/xgboost objects
    don't have that problem, so this keeps every saved artifact to those."""
    if model_type == "xgboost":
        return xgb_model.predict_proba(X)
    if model_type == "logistic":
        return logistic_model.predict_proba(X)
    return (logistic_model.predict_proba(X) + xgb_model.predict_proba(X)) / 2


def train_spread_calibration(train_df: pd.DataFrame, model_type: str, logistic_model, xgb_model) -> LinearRegression:
    """Maps the model's home win probability to an implied Vegas-style spread,
    fit on how the market actually priced games with similar win probabilities
    (Section 4.3) -- lets predict.py compare the model directly against the
    current spread_line to find an edge."""
    win_prob = predict_proba(model_type, logistic_model, xgb_model, train_df[FEATURE_COLS])[:, 1]
    return LinearRegression().fit(win_prob.reshape(-1, 1), train_df["spread_line"])


def _evaluate(name: str, logistic_model, xgb_model, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_proba = predict_proba(name, logistic_model, xgb_model, train_df[FEATURE_COLS])[:, 1]
    train_acc = accuracy_score(train_df["home_win"], (train_proba >= 0.5).astype(int))
    test_proba = predict_proba(name, logistic_model, xgb_model, test_df[FEATURE_COLS])[:, 1]
    test_acc = accuracy_score(test_df["home_win"], (test_proba >= 0.5).astype(int))
    test_loss = log_loss(test_df["home_win"], test_proba)
    print(f"  {name:12s} train acc {train_acc:.3f}  test acc {test_acc:.3f}  test log loss {test_loss:.3f}")
    return {"name": name, "test_acc": test_acc, "test_loss": test_loss}


def main():
    schedules = pd.read_parquet(SCHEDULES_PATH)
    team_stats = pd.read_parquet(TEAM_STATS_PATH)
    print("Pulling historical injury reports...")
    injuries = historical_injury_impact(HISTORICAL_SEASONS)
    print("Backfilling Elo ratings...")
    elo_per_game, _ = compute_elo_ratings(schedules)
    blowouts = blowout_loss_flags(schedules)
    lookaheads = lookahead_flags(schedules)

    games = build_feature_frame(schedules, team_stats, injuries, elo_per_game, blowouts, lookaheads)
    train_df = games[games["season"].isin(TRAIN_SEASONS)]
    test_df = games[games["season"] == TEST_SEASON]
    print(f"Train: {len(train_df)} games ({TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]})")
    print(f"Test:  {len(test_df)} games ({TEST_SEASON})")
    baseline_acc = test_df["home_win"].mean()
    print(f"Always-pick-home baseline: {max(baseline_acc, 1 - baseline_acc):.3f}")

    logistic_model = train_logistic(train_df)
    xgb_model = train_xgboost(train_df)

    print("Comparing candidate models on the 2025 holdout:")
    candidates = [
        _evaluate("logistic", logistic_model, xgb_model, train_df, test_df),
        _evaluate("xgboost", logistic_model, xgb_model, train_df, test_df),
        _evaluate("ensemble", logistic_model, xgb_model, train_df, test_df),
    ]
    winner = max(candidates, key=lambda c: (c["test_acc"], -c["test_loss"]))
    model_type = winner["name"]
    print(f"Winner: {model_type} (test acc {winner['test_acc']:.3f})")

    spread_calibration = train_spread_calibration(train_df, model_type, logistic_model, xgb_model)

    joblib.dump({
        "logistic_model": logistic_model,
        "xgb_model": xgb_model,
        "model_type": model_type,
        "feature_cols": FEATURE_COLS,
        "spread_calibration": spread_calibration,
        # So the report can show real, current numbers instead of a
        # hardcoded string that goes stale the next time this is retrained.
        "test_accuracy": winner["test_acc"],
        "baseline_accuracy": max(baseline_acc, 1 - baseline_acc),
        "test_season": TEST_SEASON,
    }, MODEL_PATH)
    print(f"Saved {model_type} model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
