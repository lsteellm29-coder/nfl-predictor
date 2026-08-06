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

from config import HISTORICAL_SEASONS
from data.fetch_injuries import historical_injury_impact

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
]
FEATURE_COLS = [f"{c}_diff" for c in STAT_COLS] + [
    "home_field_context_diff", "rest_diff", "injury_impact_diff",
]


def build_feature_frame(
    schedules: pd.DataFrame, team_stats: pd.DataFrame, injuries: pd.DataFrame,
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

    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    return games.dropna(subset=FEATURE_COLS)


def train_model(train_df: pd.DataFrame):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train_df[FEATURE_COLS], train_df["home_win"])
    return model


def train_spread_calibration(train_df: pd.DataFrame, model) -> LinearRegression:
    """Maps the model's home win probability to an implied Vegas-style spread,
    fit on how the market actually priced games with similar win probabilities
    (Section 4.3) -- lets predict.py compare the model directly against the
    current spread_line to find an edge."""
    win_prob = model.predict_proba(train_df[FEATURE_COLS])[:, 1]
    return LinearRegression().fit(win_prob.reshape(-1, 1), train_df["spread_line"])


def main():
    schedules = pd.read_parquet(SCHEDULES_PATH)
    team_stats = pd.read_parquet(TEAM_STATS_PATH)
    print("Pulling historical injury reports...")
    injuries = historical_injury_impact(HISTORICAL_SEASONS)

    games = build_feature_frame(schedules, team_stats, injuries)
    train_df = games[games["season"].isin(TRAIN_SEASONS)]
    test_df = games[games["season"] == TEST_SEASON]
    print(f"Train: {len(train_df)} games ({TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]})")
    print(f"Test:  {len(test_df)} games ({TEST_SEASON})")

    model = train_model(train_df)
    spread_calibration = train_spread_calibration(train_df, model)

    train_acc = accuracy_score(train_df["home_win"], model.predict(train_df[FEATURE_COLS]))
    test_pred = model.predict(test_df[FEATURE_COLS])
    test_proba = model.predict_proba(test_df[FEATURE_COLS])[:, 1]
    test_acc = accuracy_score(test_df["home_win"], test_pred)
    test_loss = log_loss(test_df["home_win"], test_proba)
    baseline_acc = test_df["home_win"].mean()  # always-pick-home baseline

    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")
    print(f"Test log loss:  {test_loss:.3f}")
    print(f"Always-pick-home baseline: {max(baseline_acc, 1 - baseline_acc):.3f}")

    joblib.dump({
        "model": model,
        "feature_cols": FEATURE_COLS,
        "spread_calibration": spread_calibration,
    }, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
