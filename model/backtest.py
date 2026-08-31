# Week 1 Audit & Tuning Plan Phase 5: full walk-forward Week 1 backtest
"""Every prior phase touched one season (the true 2025 holdout) or one
parameter (Phase 4.1/4.2's grids). This is the plan's own explicit ask:
prove the whole pipeline works across every season it can, not just the
one it happens to be tuned against right now.

For each season from the earliest one with enough prior history
onward, trains fresh logistic + xgboost models on every season strictly
before it (never on the test season itself, never on a later one),
predicts ONLY that season's Week 1 -- the exact case the rest of this
plan has been auditing -- and grades with the same full_backtest_metrics()
Phase 4.3 already built. Reuses model/train.py's own
train_logistic/train_xgboost/predict_proba/train_spread_calibration
directly rather than re-deriving them, so this backtest is honestly
testing the same training code path that actually ships, not a
parallel reimplementation that could quietly drift from it.

This project's cache only holds 10 seasons (2016-2025, config.py's
HISTORICAL_SEASONS), not the plan's literal "2015" -- there's no 2015
data to fit on, so the walk-forward starts at the earliest season with
model/train.py's own MIN_WALK_FORWARD_TRAIN_SEASONS (6) prior seasons
banked, same floor walk_forward_folds() already uses for model-type
selection.

Model type is pinned to whatever model/train.py's main() actually
deploys (xgboost, chosen by its own walk-forward vote) -- this backtest
is meant to answer "does the real, shipped pipeline hold up across
history," not "which of three model types is best in hindsight" (that
question belongs to select_model_type_by_walk_forward(), not here).
"""

import numpy as np
import pandas as pd

from config import HISTORICAL_SEASONS
from data.baselines import print_baselines
from data.fetch_injuries import historical_injury_impact
from data.situational import blowout_loss_flags, lookahead_flags
from model.calibration import full_backtest_metrics, print_benchmark_comparison
from model.elo import compute_elo_ratings
from model.train import (
    FEATURE_COLS, MIN_WALK_FORWARD_TRAIN_SEASONS, build_feature_frame,
    predict_proba, train_logistic, train_spread_calibration, train_xgboost,
)

MODEL_TYPE = "xgboost"


class _PrecomputedSpread:
    """Stands in for a fitted spread_calibration when pooling folds --
    each fold's implied_spread was already computed with THAT fold's own
    spread_calibration (fit only on that fold's training data), so
    pooling must reuse those per-fold values rather than fit one
    calibration across the whole backtest, which would leak later
    seasons' market pricing into earlier ones."""
    def __init__(self, implied_spread):
        self.implied_spread = np.asarray(implied_spread)

    def predict(self, X):
        return self.implied_spread


def run_backtest(games: pd.DataFrame, seasons: list[int],
                  min_train_seasons: int = MIN_WALK_FORWARD_TRAIN_SEASONS) -> list[dict]:
    fold_results = []
    for i in range(min_train_seasons, len(seasons)):
        train_seasons, test_season = seasons[:i], seasons[i]
        train_df = games[games["season"].isin(train_seasons)]
        test_week1 = games[(games["season"] == test_season) & (games["week"] == 1)]
        if train_df.empty or test_week1.empty:
            continue

        logistic_model = train_logistic(train_df)
        xgb_model = train_xgboost(train_df)
        spread_calibration = train_spread_calibration(train_df, MODEL_TYPE, logistic_model, xgb_model)
        proba = predict_proba(MODEL_TYPE, logistic_model, xgb_model, test_week1[FEATURE_COLS])[:, 1]
        implied_spread = spread_calibration.predict(proba.reshape(-1, 1))

        metrics = full_backtest_metrics(test_week1, proba, spread_calibration)
        metrics["test_season"] = test_season
        metrics["n_games"] = len(test_week1)
        metrics["_test_week1"] = test_week1
        metrics["_proba"] = proba
        metrics["_implied_spread"] = implied_spread
        fold_results.append(metrics)
    return fold_results


def pooled_metrics(fold_results: list[dict]) -> dict:
    """Genuine pooling, not an average-of-averages: concatenates every
    fold's actual test rows, win-probabilities, and (fold-specific)
    implied spreads into one set and grades that set once, so Brier/log
    loss/MAE/ATS accuracy are all computed the statistically correct
    way (on the raw pooled predictions) rather than by re-averaging
    already-summarized per-fold numbers."""
    all_test = pd.concat([f["_test_week1"] for f in fold_results], ignore_index=True)
    all_proba = np.concatenate([f["_proba"] for f in fold_results])
    all_implied_spread = np.concatenate([f["_implied_spread"] for f in fold_results])
    return full_backtest_metrics(all_test, all_proba, _PrecomputedSpread(all_implied_spread))


def main():
    schedules = pd.read_parquet("data/cache/schedules.parquet")
    team_stats = pd.read_parquet("data/cache/team_stats.parquet")
    injuries = historical_injury_impact(HISTORICAL_SEASONS)
    elo_per_game, _ = compute_elo_ratings(schedules)
    blowouts = blowout_loss_flags(schedules)
    lookaheads = lookahead_flags(schedules)
    games = build_feature_frame(schedules, team_stats, injuries, elo_per_game, blowouts, lookaheads)

    fold_results = run_backtest(games, HISTORICAL_SEASONS)

    print(f"Walk-forward Week 1 backtest ({MODEL_TYPE}, {len(fold_results)} folds, "
          f"cache covers {HISTORICAL_SEASONS[0]}-{HISTORICAL_SEASONS[-1]}):\n")
    print(f"{'season':>6}  {'n':>3}  {'straight_up':>11}  {'brier':>6}  {'ats':>6}  {'mae':>6}")
    for f in fold_results:
        print(f"{f['test_season']:>6}  {f['n_games']:>3}  {f['straight_up_accuracy']:>11.3f}  "
              f"{f['brier_score']:>6.3f}  {f['ats_accuracy']:>6.3f}  {f['mae_margin']:>6.3f}")

    pooled = pooled_metrics(fold_results)
    total_games = sum(f["n_games"] for f in fold_results)
    print(f"\nPooled across all {len(fold_results)} folds ({total_games} Week 1 games, "
          f"never fit and tested on the same season):")
    print_benchmark_comparison(pooled)

    week1_games = games[
        games["season"].isin([f["test_season"] for f in fold_results]) & (games["week"] == 1)
    ]
    print()
    print_baselines(week1_games, schedules)
    return fold_results, pooled


if __name__ == "__main__":
    main()
