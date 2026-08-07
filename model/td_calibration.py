# checks whether anytime-TD probability predictions actually mean what they say
"""Calibration audit for the anytime-TD model, matching model/calibration.py's
rigor for the win-probability model: reliability table (predicted vs.
actual rate per confidence bucket), Brier score, and AUC on
model/td_backtest.py's walk-forward output. Per-position, not one
blended curve -- RBs/WRs/TEs/QBs(rushing) score at genuinely different
rates and shapes, so a single combined calibration curve would hide a
position-specific miscalibration inside the average.

Per this spec's own framing: raw "% correct" is misleading for anytime-TD
since most players in any given week don't score, so a model that always
predicts "No" would show a deceptively high raw accuracy while being
useless. The two honest skill measures are calibration (Brier score vs.
the naive always-predict-the-base-rate score) and discrimination (AUC --
does the model correctly rank who's more likely to score above who's
less likely). Both are reported here, never raw accuracy alone.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

BUCKET_EDGES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 1.0]


def reliability_table(predicted: pd.Series, actual: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"predicted": predicted, "actual": actual})
    df["bucket"] = pd.cut(df["predicted"], BUCKET_EDGES)
    return df.groupby("bucket", observed=True).agg(
        n=("actual", "size"), predicted_mean=("predicted", "mean"), actual_rate=("actual", "mean"),
    )


def naive_baseline_brier(actual: pd.Series) -> float:
    """Brier score of always predicting the population's own base rate --
    the honest "does this beat doing nothing smarter than the average"
    bar. A model has to clear this, not just clear 0."""
    base_rate = actual.mean()
    return float(brier_score_loss(actual, np.full(len(actual), base_rate)))


def evaluate(predicted: pd.Series, actual: pd.Series) -> dict:
    if predicted.empty or actual.nunique() < 2:
        return {"n": len(predicted), "brier": None, "naive_brier": None, "auc": None}
    return {
        "n": len(predicted),
        "brier": float(brier_score_loss(actual, predicted)),
        "naive_brier": naive_baseline_brier(actual),
        "auc": float(roc_auc_score(actual, predicted)),
    }


def evaluate_by_position(results: pd.DataFrame) -> dict:
    """{position: evaluate(...)} -- see this module's docstring on why
    per-position, not pooled."""
    out = {}
    for position, group in results.groupby("position"):
        out[position] = evaluate(group["predicted_prob"], group["actual_td"])
    return out


def _game_model_brier() -> float | None:
    """The win-probability model's own Brier score, computed fresh via
    model/calibration.py's real evaluation (not a stale hardcoded number)
    -- the fairness comparison this spec asks for."""
    try:
        import joblib
        from sklearn.metrics import brier_score_loss as _bsl

        from data.fetch_injuries import historical_injury_impact
        from data.situational import blowout_loss_flags, lookahead_flags
        from model.elo import compute_elo_ratings
        from model.train import (
            FEATURE_COLS, MODEL_PATH, SCHEDULES_PATH, TEAM_STATS_PATH, TEST_SEASON, TRAIN_SEASONS,
            build_feature_frame, predict_proba,
        )
        from config import HISTORICAL_SEASONS

        schedules = pd.read_parquet(SCHEDULES_PATH)
        team_stats = pd.read_parquet(TEAM_STATS_PATH)
        injuries = historical_injury_impact(HISTORICAL_SEASONS)
        elo_per_game, _ = compute_elo_ratings(schedules)
        blowouts = blowout_loss_flags(schedules)
        lookaheads = lookahead_flags(schedules)
        games = build_feature_frame(schedules, team_stats, injuries, elo_per_game, blowouts, lookaheads)
        test_df = games[games["season"] == TEST_SEASON]

        saved = joblib.load(MODEL_PATH)
        proba = predict_proba(saved["model_type"], saved["logistic_model"], saved["xgb_model"], test_df[FEATURE_COLS])[:, 1]
        return float(_bsl(test_df["home_win"], proba))
    except Exception as e:
        print(f"(couldn't compute game-model Brier score for comparison: {e})")
        return None


def main():
    import os

    path = "data/cache/td_backtest.parquet"
    if not os.path.exists(path):
        print(f"No backtest results at {path} yet -- run `python -m model.td_backtest` first.")
        return
    results = pd.read_parquet(path)
    print(f"TD backtest: {len(results)} (player, week) predictions across "
          f"{results['season'].nunique()} season(s) ({sorted(results['season'].unique())}).")

    overall = evaluate(results["predicted_prob"], results["actual_td"])
    print(f"\nOverall: n={overall['n']}  Brier={overall['brier']:.4f}  "
          f"naive-baseline Brier={overall['naive_brier']:.4f}  AUC={overall['auc']:.4f}")
    print(f"(Lower Brier = better calibrated; AUC 0.5 = no better than random ranking, "
          f"1.0 = perfect ranking. Target: Brier meaningfully below the naive baseline, AUC >= 0.75.)")

    print("\nReliability table (overall):")
    print(reliability_table(results["predicted_prob"], results["actual_td"]))

    print("\nPer-position:")
    for position, metrics in sorted(evaluate_by_position(results).items()):
        if metrics["brier"] is None:
            continue
        print(f"  {position:4s} n={metrics['n']:5d}  Brier={metrics['brier']:.4f}  "
              f"naive={metrics['naive_brier']:.4f}  AUC={metrics['auc']:.4f}")

    game_brier = _game_model_brier()
    if game_brier is not None:
        print(f"\nFor comparison, the win-probability model's own Brier score "
              f"(2025 holdout): {game_brier:.4f}")


if __name__ == "__main__":
    main()
