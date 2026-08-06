# compute Brier score, audit confidence buckets, recalibrate if needed
"""Checks whether the model's stated win probabilities actually mean what
they say -- if games the model calls "70% confidence" only win 60% of the
time, the "edge" numbers in the report are systematically misleading, not
just imprecise.

Reports the Brier score and a bucketed reliability table (predicted vs.
actual win rate per confidence bucket) on the true holdout season. Only
recalibrates (Platt scaling or isotonic regression) if it demonstrably
improves the *held-out* Brier score -- fitting and evaluating a calibrator
on the same 233-game test set would be circular, and isotonic regression in
particular can overfit a sample this small if not done carefully. Instead,
the calibrator is fit on out-of-fold predictions from a 5-fold CV split of
the *training* seasons, then judged honestly on the untouched test season.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import KFold

from config import HISTORICAL_SEASONS
from data.fetch_injuries import historical_injury_impact
from data.situational import blowout_loss_flags, lookahead_flags
from model.elo import compute_elo_ratings
from model.train import (
    FEATURE_COLS, SCHEDULES_PATH, TEAM_STATS_PATH, TEST_SEASON, TRAIN_SEASONS,
    build_feature_frame, predict_proba, train_logistic, train_xgboost,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

BUCKET_EDGES = [0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 1.0]


def reliability_table(proba: np.ndarray, actual: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"proba": proba, "actual": actual})
    df["bucket"] = pd.cut(df["proba"], BUCKET_EDGES)
    return df.groupby("bucket", observed=True).agg(
        n=("actual", "size"), predicted_mean=("proba", "mean"), actual_rate=("actual", "mean"),
    )


def _out_of_fold_train_probs(model_type: str, train_df: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
    """Retrains model_type on 4/5 of the training seasons and predicts the
    held-out 1/5, repeated across folds -- gives predictions on the
    training set that weren't seen during their own fitting, the same
    principle sklearn's CalibratedClassifierCV uses. A calibrator fit
    directly on plain in-sample training predictions would just learn "this
    model is more confident than it should be" from overfitting, not from
    genuine miscalibration."""
    oof = np.zeros(len(train_df))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, holdout_idx in kfold.split(train_df):
        fold_train, fold_holdout = train_df.iloc[train_idx], train_df.iloc[holdout_idx]
        logistic_model = train_logistic(fold_train)
        xgb_model = train_xgboost(fold_train)
        oof[holdout_idx] = predict_proba(model_type, logistic_model, xgb_model, fold_holdout[FEATURE_COLS])[:, 1]
    return oof


# A calibrator has to beat "do nothing" by more than this relative margin
# on the true holdout to actually get deployed -- with only ~230 holdout
# games, a 0.1% Brier improvement is noise, not a real correction, and
# isotonic regression in particular can look marginally better while
# actually just fitting a jagged, unstable curve (watch for degenerate
# single-game buckets in the reliability table as a tell).
MIN_RELATIVE_IMPROVEMENT = 0.01


def fit_best_calibrator(model_type: str, train_df: pd.DataFrame, test_proba: np.ndarray, test_actual: np.ndarray):
    """Returns (name, calibrator_or_None) -- whichever of {none, platt,
    isotonic} has the best Brier score on the true held-out test set, but
    "none" wins unless a calibrator beats it by more than
    MIN_RELATIVE_IMPROVEMENT. This shouldn't add a step that doesn't
    demonstrably earn its keep."""
    oof_proba = _out_of_fold_train_probs(model_type, train_df)
    oof_actual = train_df["home_win"].values

    candidates = {"none": (None, test_proba)}

    platt = LogisticRegression().fit(oof_proba.reshape(-1, 1), oof_actual)
    candidates["platt"] = (platt, platt.predict_proba(test_proba.reshape(-1, 1))[:, 1])

    isotonic = IsotonicRegression(out_of_bounds="clip").fit(oof_proba, oof_actual)
    candidates["isotonic"] = (isotonic, isotonic.predict(test_proba))

    scores = {name: brier_score_loss(test_actual, proba) for name, (_, proba) in candidates.items()}
    print("Held-out Brier score by calibration method:")
    for name, score in sorted(scores.items(), key=lambda kv: kv[1]):
        print(f"  {name:10s} {score:.4f}")

    raw_score = scores["none"]
    best_name = min(scores, key=scores.get)
    if best_name != "none":
        improvement = (raw_score - scores[best_name]) / raw_score
        if improvement < MIN_RELATIVE_IMPROVEMENT:
            print(f"Best candidate ({best_name}) only improves Brier score by "
                  f"{improvement:.2%}, under the {MIN_RELATIVE_IMPROVEMENT:.0%} bar -- "
                  f"keeping the model uncalibrated rather than chasing noise.")
            best_name = "none"
    print(f"Decision: {best_name}")

    return best_name, candidates[best_name][0]


def apply_calibrator(calibrator, proba: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return proba
    if isinstance(calibrator, IsotonicRegression):
        return calibrator.predict(proba)
    return calibrator.predict_proba(proba.reshape(-1, 1))[:, 1]


def main():
    schedules = pd.read_parquet(SCHEDULES_PATH)
    team_stats = pd.read_parquet(TEAM_STATS_PATH)
    injuries = historical_injury_impact(HISTORICAL_SEASONS)
    elo_per_game, _ = compute_elo_ratings(schedules)
    blowouts = blowout_loss_flags(schedules)
    lookaheads = lookahead_flags(schedules)
    games = build_feature_frame(schedules, team_stats, injuries, elo_per_game, blowouts, lookaheads)
    train_df = games[games["season"].isin(TRAIN_SEASONS)]
    test_df = games[games["season"] == TEST_SEASON]

    saved = joblib.load(MODEL_PATH)
    model_type = saved["model_type"]
    raw_test_proba = predict_proba(
        model_type, saved["logistic_model"], saved["xgb_model"], test_df[FEATURE_COLS])[:, 1]
    test_actual = test_df["home_win"].values

    print(f"Model: {model_type}, test season: {TEST_SEASON}")
    print(f"Raw Brier score: {brier_score_loss(test_actual, raw_test_proba):.4f}")
    print("\nReliability table (raw, uncalibrated):")
    print(reliability_table(raw_test_proba, test_actual))
    print()

    best_name, calibrator = fit_best_calibrator(model_type, train_df, raw_test_proba, test_actual)

    if calibrator is not None:
        calibrated_proba = apply_calibrator(calibrator, raw_test_proba)
        print(f"\nReliability table ({best_name}-calibrated):")
        print(reliability_table(calibrated_proba, test_actual))

    saved["calibrator_name"] = best_name
    saved["calibrator"] = calibrator
    joblib.dump(saved, MODEL_PATH)
    print(f"\nSaved calibrator ({best_name}) -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
