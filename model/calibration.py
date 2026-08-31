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
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error
from sklearn.model_selection import KFold

from config import HISTORICAL_SEASONS
from data.baselines import print_baselines
from data.fetch_injuries import historical_injury_impact
from data.situational import blowout_loss_flags, lookahead_flags
from model.elo import compute_elo_ratings
from model.train import (
    FEATURE_COLS, SCHEDULES_PATH, TEAM_STATS_PATH, TEST_SEASON, TRAIN_SEASONS,
    build_feature_frame, predict_proba, train_logistic, train_xgboost,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

BUCKET_EDGES = [0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 1.0]

# Week 1 Audit & Tuning Plan Phase 4.3's stated benchmarks -- reference
# points to print backtest numbers next to, not a hard pass/fail gate.
# "Vegas" is what the market itself achieves; beating "Good" doesn't mean
# beating Vegas, and posting something dramatically ABOVE Vegas-level ATS
# is itself a leakage red flag (the plan's own words: "if you post 70%+
# ATS in a backtest, you have leakage. Go back to Phase 2"), not a win.
BENCHMARKS = {
    "straight_up_accuracy": {"bad": 0.60, "ok": 0.63, "good": 0.67, "vegas": 0.67, "higher_is_better": True},
    "brier_score": {"bad": 0.24, "ok": 0.22, "good": 0.20, "vegas": 0.19, "higher_is_better": False},
    "ats_accuracy": {"bad": 0.50, "ok": 0.52, "good": 0.55, "vegas": 0.50, "higher_is_better": True},
    "mae_margin": {"bad": 11.5, "ok": 10.8, "good": 10.2, "vegas": 10.0, "higher_is_better": False},
}
# Above this, a backtest result is treated as a leakage red flag rather
# than a win -- see BENCHMARKS' own comment.
ATS_LEAKAGE_SUSPICION_THRESHOLD = 0.70


def _rate_against_benchmark(metric: str, value: float) -> str:
    b = BENCHMARKS[metric]
    if metric == "ats_accuracy" and value >= ATS_LEAKAGE_SUSPICION_THRESHOLD:
        return "SUSPICIOUS -- this high an ATS number usually means leakage, not skill (Phase 2)"
    thresholds = [("good", b["good"]), ("ok", b["ok"]), ("bad", b["bad"])]
    for label, threshold in thresholds:
        if (b["higher_is_better"] and value >= threshold) or (not b["higher_is_better"] and value <= threshold):
            return label
    return "bad"


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


def full_backtest_metrics(test_df: pd.DataFrame, proba: np.ndarray, spread_calibration) -> dict:
    """Week 1 Audit & Tuning Plan Phase 4.3: the full metric suite the
    plan's benchmark table asks for, not just Brier score --
    straight-up accuracy, Brier, log loss, MAE against the actual point
    margin, and against-the-spread (ATS) accuracy.

    ATS: this codebase's own verified convention (AUDIT.md Phase 1.2,
    locked in by tests/test_spread_convention.py) is positive
    spread_line = home favored, magnitude = home's expected margin. The
    home team "covers" when their actual margin exceeds that expectation
    (actual_margin > spread_line); the model's own ATS pick is "home
    covers" when its own implied_spread (via the saved spread-calibration
    regression, the same one model/predict.py uses to compare against
    the market) is higher than the market's spread_line -- i.e. the
    model thinks home will beat the market's own number by more than
    the market itself expects. Games with no posted spread_line have
    nothing to grade ATS against and are excluded, not guessed at."""
    actual_win = test_df["home_win"].values
    actual_margin = (test_df["home_score"] - test_df["away_score"]).values
    implied_spread = spread_calibration.predict(proba.reshape(-1, 1))

    has_spread = test_df["spread_line"].notna().values
    ats_pick_home = implied_spread > test_df["spread_line"].values
    home_covered = actual_margin > test_df["spread_line"].values
    ats_correct = (ats_pick_home == home_covered)[has_spread]

    return {
        "straight_up_accuracy": accuracy_score(actual_win, (proba >= 0.5).astype(int)),
        "brier_score": brier_score_loss(actual_win, proba),
        # labels=[0, 1] explicit: sklearn's log_loss otherwise raises if
        # the sample happens to contain only one outcome class (e.g. a
        # small filtered subset, or a test fixture) -- a real backtest
        # of 200+ games will never hit this in practice, but there's no
        # reason this function should be fragile against a small sample
        # when the fix is one explicit argument.
        "log_loss": log_loss(actual_win, proba, labels=[0, 1]),
        "mae_margin": mean_absolute_error(actual_margin, implied_spread),
        "ats_accuracy": float(ats_correct.mean()) if len(ats_correct) else float("nan"),
        "ats_n": int(has_spread.sum()),
    }


def print_benchmark_comparison(metrics: dict) -> None:
    print(f"{'metric':22s} {'value':>10s}  {'rating':s}")
    for metric in ("straight_up_accuracy", "brier_score", "ats_accuracy", "mae_margin"):
        value = metrics[metric]
        rating = _rate_against_benchmark(metric, value)
        print(f"{metric:22s} {value:>10.3f}  {rating}")
    print(f"{'log_loss':22s} {metrics['log_loss']:>10.3f}  (no benchmark row in the plan's own table)")


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

    final_proba = raw_test_proba
    if calibrator is not None:
        final_proba = apply_calibrator(calibrator, raw_test_proba)
        print(f"\nReliability table ({best_name}-calibrated):")
        print(reliability_table(final_proba, test_actual))

    saved["calibrator_name"] = best_name
    saved["calibrator"] = calibrator
    joblib.dump(saved, MODEL_PATH)
    print(f"\nSaved calibrator ({best_name}) -> {MODEL_PATH}")

    # Week 1 Audit & Tuning Plan Phase 4.3 + 4.4: the full benchmark
    # suite and the three dumb baselines, printed together every time --
    # never "our model is X% accurate" reported on its own.
    print(f"\n{'=' * 60}\nFull backtest report -- {TEST_SEASON} holdout, {len(test_df)} games "
          f"({best_name}-calibrated)\n{'=' * 60}")
    metrics = full_backtest_metrics(test_df, final_proba, saved["spread_calibration"])
    print_benchmark_comparison(metrics)
    print(f"ATS graded on {metrics['ats_n']} of {len(test_df)} games (rest had no posted spread_line)\n")
    print_baselines(test_df, schedules)


if __name__ == "__main__":
    main()
