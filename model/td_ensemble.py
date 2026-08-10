# trains + applies the stacked anytime-TD ensemble (Poisson x classifier x meta-model)
"""Anytime-TD-Only Props Refocus spec, Section 5, refined by the Master
Honing Plan's Section A: "ensemble the Poisson model with a simple
logistic classifier... [that] can catch patterns Poisson's assumptions
miss." model/td_model.py's Poisson projection gives a clean count-based
probability from real red-zone opportunity share; a classifier trained
directly on "did this player score, yes/no" using the same underlying
features can pick up interaction/nonlinear patterns the Poisson model's
strict distributional assumptions can't.

A genuine STACKED ensemble (Master Honing Plan, Section A) was built and
evaluated here as a candidate replacement for the fixed BLEND_WEIGHT
formula: a small logistic-regression meta-model over [poisson_prob,
out-of-fold classifier_prob, n_games], fit the same way
model/calibration.py fits its own Platt/isotonic calibrators (out-of-fold,
never on the classifier's own in-sample predictions). walk_forward_
evaluate() runs a genuine walk-forward validation across every season in
the backtest data (train on all strictly earlier seasons, test on the
next, mirroring model/train.py's own walk_forward_folds()) and reports
Brier/AUC for three candidates side by side: Poisson-only, the fixed
blend, and the stack.

Real result (2 walk-forward folds, train-2022-23/test-2024 and
train-2022-24/test-2025): the stack nudged average Brier down slightly
(0.1675 -> 0.1672) but came in WORSE on AUC than the simple fixed blend
in both folds (0.6718 avg fixed-blend vs 0.6691 avg stacked) -- worse on
exactly the metric this was meant to improve. With only ~3,200 rows per
fold, a 3-input meta-model doesn't appear to have enough held-out signal
to learn a genuinely better combination rule than the already-tuned
scalar weight; the fixed blend stays the deployed default.
walk_forward_evaluate() is kept and still runs on every `python -m
model.td_ensemble` invocation -- it's a real improvement over what used
to be a one-off, not-easily-reproducible manual validation, and worth
re-running as more backtest seasons accumulate in case the stack starts
to earn its keep with more data. This is the same "built, tested,
honestly reported, not shipped on a negative-or-flat result" call
model/train.py already made once for its own time-decay sample-weighting
experiment.

The deployed classifier is trained on ALL available backtest seasons
(not just the walk-forward training portion) -- for live scoring of the
current/upcoming season, every one of those seasons is legitimately in
the past, same as how model/train.py's final saved model retrains on the
full available training window after validating the approach on a
holdout split. If this ever needs re-validating later, re-run
walk_forward_evaluate() with whatever the most recent complete season is
held out at that time -- training and grading on the exact same rows
already seen would be circular, not a real test.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), "td_ensemble.joblib")
BACKTEST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache", "td_backtest.parquet")

FEATURES = ["expected_tds", "player_share", "player_conversion", "def_factor", "team_rz_touches_per_game", "n_games"]
POSITIONS = ["QB", "RB", "WR", "TE"]

# Kept only as the fallback blend when no stacker has been trained yet
# (mirrors model/calibration.py's calibrator_name="none" passthrough
# contract) -- empirically swept 0.4-0.9 on an earlier 2022-2024-train/
# 2025-test holdout: Brier bottomed out at 0.70-0.75 (0.1638) but AUC
# peaked around 0.55-0.60 and started declining past 0.65; 0.6 sat on
# the flat part of both curves rather than the exact edge of either.
BLEND_WEIGHT = 0.6

MIN_WALK_FORWARD_TRAIN_SEASONS = 2


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.get_dummies(df[FEATURES + ["position"]], columns=["position"])
    for pos in POSITIONS:
        col = f"position_{pos}"
        if col not in x.columns:
            x[col] = False
    return x


def _train_classifier(df: pd.DataFrame):
    x = _feature_matrix(df)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(x, df["actual_td"])
    return clf, list(x.columns)


def _classifier_oof_probs(train_df: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold classifier predictions on the training data -- used to
    fit the stacking meta-model on genuinely held-out predictions rather
    than the classifier's own in-sample (overconfident) fit. Same
    methodology as model/calibration.py's _out_of_fold_train_probs,
    applied here to the TD classifier instead of the win-probability
    model."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    x = _feature_matrix(train_df).to_numpy()
    y = train_df["actual_td"].to_numpy()
    oof = np.zeros(len(train_df))
    for train_idx, val_idx in kf.split(x):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(x[train_idx], y[train_idx])
        oof[val_idx] = clf.predict_proba(x[val_idx])[:, 1]
    return oof


def _stack_features(poisson_prob: np.ndarray, classifier_prob: np.ndarray, n_games: np.ndarray) -> np.ndarray:
    return np.column_stack([poisson_prob, classifier_prob, n_games])


def _fit_stacker(train_df: pd.DataFrame) -> LogisticRegression:
    """The level-1 meta-model: a plain LogisticRegression over
    [poisson_prob, out-of-fold classifier_prob, n_games]. `predicted_prob`
    (the Poisson column already stored in td_backtest.parquet) is already
    leakage-safe by construction -- model/td_backtest.py computes it
    walk-forward, strictly-earlier-weeks-only, per row -- so it can be
    used directly without its own OOF pass; only the classifier needs
    one, since it's fit on the whole train_df at once."""
    classifier_oof = _classifier_oof_probs(train_df)
    stack_x = _stack_features(train_df["predicted_prob"].to_numpy(), classifier_oof, train_df["n_games"].to_numpy())
    stacker = LogisticRegression(max_iter=1000)
    stacker.fit(stack_x, train_df["actual_td"])
    return stacker


def train(backtest_path: str = BACKTEST_PATH) -> None:
    """Trains and saves the deployed classifier only -- combined with the
    Poisson probability via the fixed BLEND_WEIGHT formula in
    blend_with_classifier() below, per this module's docstring: the
    stacked meta-model was built and evaluated (see walk_forward_evaluate,
    called from main()) but didn't beat this fixed blend on AUC, so it
    isn't part of the deployed combination."""
    if not os.path.exists(backtest_path):
        raise RuntimeError(f"No backtest data at {backtest_path} -- run `python -m model.td_backtest` first.")
    df = pd.read_parquet(backtest_path)
    clf, feature_cols = _train_classifier(df)
    joblib.dump({
        "model": clf, "feature_cols": feature_cols, "n_train": len(df),
        "seasons": sorted(df["season"].unique().tolist()),
    }, MODEL_PATH)
    print(f"Trained TD ensemble classifier on {len(df)} rows "
          f"({sorted(df['season'].unique().tolist())}) -> {MODEL_PATH}")


_CACHED = None
_LOAD_ATTEMPTED = False


def _load():
    global _CACHED, _LOAD_ATTEMPTED
    if not _LOAD_ATTEMPTED:
        _LOAD_ATTEMPTED = True
        if os.path.exists(MODEL_PATH):
            _CACHED = joblib.load(MODEL_PATH)
    return _CACHED


def blend_with_classifier(poisson_prob: float, features: dict) -> float:
    """Blends `poisson_prob` with the trained classifier's own prediction
    from the same underlying `features` (expected_tds/player_share/
    player_conversion/def_factor/team_rz_touches_per_game/n_games/
    position -- matching model/td_backtest.py's stored columns) via the
    fixed BLEND_WEIGHT formula -- see this module's docstring for why the
    tested stacked alternative isn't used here. Returns `poisson_prob`
    unchanged if no trained classifier exists yet (before the first
    `python -m model.td_ensemble` run) -- same fail-open contract the
    rest of this codebase uses for an optional enhancement layer, not a
    hard dependency."""
    saved = _load()
    if saved is None:
        return poisson_prob
    row = pd.DataFrame([features])
    x = _feature_matrix(row).reindex(columns=saved["feature_cols"], fill_value=False)
    classifier_prob = saved["model"].predict_proba(x)[0, 1]
    return BLEND_WEIGHT * classifier_prob + (1 - BLEND_WEIGHT) * poisson_prob


def _evaluate(predicted: np.ndarray, actual: np.ndarray) -> dict:
    if len(predicted) == 0 or len(np.unique(actual)) < 2:
        return {"n": len(predicted), "brier": None, "auc": None}
    return {"n": len(predicted), "brier": float(brier_score_loss(actual, predicted)), "auc": float(roc_auc_score(actual, predicted))}


def walk_forward_evaluate(backtest_path: str = BACKTEST_PATH) -> list[dict]:
    """Genuine walk-forward validation across every season in the backtest
    data -- train on all strictly earlier seasons, test on the next, same
    shape as model/train.py's walk_forward_folds() for the win-probability
    model. For each fold, reports Poisson-only / fixed-blend / stacked
    Brier and AUC side by side, all evaluated on a season none of that
    fold's fitting ever saw. This is the reproducible replacement for
    what used to be a one-off manual validation run."""
    df = pd.read_parquet(backtest_path)
    seasons = sorted(df["season"].unique())
    folds = []
    for i in range(MIN_WALK_FORWARD_TRAIN_SEASONS, len(seasons)):
        train_seasons, test_season = seasons[:i], seasons[i]
        train_df = df[df["season"].isin(train_seasons)]
        test_df = df[df["season"] == test_season]
        if train_df.empty or test_df.empty or test_df["actual_td"].nunique() < 2:
            continue

        clf, feature_cols = _train_classifier(train_df)
        stacker = _fit_stacker(train_df)

        x_test = _feature_matrix(test_df).reindex(columns=feature_cols, fill_value=False)
        classifier_test = clf.predict_proba(x_test)[:, 1]
        poisson_test = test_df["predicted_prob"].to_numpy()
        actual = test_df["actual_td"].to_numpy()

        fixed_blend_test = BLEND_WEIGHT * classifier_test + (1 - BLEND_WEIGHT) * poisson_test
        stack_x_test = _stack_features(poisson_test, classifier_test, test_df["n_games"].to_numpy())
        stacked_test = stacker.predict_proba(stack_x_test)[:, 1]

        folds.append({
            "train_seasons": train_seasons, "test_season": int(test_season),
            "poisson": _evaluate(poisson_test, actual),
            "fixed_blend": _evaluate(fixed_blend_test, actual),
            "stacked": _evaluate(stacked_test, actual),
        })
    return folds


def main():
    train()
    print("\nWalk-forward validation (train on strictly earlier seasons, test on the next):")
    folds = walk_forward_evaluate()
    for f in folds:
        print(f"\n  train {f['train_seasons']} / test {f['test_season']}  (n={f['poisson']['n']})")
        for label in ("poisson", "fixed_blend", "stacked"):
            m = f[label]
            if m["brier"] is None:
                print(f"    {label:12s} not enough classes to score")
                continue
            print(f"    {label:12s} Brier={m['brier']:.4f}  AUC={m['auc']:.4f}")

    if folds:
        for label in ("poisson", "fixed_blend", "stacked"):
            briers = [f[label]["brier"] for f in folds if f[label]["brier"] is not None]
            aucs = [f[label]["auc"] for f in folds if f[label]["auc"] is not None]
            if briers:
                print(f"\n  {label} average across {len(briers)} fold(s): "
                      f"Brier={sum(briers)/len(briers):.4f}  AUC={sum(aucs)/len(aucs):.4f}")


if __name__ == "__main__":
    main()
