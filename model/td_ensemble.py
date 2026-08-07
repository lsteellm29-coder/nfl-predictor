# trains + applies the logistic-classifier half of the anytime-TD ensemble
"""Anytime-TD-Only Props Refocus spec, Section 5: "ensemble the Poisson
model with a simple logistic classifier... [that] can catch patterns
Poisson's assumptions miss." model/td_model.py's Poisson projection gives
a clean count-based probability from real red-zone opportunity share; a
classifier trained directly on "did this player score, yes/no" using the
same underlying features can pick up interaction/nonlinear patterns the
Poisson model's strict distributional assumptions can't. Blended at
BLEND_WEIGHT, same spirit as model/train.py's logistic+XGBoost ensemble
for the win-probability model.

Validated on a true holdout before shipping: trained on 2022-2024's
backtest data, evaluated on 2025 (never seen during that training) --
Poisson-only Brier 0.1676/AUC 0.6699, blended Brier 0.1638-0.1642/AUC
0.6713-0.6718 depending on blend weight. A real, if modest, improvement
on data the classifier never touched.

The deployed classifier below is trained on ALL of 2022-2025 (not just
2022-2024) -- for live scoring of the current/upcoming season, every one
of those seasons is legitimately in the past, same as how
model/train.py's final saved model retrains on the full available
training window after validating the approach on a holdout split. If
this ever needs re-validating later, hold out whatever the most recent
complete season is at that time, the same way this was first checked --
training and grading the classifier on the exact same rows it already
saw would be circular, not a real test.
"""

import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), "td_ensemble.joblib")
BACKTEST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache", "td_backtest.parquet")

FEATURES = ["expected_tds", "player_share", "player_conversion", "def_factor", "team_rz_touches_per_game", "n_games"]
POSITIONS = ["QB", "RB", "WR", "TE"]

# Empirically swept 0.4-0.9 on the 2022-2024-train/2025-test holdout:
# Brier bottoms out at 0.70-0.75 (0.1638) but AUC peaks around 0.55-0.60
# and starts declining past 0.65 -- 0.6 sits on the flat part of both
# curves rather than the exact edge of either, the same "stable point on
# a plateau, not the extreme" choice already made for
# model/td_model.py's SHRINKAGE_TOUCHES.
BLEND_WEIGHT = 0.6


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.get_dummies(df[FEATURES + ["position"]], columns=["position"])
    for pos in POSITIONS:
        col = f"position_{pos}"
        if col not in x.columns:
            x[col] = False
    return x


def train(backtest_path: str = BACKTEST_PATH) -> None:
    if not os.path.exists(backtest_path):
        raise RuntimeError(f"No backtest data at {backtest_path} -- run `python -m model.td_backtest` first.")
    df = pd.read_parquet(backtest_path)
    x = _feature_matrix(df)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(x, df["actual_td"])
    joblib.dump({"model": clf, "feature_cols": list(x.columns), "n_train": len(df),
                 "seasons": sorted(df["season"].unique().tolist())}, MODEL_PATH)
    print(f"Trained TD ensemble classifier on {len(df)} rows ({sorted(df['season'].unique().tolist())}) -> {MODEL_PATH}")


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
    position -- matching model/td_backtest.py's stored columns). Returns
    `poisson_prob` unchanged if no trained classifier exists yet (before
    the first `python -m model.td_ensemble` run) -- same fail-open
    contract the rest of this codebase uses for an optional enhancement
    layer, not a hard dependency."""
    saved = _load()
    if saved is None:
        return poisson_prob
    row = pd.DataFrame([features])
    x = _feature_matrix(row).reindex(columns=saved["feature_cols"], fill_value=False)
    logistic_prob = saved["model"].predict_proba(x)[0, 1]
    return BLEND_WEIGHT * logistic_prob + (1 - BLEND_WEIGHT) * poisson_prob


def main():
    train()


if __name__ == "__main__":
    main()
