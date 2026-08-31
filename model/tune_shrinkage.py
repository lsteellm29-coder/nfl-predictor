# Week 1 Audit & Tuning Plan Phase 4.1: grid search Elo's season-boundary regression
"""model/elo.py's SEASON_REGRESSION (1/3) is a guessed constant --
"standard FiveThirtyEight convention," never fit against this project's
own data (flagged in AUDIT.md's Phase 0 hardcoded-number table). It's
applied exactly once per team per season boundary, right before that
team's next Week 1 game -- so Week-1-specific accuracy is exactly where
getting this number right or wrong matters most, more than any other
week.

This grid-searches SEASON_REGRESSION from 0.0 (no regression -- a team
carries its full end-of-season rating straight into the new season) to
0.6 (aggressive regression toward the 1500 default) in steps of 0.05,
walk-forward: for each candidate value, train on every season strictly
before a held-out test season, then evaluate ONLY that test season's
Week 1 games (the exact case this parameter is about). Reports both
straight-up accuracy and MAE against the actual point margin (a
separate, dedicated linear regression fit straight to real margin --
not model/train.py's own spread_calibration, which maps win probability
to the MARKET's line, a different target than this plan's own "MAE
against actual margin" ask).

Picks the value that minimizes MAE, not the one that maximizes
accuracy -- the plan's own explicit instruction, since accuracy alone
can be won by a model that's confidently overconfident.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import model.elo as elo_module
from config import HISTORICAL_SEASONS
from data.fetch_injuries import historical_injury_impact
from data.situational import blowout_loss_flags, lookahead_flags
from model.train import FEATURE_COLS, build_feature_frame

SHRINKAGE_GRID = [round(x * 0.05, 2) for x in range(13)]  # 0.0 .. 0.6 step 0.05

# Test seasons need several PRIOR seasons of real accumulated Elo history
# for the regression amount to actually matter -- starting too early
# (e.g. 2017, with only one prior season on record) tests a degenerate
# case, not a realistic one.
TEST_SEASONS = HISTORICAL_SEASONS[4:]


def _games_for_shrinkage(schedules, team_stats, injuries, blowouts, lookaheads, shrinkage: float) -> pd.DataFrame:
    """Rebuilds the full training feature frame with SEASON_REGRESSION
    temporarily swapped to `shrinkage` -- restores the real value in a
    finally block so a script failure never leaves the module constant
    silently changed for anything else in the same process."""
    original = elo_module.SEASON_REGRESSION
    elo_module.SEASON_REGRESSION = shrinkage
    try:
        elo_per_game, _ = elo_module.compute_elo_ratings(schedules)
    finally:
        elo_module.SEASON_REGRESSION = original
    return build_feature_frame(schedules, team_stats, injuries, elo_per_game, blowouts, lookaheads)


def evaluate_shrinkage(shrinkage: float, schedules, team_stats, injuries, blowouts, lookaheads) -> dict:
    games = _games_for_shrinkage(schedules, team_stats, injuries, blowouts, lookaheads, shrinkage)
    margin = games["home_score"] - games["away_score"]

    accs, maes = [], []
    for test_season in TEST_SEASONS:
        train = games[games["season"] < test_season]
        test_week1 = games[(games["season"] == test_season) & (games["week"] == 1)]
        if train.empty or test_week1.empty:
            continue
        train_margin = margin.loc[train.index]

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(train[FEATURE_COLS], train["home_win"])
        margin_model = LinearRegression().fit(train[FEATURE_COLS], train_margin)

        pred_win = clf.predict(test_week1[FEATURE_COLS])
        pred_margin = margin_model.predict(test_week1[FEATURE_COLS])
        actual_margin = margin.loc[test_week1.index]

        accs.append(accuracy_score(test_week1["home_win"], pred_win))
        maes.append(mean_absolute_error(actual_margin, pred_margin))

    return {
        "shrinkage": shrinkage,
        "accuracy": float(np.mean(accs)) if accs else float("nan"),
        "mae": float(np.mean(maes)) if maes else float("nan"),
        "n_folds": len(accs),
    }


def main():
    schedules = pd.read_parquet("data/cache/schedules.parquet")
    team_stats = pd.read_parquet("data/cache/team_stats.parquet")
    injuries = historical_injury_impact(HISTORICAL_SEASONS)
    blowouts = blowout_loss_flags(schedules)
    lookaheads = lookahead_flags(schedules)

    results = []
    print(f"{'shrinkage':>9}  {'week1_acc':>9}  {'week1_MAE':>9}  n_folds")
    for s in SHRINKAGE_GRID:
        r = evaluate_shrinkage(s, schedules, team_stats, injuries, blowouts, lookaheads)
        results.append(r)
        print(f"{r['shrinkage']:>9.2f}  {r['accuracy']:>9.3f}  {r['mae']:>9.3f}  {r['n_folds']}")

    best = min(results, key=lambda r: r["mae"])
    print(f"\nBest (min MAE): shrinkage={best['shrinkage']:.2f}  "
          f"MAE={best['mae']:.3f}  accuracy={best['accuracy']:.3f}")
    print(f"Current deployed value (model/elo.py SEASON_REGRESSION): {elo_module.SEASON_REGRESSION:.4f}")
    return results


if __name__ == "__main__":
    main()
