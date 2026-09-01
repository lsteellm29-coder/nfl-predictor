# Post-freeze accuracy experiment: grid search Elo's K-factor and home-field bonus
"""model/elo.py's K_FACTOR (20.0) and HOME_FIELD_ADV (55.0) are both
guessed constants ("standard FiveThirtyEight convention," per elo.py's
own docstring) that were never actually fit against this project's own
data -- AUDIT.md's Phase 0 flagged them as hardcoded, and Phase 4.1 only
ever tuned the season-boundary regression amount (SEASON_REGRESSION),
not these two. They govern every single Elo update, so a wrong value
compounds across the whole 10-season backfill, not just at one
boundary.

Same walk-forward discipline as model/tune_shrinkage.py: for each
candidate value, rebuild Elo ratings with that constant swapped in,
train fresh on every season strictly before a held-out test season,
evaluate ONLY that season's Week 1 games. Tuned sequentially, not as a
joint 2D grid -- K_FACTOR first (holding HOME_FIELD_ADV at its current
value), then HOME_FIELD_ADV (holding K_FACTOR at whatever Step 1 found).
A full 2D sweep would cost 7x more model fits for a interaction effect
this project's ~90-game walk-forward window is unlikely to resolve
against noise anyway -- same reasoning Phase 4.1 used to justify NOT
chasing a razor-thin MAE edge on a small sample.

Picks the value that minimizes MAE against actual margin, not the one
that maximizes accuracy -- same reasoning as tune_shrinkage.py (Phase
4.1's own explicit instruction): accuracy alone can be won by a
confidently-overconfident model.
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

K_FACTOR_GRID = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
HOME_FIELD_ADV_GRID = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

# Same floor as tune_shrinkage.py's TEST_SEASONS -- early seasons don't
# have enough accumulated Elo history for either constant to meaningfully
# matter yet.
TEST_SEASONS = HISTORICAL_SEASONS[4:]


def _games_for_elo_constants(schedules, team_stats, injuries, blowouts, lookaheads,
                              k_factor: float, home_field_adv: float) -> pd.DataFrame:
    """Rebuilds the full training feature frame with K_FACTOR/HOME_FIELD_ADV
    temporarily swapped -- restores the real values in a finally block so
    a script failure never leaves the module constants silently changed
    for anything else in the same process (same pattern as
    tune_shrinkage.py's _games_for_shrinkage)."""
    original_k, original_hfa = elo_module.K_FACTOR, elo_module.HOME_FIELD_ADV
    elo_module.K_FACTOR, elo_module.HOME_FIELD_ADV = k_factor, home_field_adv
    try:
        elo_per_game, _ = elo_module.compute_elo_ratings(schedules)
    finally:
        elo_module.K_FACTOR, elo_module.HOME_FIELD_ADV = original_k, original_hfa
    return build_feature_frame(schedules, team_stats, injuries, elo_per_game, blowouts, lookaheads)


def evaluate_elo_constants(k_factor: float, home_field_adv: float,
                           schedules, team_stats, injuries, blowouts, lookaheads) -> dict:
    games = _games_for_elo_constants(schedules, team_stats, injuries, blowouts, lookaheads, k_factor, home_field_adv)
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
        "k_factor": k_factor, "home_field_adv": home_field_adv,
        "accuracy": float(np.mean(accs)) if accs else float("nan"),
        "mae": float(np.mean(maes)) if maes else float("nan"),
        "n_folds": len(accs),
    }


def _grid_search(param_name: str, grid: list, fixed_name: str, fixed_value: float,
                  schedules, team_stats, injuries, blowouts, lookaheads):
    results = []
    print(f"\n{param_name} grid (holding {fixed_name}={fixed_value:.1f} fixed):")
    print(f"{param_name:>14}  {'week1_acc':>9}  {'week1_MAE':>9}  n_folds")
    for value in grid:
        kwargs = {param_name: value, fixed_name: fixed_value}
        r = evaluate_elo_constants(kwargs["k_factor"], kwargs["home_field_adv"],
                                    schedules, team_stats, injuries, blowouts, lookaheads)
        results.append(r)
        print(f"{value:>14.1f}  {r['accuracy']:>9.3f}  {r['mae']:>9.3f}  {r['n_folds']}")
    return results


def main():
    schedules = pd.read_parquet("data/cache/schedules.parquet")
    team_stats = pd.read_parquet("data/cache/team_stats.parquet")
    injuries = historical_injury_impact(HISTORICAL_SEASONS)
    blowouts = blowout_loss_flags(schedules)
    lookaheads = lookahead_flags(schedules)

    current_k, current_hfa = elo_module.K_FACTOR, elo_module.HOME_FIELD_ADV

    k_results = _grid_search("k_factor", K_FACTOR_GRID, "home_field_adv", current_hfa,
                              schedules, team_stats, injuries, blowouts, lookaheads)
    best_k = min(k_results, key=lambda r: r["mae"])
    print(f"\nBest K_FACTOR (min MAE): {best_k['k_factor']:.1f}  MAE={best_k['mae']:.3f}  accuracy={best_k['accuracy']:.3f}")
    print(f"Current deployed K_FACTOR: {current_k:.1f}")

    hfa_results = _grid_search("home_field_adv", HOME_FIELD_ADV_GRID, "k_factor", current_k,
                                schedules, team_stats, injuries, blowouts, lookaheads)
    best_hfa = min(hfa_results, key=lambda r: r["mae"])
    print(f"\nBest HOME_FIELD_ADV (min MAE): {best_hfa['home_field_adv']:.1f}  "
          f"MAE={best_hfa['mae']:.3f}  accuracy={best_hfa['accuracy']:.3f}")
    print(f"Current deployed HOME_FIELD_ADV: {current_hfa:.1f}")

    return k_results, hfa_results


if __name__ == "__main__":
    main()
