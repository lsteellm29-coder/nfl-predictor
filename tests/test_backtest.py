# tests for the walk-forward backtest's pooling logic (Week 1 Audit & Tuning Plan Phase 5)
"""run_backtest() itself needs the real parquet caches and trains real
models, so it isn't unit-tested here (Phase 4.1/4.2's sibling scripts
follow the same pattern -- exercised by actually running them, not by
fixture-driven unit tests). What IS unit-testable and worth covering:
pooled_metrics() correctly concatenates raw per-fold predictions rather
than averaging already-summarized fold metrics, since those two give
different (and for log loss/Brier, not interchangeable) answers.
"""

import numpy as np
import pandas as pd

from model.backtest import _PrecomputedSpread, pooled_metrics


def _fold(rows, proba, implied_spread, test_season):
    test_week1 = pd.DataFrame(rows)
    return {
        "test_season": test_season,
        "n_games": len(rows),
        "_test_week1": test_week1,
        "_proba": np.array(proba),
        "_implied_spread": np.array(implied_spread),
    }


def test_pooled_metrics_concatenates_rather_than_averages_fold_accuracy():
    """Fold A: 1/1 correct (1.0 acc). Fold B: 0/1 correct (0.0 acc).
    Averaging the two fold accuracies would give 0.5; pooling the two
    raw predictions together gives 1/2 = 0.5 too for this even split --
    so use uneven folds where the two answers diverge: fold A has 3
    games (all correct), fold B has 1 game (wrong). Average-of-averages
    = (1.0 + 0.0) / 2 = 0.5. True pooled = 3/4 = 0.75."""
    fold_a = _fold(
        rows=[
            {"home_win": 1, "home_score": 24, "away_score": 10, "spread_line": 3.0},
            {"home_win": 1, "home_score": 20, "away_score": 14, "spread_line": 2.0},
            {"home_win": 1, "home_score": 30, "away_score": 20, "spread_line": 4.0},
        ],
        proba=[0.9, 0.9, 0.9],
        implied_spread=[5.0, 5.0, 5.0],
        test_season=2020,
    )
    fold_b = _fold(
        rows=[{"home_win": 0, "home_score": 10, "away_score": 24, "spread_line": 3.0}],
        proba=[0.9],
        implied_spread=[5.0],
        test_season=2021,
    )
    pooled = pooled_metrics([fold_a, fold_b])
    assert pooled["straight_up_accuracy"] == 0.75
    assert pooled["ats_n"] == 4


def test_precomputed_spread_ignores_input_and_returns_stored_values():
    stand_in = _PrecomputedSpread([1.0, 2.0, 3.0])
    result = stand_in.predict(np.zeros((3, 1)))
    assert list(result) == [1.0, 2.0, 3.0]
