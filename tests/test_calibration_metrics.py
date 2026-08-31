# tests for the full backtest-metric suite (Week 1 Audit & Tuning Plan Phase 4.3)
"""model/calibration.py had no test coverage before this -- rather than
retroactively covering the whole pre-existing module (Brier score,
reliability table, Platt/isotonic selection), this covers just the new
additions this build adds: full_backtest_metrics() (log loss, MAE
against actual margin, ATS accuracy -- the plan's benchmark table asks
for all of these, not just Brier score) and the benchmark-rating logic.
"""

import numpy as np
import pandas as pd

from model.calibration import BENCHMARKS, _rate_against_benchmark, full_backtest_metrics


class _IdentitySpreadCalibration:
    """A stand-in for the real spread_calibration LinearRegression --
    maps win probability directly to an implied spread via a fixed
    linear rule, so the ATS test below can hand-compute the expected
    answer instead of depending on a real fitted model's coefficients."""
    def predict(self, X):
        # win_prob 0.5 -> spread 0; each 0.1 of probability above/below
        # 0.5 -> 4 points of implied spread, wide enough to be decisive.
        return (np.asarray(X).ravel() - 0.5) * 40


def _test_df(rows):
    return pd.DataFrame(rows)


def test_full_backtest_metrics_computes_all_five_fields():
    test_df = _test_df([
        {"home_win": 1, "home_score": 24, "away_score": 17, "spread_line": 3.0},
        {"home_win": 0, "home_score": 14, "away_score": 21, "spread_line": -2.0},
        {"home_win": 1, "home_score": 27, "away_score": 10, "spread_line": 6.0},
    ])
    proba = np.array([0.65, 0.40, 0.70])
    metrics = full_backtest_metrics(test_df, proba, _IdentitySpreadCalibration())
    for key in ("straight_up_accuracy", "brier_score", "log_loss", "mae_margin", "ats_accuracy", "ats_n"):
        assert key in metrics


def test_ats_accuracy_matches_hand_computed_case():
    """proba=0.65 -> implied_spread = (0.65-0.5)*40 = 6.0. spread_line=3.0,
    so the model's own pick is "home covers" (6.0 > 3.0). Actual margin
    24-17=7, which IS > 3.0 -- home covered -- so this game's ATS pick
    is correct."""
    test_df = _test_df([{"home_win": 1, "home_score": 24, "away_score": 17, "spread_line": 3.0}])
    proba = np.array([0.65])
    metrics = full_backtest_metrics(test_df, proba, _IdentitySpreadCalibration())
    assert metrics["ats_accuracy"] == 1.0
    assert metrics["ats_n"] == 1


def test_ats_accuracy_flags_a_wrong_pick():
    """Same math as above, but the actual margin (24-23=1) falls SHORT
    of the market's 3.0 -- home did NOT cover, so the model's "home
    covers" pick (implied_spread 6.0 > 3.0) is wrong here."""
    test_df = _test_df([{"home_win": 1, "home_score": 24, "away_score": 23, "spread_line": 3.0}])
    proba = np.array([0.65])
    metrics = full_backtest_metrics(test_df, proba, _IdentitySpreadCalibration())
    assert metrics["ats_accuracy"] == 0.0


def test_games_with_no_spread_line_excluded_from_ats_but_not_other_metrics():
    test_df = _test_df([
        {"home_win": 1, "home_score": 24, "away_score": 17, "spread_line": 3.0},
        {"home_win": 0, "home_score": 14, "away_score": 21, "spread_line": float("nan")},
    ])
    proba = np.array([0.65, 0.40])
    metrics = full_backtest_metrics(test_df, proba, _IdentitySpreadCalibration())
    assert metrics["ats_n"] == 1  # only the game with a real spread_line graded
    assert not np.isnan(metrics["straight_up_accuracy"])  # the other metrics still use both games


def test_rate_against_benchmark_boundaries():
    assert _rate_against_benchmark("straight_up_accuracy", 0.68) == "good"
    assert _rate_against_benchmark("straight_up_accuracy", 0.64) == "ok"
    assert _rate_against_benchmark("straight_up_accuracy", 0.55) == "bad"
    # lower-is-better metric: brier_score
    assert _rate_against_benchmark("brier_score", 0.19) == "good"
    assert _rate_against_benchmark("brier_score", 0.25) == "bad"


def test_rate_against_benchmark_flags_suspiciously_high_ats_as_leakage():
    """The plan's own explicit warning: 70%+ ATS in a backtest means
    leakage, not skill -- must be flagged distinctly from a genuine
    "good" rating, not just silently rated "good"."""
    rating = _rate_against_benchmark("ats_accuracy", 0.72)
    assert "SUSPICIOUS" in rating
    assert "leakage" in rating.lower()


def test_benchmarks_table_has_all_four_plan_metrics():
    assert set(BENCHMARKS.keys()) == {"straight_up_accuracy", "brier_score", "ats_accuracy", "mae_margin"}
