# tests for run_week.py's grading logic
"""run_week.py has no existing test coverage (heavy external
dependencies: live odds/injury/news fetches, report building) -- this
covers just the one fixed bug worth locking in: _grade_pending() must
not grade a real NFL tie as an away-team win.
"""

import pandas as pd
import pytest

import run_week


def _log_df(**overrides):
    row = {
        "season": 2025, "week": 11, "away_team": "NE", "home_team": "SEA",
        "predicted_winner": "SEA", "home_win_prob": 0.6,
        "vegas_spread": 1.0, "model_spread": 3.0, "edge": 2.0,
        "actual_away_score": pd.NA, "actual_home_score": pd.NA, "actual_winner": pd.NA,
        "correct": pd.NA, "model_beat_market": pd.NA,
    }
    row.update(overrides)
    return pd.DataFrame([row], columns=run_week.LOG_COLS)


def _fake_schedule(home_score, away_score):
    return pd.DataFrame([{
        "season": 2025, "week": 11, "game_type": "REG",
        "home_team": "SEA", "away_team": "NE",
        "home_score": home_score, "away_score": away_score,
    }])


def test_grade_pending_does_not_grade_a_real_tie_as_an_away_team_win(monkeypatch):
    monkeypatch.setattr(run_week.nfl, "import_schedules", lambda seasons: _fake_schedule(20, 20))
    result = run_week._grade_pending(_log_df())
    row = result.iloc[0]
    assert row["actual_winner"] == "TIE"
    assert pd.isna(row["correct"])


def test_grade_pending_still_grades_a_real_win_correctly(monkeypatch):
    monkeypatch.setattr(run_week.nfl, "import_schedules", lambda seasons: _fake_schedule(24, 17))
    result = run_week._grade_pending(_log_df())
    row = result.iloc[0]
    assert row["actual_winner"] == "SEA"
    assert row["correct"] == True  # noqa: E712 -- confirms a real bool, not just truthy
