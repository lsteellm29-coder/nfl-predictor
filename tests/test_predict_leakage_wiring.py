# tests confirming assert_no_leakage is actually wired into the two
# narrative-only pbp filters that were missing it (found in a security/
# correctness review after the Week 1 Audit & Tuning Plan's Phase 2 --
# these two filters do the identical `pbp[pbp["week"] < week]` pattern
# every other week-boundary filter in this codebase already tripwires,
# but had no assert_no_leakage call of their own).
#
# The filter itself (`pbp[pbp["week"] < week]`) already excludes any
# leaking row before assert_no_leakage ever sees the result, so feeding
# "bad" raw data can't actually exercise the tripwire firing -- what's
# worth locking in instead is that the call is actually THERE, so a
# future refactor that weakens the filter (e.g. `<` -> `<=`) trips a
# test failure here rather than silently shipping. Spies on
# assert_no_leakage rather than data/leakage.py's own logic, which
# tests/test_leakage.py already covers directly.
from unittest.mock import MagicMock

import pandas as pd

import model.predict as predict


def _schedule(season, played_weeks):
    return pd.DataFrame([
        {"season": season, "game_type": "REG", "week": w, "home_score": 20.0} for w in played_weeks
    ])


def test_current_season_pbp_for_td_calls_the_leakage_tripwire(monkeypatch):
    monkeypatch.setattr(predict.nfl, "import_schedules", lambda seasons: _schedule(2026, [1]))
    monkeypatch.setattr(predict.nfl, "import_pbp_data", lambda seasons, downcast=True: pd.DataFrame({"week": [1]}))
    spy = MagicMock()
    monkeypatch.setattr(predict, "assert_no_leakage", spy)

    fallback = pd.DataFrame({"week": []})
    predict._current_season_pbp_for_td(2026, week=2, fallback_pbp=fallback)

    spy.assert_called_once()
    args, kwargs = spy.call_args
    assert args[1] == 2  # week
    assert kwargs.get("context") == "_current_season_pbp_for_td"


def test_current_season_pbp_calls_the_leakage_tripwire(monkeypatch):
    monkeypatch.setattr(predict.nfl, "import_schedules", lambda seasons: _schedule(2026, [1]))
    monkeypatch.setattr(predict.nfl, "import_pbp_data", lambda seasons, downcast=True: pd.DataFrame({"week": [1]}))
    spy = MagicMock()
    monkeypatch.setattr(predict, "assert_no_leakage", spy)

    fallback = pd.DataFrame({"week": []})
    predict._current_season_pbp(2026, week=2, fallback_pbp=fallback)

    spy.assert_called_once()
    args, kwargs = spy.call_args
    assert args[1] == 2  # week
    assert kwargs.get("context") == "_current_season_pbp"
