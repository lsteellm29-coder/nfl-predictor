# get_current_elo_ratings(season, week): ratings as of the START of the week, so a
# game already played this week can't leak its own result into its own "prediction".
import numpy as np
import pandas as pd

import model.predict as predict


def _game(season, week, home, away, hs, as_):
    return {"game_id": f"{season}_{week:02d}_{away}_{home}", "season": season, "week": week, "game_type": "REG",
            "gameday": f"{season}-09-{week:02d}",
            "home_team": home, "away_team": away, "home_score": hs, "away_score": as_, "location": "Home"}


def test_elo_as_of_week_excludes_games_already_played_that_week(tmp_path, monkeypatch):
    hist = pd.DataFrame([_game(2025, 1, "BUF", "DET", 27.0, 24.0)])
    hist_path = tmp_path / "schedules.parquet"
    hist.to_parquet(hist_path)
    monkeypatch.setattr(predict, "SCHEDULES_PATH", str(hist_path))

    week1 = _game(2026, 1, "BUF", "NYJ", 31.0, 10.0)
    thursday_wk2 = _game(2026, 2, "DET", "BUF", 31.0, 41.0)     # already played -- must not count for week 2
    current = pd.DataFrame([week1, thursday_wk2, _game(2026, 3, "BUF", "MIA", np.nan, np.nan)])
    monkeypatch.setattr(predict.nfl, "import_schedules", lambda seasons: current.copy())

    as_of_wk2 = predict.get_current_elo_ratings(2026, week=2)
    as_of_wk3 = predict.get_current_elo_ratings(2026, week=3)
    latest = predict.get_current_elo_ratings(2026, week=None)

    # as of the start of week 2: only the week-1 game counts, so it must equal a replay in which the
    # Thursday game (and everything after it) never happened
    without_thursday = current.copy()
    without_thursday.loc[without_thursday["week"] >= 2, ["home_score", "away_score"]] = np.nan
    monkeypatch.setattr(predict.nfl, "import_schedules", lambda seasons: without_thursday.copy())
    assert as_of_wk2 == predict.get_current_elo_ratings(2026, week=None)
    assert as_of_wk2["BUF"] != as_of_wk3["BUF"]          # week 3 has replayed the Thursday game; week 2 hasn't
    assert as_of_wk3 == latest                            # nothing at/after week 3 has a score
    assert as_of_wk2["DET"] != as_of_wk3["DET"]           # the week-2 game moved DET, but not as of week 2


def test_week_is_a_required_argument():
    # so a caller (score_week) can't quietly drop it and silently go back to hindsight ratings
    import inspect
    assert inspect.signature(predict.get_current_elo_ratings).parameters["week"].default is inspect.Parameter.empty


def test_a_replayed_cached_season_is_not_counted_twice(tmp_path, monkeypatch):
    # scoring week 2 OF a season that is also in the historical cache: the cached copy of that
    # season must not be replayed un-blanked (it contains week 2's result) on top of the live one
    season = pd.DataFrame([_game(2025, 1, "BUF", "DET", 27.0, 24.0), _game(2025, 2, "BUF", "NYJ", 30.0, 3.0)])
    path = tmp_path / "schedules.parquet"
    season.to_parquet(path)
    monkeypatch.setattr(predict, "SCHEDULES_PATH", str(path))
    monkeypatch.setattr(predict.nfl, "import_schedules", lambda seasons: season.copy())

    as_of_wk2 = predict.get_current_elo_ratings(2025, week=2)

    week1_only = season.copy()
    week1_only.loc[week1_only["week"] >= 2, ["home_score", "away_score"]] = np.nan
    _, expected = predict.compute_elo_ratings(week1_only)
    assert as_of_wk2 == expected
    assert "NYJ" not in as_of_wk2          # NYJ's only game is week 2 -- it hasn't happened as of week 2
