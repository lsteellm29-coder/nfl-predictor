# Live pre-game team stats: the week-boundary selection that used to hand the
# model all-NaN (Week 2) / one-game-stale (Week 3+) inputs.
import numpy as np
import pandas as pd
import pytest

import data.live_stats as live_stats
import model.predict as predict
from data.leakage import assert_pregame_selection
from data.live_stats import assert_pbp_covers_played_games, pregame_team_stats, select_pregame_rows
from model.predict import SPLIT_COLS, fill_missing_stat_values
from model.train import STAT_COLS


EMPTY_PBP = pd.DataFrame({"game_id": [], "season": [], "week": []})


def _rolling():
    # Each row is PRE-game (build_rolling_team_stats' expanding().mean().shift(1)):
    # the row for week W holds stats from games before W. B is on a bye in week 2.
    rows = [
        # team, week, games_played, value (== "stat through the previous game")
        ("A", 1, 0, np.nan), ("A", 2, 1, 10.0), ("A", 3, 2, 20.0),
        ("B", 1, 0, np.nan), ("B", 3, 1, 11.0), ("B", 4, 2, 22.0),
    ]
    return pd.DataFrame([{"season": 2026, "week": w, "team": t, "games_played": g, "x_avg": v}
                         for t, w, g, v in rows])


def _schedule(season=2026):
    games = [(1, "A", "B"), (2, "A", "B"), (3, "B", "A")]
    return pd.DataFrame([{"game_id": f"{season}_{w:02d}_{a}_{h}", "season": season, "game_type": "REG", "week": w,
                          "home_team": h, "away_team": a, "home_score": 20.0, "away_score": 10.0}
                         for w, h, a in games])


def _pbp_for(schedule, season=2026, through_week=1):
    ids = schedule.loc[(schedule["season"] == season) & (schedule["week"] <= through_week), "game_id"]
    return pd.DataFrame({"game_id": list(ids), "season": season, "week": list(schedule.loc[ids.index, "week"])})


def test_select_takes_first_row_at_or_after_the_target_week():
    rows = select_pregame_rows(_rolling(), season=2026, week=2).set_index("team")
    assert rows.loc["A", "week"] == 2 and rows.loc["A", "x_avg"] == 10.0   # plays week 2: its own row
    assert rows.loc["B", "week"] == 3 and rows.loc["B", "x_avg"] == 11.0   # bye in 2: next game's row


def test_select_never_returns_the_empty_week_one_row_for_week_two():
    # the original `week < W` filter returned exactly these rows
    rows = select_pregame_rows(_rolling(), season=2026, week=2)
    assert rows["x_avg"].notna().all()


def test_select_ignores_other_seasons():
    other = _rolling().assign(season=2025)
    assert select_pregame_rows(other, season=2026, week=2).empty


def test_pregame_selection_tripwire_accepts_correct_rows():
    rows = select_pregame_rows(_rolling(), season=2026, week=2)
    sched = _schedule()
    # A played weeks 1 (before wk 2) = 1 game; B played week 1 = 1 game
    assert_pregame_selection(rows, sched, 2026, 2)


def test_pregame_selection_tripwire_catches_the_old_stale_selection():
    stale = _rolling()
    stale = stale[stale["week"] < 2].sort_values("week").groupby("team").tail(1)  # the original filter
    with pytest.raises(AssertionError, match="PRE-GAME SELECTION"):
        assert_pregame_selection(stale, _schedule(), 2026, 2)


def test_pregame_selection_tripwire_catches_a_row_that_includes_the_target_week():
    leaky = select_pregame_rows(_rolling(), season=2026, week=2)
    leaky = leaky.assign(games_played=leaky["games_played"] + 1)
    with pytest.raises(AssertionError, match="PRE-GAME SELECTION"):
        assert_pregame_selection(leaky, _schedule(), 2026, 2)


def test_pregame_selection_tripwire_catches_a_team_missing_from_the_selection():
    rows = select_pregame_rows(_rolling(), season=2026, week=2)
    rows = rows[rows["team"] != "B"]          # B still has games at/after week 2 but no row was selected
    with pytest.raises(AssertionError, match="no pre-game row selected for \\['B'\\]"):
        assert_pregame_selection(rows, _schedule(), 2026, 2)


def test_pbp_coverage_passes_when_every_played_game_is_in_the_pbp():
    sched = _schedule()
    assert_pbp_covers_played_games(sched, _pbp_for(sched, through_week=1), 2026, 2)   # week 1 played, in pbp


def test_pbp_coverage_refuses_when_pbp_lags_the_schedule():
    sched = _schedule()
    lagging = _pbp_for(sched, through_week=1).iloc[0:0]                                # schedule has the score, pbp doesn't
    with pytest.raises(AssertionError, match="missing 1 game"):
        assert_pbp_covers_played_games(sched, lagging, 2026, 2)


def test_pbp_coverage_ignores_games_at_or_after_the_target_week():
    sched = _schedule()
    assert_pbp_covers_played_games(sched, _pbp_for(sched, through_week=0), 2026, 1)    # nothing before week 1


def _capture_builders(monkeypatch, rolling):
    seen = {}

    def fake_team_game_stats(schedule, pbp):
        seen["schedule"], seen["pbp"] = schedule, pbp
        return pd.DataFrame()

    monkeypatch.setattr(live_stats, "build_team_game_stats", fake_team_game_stats)
    monkeypatch.setattr(live_stats, "build_rolling_team_stats", lambda tgs: rolling)
    return seen


def test_pregame_team_stats_cuts_inputs_to_the_start_of_the_week(monkeypatch):
    seen = _capture_builders(monkeypatch, _rolling())
    prior_sched = _schedule(2025)
    cur_sched = _schedule(2026)
    cur_pbp = pd.DataFrame({"game_id": ["2026_01_B_A", "2026_02_B_A", "2026_02_B_A"], "season": [2026] * 3,
                            "week": [1, 2, 2], "epa": [1.0, 2.0, 3.0]})
    prior_pbp = pd.DataFrame({"game_id": ["2025_18_B_A"], "season": [2025], "week": [18], "epa": [9.0]})

    pregame_team_stats(cur_sched, cur_pbp, 2026, 2, prior_schedule=prior_sched, prior_pbp=prior_pbp)

    s = seen["schedule"]
    cur = s[s["season"] == 2026]
    assert cur.loc[cur["week"] >= 2, ["home_score", "away_score"]].isna().all().all()  # Thursday etc. blanked
    assert cur.loc[cur["week"] < 2, ["home_score", "away_score"]].notna().all().all()
    assert s.loc[s["season"] == 2025, "home_score"].notna().all()                       # prior season intact
    p = seen["pbp"]
    assert not ((p["season"] == 2026) & (p["week"] >= 2)).any()                          # no week-2 plays
    assert (p["season"] == 2025).any()                                                   # prior season included


def test_pregame_team_stats_rejects_unknown_team_codes(monkeypatch):
    _capture_builders(monkeypatch, _rolling())
    cur = _schedule(2026).assign(home_team="ZZZ")
    with pytest.raises(ValueError, match="ZZZ"):
        pregame_team_stats(cur, EMPTY_PBP, 2026, 2, prior_schedule=_schedule(2025), prior_pbp=EMPTY_PBP)


def test_pregame_team_stats_refuses_when_required_pbp_stats_are_undefined(monkeypatch):
    rolling = _rolling().assign(epa_avg=np.nan)
    _capture_builders(monkeypatch, rolling)
    sched = _schedule(2026)
    with pytest.raises(AssertionError, match="undefined"):
        pregame_team_stats(sched, _pbp_for(sched), 2026, 2, prior_schedule=_schedule(2025), prior_pbp=EMPTY_PBP,
                           required_cols=["epa_avg"])


def test_pregame_team_stats_accepts_defined_required_stats(monkeypatch):
    rolling = _rolling().assign(epa_avg=0.1)
    _capture_builders(monkeypatch, rolling)
    sched = _schedule(2026)
    rows = pregame_team_stats(sched, _pbp_for(sched), 2026, 2, prior_schedule=_schedule(2025), prior_pbp=EMPTY_PBP,
                              required_cols=["epa_avg"])
    assert sorted(rows["team"]) == ["A", "B"]


def test_pregame_team_stats_refuses_when_pbp_is_missing_a_played_game(monkeypatch):
    _capture_builders(monkeypatch, _rolling())
    sched = _schedule(2026)
    with pytest.raises(AssertionError, match="play-by-play is missing"):
        pregame_team_stats(sched, _pbp_for(sched).iloc[0:0], 2026, 2, prior_schedule=_schedule(2025),
                           prior_pbp=EMPTY_PBP)


# ---- wiring into model/predict.py: the tests that would catch `week < W` being re-inlined --------

def test_current_season_stats_delegates_to_pregame_team_stats_with_the_week(monkeypatch):
    sched = _schedule(2026)
    monkeypatch.setattr(predict.nfl, "import_schedules", lambda seasons: sched)
    monkeypatch.setattr(predict, "load_season_pbp", lambda season: _pbp_for(sched))
    seen = {}

    def spy(schedule, pbp, season, week, **kwargs):
        seen.update(season=season, week=week, **kwargs)
        return pd.DataFrame({"team": ["A"]})

    monkeypatch.setattr(predict, "pregame_team_stats", spy)
    predict._current_season_stats(2026, 2)
    assert seen["season"] == 2026 and seen["week"] == 2
    assert seen["required_cols"] == predict.PBP_DERIVED_STAT_COLS


def test_current_season_stats_is_empty_before_any_game_is_played(monkeypatch):
    sched = _schedule(2026).assign(home_score=np.nan, away_score=np.nan)
    monkeypatch.setattr(predict.nfl, "import_schedules", lambda seasons: sched)
    out = predict._current_season_stats(2026, 1)
    assert out.empty and "team" in out.columns


def test_get_pregame_stats_uses_current_rows_then_fills_gaps_from_the_prior_season(monkeypatch):
    col = STAT_COLS[0]
    current = pd.DataFrame({"team": ["A"], **{c: 1.0 for c in STAT_COLS}, "home_point_diff_avg": np.nan,
                            "away_point_diff_avg": 2.0}).assign(**{col: np.nan})
    fallback = pd.DataFrame({"team": ["A", "Z"], **{c: 0.5 for c in STAT_COLS},
                             "home_point_diff_avg": 3.0, "away_point_diff_avg": 4.0})
    monkeypatch.setattr(predict, "_current_season_stats", lambda season, week: current)
    monkeypatch.setattr(predict, "_fallback_stats", lambda season: fallback)
    stats = predict.get_pregame_stats(2026, 2)
    assert set(stats.index) == {"A", "Z"}                      # Z has no current row: whole fallback row
    assert stats.at["A", col] == 0.5                           # undefined -> prior season
    assert stats.at["A", "home_point_diff_avg"] == 3.0         # split column filled from the prior season too
    assert stats.at["A", "away_point_diff_avg"] == 2.0         # a real current value is never overwritten


def _stats(**overrides):
    base = {c: 1.0 for c in STAT_COLS}
    return {**base, **overrides}


def test_fill_missing_stat_values_uses_prior_season_then_league_mean():
    col = STAT_COLS[0]
    stats = pd.DataFrame({"ATL": _stats(**{col: np.nan}), "CAR": _stats(**{col: 0.4}),
                          "NEW": _stats(**{col: np.nan})}).T
    fallback = pd.DataFrame({"ATL": _stats(**{col: 0.55})}).T
    filled, log = fill_missing_stat_values(stats, fallback)
    assert filled.at["ATL", col] == 0.55                      # prior-season value
    assert filled.at["NEW", col] == pytest.approx(0.4)        # no prior value -> mean of what's known
    assert sorted(log) == [("ATL", col), ("NEW", col)]
    assert pd.isna(stats.at["ATL", col])                      # input not mutated


def test_fill_missing_stat_values_fills_split_columns_from_the_prior_season_but_never_the_league_mean():
    stats = pd.DataFrame({"ATL": {**_stats(), "home_point_diff_avg": np.nan, "away_point_diff_avg": np.nan},
                          "CAR": {**_stats(), "home_point_diff_avg": 4.0, "away_point_diff_avg": 6.0}}).T
    fallback = pd.DataFrame({"ATL": {**_stats(), "home_point_diff_avg": 2.5, "away_point_diff_avg": np.nan}}).T
    filled, log = fill_missing_stat_values(stats, fallback)
    assert filled.at["ATL", "home_point_diff_avg"] == 2.5             # prior season value (as in training)
    assert pd.isna(filled.at["ATL", "away_point_diff_avg"])           # no prior either: stays NaN -> 0.0 downstream
    assert filled.at["CAR", "home_point_diff_avg"] == 4.0             # real values untouched
    assert ("ATL", "home_point_diff_avg") in log and ("ATL", "away_point_diff_avg") not in log
    assert set(SPLIT_COLS) == {"home_point_diff_avg", "away_point_diff_avg"}


def test_fill_missing_stat_values_leaves_complete_stats_alone():
    stats = pd.DataFrame({"ATL": {**_stats(), "home_point_diff_avg": 1.0, "away_point_diff_avg": 2.0}}).T
    filled, log = fill_missing_stat_values(stats, stats)
    assert log == [] and filled.equals(stats)
