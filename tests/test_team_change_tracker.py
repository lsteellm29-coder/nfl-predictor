# tests for coaching + per-unit roster continuity (Combined Build Plan Part 3)
"""data/team_change_tracker.py's functions are pure decision logic over
nfl_data_py calls (import_schedules, import_snap_counts) and data/
rosters.py's fetch_rosters() -- all three are monkeypatched with small
synthetic frames rather than hitting the network, since the shape being
tested (turnover math, coach-change detection, callout phrasing) doesn't
depend on real season data.
"""

import pandas as pd

import data.team_change_tracker as tracker


def _schedule(team_coach_pairs):
    rows = []
    for i, (home, home_coach, away, away_coach) in enumerate(team_coach_pairs):
        rows.append({"home_team": home, "home_coach": home_coach, "away_team": away, "away_coach": away_coach})
    return pd.DataFrame(rows)


def test_head_coach_changes_detects_a_real_change(monkeypatch):
    def fake_schedules(seasons):
        season = seasons[0]
        if season == 2025:
            return _schedule([("SEA", "Coach A", "NE", "Coach B")])
        return _schedule([("SEA", "Coach C", "NE", "Coach B")])

    monkeypatch.setattr(tracker.nfl, "import_schedules", fake_schedules)
    changes = tracker.head_coach_changes(2026)
    assert changes == {"SEA": {"prior_coach": "Coach A", "current_coach": "Coach C"}}


def test_head_coach_changes_empty_when_nobody_changed(monkeypatch):
    def fake_schedules(seasons):
        return _schedule([("SEA", "Coach A", "NE", "Coach B")])

    monkeypatch.setattr(tracker.nfl, "import_schedules", fake_schedules)
    assert tracker.head_coach_changes(2026) == {}


def _snap_counts(rows):
    """rows: list of (team, player, position, snap_pct) x however many
    games -- caller repeats a row MEANINGFUL_MIN_GAMES times to clear
    the games-played bar."""
    data = []
    for team, player, position, pct, games in rows:
        for week in range(games):
            data.append({
                "team": team, "player": player, "position": position, "game_type": "REG",
                "offense_pct": pct, "defense_pct": 0.0,
            })
    return pd.DataFrame(data)


def test_team_unit_turnover_flags_departed_starter(monkeypatch):
    prior_snaps = _snap_counts([
        ("SEA", "Player One", "WR", 0.9, tracker.MEANINGFUL_MIN_GAMES),
        ("SEA", "Player Two", "WR", 0.9, tracker.MEANINGFUL_MIN_GAMES),
    ])
    monkeypatch.setattr(tracker.nfl, "import_snap_counts", lambda seasons: prior_snaps)

    current_roster = pd.DataFrame([
        {"player_id": "P1", "player_name": "Player One", "team": "SEA", "position": "WR", "season": 2026},
        # Player Two is gone -- signed elsewhere or released.
    ])
    monkeypatch.setattr(tracker, "fetch_rosters", lambda seasons: current_roster)

    result = tracker.team_unit_turnover(2026)
    assert result["SEA"]["offensive skill"]["departed"] == 1
    assert result["SEA"]["offensive skill"]["total"] == 2
    assert result["SEA"]["offensive skill"]["turnover_pct"] == 0.5


def test_team_unit_turnover_no_prior_snap_data_returns_empty(monkeypatch):
    monkeypatch.setattr(tracker.nfl, "import_snap_counts", lambda seasons: pd.DataFrame(
        columns=["team", "player", "position", "game_type", "offense_pct", "defense_pct"]))
    assert tracker.team_unit_turnover(2026) == {}


def test_below_meaningful_games_threshold_is_not_a_starter(monkeypatch):
    """A player who only played one game at a high snap share (a spot
    start, not a real starter) shouldn't count -- MEANINGFUL_MIN_GAMES
    is there specifically to filter this out."""
    prior_snaps = _snap_counts([("SEA", "Player One", "WR", 0.9, 1)])
    monkeypatch.setattr(tracker.nfl, "import_snap_counts", lambda seasons: prior_snaps)
    monkeypatch.setattr(tracker, "fetch_rosters", lambda seasons: pd.DataFrame(
        columns=["player_id", "player_name", "team", "position", "season"]))

    assert tracker.team_unit_turnover(2026) == {}


def test_notable_player_moves_detects_departure_and_arrival(monkeypatch):
    prior_snaps = _snap_counts([("SEA", "Star Player", "WR", 0.85, tracker.MEANINGFUL_MIN_GAMES)])
    monkeypatch.setattr(tracker.nfl, "import_snap_counts", lambda seasons: prior_snaps)

    current_roster = pd.DataFrame([
        {"player_id": "P1", "player_name": "Star Player", "team": "NE", "position": "WR", "season": 2026},
    ])
    monkeypatch.setattr(tracker, "fetch_rosters", lambda seasons: current_roster)

    result = tracker.notable_player_moves(2026)
    assert "Star Player" in result["SEA"]["departed"]
    assert any("Star Player" in a for a in result["NE"]["arrived"])


def test_notable_player_moves_same_team_is_not_a_move(monkeypatch):
    prior_snaps = _snap_counts([("SEA", "Star Player", "WR", 0.85, tracker.MEANINGFUL_MIN_GAMES)])
    monkeypatch.setattr(tracker.nfl, "import_snap_counts", lambda seasons: prior_snaps)
    current_roster = pd.DataFrame([
        {"player_id": "P1", "player_name": "Star Player", "team": "SEA", "position": "WR", "season": 2026},
    ])
    monkeypatch.setattr(tracker, "fetch_rosters", lambda seasons: current_roster)

    assert tracker.notable_player_moves(2026) == {}


def test_team_change_callouts_combines_all_three_signals():
    coach_changes = {"SEA": {"prior_coach": "Coach A", "current_coach": "Coach B"}}
    unit_turnover = {"SEA": {"secondary": {"turnover_pct": 0.6, "departed": 3, "total": 5}}}
    notable_moves = {"SEA": {"departed": ["Star Player"], "arrived": ["New Guy (from NE)"]}}

    callouts = tracker.team_change_callouts("SEA", coach_changes, unit_turnover, notable_moves)
    assert any("New head coach" in c for c in callouts)
    assert any("secondary" in c and "3 of 5" in c for c in callouts)
    assert any("Star Player" in c for c in callouts)
    assert any("New Guy" in c for c in callouts)


def test_team_change_callouts_empty_for_unaffected_team():
    assert tracker.team_change_callouts("KC", {}, {}, {}) == []


def test_unit_turnover_below_threshold_is_not_a_callout():
    unit_turnover = {"SEA": {"secondary": {"turnover_pct": 0.2, "departed": 1, "total": 5}}}
    callouts = tracker.team_change_callouts("SEA", {}, unit_turnover, {})
    assert callouts == []


# Combined Build Plan Part 3 step 4/5: the confidence-caveat tie-in.
COACH_CHANGE = {"prior_coach": "Coach A", "current_coach": "Coach B"}
HIGH_TURNOVER = {"secondary": {"turnover_pct": tracker.UNIT_TURNOVER_NOTABLE + 0.05, "departed": 5, "total": 8}}
LOW_TURNOVER = {"secondary": {"turnover_pct": 0.1, "departed": 1, "total": 8}}


def test_confidence_flag_requires_both_coach_change_and_high_turnover():
    assert tracker.team_change_confidence_flag(0, COACH_CHANGE, HIGH_TURNOVER) is True


def test_confidence_flag_false_without_coach_change():
    assert tracker.team_change_confidence_flag(0, None, HIGH_TURNOVER) is False


def test_confidence_flag_false_without_high_turnover():
    assert tracker.team_change_confidence_flag(0, COACH_CHANGE, LOW_TURNOVER) is False


def test_confidence_flag_expires_after_games_window():
    assert tracker.team_change_confidence_flag(
        tracker.COACH_CHANGE_GAMES_WINDOW, COACH_CHANGE, HIGH_TURNOVER) is False
    assert tracker.team_change_confidence_flag(
        tracker.COACH_CHANGE_GAMES_WINDOW - 1, COACH_CHANGE, HIGH_TURNOVER) is True
