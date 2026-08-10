# tests for the "Cleared to Play" returned-from-injury detection (Combined Build Plan Part 4)
"""data/fetch_injuries.py's fetch_current_player_injury_status() is a
live-only ESPN snapshot -- there's no historical query on that endpoint,
so "was this player on the report last week" is only answerable by
persisting each week's snapshot ourselves and diffing against it later.
save_injury_snapshot()/detect_returned_players() are pure file-I/O +
set logic once a snapshot exists, so they're tested directly against a
temp cache path rather than mocking the live ESPN fetch.
"""

import data.fetch_injuries as fetch_injuries


def _isolate_cache(monkeypatch, tmp_path):
    path = tmp_path / "injury_history.parquet"
    monkeypatch.setattr(fetch_injuries, "INJURY_HISTORY_PATH", str(path))
    return path


def test_no_history_file_returns_empty_set(monkeypatch, tmp_path):
    """First time this has ever run -- nothing to compare against yet,
    not an error."""
    _isolate_cache(monkeypatch, tmp_path)
    assert fetch_injuries.detect_returned_players({}, 2026, 3) == set()


def test_player_hurt_last_week_and_absent_now_is_returned(monkeypatch, tmp_path):
    _isolate_cache(monkeypatch, tmp_path)
    fetch_injuries.save_injury_snapshot(
        {"P1": {"status": "Questionable", "position": "RB", "usage_multiplier": 0.85}}, 2026, 2)

    returned = fetch_injuries.detect_returned_players({}, 2026, 3)
    assert returned == {"P1"}


def test_player_still_on_report_is_not_returned(monkeypatch, tmp_path):
    """The core distinction from a simple "were they ever hurt" check --
    still-injured players must not show as cleared."""
    _isolate_cache(monkeypatch, tmp_path)
    fetch_injuries.save_injury_snapshot(
        {"P1": {"status": "Questionable", "position": "RB", "usage_multiplier": 0.85}}, 2026, 2)

    current_status = {"P1": {"status": "Questionable", "position": "RB", "usage_multiplier": 0.85}}
    returned = fetch_injuries.detect_returned_players(current_status, 2026, 3)
    assert returned == set()


def test_badge_expires_after_return_window(monkeypatch, tmp_path):
    """The "first game or two back" window -- a player hurt several weeks
    ago (older than RETURN_WINDOW_WEEKS) shouldn't still read as
    "just cleared" indefinitely."""
    _isolate_cache(monkeypatch, tmp_path)
    fetch_injuries.save_injury_snapshot(
        {"P1": {"status": "Out", "position": "WR", "usage_multiplier": 0.0}}, 2026, 1)

    # Week 1 + RETURN_WINDOW_WEEKS + 1 is outside the lookback window.
    too_late_week = 1 + fetch_injuries.RETURN_WINDOW_WEEKS + 1
    assert fetch_injuries.detect_returned_players({}, 2026, too_late_week) == set()

    # But right at the edge of the window, still returned.
    edge_week = 1 + fetch_injuries.RETURN_WINDOW_WEEKS
    assert fetch_injuries.detect_returned_players({}, 2026, edge_week) == {"P1"}


def test_save_injury_snapshot_replaces_same_week_not_duplicates(monkeypatch, tmp_path):
    """A mid-week rerun (e.g. after a late scratch changes the report)
    must overwrite that week's rows, not accumulate them."""
    path = _isolate_cache(monkeypatch, tmp_path)
    fetch_injuries.save_injury_snapshot(
        {"P1": {"status": "Questionable", "position": "RB", "usage_multiplier": 0.85}}, 2026, 2)
    fetch_injuries.save_injury_snapshot(
        {"P2": {"status": "Out", "position": "WR", "usage_multiplier": 0.0}}, 2026, 2)

    import pandas as pd
    history = pd.read_parquet(path)
    week_2_rows = history[history["week"] == 2]
    assert len(week_2_rows) == 1
    assert week_2_rows.iloc[0]["player_id"] == "P2"


def test_save_injury_snapshot_preserves_other_weeks(monkeypatch, tmp_path):
    _isolate_cache(monkeypatch, tmp_path)
    fetch_injuries.save_injury_snapshot(
        {"P1": {"status": "Questionable", "position": "RB", "usage_multiplier": 0.85}}, 2026, 1)
    fetch_injuries.save_injury_snapshot(
        {"P2": {"status": "Out", "position": "WR", "usage_multiplier": 0.0}}, 2026, 2)

    returned = fetch_injuries.detect_returned_players({}, 2026, 3)
    assert returned == {"P1", "P2"}
