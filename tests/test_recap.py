# tests for report/recap.py's weekly recap generation
"""build_recap() must (a) say nothing when nothing was actually graded
(no fabricated recap about zero real outcomes) and (b) never mangle a
player's name -- caught a real bug testing this live: extracting a
"last name" via name.split()[-1] grabs a suffix ("III", "Jr.") for any
real player who has one, producing "and III scored" instead of "and
Walker scored"."""

import pandas as pd

from report.recap import build_recap

EMPTY_GAMES = pd.DataFrame(columns=["season", "week", "correct", "predicted_winner", "home_team", "away_team",
                                     "home_win_prob", "edge", "model_beat_market"])
EMPTY_PROPS = pd.DataFrame(columns=["season", "week", "player", "edge", "model_correct", "actual_over"])


def test_no_recap_when_nothing_graded():
    assert build_recap(EMPTY_GAMES, EMPTY_PROPS, 2026, 1) is None


def test_recap_never_uses_last_token_as_a_name():
    """The exact regression case: a player whose name ends in a suffix
    ("III") must appear in full, not truncated to the suffix alone."""
    props = pd.DataFrame([{
        "season": 2026, "week": 1, "player": "Kenneth Walker III", "edge": 0.15,
        "model_correct": True, "actual_over": True,
    }])
    text = build_recap(EMPTY_GAMES, props, 2026, 1)
    assert text is not None
    assert "Kenneth Walker III" in text
    assert "and III scored" not in text


def test_recap_reports_real_record_not_a_guess():
    games = pd.DataFrame([
        {"season": 2026, "week": 1, "correct": True, "predicted_winner": "SEA", "home_team": "SEA",
         "away_team": "NE", "home_win_prob": 0.61, "edge": -0.2, "model_beat_market": True},
        {"season": 2026, "week": 1, "correct": False, "predicted_winner": "DAL", "home_team": "NYG",
         "away_team": "DAL", "home_win_prob": 0.30, "edge": -1.5, "model_beat_market": False},
    ])
    text = build_recap(games, EMPTY_PROPS, 2026, 1)
    assert "1-1" in text  # exactly one correct of two graded, not a rounded/approximate figure


def test_recap_skips_games_still_pending():
    """A row with no `correct` value yet (game hasn't finished) must not
    be counted toward the week's record."""
    games = pd.DataFrame([
        {"season": 2026, "week": 1, "correct": True, "predicted_winner": "SEA", "home_team": "SEA",
         "away_team": "NE", "home_win_prob": 0.61, "edge": -0.2, "model_beat_market": True},
        {"season": 2026, "week": 1, "correct": None, "predicted_winner": "DAL", "home_team": "NYG",
         "away_team": "DAL", "home_win_prob": 0.30, "edge": None, "model_beat_market": None},
    ])
    text = build_recap(games, EMPTY_PROPS, 2026, 1)
    assert "1-0" in text
