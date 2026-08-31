# tests for the three dumb-model baselines (Week 1 Audit & Tuning Plan Phase 4.4)
import pandas as pd

from data.baselines import always_home, always_vegas_favorite, prior_season_win_pct


def _games(rows):
    return pd.DataFrame(rows)


def test_always_home_counts_correctly():
    games = _games([
        {"home_score": 24, "away_score": 20},  # home win
        {"home_score": 10, "away_score": 30},  # away win
        {"home_score": 17, "away_score": 14},  # home win
    ])
    result = always_home(games)
    assert result["accuracy"] == 2 / 3
    assert result["n"] == 3


def test_always_vegas_favorite_uses_positive_spread_is_home_favored_convention():
    games = _games([
        # home favored (spread_line > 0) and home wins -> correct
        {"home_score": 24, "away_score": 20, "spread_line": 3.5},
        # away favored (spread_line < 0) but home wins anyway -> wrong
        {"home_score": 24, "away_score": 20, "spread_line": -3.5},
    ])
    result = always_vegas_favorite(games)
    assert result["accuracy"] == 0.5
    assert result["n"] == 2


def test_always_vegas_favorite_excludes_pick_em_games():
    games = _games([
        {"home_score": 24, "away_score": 20, "spread_line": 0.0},  # no favorite to pick
        {"home_score": 24, "away_score": 20, "spread_line": 3.0},
    ])
    result = always_vegas_favorite(games)
    assert result["n"] == 1  # pick'em excluded, not guessed


def test_prior_season_win_pct_picks_the_better_record():
    schedules = _games([
        # 2024: SEA 2-0, NE 0-2 (SEA has the better prior record entering 2025)
        {"season": 2024, "week": 1, "game_type": "REG", "home_team": "SEA", "away_team": "NE",
         "home_score": 24, "away_score": 10},
        {"season": 2024, "week": 2, "game_type": "REG", "home_team": "NE", "away_team": "SEA",
         "home_score": 10, "away_score": 24},
    ])
    games_2025 = _games([
        {"season": 2025, "home_team": "SEA", "away_team": "NE", "home_score": 20, "away_score": 17},
    ])
    result = prior_season_win_pct(games_2025, schedules)
    assert result["accuracy"] == 1.0  # picked SEA (better 2024 record), and SEA won
    assert result["n"] == 1


def test_prior_season_win_pct_skips_games_with_no_prior_record():
    """The very first cached season (or a genuine expansion team) has
    nothing to compare -- must be skipped, not guessed at."""
    schedules = pd.DataFrame(columns=["season", "week", "game_type", "home_team", "away_team", "home_score", "away_score"])
    games = _games([
        {"season": 2016, "home_team": "SEA", "away_team": "NE", "home_score": 20, "away_score": 17},
    ])
    result = prior_season_win_pct(games, schedules)
    assert result["n"] == 0


def test_prior_season_win_pct_skips_a_genuine_tie_in_prior_record():
    schedules = _games([
        {"season": 2024, "week": 1, "game_type": "REG", "home_team": "SEA", "away_team": "NE",
         "home_score": 24, "away_score": 24},  # tie -> both teams end up 0.5
    ])
    games_2025 = _games([
        {"season": 2025, "home_team": "SEA", "away_team": "NE", "home_score": 20, "away_score": 17},
    ])
    result = prior_season_win_pct(games_2025, schedules)
    assert result["n"] == 0  # identical prior records -- nothing to pick
