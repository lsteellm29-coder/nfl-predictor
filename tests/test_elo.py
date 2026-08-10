# tests for model/elo.py's split offense/defense Elo formulas
"""_expected_share/_mov_multiplier are checked against hand-computed
values; compute_elo_ratings is checked end-to-end on small synthetic
schedules for its stated no-leakage guarantee (pre-game ratings must
never reflect that same game's own result), the neutral-site home-bonus
exception, and season-boundary regression toward 1500."""

import math

import pandas as pd
import pytest

from model.elo import HOME_FIELD_ADV, INITIAL_RATING, K_FACTOR, SEASON_REGRESSION, _expected_share, _mov_multiplier, compute_elo_ratings


def test_expected_share_equal_ratings_no_bonus_is_half():
    assert _expected_share(1500, 1500, home_bonus=0.0) == pytest.approx(0.5, abs=1e-9)


def test_expected_share_higher_offense_favored_above_half():
    assert _expected_share(1700, 1500, home_bonus=0.0) > 0.5


def test_expected_share_home_bonus_shifts_expectation_up():
    no_bonus = _expected_share(1500, 1500, home_bonus=0.0)
    with_bonus = _expected_share(1500, 1500, home_bonus=HOME_FIELD_ADV)
    assert with_bonus > no_bonus


def test_mov_multiplier_bigger_margin_moves_more():
    small = _mov_multiplier(margin=3, elo_diff=0)
    big = _mov_multiplier(margin=28, elo_diff=0)
    assert big > small


def test_mov_multiplier_expected_blowout_dampened_vs_surprise_blowout():
    """A blowout that matches a huge pre-game Elo gap (expected) should
    move ratings less than the same-margin blowout between evenly-rated
    teams (a surprise) -- the whole point of a MOV multiplier."""
    surprise = _mov_multiplier(margin=21, elo_diff=0)
    expected_blowout = _mov_multiplier(margin=21, elo_diff=800)
    assert surprise > expected_blowout


def _game_row(game_id, season, week, home, away, home_score, away_score, location="Home"):
    return {"game_id": game_id, "season": season, "week": week, "game_type": "REG",
            "home_team": home, "away_team": away, "home_score": home_score, "away_score": away_score,
            "gameday": f"{season}-09-{week:02d}", "location": location}


def test_first_game_pre_ratings_are_untouched_initial_values():
    """No-leakage guarantee: a team's very first game of the dataset must
    show the pre-game rating as the pure INITIAL_RATING, never influenced
    by that same game's own result."""
    schedules = pd.DataFrame([_game_row("g1", 2025, 1, "SEA", "NE", 24, 17)])
    per_game, _ = compute_elo_ratings(schedules)
    row = per_game.iloc[0]
    assert row["home_off_elo_pre"] == INITIAL_RATING
    assert row["home_def_elo_pre"] == INITIAL_RATING
    assert row["away_off_elo_pre"] == INITIAL_RATING
    assert row["away_def_elo_pre"] == INITIAL_RATING


def test_second_game_pre_rating_reflects_first_games_result_not_initial():
    """The team's SECOND game must show a pre-game rating that moved off
    1500 -- confirms the update from game 1 actually persists forward
    (and isn't itself leaked backward into game 1's own pre-game value,
    checked above)."""
    schedules = pd.DataFrame([
        _game_row("g1", 2025, 1, "SEA", "NE", 30, 10),   # SEA wins big -> SEA off_elo should rise
        _game_row("g2", 2025, 2, "SEA", "KC", 20, 20),   # tie, skipped for rating update, but still a pre-game snapshot
    ])
    per_game, _ = compute_elo_ratings(schedules)
    week2 = per_game[per_game["week"] == 2].iloc[0]
    assert week2["home_off_elo_pre"] != INITIAL_RATING
    assert week2["home_off_elo_pre"] > INITIAL_RATING  # SEA blew out NE, should have gained offensive rating


def test_tie_or_scoreless_game_does_not_update_ratings():
    schedules = pd.DataFrame([
        _game_row("g1", 2025, 1, "SEA", "NE", 20, 20),  # tie -> margin=0, must be skipped
        _game_row("g2", 2025, 2, "SEA", "KC", 24, 10),
    ])
    per_game, _ = compute_elo_ratings(schedules)
    week2 = per_game[per_game["week"] == 2].iloc[0]
    # if the tie had (incorrectly) updated ratings, home_off_elo_pre would differ from INITIAL_RATING
    assert week2["home_off_elo_pre"] == INITIAL_RATING


def test_neutral_site_game_gets_no_home_bonus():
    """Two otherwise-identical games (same score, same pre-game ratings)
    differing only in location=Neutral vs Home must produce different
    off_elo updates for the home team, since the expected-share
    calculation drops the +55 home-field bonus on a neutral site."""
    home_game = pd.DataFrame([_game_row("g1", 2025, 1, "SEA", "NE", 24, 17, location="Home")])
    neutral_game = pd.DataFrame([_game_row("g1", 2025, 1, "SEA", "NE", 24, 17, location="Neutral")])

    _, final_home = compute_elo_ratings(home_game)
    _, final_neutral = compute_elo_ratings(neutral_game)

    assert final_home["SEA"]["off"] != final_neutral["SEA"]["off"]


def test_season_boundary_regresses_ratings_toward_initial():
    """A team that finished a season well above 1500 should start the
    next season's PRE-game rating pulled back toward 1500 by exactly
    SEASON_REGRESSION -- the module's own stated FiveThirtyEight-style
    convention, checked by hand."""
    schedules = pd.DataFrame([
        _game_row("g1", 2025, 1, "SEA", "NE", 40, 3),     # SEA blows out NE -> big off_elo gain
        _game_row("g2", 2026, 1, "SEA", "KC", 20, 20),    # tie in the new season, no update, just a snapshot
    ])
    per_game, final_2025 = compute_elo_ratings(schedules[schedules["season"] == 2025])
    full_per_game, _ = compute_elo_ratings(schedules)

    end_of_2025_off = final_2025["SEA"]["off"]
    assert end_of_2025_off > INITIAL_RATING  # confirm there's something real to regress

    start_of_2026_pre = full_per_game[(full_per_game["season"] == 2026)].iloc[0]["home_off_elo_pre"]
    expected = end_of_2025_off * (1 - SEASON_REGRESSION) + INITIAL_RATING * SEASON_REGRESSION
    assert start_of_2026_pre == pytest.approx(expected, abs=1e-6)
