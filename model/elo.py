# standard NFL Elo (start 1500, K~20, home field adj ~+55, MOV multiplier)
"""Standard NFL Elo rating system (FiveThirtyEight-style): every team starts
at 1500, and ratings update after each game based on margin of victory and
how surprising the result was. This reacts much faster to hot/cold streaks
than the season-to-date rolling stats elsewhere in this project -- that's
the complementary signal Phase 2 is after. Both get fed to the model, which
learns how much to weigh each.
"""

import math

import pandas as pd

INITIAL_RATING = 1500.0
K_FACTOR = 20.0
HOME_FIELD_ADV = 55.0
# How far a team's rating drifts back toward 1500 between seasons, to
# account for roster turnover -- a team's 2016 rating shouldn't still be
# fully "live" in 2025 with zero decay. 1/3 is the standard FiveThirtyEight
# convention for NFL Elo.
SEASON_REGRESSION = 1 / 3


def _expected_home_win_prob(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** (-((home_elo + HOME_FIELD_ADV) - away_elo) / 400.0))


def _mov_multiplier(margin: float, elo_diff: float) -> float:
    """Bigger, more surprising blowouts move ratings more than narrow or
    expected wins."""
    return math.log(abs(margin) + 1) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))


def compute_elo_ratings(schedules: pd.DataFrame):
    """Processes every REG-season game in `schedules` in chronological order.

    Returns (per_game_df, final_ratings):
      - per_game_df: one row per game with each team's Elo *before* that
        game -- safe to use as a pre-game feature, no leakage.
      - final_ratings: dict of team -> rating after the last completed game
        in the input, i.e. "current" ratings for scoring an upcoming game.
    """
    reg = schedules[schedules["game_type"] == "REG"].sort_values(["season", "week", "gameday"]).copy()

    ratings = {}
    records = []
    current_season = None

    for _, g in reg.iterrows():
        season, home, away = g["season"], g["home_team"], g["away_team"]

        if season != current_season:
            if current_season is not None:
                for team in ratings:
                    ratings[team] = (
                        ratings[team] * (1 - SEASON_REGRESSION) + INITIAL_RATING * SEASON_REGRESSION
                    )
            current_season = season

        home_elo = ratings.get(home, INITIAL_RATING)
        away_elo = ratings.get(away, INITIAL_RATING)

        records.append({
            "game_id": g["game_id"], "season": season, "week": g["week"],
            "home_team": home, "away_team": away,
            "home_elo_pre": home_elo, "away_elo_pre": away_elo,
        })

        if pd.isna(g["home_score"]):
            continue  # not played yet -- nothing to update

        margin = g["home_score"] - g["away_score"]
        if margin == 0:
            continue  # tie (rare), skip update

        actual_home = 1.0 if margin > 0 else 0.0
        expected_home = _expected_home_win_prob(home_elo, away_elo)
        mult = _mov_multiplier(margin, home_elo - away_elo)
        change = K_FACTOR * mult * (actual_home - expected_home)

        ratings[home] = home_elo + change
        ratings[away] = away_elo - change

    return pd.DataFrame(records), ratings
