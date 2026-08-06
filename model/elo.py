# split offense/defense NFL Elo (start 1500, K~20, home field +55, MOV multiplier)
"""Offense/defense-split NFL Elo (v3 spec Phase 5) -- every team starts at
1500 on *both* sides, and each side updates independently after every game.
A single blended rating hides mismatched teams: a great-offense/bad-defense
team and a balanced team can land on a similar combined number, but they're
genuinely different matchups. This reacts much faster to hot/cold streaks
than the season-to-date rolling stats elsewhere in this project -- that's
the complementary signal the split ratings are after. Both offensive and
defensive diffs get fed to the model separately, which learns how much to
weigh each.
"""

import math

import pandas as pd

INITIAL_RATING = 1500.0
K_FACTOR = 20.0
HOME_FIELD_ADV = 55.0
# How far a team's ratings drift back toward 1500 between seasons, to
# account for roster turnover. 1/3 is the standard FiveThirtyEight
# convention for NFL Elo, applied to both sides independently here.
SEASON_REGRESSION = 1 / 3


def _expected_share(off_elo: float, def_elo: float, home_bonus: float = 0.0) -> float:
    """Same logistic S-curve as a standard win-probability Elo, but mapped
    onto a [0,1] *scoring share* instead of a binary win/loss -- see
    compute_elo_ratings for how "share" is defined."""
    return 1.0 / (1.0 + 10 ** (-((off_elo + home_bonus) - def_elo) / 400.0))


def _mov_multiplier(margin: float, elo_diff: float) -> float:
    """Bigger, more surprising blowouts move ratings more than narrow or
    expected wins."""
    return math.log(abs(margin) + 1) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))


def compute_elo_ratings(schedules: pd.DataFrame):
    """Processes every REG-season game in `schedules` in chronological
    order, updating four independent ratings per game: each team's offense
    against the other's defense.

    Each side's "result" is that team's share of the game's total points
    (home_score / (home_score + away_score)) rather than a plain win/loss
    -- this avoids needing an arbitrary league-average scoring baseline to
    judge a single side's performance against, and stays entirely
    self-referential to the two teams actually on the field. The home
    side's offense gets the usual home-field bump; the away side's does
    not (adding it to both would just cancel out).

    Returns (per_game_df, final_ratings):
      - per_game_df: one row per game with each team's off/def Elo
        *before* that game -- safe to use as a pre-game feature, no
        leakage.
      - final_ratings: dict of team -> {"off": ..., "def": ...} after the
        last completed game in the input, i.e. "current" ratings for
        scoring an upcoming game.
    """
    reg = schedules[schedules["game_type"] == "REG"].sort_values(["season", "week", "gameday"]).copy()

    off_ratings, def_ratings = {}, {}
    records = []
    current_season = None

    for _, g in reg.iterrows():
        season, home, away = g["season"], g["home_team"], g["away_team"]

        if season != current_season:
            if current_season is not None:
                for team in set(off_ratings) | set(def_ratings):
                    off_ratings[team] = (
                        off_ratings.get(team, INITIAL_RATING) * (1 - SEASON_REGRESSION)
                        + INITIAL_RATING * SEASON_REGRESSION
                    )
                    def_ratings[team] = (
                        def_ratings.get(team, INITIAL_RATING) * (1 - SEASON_REGRESSION)
                        + INITIAL_RATING * SEASON_REGRESSION
                    )
            current_season = season

        home_off, home_def = off_ratings.get(home, INITIAL_RATING), def_ratings.get(home, INITIAL_RATING)
        away_off, away_def = off_ratings.get(away, INITIAL_RATING), def_ratings.get(away, INITIAL_RATING)

        records.append({
            "game_id": g["game_id"], "season": season, "week": g["week"],
            "home_team": home, "away_team": away,
            "home_off_elo_pre": home_off, "home_def_elo_pre": home_def,
            "away_off_elo_pre": away_off, "away_def_elo_pre": away_def,
        })

        if pd.isna(g["home_score"]):
            continue  # not played yet -- nothing to update

        home_score, away_score = g["home_score"], g["away_score"]
        margin = home_score - away_score
        total_points = home_score + away_score
        if margin == 0 or total_points == 0:
            continue  # tie or scoreless game (both vanishingly rare), skip update

        home_share = home_score / total_points
        away_share = 1 - home_share
        mult = _mov_multiplier(margin, home_off - away_def)

        # International/neutral-site games (schedules' own "location"
        # field) get no home-field bonus -- the "home" team is still the
        # designated home team for scheduling purposes, but doesn't
        # actually get the crowd/travel advantage that bonus represents.
        home_bonus = 0.0 if g.get("location") == "Neutral" else HOME_FIELD_ADV
        exp_home_share = _expected_share(home_off, away_def, home_bonus)
        home_off_change = K_FACTOR * mult * (home_share - exp_home_share)

        exp_away_share = _expected_share(away_off, home_def)
        away_off_change = K_FACTOR * mult * (away_share - exp_away_share)

        off_ratings[home] = home_off + home_off_change
        def_ratings[away] = away_def - home_off_change  # away D gave up more/less than expected
        off_ratings[away] = away_off + away_off_change
        def_ratings[home] = home_def - away_off_change  # home D gave up more/less than expected

    final = {
        team: {"off": off_ratings.get(team, INITIAL_RATING), "def": def_ratings.get(team, INITIAL_RATING)}
        for team in set(off_ratings) | set(def_ratings)
    }
    return pd.DataFrame(records), final
