# three dumb models this project's own model has to beat (Week 1 Audit & Tuning Plan Phase 4.4)
"""If the real model doesn't beat all three of these, it isn't adding
value yet. Always printed alongside real backtest metrics
(model/calibration.py's main()), never computed in isolation -- "our
model is X% accurate" means nothing on its own without "and here's what
a coin flip / the market / last year's record would have gotten too."

All three operate on a `games`-shaped dataframe with at least
season/week/home_team/away_team/home_score/away_score/spread_line
(model/train.py's build_feature_frame() output already has this shape),
except prior_season_win_pct(), which needs the raw schedule directly
since it looks at a DIFFERENT season than the one being scored.
"""

import pandas as pd


def always_home(games: pd.DataFrame) -> dict:
    """Straight-up accuracy of "the home team always wins" -- real NFL
    home-field advantage, not a coin flip."""
    accuracy = float((games["home_score"] > games["away_score"]).mean())
    return {"name": "always_home", "accuracy": accuracy, "n": len(games)}


def always_vegas_favorite(games: pd.DataFrame) -> dict:
    """Straight-up accuracy of "whichever side the market's own spread
    favors" -- this codebase's verified convention (AUDIT.md Phase 1.2):
    positive spread_line = home favored. A pick'em (spread_line == 0)
    has no favorite to pick, so it's excluded rather than guessed."""
    valid = games[games["spread_line"] != 0]
    home_favored = valid["spread_line"] > 0
    home_won = valid["home_score"] > valid["away_score"]
    accuracy = float((home_favored == home_won).mean()) if len(valid) else float("nan")
    return {"name": "always_vegas_favorite", "accuracy": accuracy, "n": len(valid)}


def _team_season_win_pct(schedules: pd.DataFrame, season: int) -> dict[str, float]:
    """team -> that season's final win percentage (ties count as half a
    win, the standard NFL standings convention) -- computed directly
    from the schedule, not team_stats.parquet's ats_win_pct_season
    (a DIFFERENT thing: against-the-spread record, not straight win%)."""
    reg = schedules[
        (schedules["season"] == season) & (schedules["game_type"] == "REG") & schedules["home_score"].notna()
    ]
    wins: dict[str, float] = {}
    games: dict[str, int] = {}
    for _, g in reg.iterrows():
        home, away = g["home_team"], g["away_team"]
        if g["home_score"] > g["away_score"]:
            home_result, away_result = 1.0, 0.0
        elif g["home_score"] < g["away_score"]:
            home_result, away_result = 0.0, 1.0
        else:
            home_result = away_result = 0.5
        for team, result in ((home, home_result), (away, away_result)):
            wins[team] = wins.get(team, 0.0) + result
            games[team] = games.get(team, 0) + 1
    return {team: wins[team] / games[team] for team in games}


def prior_season_win_pct(games: pd.DataFrame, schedules: pd.DataFrame) -> dict:
    """Straight-up accuracy of "pick whichever team had the better prior-
    season win percentage" -- a team with no prior-season record at all
    (the very first cached season) is skipped for that game rather than
    guessed at, the same honest-data-limit rule this whole project
    already applies everywhere else."""
    correct = 0
    n = 0
    for season in sorted(games["season"].unique()):
        prior_pct = _team_season_win_pct(schedules, season - 1)
        season_games = games[games["season"] == season]
        for _, g in season_games.iterrows():
            home_pct, away_pct = prior_pct.get(g["home_team"]), prior_pct.get(g["away_team"])
            if home_pct is None or away_pct is None or home_pct == away_pct:
                continue  # no prior record, or a genuine tie with nothing to pick
            picked_home = home_pct > away_pct
            home_won = g["home_score"] > g["away_score"]
            correct += int(picked_home == home_won)
            n += 1
    return {"name": "prior_season_win_pct", "accuracy": correct / n if n else float("nan"), "n": n}


def all_baselines(games: pd.DataFrame, schedules: pd.DataFrame) -> list[dict]:
    return [
        always_home(games),
        always_vegas_favorite(games),
        prior_season_win_pct(games, schedules),
    ]


def print_baselines(games: pd.DataFrame, schedules: pd.DataFrame) -> None:
    print("Baselines (must beat all three for the model to be adding value):")
    for b in all_baselines(games, schedules):
        print(f"  {b['name']:22s} accuracy {b['accuracy']:.3f}  (n={b['n']})")
