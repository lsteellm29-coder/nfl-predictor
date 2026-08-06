# coach and team head-to-head history, narrative only
"""Coach and team head-to-head history for the game-preview narrative --
narrative only, doesn't touch the win-probability model. `schedules`
should be whatever window the caller wants searched; predict.py passes the
cached historical seasons combined with the live current-season schedule,
same pattern as model/predict.py's get_current_elo_ratings.
"""

import pandas as pd

# Below this many meetings, a head-to-head "record" is really just a
# couple of data points wearing a stats costume.
MIN_COACH_MEETINGS = 3
MIN_TEAM_MEETINGS = 3


def _played_games(schedules: pd.DataFrame) -> pd.DataFrame:
    return schedules[(schedules["game_type"] == "REG") & schedules["home_score"].notna()]


def _record(games: pd.DataFrame, a: str, home_col: str) -> tuple[int, int, int]:
    """(a_wins, b_wins, ties) across `games`, crediting each game to
    whichever of a/b was actually the winner regardless of which side of
    home/away they were on that particular game."""
    home_won = games["home_score"] > games["away_score"]
    tied = games["home_score"] == games["away_score"]
    a_was_home = games[home_col] == a
    a_wins = int(((a_was_home & home_won) | (~a_was_home & ~home_won & ~tied)).sum())
    ties = int(tied.sum())
    b_wins = len(games) - a_wins - ties
    return a_wins, b_wins, ties


def coach_h2h(coach_a: str, coach_b: str, schedules: pd.DataFrame) -> dict | None:
    """coach_a/coach_b's all-time record against each other, any teams
    either has coached -- bounded to whatever seasons are in `schedules`,
    not literally a full career if a coach's tenure predates that window."""
    if not coach_a or not coach_b or coach_a == coach_b:
        return None
    games = _played_games(schedules)
    matchups = games[
        ((games["home_coach"] == coach_a) & (games["away_coach"] == coach_b))
        | ((games["home_coach"] == coach_b) & (games["away_coach"] == coach_a))
    ]
    if len(matchups) < MIN_COACH_MEETINGS:
        return None
    a_wins, b_wins, ties = _record(matchups, coach_a, "home_coach")
    return {
        "type": "coach_h2h", "coach_a": coach_a, "coach_b": coach_b,
        "a_wins": a_wins, "b_wins": b_wins, "ties": ties, "n": len(matchups),
    }


def team_last_n_meetings(team_a: str, team_b: str, schedules: pd.DataFrame, n: int = 5) -> dict | None:
    """team_a/team_b's record over their last `n` meetings, whoever was
    home or away each time."""
    games = _played_games(schedules)
    matchups = games[
        ((games["home_team"] == team_a) & (games["away_team"] == team_b))
        | ((games["home_team"] == team_b) & (games["away_team"] == team_a))
    ].sort_values(["season", "week"])
    if len(matchups) < MIN_TEAM_MEETINGS:
        return None
    recent = matchups.tail(n)
    a_wins, b_wins, ties = _record(recent, team_a, "home_team")
    last = recent.iloc[-1]
    return {
        "type": "team_h2h", "team_a": team_a, "team_b": team_b,
        "a_wins": a_wins, "b_wins": b_wins, "ties": ties, "n": len(recent),
        "last_season": int(last["season"]), "last_week": int(last["week"]),
    }
