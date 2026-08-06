# QB and lead-RB last-4-game trend vs season average, narrative only
"""Hot/cold streak detection for the game-preview narrative -- like
data/positional_matchups.py, this never touches the win-probability model,
just report/narrative.py's search for a genuinely convincing "X is playing
out of his mind right now" storyline.
"""

import pandas as pd

MIN_GAMES_FOR_TREND = 6
TREND_WINDOW = 4

# Calibrated empirically against a full season of real last-4-vs-season
# diffs (computed for every qualifying QB/RB), set at roughly the 85th
# percentile of each -- a real, unmistakable streak, not just normal
# week-to-week variance (game-to-game EPA has a standard deviation around
# 0.35 even for a perfectly steady player).
QB_TREND_THRESHOLD = 0.24
RB_TREND_THRESHOLD = 0.25


def qb_game_log(pbp: pd.DataFrame) -> pd.DataFrame:
    dropbacks = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna()]
    return (
        dropbacks.groupby(["season", "week", "posteam", "passer_player_id", "passer_player_name"])
        .agg(epa=("epa", "mean"), dropbacks=("epa", "size"))
        .reset_index()
    )


def rb_game_log(pbp: pd.DataFrame, pos_map: dict) -> pd.DataFrame:
    """Rushes by players rostered at RB -- excludes QB scrambles/kneels,
    which `play_type == "run"` alone would otherwise include."""
    rushes = pbp[(pbp["play_type"] == "run") & pbp["rusher_player_id"].notna()].copy()
    rushes["rusher_pos"] = rushes["rusher_player_id"].map(pos_map)
    rushes = rushes[rushes["rusher_pos"] == "RB"]
    return (
        rushes.groupby(["season", "week", "posteam", "rusher_player_id", "rusher_player_name"])
        .agg(epa=("epa", "mean"), yards=("yards_gained", "sum"), attempts=("epa", "size"))
        .reset_index()
    )


def _lead_back(game_log: pd.DataFrame, team: str) -> str | None:
    team_log = game_log[game_log["posteam"] == team]
    if team_log.empty:
        return None
    totals = team_log.groupby("rusher_player_id")["attempts"].sum()
    return totals.idxmax()


def _trend(player_log: pd.DataFrame, name_col: str) -> dict | None:
    """player_log: one player's game log for one window (current season or
    a fallback prior season), sorted season/week ascending."""
    if len(player_log) < MIN_GAMES_FOR_TREND:
        return None
    season_avg = player_log["epa"].mean()
    recent = player_log.tail(TREND_WINDOW)
    recent_avg = recent["epa"].mean()
    return {
        "player": player_log[name_col].iloc[-1],
        "season_avg": float(season_avg), "recent_avg": float(recent_avg),
        "diff": float(recent_avg - season_avg), "games": len(player_log),
    }


def qb_streak(team: str, qb_id: str, current_log: pd.DataFrame, fallback_log: pd.DataFrame) -> dict | None:
    """Tries the named starter's current-season log first; only drops to
    the fallback (prior season) log if the current season doesn't have
    enough games yet. Once a window has enough games, a trend that doesn't
    clear the threshold ends the search there -- a stale streak from last
    season shouldn't override what this season's data already shows."""
    if qb_id is None:
        return None
    for log in (current_log, fallback_log):
        player_log = log[(log["posteam"] == team) & (log["passer_player_id"] == qb_id)].sort_values(["season", "week"])
        trend = _trend(player_log, "passer_player_name")
        if trend is None:
            continue
        if abs(trend["diff"]) < QB_TREND_THRESHOLD:
            return None
        trend.update({"type": "qb_streak", "team": team, "direction": "hot" if trend["diff"] > 0 else "cold"})
        return trend
    return None


def rb_streak(team: str, current_log: pd.DataFrame, fallback_log: pd.DataFrame) -> dict | None:
    for log in (current_log, fallback_log):
        rb_id = _lead_back(log, team)
        if rb_id is None:
            continue
        player_log = log[(log["posteam"] == team) & (log["rusher_player_id"] == rb_id)].sort_values(["season", "week"])
        trend = _trend(player_log, "rusher_player_name")
        if trend is None:
            continue
        if abs(trend["diff"]) < RB_TREND_THRESHOLD:
            return None
        trend.update({"type": "rb_streak", "team": team, "direction": "hot" if trend["diff"] > 0 else "cold"})
        return trend
    return None
