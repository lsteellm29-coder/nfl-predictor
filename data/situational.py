# short week, travel, division rivalry, bounce-back, lookahead spots
"""Situational "spot" features -- schedule quirks that betting markets
famously price in but a pure stats-and-Elo model has no way to see on its
own: short rest, a cross-country early kickoff, division rivalry, bouncing
back from a blowout loss, and a lookahead trap before a bigger game next
week.
"""

import pandas as pd

SHORT_REST_DAYS = 5  # a Thursday game (Sun -> Thu) is always <=5
BLOWOUT_MARGIN = 17

# Offset from Eastern time (0 = ET) -- used only to gauge how far a team is
# traveling and what an early kickoff feels like on their body clock, not
# meant as a precise legal timezone map (e.g. Arizona doesn't observe DST,
# but Mountain is a fine approximation for this purpose).
TEAM_TZ_OFFSET = {
    **dict.fromkeys(["ATL", "BAL", "BUF", "CAR", "CIN", "CLE", "DET", "IND", "JAX",
                      "MIA", "NE", "NYG", "NYJ", "PHI", "PIT", "TB", "WAS"], 0),
    **dict.fromkeys(["CHI", "DAL", "GB", "HOU", "KC", "MIN", "NO", "TEN"], -1),
    **dict.fromkeys(["DEN", "ARI"], -2),
    **dict.fromkeys(["LA", "LAC", "LV", "SF", "SEA"], -3),
}
CROSS_COUNTRY_TZ_DIFF = 2
EARLY_KICKOFF_LOCAL_HOUR = 13  # 1pm


def is_short_week(rest_days) -> bool:
    return pd.notna(rest_days) and rest_days <= SHORT_REST_DAYS


def away_travel_penalty(home_team: str, away_team: str, gametime) -> bool:
    """True if the away team is crossing >=2 timezones AND kickoff lands
    before 1pm on their body clock -- the classic "West Coast team flying
    east for an early game" spot."""
    home_tz, away_tz = TEAM_TZ_OFFSET.get(home_team), TEAM_TZ_OFFSET.get(away_team)
    if home_tz is None or away_tz is None or pd.isna(gametime):
        return False
    if (home_tz - away_tz) < CROSS_COUNTRY_TZ_DIFF:
        return False
    try:
        kickoff_et_hour = int(str(gametime).split(":")[0])
    except (ValueError, IndexError):
        return False
    away_local_hour = kickoff_et_hour + away_tz  # away_tz <= 0, shifts earlier
    return away_local_hour < EARLY_KICKOFF_LOCAL_HOUR


def _team_game_margins(schedules: pd.DataFrame) -> pd.DataFrame:
    """season, week, team, margin (that team's own point differential) for
    every completed REG game, one row per team per game."""
    reg = schedules[(schedules["game_type"] == "REG") & schedules["home_score"].notna()].copy()
    reg["result"] = reg["home_score"] - reg["away_score"]

    home = reg[["season", "week", "home_team", "result"]].rename(
        columns={"home_team": "team", "result": "margin"})
    away = reg[["season", "week", "away_team"]].rename(columns={"away_team": "team"})
    away["margin"] = -reg["result"].values

    return pd.concat([home, away], ignore_index=True)


def blowout_loss_flags(schedules: pd.DataFrame) -> pd.DataFrame:
    """season, week, team -> 1 if that team's most recent game *within the
    same season* was a loss by more than BLOWOUT_MARGIN points, else 0.
    Week 1 (no prior game that season) is always 0."""
    games = _team_game_margins(schedules).sort_values(["team", "season", "week"])
    games["prev_margin"] = games.groupby(["team", "season"])["margin"].shift(1)
    games["blowout_loss_last_game"] = (games["prev_margin"] < -BLOWOUT_MARGIN).astype(int)
    return games[["season", "week", "team", "blowout_loss_last_game"]]


def lookahead_flags(schedules: pd.DataFrame) -> pd.DataFrame:
    """season, week, team -> 1 if that team's game the *following* week is
    a divisional matchup -- a classic lookahead trap spot. Known in advance
    from the published schedule, not derived from any future result, so
    this isn't leakage."""
    reg = schedules[schedules["game_type"] == "REG"][
        ["season", "week", "home_team", "away_team", "div_game"]].copy()
    home = reg.rename(columns={"home_team": "team"})[["season", "week", "team", "div_game"]]
    away = reg.rename(columns={"away_team": "team"})[["season", "week", "team", "div_game"]]
    games = pd.concat([home, away], ignore_index=True).sort_values(["team", "season", "week"])
    games["lookahead_spot"] = games.groupby(["team", "season"])["div_game"].shift(-1).fillna(0).astype(int)
    return games[["season", "week", "team", "lookahead_spot"]]
