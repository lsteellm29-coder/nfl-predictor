# pulls current injury report per team, no API key needed
"""Injury impact scoring, from two different sources depending on use case:

- Historical (training): nfl_data_py's weekly injury reports, which go back
  to 2016 with an official Out/Doubtful/Questionable designation per player
  per team per week -- exactly aligned to the games in the training set.
- Live (prediction): ESPN's unofficial API
  (https://github.com/pseudo-r/Public-ESPN-API), since nfl_data_py's injury
  data is a periodic dump, not built for "what does the report look like
  right now, this week." There's no historical query on this endpoint --
  it only reflects the current moment, which is exactly what's needed to
  score an upcoming week's games.

Both sides compute the same underlying score: each injured player
contributes status_weight * position_weight, summed per team. Higher score
= more/worse injuries.
"""

import nfl_data_py as nfl
import pandas as pd
import requests

# Historical reports use the official designation. "Probable" was retired
# from the NFL's official report after 2016 and essentially never appears,
# but is kept here in case it shows up in older data.
STATUS_WEIGHTS = {
    "Out": 1.0,
    "Injured Reserve": 1.0,
    "Doubtful": 0.75,
    "Questionable": 0.35,
    "Probable": 0.1,
}

# Importance multiplier by position -- a starting QB out matters far more
# than a backup LS. Covers both nfl_data_py's and ESPN's position labels
# (they don't use identical abbreviations for the same positions).
POSITION_WEIGHTS = {
    "QB": 5.0,
    "WR": 2.0, "RB": 2.0, "TE": 2.0,
    "T": 1.5, "OT": 1.5, "G": 1.5, "C": 1.5,
    "DE": 1.5, "DT": 1.5, "LB": 1.5, "CB": 1.5, "S": 1.5,
    "FB": 0.5, "LS": 0.5,
    "K": 0.75, "PK": 0.75, "P": 0.75,
}
DEFAULT_POSITION_WEIGHT = 1.0

# ESPN's team abbreviations differ from nflverse's for a couple of teams.
ESPN_ABBR_FIX = {"WSH": "WAS", "LAR": "LA"}

ESPN_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"


def _impact_score(position, status) -> float:
    status_w = STATUS_WEIGHTS.get(status, 0.0)
    if status_w == 0.0:
        return 0.0
    pos_w = POSITION_WEIGHTS.get(str(position).strip(), DEFAULT_POSITION_WEIGHT)
    return status_w * pos_w


def historical_injury_impact(seasons: list[int]) -> pd.DataFrame:
    """season, week, team -> injury_impact, from nfl_data_py's weekly
    injury reports. One row per player per team per week; summed per team.
    """
    inj = nfl.import_injuries(seasons)
    inj = inj.copy()
    inj["impact"] = [
        _impact_score(pos, status)
        for pos, status in zip(inj["position"], inj["report_status"])
    ]
    return (
        inj.groupby(["season", "week", "team"])["impact"]
        .sum()
        .reset_index()
        .rename(columns={"impact": "injury_impact"})
    )


def _espn_team_id_to_abbr() -> dict:
    resp = requests.get(ESPN_TEAMS_URL, timeout=15)
    resp.raise_for_status()
    teams = resp.json()["sports"][0]["leagues"][0]["teams"]
    mapping = {}
    for t in teams:
        abbr = t["team"]["abbreviation"]
        mapping[t["team"]["id"]] = ESPN_ABBR_FIX.get(abbr, abbr)
    return mapping


def fetch_current_injury_impact() -> dict:
    """team -> injury_impact, live from ESPN. Teams with no listed injuries
    (or not present in the response) are simply absent -- callers should
    treat a missing key as 0 impact."""
    id_to_abbr = _espn_team_id_to_abbr()

    resp = requests.get(ESPN_INJURIES_URL, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    impact = {}
    for team_block in data.get("injuries", []):
        abbr = id_to_abbr.get(team_block.get("id"))
        if not abbr:
            continue
        total = 0.0
        for entry in team_block.get("injuries", []):
            position = ((entry.get("athlete") or {}).get("position") or {}).get("abbreviation")
            total += _impact_score(position, entry.get("status"))
        impact[abbr] = total
    return impact
