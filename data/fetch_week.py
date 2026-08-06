# pulls this week's schedule + lines
"""Pulls one week's NFL matchups (nfl_data_py) and the current spread/
moneyline/total for each game (The Odds API), merged into a single per-game
table for run_week.py to score.

spread_line follows the same convention as data/team_stats.py: the home
team's market-implied margin of victory (positive = home favored).
"""

import nfl_data_py as nfl
import pandas as pd
import requests

from config import CURRENT_SEASON, ODDS_API_KEY

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
PREFERRED_BOOK = "draftkings"

SCHEDULE_COLS = [
    "game_id", "season", "week", "gameday", "gametime", "weekday",
    "away_team", "home_team", "home_rest", "away_rest", "div_game",
    "roof", "location", "away_qb_id", "away_qb_name", "home_qb_id", "home_qb_name",
    "away_coach", "home_coach",
]


def fetch_schedule(season: int, week: int) -> pd.DataFrame:
    sched = nfl.import_schedules([season])
    return sched.loc[sched["week"] == week, SCHEDULE_COLS].copy()


def _team_name_to_abbr(season: int) -> dict:
    """nfl_data_py's team_desc has legacy abbreviations too (OAK/SD/STL/LAR);
    restrict to abbreviations actually used in this season's schedule so each
    full team name maps to exactly one current abbreviation."""
    sched = nfl.import_schedules([season])
    abbrs = set(sched["home_team"]) | set(sched["away_team"])
    team_desc = nfl.import_team_desc()
    current = team_desc[team_desc["team_abbr"].isin(abbrs)]
    return dict(zip(current["team_name"], current["team_abbr"]))


def fetch_odds_events() -> list[dict]:
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY not set (expected in .env)")
    resp = requests.get(
        ODDS_API_URL,
        params={
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _pick_bookmaker(bookmakers: list[dict]) -> dict | None:
    if not bookmakers:
        return None
    for book in bookmakers:
        if book["key"] == PREFERRED_BOOK:
            return book
    return bookmakers[0]


def _parse_event(event: dict, name_to_abbr: dict) -> dict:
    home_name, away_name = event["home_team"], event["away_team"]
    row = {
        "event_id": event.get("id"),
        "home_team": name_to_abbr.get(home_name, home_name),
        "away_team": name_to_abbr.get(away_name, away_name),
        "book": None,
        "home_moneyline": None,
        "away_moneyline": None,
        "spread_line": None,
        "total_line": None,
    }

    book = _pick_bookmaker(event.get("bookmakers", []))
    if not book:
        return row
    row["book"] = book["key"]

    for market in book["markets"]:
        outcomes = market["outcomes"]
        if market["key"] == "h2h":
            for o in outcomes:
                if o["name"] == home_name:
                    row["home_moneyline"] = o["price"]
                elif o["name"] == away_name:
                    row["away_moneyline"] = o["price"]
        elif market["key"] == "spreads":
            for o in outcomes:
                if o["name"] == home_name:
                    # sportsbooks quote negative = favorite; flip so
                    # positive = home favored, matching team_stats.py
                    row["spread_line"] = -o["point"]
        elif market["key"] == "totals":
            row["total_line"] = outcomes[0]["point"]

    return row


def fetch_odds(season: int) -> pd.DataFrame:
    name_to_abbr = _team_name_to_abbr(season)
    events = fetch_odds_events()
    rows = [_parse_event(e, name_to_abbr) for e in events]
    return pd.DataFrame(rows)


def fetch_week(week: int, season: int = CURRENT_SEASON) -> pd.DataFrame:
    schedule = fetch_schedule(season, week)
    odds = fetch_odds(season)
    return schedule.merge(odds, on=["home_team", "away_team"], how="left")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    args = parser.parse_args()

    games = fetch_week(args.week, args.season)
    cols = ["away_team", "home_team", "gameday", "spread_line",
            "home_moneyline", "away_moneyline", "total_line", "book"]
    print(games[cols].to_string(index=False))


if __name__ == "__main__":
    main()
