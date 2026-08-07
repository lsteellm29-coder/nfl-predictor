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
from data.odds_aggregation import aggregate_two_sided

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"

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
            # us2 (BetRivers, Bally Bet, Hard Rock, etc.) roughly doubles
            # book coverage over us alone -- more books for _parse_event's
            # aggregation to average across, same fix as the multi-region
            # player-props call already makes.
            "regions": "us,us2",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_event(event: dict, name_to_abbr: dict) -> dict:
    """Aggregates every book in the response rather than picking a single
    one (see data/odds_aggregation.py) -- moneyline/spread/total each get
    the consensus line and a de-vigged probability averaged across
    whichever books agree on it, so one book's stale or outlier price
    can't single-handedly become "the market" for this game."""
    home_name, away_name = event["home_team"], event["away_team"]
    row = {
        "event_id": event.get("id"),
        "home_team": name_to_abbr.get(home_name, home_name),
        "away_team": name_to_abbr.get(away_name, away_name),
        "n_books": len(event.get("bookmakers", [])),
        "home_moneyline": None, "away_moneyline": None, "home_ml_prob": None,
        "spread_line": None, "spread_n_books": None,
        "total_line": None, "total_n_books": None,
    }

    h2h_quotes, spread_quotes, total_quotes = [], [], []
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            outcomes = market["outcomes"]
            if market["key"] == "h2h":
                home_o = next((o for o in outcomes if o["name"] == home_name), None)
                away_o = next((o for o in outcomes if o["name"] == away_name), None)
                if home_o and away_o:
                    h2h_quotes.append({"point": 0.0, "price_a": home_o["price"], "price_b": away_o["price"]})
            elif market["key"] == "spreads":
                home_o = next((o for o in outcomes if o["name"] == home_name), None)
                away_o = next((o for o in outcomes if o["name"] == away_name), None)
                if home_o and away_o:
                    # sportsbooks quote negative = favorite; flip so
                    # positive = home favored, matching team_stats.py
                    spread_quotes.append({"point": -home_o["point"], "price_a": home_o["price"], "price_b": away_o["price"]})
            elif market["key"] == "totals":
                over_o = next((o for o in outcomes if o["name"] == "Over"), None)
                under_o = next((o for o in outcomes if o["name"] == "Under"), None)
                if over_o and under_o:
                    total_quotes.append({"point": over_o["point"], "price_a": over_o["price"], "price_b": under_o["price"]})

    h2h_agg = aggregate_two_sided(h2h_quotes)
    if h2h_agg:
        row["home_moneyline"] = h2h_agg["price_a"]
        row["away_moneyline"] = h2h_agg["price_b"]
        row["home_ml_prob"] = h2h_agg["prob_a"]

    spread_agg = aggregate_two_sided(spread_quotes)
    if spread_agg:
        row["spread_line"] = spread_agg["point"]
        row["spread_n_books"] = spread_agg["n_books"]

    total_agg = aggregate_two_sided(total_quotes)
    if total_agg:
        row["total_line"] = total_agg["point"]
        row["total_n_books"] = total_agg["n_books"]

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
    cols = ["away_team", "home_team", "gameday", "spread_line", "spread_n_books",
            "home_moneyline", "away_moneyline", "total_line", "total_n_books"]
    print(games[cols].to_string(index=False))


if __name__ == "__main__":
    main()
