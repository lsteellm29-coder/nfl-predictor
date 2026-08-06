# pulls player prop lines (yards, receptions, anytime TD) for the week
"""Player prop odds from The Odds API. Unlike the game lines in
data/fetch_week.py (one bulk /odds call for the whole week), player props
only live on the per-event odds endpoint, so this costs one API call per
game -- games (with an `event_id` column) come from data/fetch_week.py's
output, which already threads the event id through from the same bulk
call used for game lines, at no extra cost.
"""

import pandas as pd
import requests

from config import ODDS_API_KEY

EVENT_ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"
PREFERRED_BOOK = "draftkings"

# Odds API market key -> our internal stat name.
PROP_MARKETS = {
    "player_pass_yds": "pass_yds",
    "player_rush_yds": "rush_yds",
    "player_receptions": "receptions",
    "player_reception_yds": "rec_yds",
    "player_anytime_td": "anytime_td",
}


def fetch_event_props(event_id: str) -> dict:
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY not set (expected in .env)")
    resp = requests.get(
        EVENT_ODDS_URL.format(event_id=event_id),
        params={
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": ",".join(PROP_MARKETS),
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


def american_to_prob(price: float) -> float:
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def _devig_pair(over_price: float, under_price: float) -> tuple[float, float]:
    """Removes the bookmaker's overround by normalizing both sides'
    implied probabilities to sum to 1 -- without this, comparing a model
    probability against a single raw vig-inflated line probability would
    understate every "edge" by the book's built-in margin."""
    over_p, under_p = american_to_prob(over_price), american_to_prob(under_price)
    total = over_p + under_p
    return over_p / total, under_p / total


def parse_event_props(raw: dict) -> pd.DataFrame:
    """One row per (player, stat, side) -- side is "over"/"under" for
    yardage/reception markets, "yes" for anytime_td (usually one-sided)."""
    book = _pick_bookmaker(raw.get("bookmakers", []))
    if not book:
        return pd.DataFrame(columns=["player", "stat", "side", "line", "price", "prob"])

    rows = []
    for market in book.get("markets", []):
        stat = PROP_MARKETS.get(market["key"])
        if stat is None:
            continue
        outcomes = market["outcomes"]
        if stat == "anytime_td":
            for o in outcomes:
                if o["name"] != "Yes":
                    continue
                rows.append({
                    "player": o["description"], "stat": stat, "side": "yes",
                    "line": None, "price": o["price"], "prob": american_to_prob(o["price"]),
                })
            continue

        by_player: dict[str, dict] = {}
        for o in outcomes:
            by_player.setdefault(o["description"], {})[o["name"]] = o
        for player, sides in by_player.items():
            over, under = sides.get("Over"), sides.get("Under")
            if not over or not under:
                continue
            over_prob, under_prob = _devig_pair(over["price"], under["price"])
            rows.append({"player": player, "stat": stat, "side": "over",
                         "line": over["point"], "price": over["price"], "prob": over_prob})
            rows.append({"player": player, "stat": stat, "side": "under",
                         "line": under["point"], "price": under["price"], "prob": under_prob})

    return pd.DataFrame(rows)


def fetch_props_for_week(games: pd.DataFrame) -> pd.DataFrame:
    """games: data/fetch_week.py's output (needs event_id, home_team,
    away_team). Returns every parsed prop row across the week's games, or
    an empty frame for games whose event_id is missing (no odds posted
    yet) or whose prop call errors -- a missing week of props shouldn't
    take down the whole report."""
    frames = []
    for _, game in games.iterrows():
        event_id = game.get("event_id")
        if pd.isna(event_id):
            continue
        try:
            raw = fetch_event_props(event_id)
        except requests.RequestException:
            continue
        props = parse_event_props(raw)
        if props.empty:
            continue
        props["home_team"] = game["home_team"]
        props["away_team"] = game["away_team"]
        frames.append(props)
    if not frames:
        return pd.DataFrame(columns=["player", "stat", "side", "line", "price", "prob", "home_team", "away_team"])
    return pd.concat(frames, ignore_index=True)
