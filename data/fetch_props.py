# pulls anytime-TD prop lines + alternate game lines for the week
"""Player prop odds (anytime-TD only -- Anytime-TD-Only Props Refocus
spec removed yardage/reception props) and alternate game lines (alternate
spreads/totals, team totals) from The Odds API. Unlike the core game
lines in data/fetch_week.py (one bulk /odds call for the whole week), all
of these only live on the per-event odds endpoint, so this costs one API
call per game -- games (with an `event_id` column) come from
data/fetch_week.py's output, which already threads the event id through
from the same bulk call used for game lines, at no extra cost. Alternate
lines are requested in the same per-event call as player props (The Odds
API prices both at the event-scoped endpoint) so adding them costs
nothing beyond what props already needed.
"""

import pandas as pd
import requests

from config import ODDS_API_KEY
from data.odds_aggregation import aggregate_one_sided, ladder_by_point

EVENT_ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"

# Odds API market key -> our internal stat name. anytime-TD only --
# confirmed live against The Odds API that this market's "Yes" outcomes
# are rush/receiving TDs (the standard real-world "anytime touchdown
# scorer" definition), never passing TDs.
PROP_MARKETS = {
    "player_anytime_td": "anytime_td",
}

# Alternate game lines -- game-level, not player-level, so they're parsed
# separately (parse_alt_lines) from player props. team_totals currently
# returns zero books this far before kickoff (checked live -- every book
# queried has it absent, not just thin), but the request costs nothing
# extra to include, so it's left in: it'll start returning real data on
# its own once books post it, with no code change needed here.
ALT_LINE_MARKETS = ["alternate_spreads", "alternate_totals", "team_totals"]


def fetch_event_odds(event_id: str) -> dict:
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY not set (expected in .env)")
    resp = requests.get(
        EVENT_ODDS_URL.format(event_id=event_id),
        params={
            "apiKey": ODDS_API_KEY,
            # us_dfs (PrizePicks/Underdog/betr) is where yardage and
            # reception props actually get posted this far out -- us/us2
            # sportsbooks mostly only have anytime-TD priced yet.
            # Fetching all three regions gets every real line that's live
            # anywhere, instead of a single-region call silently missing
            # markets that exist, just not at a traditional sportsbook.
            "regions": "us,us2,us_dfs",
            "markets": ",".join(list(PROP_MARKETS) + ALT_LINE_MARKETS),
            "oddsFormat": "american",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def parse_event_props(raw: dict) -> pd.DataFrame:
    """One row per player: the consensus anytime-TD "Yes" probability
    averaged across every book quoting it (data/odds_aggregation.py's
    aggregate_one_sided -- there's no "No" leg to de-vig against, same
    limitation the single-book version always had), rather than picking a
    single book. Most players only have one book quoting them yet this
    far before kickoff (see fetch_event_odds); aggregation naturally
    reduces to that one book's number today and will average across more
    as books add coverage closer to kickoff, with no code change needed."""
    books = raw.get("bookmakers", [])
    if not books:
        return pd.DataFrame(columns=["player", "stat", "side", "line", "price", "prob", "n_books"])

    td_prices: dict[str, list[float]] = {}
    for book in books:
        for market in book.get("markets", []):
            if PROP_MARKETS.get(market["key"]) != "anytime_td":
                continue
            for o in market["outcomes"]:
                if o["name"] != "Yes":
                    continue
                td_prices.setdefault(o["description"], []).append(o["price"])

    rows = []
    for player, prices in td_prices.items():
        agg = aggregate_one_sided(prices)
        rows.append({"player": player, "stat": "anytime_td", "side": "yes",
                     "line": None, "price": agg["price"], "prob": agg["prob"], "n_books": agg["n_books"]})
    return pd.DataFrame(rows)


def parse_alt_lines(raw: dict, home_team: str, away_team: str) -> dict:
    """Alternate spreads/totals as full ladders (data/odds_aggregation.py's
    ladder_by_point -- one multi-book-aggregated rung per point value),
    plus team totals whenever a book actually has them priced. Game-level,
    not player-level, so this is kept separate from parse_event_props.
    Each rung's n_books tells a caller how many books actually agree that
    specific point exists -- most rungs are n_books=1 this far before
    kickoff (see ALT_LINE_MARKETS), which is real, not a bug: only a
    couple of books have posted a genuine ladder yet this far out."""
    home_name = raw.get("home_team")
    result = {"alternate_spreads": [], "alternate_totals": [], "team_totals": []}

    spread_quotes, total_quotes = [], []
    home_team_total_quotes, away_team_total_quotes = [], []
    for book in raw.get("bookmakers", []):
        for market in book.get("markets", []):
            outcomes = market["outcomes"]
            if market["key"] == "alternate_spreads":
                by_point: dict[float, dict] = {}
                for o in outcomes:
                    by_point.setdefault(o["point"] if o["name"] == home_name else -o["point"], {})[
                        "home" if o["name"] == home_name else "away"] = o
                for point, sides in by_point.items():
                    if "home" in sides and "away" in sides:
                        spread_quotes.append({"point": point, "price_a": sides["home"]["price"], "price_b": sides["away"]["price"]})
            elif market["key"] == "alternate_totals":
                by_point: dict[float, dict] = {}
                for o in outcomes:
                    by_point.setdefault(o["point"], {})[o["name"]] = o
                for point, sides in by_point.items():
                    if "Over" in sides and "Under" in sides:
                        total_quotes.append({"point": point, "price_a": sides["Over"]["price"], "price_b": sides["Under"]["price"]})
            elif market["key"] == "team_totals":
                by_team_point: dict[tuple[str, float], dict] = {}
                for o in outcomes:
                    team = o.get("description")
                    by_team_point.setdefault((team, o["point"]), {})[o["name"]] = o
                for (team, point), sides in by_team_point.items():
                    if "Over" not in sides or "Under" not in sides:
                        continue
                    quote = {"point": point, "price_a": sides["Over"]["price"], "price_b": sides["Under"]["price"]}
                    (home_team_total_quotes if team == home_name else away_team_total_quotes).append(quote)

    result["alternate_spreads"] = ladder_by_point(spread_quotes)
    result["alternate_totals"] = ladder_by_point(total_quotes)
    result["team_totals"] = {
        "home": ladder_by_point(home_team_total_quotes),
        "away": ladder_by_point(away_team_total_quotes),
    }
    return result


def fetch_props_for_week(games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """games: data/fetch_week.py's output (needs event_id, home_team,
    away_team). Returns (props_df, alt_lines_by_event) -- props_df is
    every parsed prop row across the week's games (empty frame for games
    whose event_id is missing or whose call errors, so a missing week of
    props shouldn't take down the whole report); alt_lines_by_event is
    keyed by event_id, each value parse_alt_lines' output. Alt-line data
    is fetched and available here but not yet wired into a UI display --
    see parse_alt_lines' docstring on why most rungs are thin (n_books=1)
    this far before kickoff; building a full ladder display around data
    that's mostly one book's number yet would be featuring something
    that isn't really live."""
    frames = []
    alt_lines_by_event = {}
    for _, game in games.iterrows():
        event_id = game.get("event_id")
        if pd.isna(event_id):
            continue
        try:
            raw = fetch_event_odds(event_id)
        except requests.RequestException:
            continue
        props = parse_event_props(raw)
        if not props.empty:
            props["home_team"] = game["home_team"]
            props["away_team"] = game["away_team"]
            frames.append(props)
        alt_lines_by_event[event_id] = parse_alt_lines(raw, game["home_team"], game["away_team"])
    if not frames:
        props_df = pd.DataFrame(columns=["player", "stat", "side", "line", "price", "prob", "n_books", "home_team", "away_team"])
    else:
        props_df = pd.concat(frames, ignore_index=True)
    return props_df, alt_lines_by_event
