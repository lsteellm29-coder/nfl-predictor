# independent second roster source, for QA cross-validation only
"""NFL roster data from balldontlie.io -- an independent second source
from nfl_data_py, used only for QA validation (qa/validate_rosters.py).
Never a primary data source for the model or props themselves: this
project's whole feature set is built on nfl_data_py/nflfastR's schema, and
duplicating that dependency here would be real added risk for no benefit.
The one thing a second, differently-sourced roster pull is genuinely good
for is telling a real team change apart from a single provider's bad data
-- two independent sources agreeing is a much stronger signal than either
one being merely self-consistent.
"""

import time

import requests

from config import BALLDONTLIE_API_KEY

BASE_URL = "https://api.balldontlie.io/nfl/v1"

# balldontlie uses different abbreviations than nflverse for these two
# franchises; every other team abbreviation matches.
ABBR_FIX = {"LAR": "LA", "WSH": "WAS"}

# The free tier allows 5 requests per short window and returns a
# Retry-After header when exceeded -- honoring it turns a full
# ~20-page pull into a slow-but-reliable minute or so, rather than
# erroring out partway through.
MAX_RETRIES = 5


def _get(url: str, headers: dict, params: dict) -> dict:
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 12))
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"balldontlie: still rate-limited after {MAX_RETRIES} retries")


def fetch_all_players(max_seconds: float = 180) -> dict:
    """player full name ("First Last") -> current team abbreviation
    (nflverse convention), for every player balldontlie has assigned to a
    team. Paginates through the full player list (~2000+ players, 100 per
    page) -- slow (rate-limit-bound) but this is a weekly QA step, not a
    hot path. `max_seconds` bounds the worst case: balldontlie's players
    endpoint isn't scoped to just current-season active players, so its
    true page count is unpredictable, and this feeds a QA gate that must
    never hang the weekly pipeline indefinitely waiting on a third-party
    API. Hitting the budget returns whatever's been collected so far
    rather than raising -- a partial cross-check is still useful; an
    unbounded stall is not."""
    if not BALLDONTLIE_API_KEY:
        raise RuntimeError("BALLDONTLIE_API_KEY not set (expected in .env)")
    headers = {"Authorization": BALLDONTLIE_API_KEY}
    out = {}
    cursor = None
    start = time.time()
    while True:
        if time.time() - start > max_seconds:
            break
        params = {"per_page": 100}
        if cursor is not None:
            params["cursor"] = cursor
        payload = _get(f"{BASE_URL}/players", headers, params)
        for p in payload["data"]:
            team = p.get("team")
            if not team:
                continue
            name = f"{p['first_name']} {p['last_name']}"
            abbr = team["abbreviation"]
            out[name] = ABBR_FIX.get(abbr, abbr)
        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            break
    return out
