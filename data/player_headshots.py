# caches whether a player's ESPN headshot actually resolves
"""Player headshot URL caching (QA spec Section 3). The URL itself is
cheap to compute -- same predictable ESPN CDN pattern report/cards.py's
espn_headshot_url() uses for live display -- what's actually worth
caching is *whether a photo resolves there*, since that needs a real
network check. Refreshed on the same cadence as roster data (whenever
this runs as part of a weekly props generation).
"""

import os

import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "headshot_cache.parquet")

ESPN_HEADSHOT_URL = "https://a.espncdn.com/i/headshots/nfl/players/full/{espn_id}.png"


def headshot_url(espn_id) -> str | None:
    if not espn_id or pd.isna(espn_id):
        return None
    return ESPN_HEADSHOT_URL.format(espn_id=espn_id)


def _resolves(url: str) -> bool:
    try:
        resp = requests.head(url, timeout=5, allow_redirects=True)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        df = pd.read_parquet(CACHE_PATH)
        return dict(zip(df["espn_id"], df["has_photo"]))
    return {}


def save_cache(cache: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    pd.DataFrame({"espn_id": list(cache.keys()), "has_photo": list(cache.values())}).to_parquet(CACHE_PATH)


def check_headshots(espn_ids: list) -> dict:
    """espn_id -> bool (True if a real photo resolves at ESPN's CDN).
    Checks the local cache first, so re-running this weekly only makes a
    network request for players not already verified in a prior run."""
    cache = load_cache()
    ids = {e for e in espn_ids if e and pd.notna(e)}
    to_check = ids - set(cache.keys())
    for espn_id in to_check:
        cache[espn_id] = _resolves(headshot_url(espn_id))
    save_cache(cache)
    return {e: cache.get(e, False) for e in ids}
