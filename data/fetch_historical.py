# pulls 10 seasons via nfl_data_py
"""Pulls the last 10 completed NFL seasons (schedules + play-by-play) via
nfl_data_py and caches them locally as parquet, so team_stats.py and train.py
aren't re-pulling from the network every run.

Week 1 Audit & Tuning Plan Phase 1.1: this is the one place team-code
normalization actually has to happen for it to matter. nfl_data_py's raw
data uses whichever code was in use in a given season (OAK through 2019,
LV from 2020 on, for the same Raiders franchise; SD/LAC for the Chargers
the same way across 2017) -- three real relocations sit inside this
10-season cache window. Left unnormalized, that silently broke
model/elo.py's Elo continuity (a plain dict keyed by the raw code string
resets a relocated franchise's entire rating history to the 1500
default), data/team_history.py's head-to-head lookups (undercounts real
meetings across a code change), and data/team_stats.py's
ats_win_pct_last5 (the one rolling stat that deliberately bridges season
boundaries, so it also bridges -- incorrectly -- a code change). See
data/team_codes.py's own docstring for the full writeup. Normalizing
here, at the cache-build step, means every downstream consumer of
schedules.parquet/pbp.parquet is correct automatically, with no separate
patch needed at each of those three call sites.
"""

import os
import time

import nfl_data_py as nfl

from config import HISTORICAL_SEASONS
from data.team_codes import normalize_team_codes

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SCHEDULES_PATH = os.path.join(CACHE_DIR, "schedules.parquet")
PBP_PATH = os.path.join(CACHE_DIR, "pbp.parquet")


def fetch_schedules(seasons=HISTORICAL_SEASONS):
    return normalize_team_codes(nfl.import_schedules(seasons))


def fetch_pbp(seasons=HISTORICAL_SEASONS):
    return normalize_team_codes(nfl.import_pbp_data(seasons, downcast=True))


FETCH_ATTEMPTS = 4
RETRY_DELAY_SECONDS = 10


def _with_retries(fetch, what: str, attempts: int = FETCH_ATTEMPTS, delay: float = RETRY_DELAY_SECONDS,
                  sleep=time.sleep):
    """Calls `fetch()`, retrying with a growing delay. Downloading ten seasons of pbp is
    many separate requests, and one dropped connection used to kill the whole weekly run --
    nfl_data_py's own handler for it is broken (`except Error` is an undefined name), so it
    surfaced as an unrelated NameError."""
    for attempt in range(1, attempts + 1):
        try:
            return fetch()
        except Exception as e:
            if attempt == attempts:
                raise
            wait = delay * attempt
            print(f"  {what}: attempt {attempt}/{attempts} failed ({type(e).__name__}); retrying in {wait:.0f}s")
            sleep(wait)


def _refresh(what: str, fetch, path: str, **retry_kwargs):
    """Fetch and cache one table. Completed seasons don't change, so if the refresh fails
    after its retries and a cache already exists, keep that cache (nothing partial is ever
    written) and carry on instead of failing the run; with no cache to fall back on it still
    raises. Returns the fetched frame, or None if the existing cache was kept."""
    try:
        df = _with_retries(fetch, what, **retry_kwargs)
    except Exception as e:
        if os.path.exists(path):
            print(f"WARNING: couldn't refresh {what} ({type(e).__name__}); keeping the existing cache at {path}. "
                  f"Completed seasons don't change, so this is safe.")
            return None
        raise
    df.to_parquet(path)
    print(f"  saved {len(df)} rows -> {path}")
    return df


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"Pulling schedules for {HISTORICAL_SEASONS[0]}-{HISTORICAL_SEASONS[-1]}...")
    _refresh("schedules", fetch_schedules, SCHEDULES_PATH)

    print(f"Pulling play-by-play for {HISTORICAL_SEASONS[0]}-{HISTORICAL_SEASONS[-1]}...")
    _refresh("play-by-play", fetch_pbp, PBP_PATH)


if __name__ == "__main__":
    main()
