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


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"Pulling schedules for {HISTORICAL_SEASONS[0]}-{HISTORICAL_SEASONS[-1]}...")
    schedules = fetch_schedules()
    schedules.to_parquet(SCHEDULES_PATH)
    print(f"  saved {len(schedules)} games -> {SCHEDULES_PATH}")

    print(f"Pulling play-by-play for {HISTORICAL_SEASONS[0]}-{HISTORICAL_SEASONS[-1]}...")
    pbp = fetch_pbp()
    pbp.to_parquet(PBP_PATH)
    print(f"  saved {len(pbp)} plays -> {PBP_PATH}")


if __name__ == "__main__":
    main()
