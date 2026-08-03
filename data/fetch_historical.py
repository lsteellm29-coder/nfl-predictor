# pulls 10 seasons via nfl_data_py
"""Pulls the last 10 completed NFL seasons (schedules + play-by-play) via
nfl_data_py and caches them locally as parquet, so team_stats.py and train.py
aren't re-pulling from the network every run.
"""

import os

import nfl_data_py as nfl

from config import HISTORICAL_SEASONS

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SCHEDULES_PATH = os.path.join(CACHE_DIR, "schedules.parquet")
PBP_PATH = os.path.join(CACHE_DIR, "pbp.parquet")


def fetch_schedules(seasons=HISTORICAL_SEASONS):
    return nfl.import_schedules(seasons)


def fetch_pbp(seasons=HISTORICAL_SEASONS):
    return nfl.import_pbp_data(seasons, downcast=True)


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
