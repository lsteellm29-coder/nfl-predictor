# API keys, current season/week

import os

from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
BALLDONTLIE_API_KEY = os.environ.get("BALLDONTLIE_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")

CURRENT_SEASON = 2026

# Last 10 completed seasons -- the stable sample window from the project spec.
HISTORICAL_SEASONS = list(range(CURRENT_SEASON - 10, CURRENT_SEASON))

# nfl_data_py has no retries and a broken error handler; add retries to every download at once
# (see data/nfl_net.py). Every entry point imports config, so this runs before any fetch.
from data.nfl_net import install_retries  # noqa: E402

install_retries()
