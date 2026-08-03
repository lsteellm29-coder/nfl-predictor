# API keys, current season/week

import os

from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.environ.get("ODDS_API_KEY")

CURRENT_SEASON = 2026

# Last 10 completed seasons -- the stable sample window from the project spec.
HISTORICAL_SEASONS = list(range(CURRENT_SEASON - 10, CURRENT_SEASON))
