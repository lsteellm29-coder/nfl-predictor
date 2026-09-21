# Real-data regression: the pre-game stats live scoring feeds the model must reproduce, exactly,
# the rows the model was trained on (data/cache/team_stats.parquet) -- for a fully played cached
# season, built the live way (state cut to the start of the week, prior season included).
# This is what would have caught model/predict.py's original `week < W` off-by-one. Reads only
# local cache files; skipped where the cache isn't present (e.g. a fresh clone).
import os

import pytest

from data.live_stats import PBP_PATH, SCHEDULES_PATH

TEAM_STATS_PATH = os.path.join(os.path.dirname(SCHEDULES_PATH), "team_stats.parquet")

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (PBP_PATH, SCHEDULES_PATH, TEAM_STATS_PATH)),
    reason="historical data cache not present")


def test_live_pregame_stats_reproduce_the_training_rows():
    from qa.validate_live_stats import run
    assert run() == []
