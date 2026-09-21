# QA gate: live pre-game team stats must equal what the model was trained on
"""Re-derives, the way live scoring does (data/live_stats.py -- state cut to the
start of week W, prior season included), the pre-game rolling stats for a
fully-played cached season, and compares them to the training-time rows in
data/cache/team_stats.parquet for the same (season, week, team).

This is the check that would have caught model/predict.py's original
`week < W` off-by-one, which handed the model all-NaN stats for Week 2 and one-
game-stale stats after that. It exits 1 on any mismatch, so weekly_pipeline.sh
(set -e) stops before publishing picks scored off wrong inputs.

Weeks checked: 2 (the first week that uses current-season stats at all) and 9
(mid-season, includes bye-week teams that have no row for the target week).
"""

import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import CURRENT_SEASON
from data.live_stats import PBP_PATH, SCHEDULES_PATH, pregame_team_stats
from model.train import STAT_COLS

TOLERANCE = 1e-9
SPLIT_COLS = ["home_point_diff_avg", "away_point_diff_avg"]
CHECK_WEEKS = (2, 9)


def run(season: int = CURRENT_SEASON - 1, weeks=CHECK_WEEKS) -> list[str]:
    """Returns a list of human-readable failures (empty = all good)."""
    schedules = pd.read_parquet(SCHEDULES_PATH)
    cur_schedule = schedules[schedules["season"] == season]
    prior_schedule = schedules[schedules["season"] == season - 1]
    cur_pbp = pq.read_table(PBP_PATH, filters=[("season", "=", season)]).to_pandas()
    prior_pbp = pq.read_table(PBP_PATH, filters=[("season", "=", season - 1)]).to_pandas()
    trained = pd.read_parquet(str(SCHEDULES_PATH).replace("schedules.parquet", "team_stats.parquet"))
    trained = trained[trained["season"] == season]

    cols = STAT_COLS + SPLIT_COLS
    failures = []
    for week in weeks:
        live = pregame_team_stats(cur_schedule, cur_pbp, season, week,
                                  prior_schedule=prior_schedule, prior_pbp=prior_pbp).set_index("team")
        ref = trained[trained["week"] == week].set_index("team")
        teams = ref.index.intersection(live.index)
        if len(teams) < 20:
            failures.append(f"week {week}: only {len(teams)} teams comparable")
            continue
        nan_mismatch = (live.loc[teams, cols].isna() != ref.loc[teams, cols].isna()).to_numpy().sum()
        diff = (live.loc[teams, cols] - ref.loc[teams, cols]).abs().to_numpy(dtype=float)
        worst = float(np.nanmax(diff)) if diff.size else 0.0
        print(f"  {season} week {week}: {len(teams)} teams, max|live - trained| = {worst:.2e}, "
              f"NaN-pattern mismatches = {int(nan_mismatch)}")
        if worst > TOLERANCE or nan_mismatch:
            failures.append(f"week {week}: live stats differ from training rows "
                            f"(max diff {worst:.3g}, {int(nan_mismatch)} NaN mismatches)")
    return failures


def main():
    print("Validating live pre-game stats against training-time rows...")
    failures = run()
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("OK -- live pre-game stats reproduce the training rows exactly.")


if __name__ == "__main__":
    main()
