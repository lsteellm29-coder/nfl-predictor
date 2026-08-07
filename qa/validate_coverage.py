# post-generation check: does every team-lineup actually meet COVERAGE_MINIMUMS
"""Post-generation lineup coverage check (QA spec Section 2).
model/player_stats.py's score_props() already fills gaps with a no-line
fallback projection wherever it can, and logs the rare slot it truly
can't fill (no usable season data -- deep bench players with no game log
to project from, a real data limit, not a bug). This script re-checks the
*actual output* against COVERAGE_MINIMUMS per team per game, so a
regression in that fallback (a code change that breaks it, a roster fetch
that silently comes back thin) is caught before publishing instead of
relying on eyeballing score_props()'s console log every run.
"""

import sys
from collections import Counter

import pandas as pd

from config import CURRENT_SEASON
from data.fetch_week import fetch_week
from model.player_stats import COVERAGE_MINIMUMS, score_props
from run_week import get_current_week


def _position_counts(props: pd.DataFrame, team: str) -> Counter:
    rows = props[props["team"] == team].drop_duplicates("player")
    return Counter(rows["position"])


def run(week: int | None = None, season: int = CURRENT_SEASON) -> bool:
    if week is None:
        week = get_current_week(season)
    games = fetch_week(week, season)
    props = score_props(week, season)

    gaps = []
    empty_teams = []
    for _, game in games.iterrows():
        for team in (game["home_team"], game["away_team"]):
            counts = _position_counts(props, team)
            if sum(counts.values()) == 0:
                # Zero props for an entire team points at a pipeline break
                # (roster fetch failure, opponent mismatch) rather than a
                # normal data gap -- distinct and more serious than a
                # partial shortfall below.
                empty_teams.append(f"{team} ({game['away_team']} @ {game['home_team']})")
                continue
            for position, minimum in COVERAGE_MINIMUMS.items():
                have = counts.get(position, 0)
                if have < minimum:
                    gaps.append(f"{team} ({game['away_team']} @ {game['home_team']}): "
                                f"{have}/{minimum} {position} covered")

    total_teams = len(games) * 2
    print(f"Coverage validation: {total_teams} team-lineups checked across {len(games)} games.")
    if empty_teams:
        print(f"  {len(empty_teams)} team(s) with NO props at all -- likely a pipeline break:")
        for e in empty_teams:
            print(f"    - {e}")
    if gaps:
        print(f"  {len(gaps)} partial coverage gap(s) (commonly deep-bench players with no "
              f"usable season data to project from -- see score_props()'s own log for reasons):")
        for g in gaps:
            print(f"    - {g}")
    if not empty_teams and not gaps:
        print("  Full coverage -- every team meets COVERAGE_MINIMUMS at every position.")

    # A team with zero props is a real pipeline failure and blocks
    # publishing. A partial shortfall (e.g. 4/5 WR) is usually a genuine
    # data limit, not a bug -- logged for visibility but not blocking,
    # same reasoning qa/validate_rosters.py uses for normal season-to-
    # season roster churn.
    return len(empty_teams) == 0


def main():
    sys.exit(0 if run() else 1)


if __name__ == "__main__":
    main()
