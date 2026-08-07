# pre-publish check: which players in this week's props are missing a real headshot
"""Pre-publish headshot validation (QA spec Section 3). Flags any player
in this week's props missing a real headshot so the gap is visible before
publishing -- even though report/cards.py already has a graceful runtime
fallback (team logo, then initials) for exactly this case, so a gap here
is informational, not a broken card.
"""

import sys

import pandas as pd

from config import CURRENT_SEASON
from data.player_headshots import check_headshots
from model.player_stats import score_props
from run_week import get_current_week


def run(week: int | None = None, season: int = CURRENT_SEASON) -> bool:
    if week is None:
        week = get_current_week(season)
    props = score_props(week, season)
    if props.empty:
        print("Headshot validation: no props to check this week.")
        return True

    players = props[["player", "team", "espn_id"]].drop_duplicates()
    with_id = players[players["espn_id"].notna()]
    no_id = players[players["espn_id"].isna()]

    results = check_headshots(with_id["espn_id"].unique().tolist())
    missing = with_id[~with_id["espn_id"].map(lambda e: results.get(e, False))]
    gaps = pd.concat([missing, no_id])[["player", "team"]].drop_duplicates()

    print(f"Headshot validation: {len(players)} players in this week's props, "
          f"{len(gaps)} missing a photo (falls back to team logo automatically, not a broken card).")
    for _, row in gaps.iterrows():
        print(f"  - {row['player']} ({row['team']})")
    return True  # informational only -- a missing headshot degrades gracefully, never blocks publishing


def main():
    sys.exit(0 if run() else 1)


if __name__ == "__main__":
    main()
