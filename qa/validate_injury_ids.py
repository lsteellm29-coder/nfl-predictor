# flags current-week starters with no espn_id, who'd silently get 0 injury impact in props
"""Injury-ID coverage check (Audit Fix Plan Step 2). data/fetch_injuries.py's
fetch_current_player_injury_status() can only apply a real injury status to
a player whose roster row has an espn_id -- that's the only field linking
ESPN's athlete id space to nflverse's GSIS player id space
(_espn_id_to_gsis_id). A required-lineup starter with no espn_id is
invisible to that crosswalk: if ESPN lists them as questionable/out, model/
player_stats.py's apply_injury_usage() would never see it and would
silently score them at full healthy usage.

This never touches how injury impact is calculated -- it only reports which
current-week starters are exposed to that gap, so a real "Kupp is
questionable but his prop shows no discount" case doesn't go unnoticed.

Note this gap is specific to *player*-level injury status (props). The
*team*-level injury_impact feature that actually feeds the win-probability
model (fetch_current_injury_impact()) reads ESPN's own team/athlete
structure directly and never needs an espn_id crosswalk, so it isn't
affected by anything this script finds.
"""

import sys

import nfl_data_py as nfl
import pandas as pd

from config import CURRENT_SEASON
from data.fetch_week import fetch_week
from model.player_stats import (
    COVERAGE_MINIMUMS, _player_season_avg, qb_passing_game_log,
    rb_rushing_game_log, receiving_game_log, required_lineup,
    _fallback_season_pbp, _current_season_pbp,
)
from run_week import get_current_week


def _required_starters(week: int, season: int) -> list[dict]:
    """Every required-lineup starter (QA spec's own COVERAGE_MINIMUMS) for
    every team playing this week, with their roster row's espn_id."""
    games = fetch_week(week, season)
    rosters = nfl.import_seasonal_rosters([season, season - 1])
    rosters = rosters[rosters["player_id"] != ""]
    pos_map = dict(zip(rosters["player_id"], rosters["position"]))
    id_to_name = dict(zip(rosters["player_id"], rosters["player_name"]))
    id_to_espn_id = dict(zip(rosters["player_id"], rosters["espn_id"]))

    fallback_pbp = _fallback_season_pbp(season)
    current_pbp = _current_season_pbp(season, week, fallback_pbp)
    qb_current, qb_fallback = qb_passing_game_log(current_pbp), qb_passing_game_log(fallback_pbp)
    rb_current, rb_fallback = rb_rushing_game_log(current_pbp, pos_map), rb_rushing_game_log(fallback_pbp, pos_map)
    rec_current, rec_fallback = receiving_game_log(current_pbp, pos_map), receiving_game_log(fallback_pbp, pos_map)
    qb_avgs = (_player_season_avg(qb_current, "passer_player_id", "passer_player_name", ["pass_yards", "pass_tds", "attempts"]),
               _player_season_avg(qb_fallback, "passer_player_id", "passer_player_name", ["pass_yards", "pass_tds", "attempts"]))
    rb_avgs = (_player_season_avg(rb_current, "rusher_player_id", "rusher_player_name", ["rush_yards", "rush_tds", "attempts"]),
               _player_season_avg(rb_fallback, "rusher_player_id", "rusher_player_name", ["rush_yards", "rush_tds", "attempts"]))
    rec_avgs = (_player_season_avg(rec_current, "receiver_player_id", "receiver_player_name", ["receptions", "rec_yards", "rec_tds", "targets"]),
                _player_season_avg(rec_fallback, "receiver_player_id", "receiver_player_name", ["receptions", "rec_yards", "rec_tds", "targets"]))

    starters = []
    seen = set()
    for _, game in games.iterrows():
        for team in (game["home_team"], game["away_team"]):
            lineup = required_lineup(team, rosters, qb_avgs, rb_avgs, rec_avgs)
            for position in COVERAGE_MINIMUMS:
                for player_id in lineup.get(position, []):
                    if (team, player_id) in seen:
                        continue
                    seen.add((team, player_id))
                    starters.append({
                        "player_id": player_id, "player": id_to_name.get(player_id, player_id),
                        "team": team, "position": position, "espn_id": id_to_espn_id.get(player_id),
                    })
    return starters


def run(week: int | None = None, season: int = CURRENT_SEASON) -> bool:
    if week is None:
        week = get_current_week(season)
    starters = _required_starters(week, season)
    if not starters:
        print("Injury-ID validation: no required-lineup starters found this week.")
        return True

    missing = [s for s in starters if pd.isna(s["espn_id"]) or not s["espn_id"]]

    print(f"Injury-ID validation: {len(starters)} required-lineup starters checked, "
          f"{len(missing)} missing an espn_id (injury status for these players can't be "
          f"looked up -- they'll silently score at full healthy usage regardless of their "
          f"real report status).")
    for s in missing:
        print(f"  - {s['player']} ({s['team']}, {s['position']}) -- missing espn_id")
    return True  # informational only -- doesn't change how injury impact is calculated


def main():
    sys.exit(0 if run() else 1)


if __name__ == "__main__":
    main()
