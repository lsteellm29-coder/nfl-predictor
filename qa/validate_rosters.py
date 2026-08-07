# pre-props-generation roster staleness/diff check, hard-fails on problems
"""Pre-props-generation roster validation (QA spec Section 1). Hard-fails
(exits non-zero) rather than letting the week publish if this week's
roster pull looks stale, empty, or partial -- a props run built on stale
rosters would show players as still on a team they've been traded off,
cut from, or retired from.
"""

import os
import sys

import nfl_data_py as nfl
import pandas as pd

from config import CURRENT_SEASON
from data.fetch_balldontlie import fetch_all_players

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
SNAPSHOT_PATH = os.path.join(CACHE_DIR, "roster_snapshot.parquet")

# An active NFL roster is 53 (plus practice squad) -- well under this is a
# broken/partial API pull, not a real team thinned by injury or cuts.
MIN_ROSTER_SIZE = 40


def fetch_current_roster(season: int = CURRENT_SEASON) -> pd.DataFrame:
    rosters = nfl.import_seasonal_rosters([season])
    return rosters[rosters["player_id"] != ""][["player_id", "player_name", "team", "position"]].copy()


def check_staleness(roster: pd.DataFrame) -> list[str]:
    problems = []
    if roster.empty:
        problems.append("Roster pull returned zero players -- API likely broken or season not recognized.")
        return problems
    counts = roster.groupby("team").size()
    for team, n in counts[counts < MIN_ROSTER_SIZE].items():
        problems.append(f"{team}: only {n} players on roster (expected 40+) -- looks like a partial/stale pull.")
    return problems


def diff_against_last_snapshot(roster: pd.DataFrame) -> list[str]:
    """Compares this pull against the last cached snapshot -- flags any
    player whose team changed (trade/waiver claim) or who dropped off
    every roster (release/retirement), so a prop card built on stale
    cached team data doesn't silently persist. Informational, not a
    hard-fail on its own -- roster churn is normal, especially in-season."""
    if not os.path.exists(SNAPSHOT_PATH):
        return []
    previous = pd.read_parquet(SNAPSHOT_PATH)
    prev_team = dict(zip(previous["player_id"], previous["team"]))

    changes = []
    for _, row in roster.iterrows():
        old_team = prev_team.get(row["player_id"])
        if old_team and old_team != row["team"]:
            changes.append(f"{row['player_name']}: {old_team} -> {row['team']}")

    dropped_ids = set(previous["player_id"]) - set(roster["player_id"])
    if dropped_ids:
        dropped_names = previous[previous["player_id"].isin(dropped_ids)]["player_name"].tolist()
        shown = ", ".join(dropped_names[:10]) + ("..." if len(dropped_names) > 10 else "")
        changes.append(f"{len(dropped_names)} player(s) no longer on any roster (released/retired): {shown}")
    return changes


def year_over_year_changes(season: int = CURRENT_SEASON) -> list[str]:
    """Compares this season's team assignment against last season's, for
    players present in both -- catches things a week-over-week diff can't
    (there's no prior week to compare against on the very first run of a
    season). Informational only, not a hard-fail: real season-over-season
    NFL roster turnover (free agency, trades, cuts) commonly runs in the
    15-25% range on its own, so a high count here isn't reliably
    distinguishable from normal offseason movement without an independent
    source this codebase doesn't have -- surfaced for human review, not
    auto-judged."""
    rosters = nfl.import_seasonal_rosters([season, season - 1])
    rosters = rosters[rosters["player_id"] != ""]
    cur = rosters[rosters["season"] == season][["player_id", "player_name", "team"]].drop_duplicates("player_id")
    prev = rosters[rosters["season"] == season - 1][["player_id", "team"]].drop_duplicates("player_id")
    merged = cur.merge(prev, on="player_id", suffixes=("_cur", "_prev"))
    changed = merged[merged["team_cur"] != merged["team_prev"]]
    if changed.empty:
        return []
    rate = len(changed) / len(merged)
    lines = [f"{len(changed)}/{len(merged)} players ({rate:.0%}) show a different team than last season "
             f"-- normal offseason churn is commonly 15-25%, so this alone isn't necessarily a data problem."]
    return lines


def cross_check_with_balldontlie(roster: pd.DataFrame) -> list[str]:
    """Cross-references nfl_data_py's roster against balldontlie's
    independent NFL API (data/fetch_balldontlie.py) -- a genuine second
    source, unlike year_over_year_changes()'s same-source comparison.
    Two sources actually disagreeing about a player's team is a much
    stronger signal of a real data problem than either source's own
    season-over-season churn, which normal offseason movement also
    produces. Best-effort and informational only, never a hard-fail:
    balldontlie's free tier is aggressively rate-limited, so
    fetch_all_players() is time-boxed and may only cover part of the
    league on a given run -- a missing balldontlie entry means "not
    compared," not "wrong," and an API outage here shouldn't be able to
    block publishing over a third-party dependency this project doesn't
    otherwise rely on."""
    try:
        bdl_teams = fetch_all_players()
    except Exception as e:
        return [f"balldontlie cross-check skipped ({type(e).__name__}: {e})"]
    if not bdl_teams:
        return ["balldontlie cross-check: no data returned (rate-limited before any page landed?) -- skipped"]

    mismatches = []
    checked = 0
    for _, row in roster.iterrows():
        bdl_team = bdl_teams.get(row["player_name"])
        if bdl_team is None:
            continue
        checked += 1
        if bdl_team != row["team"]:
            mismatches.append(f"{row['player_name']}: nfl_data_py has them at {row['team']}, balldontlie has them at {bdl_team}")

    if checked == 0:
        return ["balldontlie cross-check: fetched player data but matched none by name -- skipped"]
    lines = [f"balldontlie cross-check: {checked} players compared, {len(mismatches)} disagreement(s)."]
    lines.extend(f"  - {m}" for m in mismatches[:20])
    if len(mismatches) > 20:
        lines.append(f"  ...and {len(mismatches) - 20} more")
    return lines


def cross_check_with_ourlads(roster: pd.DataFrame) -> list[str]:
    """Cross-references nfl_data_py's roster against ourlads.com's own
    independently-maintained depth charts (data/fetch_firecrawl_sources.py,
    MCP Integration spec) -- another genuine second source, same spirit
    as cross_check_with_balldontlie: two independently-maintained sources
    disagreeing about who's on a team is a much stronger signal than
    either source's own season-over-season churn. Flags a depth-chart
    player nfl_data_py's roster doesn't have anywhere on that team at
    all -- name-format differences between the two sources (nicknames,
    suffixes) mean some false positives are expected here, same
    tolerance the balldontlie check already accepts; this is informational,
    never a hard-fail, and a Firecrawl outage/rate-limit/missing API key
    degrades to "skipped" rather than blocking the run."""
    try:
        from data.fetch_firecrawl_sources import TEAM_NEWS_URLS, cross_check_depth_chart
    except Exception as e:
        return [f"ourlads.com cross-check skipped ({type(e).__name__}: {e})"]

    names_by_team: dict[str, set] = {}
    for team, names in roster.groupby("team")["player_name"]:
        names_by_team[team] = set(names)

    all_flags = []
    checked_teams = 0
    for team in TEAM_NEWS_URLS:
        if team not in names_by_team:
            continue
        try:
            flags = cross_check_depth_chart(team, names_by_team[team])
        except Exception as e:
            all_flags.append(f"  - {team}: cross-check failed ({type(e).__name__}: {e})")
            continue
        checked_teams += 1
        all_flags.extend(f"  - {f}" for f in flags)

    if checked_teams == 0:
        return ["ourlads.com cross-check: no teams could be checked -- skipped"]
    lines = [f"ourlads.com depth-chart cross-check: {checked_teams} team(s) checked, "
             f"{len(all_flags)} discrepancy flag(s)."]
    lines.extend(all_flags[:20])
    if len(all_flags) > 20:
        lines.append(f"  ...and {len(all_flags) - 20} more")
    return lines


def run(season: int = CURRENT_SEASON, check_ourlads: bool = True) -> bool:
    """Returns True if safe to proceed. Always overwrites the snapshot
    with this pull (even on failure) so next week's diff compares against
    the most recent real data, not a stale baseline."""
    roster = fetch_current_roster(season)
    problems = check_staleness(roster)
    changes = diff_against_last_snapshot(roster)
    yoy = year_over_year_changes(season)

    print(f"Roster validation: {len(roster)} players across {roster['team'].nunique() if not roster.empty else 0} teams.")
    if yoy:
        print("Season-over-season team-assignment check:")
        for line in yoy:
            print(f"  - {line}")
    if changes:
        print(f"{len(changes)} roster change(s) since last pull:")
        for c in changes[:30]:
            print(f"  - {c}")
    if not roster.empty:
        for line in cross_check_with_balldontlie(roster):
            print(line)
        # 32 sequential Firecrawl calls, rate-limited to 11/min on the
        # free tier -- real but bounded time cost (a few minutes) for a
        # weekly job, so on by default; check_ourlads=False is there for
        # a quick local test run that shouldn't wait on it.
        if check_ourlads:
            for line in cross_check_with_ourlads(roster):
                print(line)

    os.makedirs(CACHE_DIR, exist_ok=True)
    if not roster.empty:
        roster.to_parquet(SNAPSHOT_PATH)

    if problems:
        print("ROSTER VALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        return False

    print("Roster validation passed.")
    return True


def main():
    sys.exit(0 if run() else 1)


if __name__ == "__main__":
    main()
