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
from data.rosters import TEAM_CODE_ALIASES, fetch_rosters

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
SNAPSHOT_PATH = os.path.join(CACHE_DIR, "roster_snapshot.parquet")
BLOCKED_PLAYERS_PATH = os.path.join(CACHE_DIR, "blocked_players.json")

# An active NFL roster is 53 (plus practice squad) -- well under this is a
# broken/partial API pull, not a real team thinned by injury or cuts.
MIN_ROSTER_SIZE = 40


def fetch_current_roster(season: int = CURRENT_SEASON) -> pd.DataFrame:
    # data/rosters.py's fetch_rosters() (not a raw nfl.import_seasonal_rosters
    # call) -- its team-code canonicalization (TEAM_CODE_ALIASES) matters
    # here specifically: without it, this sweep's own balldontlie/ourlads
    # cross-checks below would flag ~90 Arizona players as "mismatched"
    # every single run purely because nfl_data_py's current-season pull
    # uses "AZ" while balldontlie/ourlads both use "ARI" -- a false
    # positive on this exact scale, not a real discrepancy to review.
    rosters = fetch_rosters([season])
    return rosters[["player_id", "player_name", "team", "position"]].copy()


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
    rosters = rosters[rosters["player_id"] != ""].copy()
    # data/rosters.py's TEAM_CODE_ALIASES -- this function deliberately
    # keeps BOTH seasons' rows (unlike fetch_rosters(), which collapses
    # to one row per player_id), so it can't just call fetch_rosters()
    # here. Without this, a source-side team-code rename between the two
    # seasons (nfl_data_py's current-season pull using "AZ" for Arizona
    # while last season's still says "ARI") would show every returning
    # player on that team as a "team change," burying any real trades
    # in noise.
    rosters["team"] = rosters["team"].replace(TEAM_CODE_ALIASES)
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


def cross_check_with_balldontlie(roster: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """Cross-references nfl_data_py's roster against balldontlie's
    independent NFL API (data/fetch_balldontlie.py) -- a genuine second
    source, unlike year_over_year_changes()'s same-source comparison.
    Two sources actually disagreeing about a player's team is a much
    stronger signal of a real data problem than either source's own
    season-over-season churn, which normal offseason movement also
    produces. Best-effort, never a hard-fail on its own: balldontlie's
    free tier is aggressively rate-limited, so fetch_all_players() is
    time-boxed and may only cover part of the league on a given run --
    a missing balldontlie entry means "not compared," not "wrong," and
    an API outage here shouldn't be able to block publishing over a
    third-party dependency this project doesn't otherwise rely on.

    Returns (print_lines, mismatches) -- `mismatches` is structured
    ({"player", "nfl_team", "other_team"} dicts) for
    compute_blocked_players() to cross-reference against the ourlads
    check (Combined Build Plan Part 1 step 3: a single source's
    disagreement stays informational; two independent sources agreeing
    on the SAME alternate team is what actually blocks a player)."""
    try:
        bdl_teams = fetch_all_players()
    except Exception as e:
        return [f"balldontlie cross-check skipped ({type(e).__name__}: {e})"], []
    if not bdl_teams:
        return ["balldontlie cross-check: no data returned (rate-limited before any page landed?) -- skipped"], []

    mismatches = []
    checked = 0
    for _, row in roster.iterrows():
        bdl_team = bdl_teams.get(row["player_name"])
        if bdl_team is None:
            continue
        checked += 1
        if bdl_team != row["team"]:
            mismatches.append({"player": row["player_name"], "nfl_team": row["team"], "other_team": bdl_team})

    if checked == 0:
        return ["balldontlie cross-check: fetched player data but matched none by name -- skipped"], []
    lines = [f"balldontlie cross-check: {checked} players compared, {len(mismatches)} disagreement(s)."]
    lines.extend(f"  - {m['player']}: nfl_data_py has them at {m['nfl_team']}, balldontlie has them at {m['other_team']}"
                 for m in mismatches[:20])
    if len(mismatches) > 20:
        lines.append(f"  ...and {len(mismatches) - 20} more")
    return lines, mismatches


def cross_check_with_ourlads(roster: pd.DataFrame) -> tuple[list[str], set[tuple[str, str]]]:
    """Cross-references nfl_data_py's roster against ourlads.com's own
    independently-maintained depth charts (data/fetch_firecrawl_sources.py,
    MCP Integration spec) -- another genuine second source, same spirit
    as cross_check_with_balldontlie: two independently-maintained sources
    disagreeing about who's on a team is a much stronger signal than
    either source's own season-over-season churn. Flags a depth-chart
    player nfl_data_py's roster doesn't have anywhere on that team at
    all -- name-format differences between the two sources (nicknames,
    suffixes) mean some false positives are expected here, same
    tolerance the balldontlie check already accepts; never a hard-fail
    on its own, and a Firecrawl outage/rate-limit/missing API key
    degrades to "skipped" rather than blocking the run.

    Returns (print_lines, phantom_pairs) -- `phantom_pairs` is a set of
    (team, normalized_player_name) ourlads has on a team's depth chart
    but nfl_data_py's roster doesn't have there at all, for
    compute_blocked_players() to cross-reference."""
    try:
        from data.fetch_firecrawl_sources import TEAM_NEWS_URLS, _normalize_name, cross_check_depth_chart
    except Exception as e:
        return [f"ourlads.com cross-check skipped ({type(e).__name__}: {e})"], set()

    names_by_team: dict[str, set] = {}
    for team, names in roster.groupby("team")["player_name"]:
        names_by_team[team] = set(names)

    all_flags = []
    phantom_pairs: set[tuple[str, str]] = set()
    checked_teams = 0
    for team in TEAM_NEWS_URLS:
        if team not in names_by_team:
            continue
        try:
            flags = cross_check_depth_chart(team, names_by_team[team])
        except Exception as e:
            all_flags.append(f"{team}: cross-check failed ({type(e).__name__}: {e})")
            continue
        checked_teams += 1
        for player, position, flagged_team in flags:
            all_flags.append(f"{player} ({position}) listed on {flagged_team}'s ourlads.com depth chart, "
                              f"not found on {flagged_team}'s nfl_data_py roster")
            phantom_pairs.add((flagged_team, _normalize_name(player)))

    if checked_teams == 0:
        return ["ourlads.com cross-check: no teams could be checked -- skipped"], set()
    lines = [f"ourlads.com depth-chart cross-check: {checked_teams} team(s) checked, "
             f"{len(all_flags)} discrepancy flag(s)."]
    lines.extend(f"  - {f}" for f in all_flags[:20])
    if len(all_flags) > 20:
        lines.append(f"  ...and {len(all_flags) - 20} more")
    return lines, phantom_pairs


def compute_blocked_players(bdl_mismatches: list[dict], ourlads_phantoms: set[tuple[str, str]]) -> list[dict]:
    """Combined Build Plan Part 1 step 3: upgrades the two cross-checks
    above from informational-only to actionable. A single source
    disagreeing with nfl_data_py isn't enough on its own -- balldontlie
    and ourlads.com both lag real roster moves sometimes, and the spec's
    own words are explicit about avoiding "false positives from one
    lagging source." But when balldontlie says a player is really on
    team B (not nfl_data_py's team A) AND ourlads.com independently
    lists that same player on team B's depth chart (not found on team
    A's), that's two independent sources agreeing on the same specific
    correction -- a strong enough signal to hard-block the player from
    that week's props until a human confirms it, rather than continuing
    to show them under a team two other sources both say is wrong."""
    from data.fetch_firecrawl_sources import _normalize_name

    blocked = []
    for m in bdl_mismatches:
        key = (m["other_team"], _normalize_name(m["player"]))
        if key in ourlads_phantoms:
            blocked.append({
                "player": m["player"], "nfl_team": m["nfl_team"], "confirmed_team": m["other_team"],
                "reason": f"nfl_data_py has them at {m['nfl_team']}; both balldontlie and ourlads.com "
                          f"independently say {m['other_team']}",
            })
    return blocked


def save_blocked_players(blocked: list[dict]) -> None:
    """Persists this run's blocked-player list (Combined Build Plan
    Part 1 step 3) so model/player_stats.py's score_props() -- a
    separate process, run later in the weekly pipeline -- can read it
    without re-running both cross-checks itself. Always overwrites: a
    player confirmed correct on a later run (or an upstream fix landing)
    should stop being blocked, not accumulate forever."""
    import json
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(BLOCKED_PLAYERS_PATH, "w") as f:
        json.dump(blocked, f, indent=2)


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

    bdl_lines, bdl_mismatches = [], []
    ourlads_lines, ourlads_phantoms = [], set()
    if not roster.empty:
        bdl_lines, bdl_mismatches = cross_check_with_balldontlie(roster)
        for line in bdl_lines:
            print(line)
        # 32 sequential Firecrawl calls, rate-limited to 11/min on the
        # free tier -- real but bounded time cost (a few minutes) for a
        # weekly job, so on by default; check_ourlads=False is there for
        # a quick local test run that shouldn't wait on it.
        if check_ourlads:
            ourlads_lines, ourlads_phantoms = cross_check_with_ourlads(roster)
            for line in ourlads_lines:
                print(line)

    blocked = compute_blocked_players(bdl_mismatches, ourlads_phantoms)
    save_blocked_players(blocked)
    if blocked:
        print(f"{len(blocked)} player(s) BLOCKED from this week's props (balldontlie and ourlads.com "
              f"both independently disagree with nfl_data_py's team assignment -- manual confirmation needed):")
        for b in blocked:
            print(f"  - {b['player']}: {b['reason']}")

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
