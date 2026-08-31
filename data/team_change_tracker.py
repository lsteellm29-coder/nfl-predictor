# coaching + per-unit roster continuity, narrative-only (Combined Build Plan Part 3)
"""Extends Part 2's roster-turnover concept (model/predict.py's
team_roster_turnover(), one flat number per team) to the full picture:
head-coach changes and turnover broken out by position UNIT, not just
lumped into one team-wide percentage -- "6 of 11 defensive starters are
new" is a materially different, more citable signal than "43% turnover"
with no sense of where.

Same narrative-only architecture as data/positional_matchups.py,
data/player_trends.py, data/team_history.py: never touches
model/train.py's FEATURE_COLS, runs after model/predict.py has already
scored the week, exists purely to inform a viewer and (via
model/predict.py's own confidence-caveat system) calibrate how much
visual confidence a pick displays with.

Known, honest scope limit: nfl_data_py exposes a head coach per team
per game (schedules' home_coach/away_coach column, the same field
data/team_history.py's coach_h2h() already reads) but has no offensive/
defensive-coordinator data anywhere in this codebase's data sources.
"Coaching changes: HC, OC, DC" in the spec is tracked here as HC only --
building a real OC/DC data source would mean standing up a new scraper
this codebase doesn't have, not reusing an existing one, so it's left
alone rather than faked.
"""

import re

import nfl_data_py as nfl
import pandas as pd

from data.rosters import fetch_rosters
from data.team_codes import normalize_team_codes

# Same usage definition as model/predict.py's MEANINGFUL_SNAP_PCT/
# MEANINGFUL_MIN_GAMES (kept as separate constants, not a cross-import,
# to avoid a data/ module depending on model/ -- this codebase's existing
# layering runs the other way) -- "starters/high-snap, not depth-chart
# bottom."
MEANINGFUL_SNAP_PCT = 0.50
MEANINGFUL_MIN_GAMES = 4
# A stricter bar for "genuinely key player," not just a rotational
# starter -- true every-down players.
NOTABLE_SNAP_PCT = 0.80

# A full position unit turning over 40%+ of its real starters is a real,
# citable signal on its own -- clearly above ordinary year-to-year
# rotation within one unit.
UNIT_TURNOVER_NOTABLE = 0.40

# Spec step 5: "stays active for the first several weeks of real games,
# not just preseason." A new coach's scheme (and a heavily turned-over
# unit gelling under it) takes longer to stabilize than model/predict.py's
# own FALLBACK_GAMES_THRESHOLD (3, Part 2's plain stat-sample-size bar) --
# deliberately a looser window than that one, roughly a quarter-season,
# not a hard cutoff backed by anything more precise than "a new scheme
# needs more than 2-3 games to prove itself either way."
COACH_CHANGE_GAMES_WINDOW = 6

POSITION_UNITS = {
    "offensive skill": {"RB", "WR", "TE", "FB"},
    "offensive line": {"T", "OT", "G", "OG", "C", "OL"},
    "defensive front": {"DE", "DT", "NT", "DL"},
    "linebackers": {"LB", "ILB", "OLB", "MLB"},
    "secondary": {"CB", "S", "FS", "SS", "DB"},
    "special teams": {"K", "P", "LS"},
}


def _normalize_name(name: str) -> str:
    """Same normalization data/fetch_firecrawl_sources.py's own copy
    applies -- kept as a local copy rather than a shared import, same
    precedent that module's own copy already set (and model/predict.py's
    copy repeated) for this codebase's independent-source name matching."""
    name = name.replace(".", "")
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip().lower()


def _unit_of(position) -> str | None:
    position = str(position).upper().strip()
    for unit, positions in POSITION_UNITS.items():
        if position in positions:
            return unit
    return None


def _team_coach_map(schedule: pd.DataFrame) -> dict[str, str]:
    """team -> head coach, from that team's most recent scheduled game
    with a coach listed (a season can have an in-season coaching change;
    the LAST entry is the most current one)."""
    rows = []
    for _, g in schedule.iterrows():
        rows.append((g["home_team"], g.get("home_coach")))
        rows.append((g["away_team"], g.get("away_coach")))
    df = pd.DataFrame(rows, columns=["team", "coach"]).dropna()
    if df.empty:
        return {}
    return df.drop_duplicates("team", keep="last").set_index("team")["coach"].to_dict()


def head_coach_changes(season: int) -> dict[str, dict]:
    """team -> {"prior_coach", "current_coach"} for every team whose
    head coach differs from last season's final one. A team with no
    coach listed in either season (a real, if rare, data gap) is simply
    absent rather than guessed at.

    Week 1 Audit & Tuning Plan Phase 1.1: these are two independent live
    fetches, one season apart, so they don't go through
    data/fetch_historical.py's cache (already normalized there) --
    normalizing here directly is what keeps a real relocation (e.g.
    Raiders OAK->LV in 2020) from making this function think every team
    in the league got a new head coach that year, purely because the
    two seasons' team-code strings didn't match."""
    prior = _team_coach_map(normalize_team_codes(nfl.import_schedules([season - 1])))
    current = _team_coach_map(normalize_team_codes(nfl.import_schedules([season])))
    return {
        team: {"prior_coach": prior[team], "current_coach": coach}
        for team, coach in current.items()
        if prior.get(team) and prior[team] != coach
    }


def _snap_share_by_player(season: int) -> pd.DataFrame:
    """One row per (team, player, position) with that player's average
    offense-or-defense snap share and games played, `season`'s REG-
    season games only. Empty if snap counts aren't available yet for
    this season (e.g. it hasn't started)."""
    snaps = normalize_team_codes(nfl.import_snap_counts([season]))
    snaps = snaps[snaps["game_type"] == "REG"]
    if snaps.empty:
        return pd.DataFrame(columns=["team", "player", "position", "mean", "size"])
    snap_pct = snaps[["offense_pct", "defense_pct"]].max(axis=1)
    snaps = snaps.assign(snap_pct=snap_pct)
    return snaps.groupby(["team", "player", "position"])["snap_pct"].agg(["mean", "size"]).reset_index()


def _meaningful_usage_by_unit(season: int) -> dict[str, dict[str, set[tuple[str, str]]]]:
    """team -> unit -> {(normalized_name, real_name)} of that unit's real
    starters/high-snap players in `season`."""
    usage = _snap_share_by_player(season)
    starters = usage[(usage["mean"] >= MEANINGFUL_SNAP_PCT) & (usage["size"] >= MEANINGFUL_MIN_GAMES)]

    result: dict[str, dict[str, set[tuple[str, str]]]] = {}
    for _, row in starters.iterrows():
        unit = _unit_of(row["position"])
        if not unit:
            continue
        result.setdefault(row["team"], {}).setdefault(unit, set()).add(
            (_normalize_name(row["player"]), row["player"]))
    return result


def team_unit_turnover(season: int) -> dict[str, dict[str, dict]]:
    """team -> unit -> {"turnover_pct", "departed", "total"} -- last
    season's real starters per position unit vs. this season's current
    roster (data/rosters.py's fetch_rosters(), so this inherits the
    current-season-wins + team-code-canonicalization guarantees the rest
    of this build already established for exactly this kind of cross-
    season/cross-source comparison). Empty dict for a team/season with
    no recorded prior-season starters at all."""
    prior_by_unit = _meaningful_usage_by_unit(season - 1)
    if not prior_by_unit:
        return {}

    current_rosters = fetch_rosters([season])
    current_by_team: dict[str, set] = {}
    for team, names in current_rosters.groupby("team")["player_name"]:
        current_by_team[team] = {_normalize_name(n) for n in names}

    result: dict[str, dict[str, dict]] = {}
    for team, units in prior_by_unit.items():
        current_names = current_by_team.get(team, set())
        team_result = {}
        for unit, starters in units.items():
            if not starters:
                continue
            departed = sum(1 for norm, _ in starters if norm not in current_names)
            team_result[unit] = {
                "turnover_pct": departed / len(starters),
                "departed": departed, "total": len(starters),
            }
        if team_result:
            result[team] = team_result
    return result


def notable_player_moves(season: int) -> dict[str, dict]:
    """team -> {"departed": [names], "arrived": [names]} for players who
    cleared NOTABLE_SNAP_PCT (true every-down usage, not just a
    rotational starter) for SOME team last season -- "departed" if
    they're no longer on that same team's current roster at all,
    "arrived" for a DIFFERENT team if this team's current roster now has
    them. Both computable from last season's usage + this season's
    roster alone -- neither needs this season's snap counts to exist
    yet, which matters since this runs before the season (and its own
    snap data) has started."""
    usage = _snap_share_by_player(season - 1)
    notable = usage[(usage["mean"] >= NOTABLE_SNAP_PCT) & (usage["size"] >= MEANINGFUL_MIN_GAMES)]
    if notable.empty:
        return {}

    # normalized_name -> (real_name, prior_team) -- last row wins on a
    # rare duplicate (a player who split significant snaps across two
    # teams last season), same "most recent/most complete row" tolerance
    # the rest of this codebase already accepts for this class of edge case.
    prior_notable: dict[str, tuple[str, str]] = {}
    for _, row in notable.iterrows():
        prior_notable[_normalize_name(row["player"])] = (row["player"], row["team"])

    current_rosters = fetch_rosters([season])
    current_team_of = dict(zip(current_rosters["player_name"].map(_normalize_name), current_rosters["team"]))

    result: dict[str, dict] = {}
    for norm, (real_name, prior_team) in prior_notable.items():
        current_team = current_team_of.get(norm)
        if current_team is None or current_team == prior_team:
            continue
        result.setdefault(prior_team, {}).setdefault("departed", []).append(real_name)
        result.setdefault(current_team, {}).setdefault("arrived", []).append(f"{real_name} (from {prior_team})")
    return result


def team_change_confidence_flag(games_played: int, coach_change: dict | None, unit_turnover: dict) -> bool:
    """Spec step 4: "ties into the same confidence-caveat system from
    Part 2 -- new [head coach] + high positional turnover pushes toward
    the yellow tier the same way preseason-fallback reliance does."
    Deliberately requires BOTH a coach change AND meaningfully high unit
    turnover, same AND-gate discipline model/predict.py's
    fallback_confidence_caveat() already applies for Part 2 -- a new
    coach alone with a stable roster, or high turnover under a returning
    coach, are each a real but smaller risk than both together. Active
    through COACH_CHANGE_GAMES_WINDOW games (spec step 5: "the first
    several weeks of real games, not just preseason"), not just while
    games_played is 0."""
    if games_played >= COACH_CHANGE_GAMES_WINDOW:
        return False
    if not coach_change:
        return False
    max_turnover = max((u["turnover_pct"] for u in unit_turnover.values()), default=0.0)
    return max_turnover >= UNIT_TURNOVER_NOTABLE


def team_change_callouts(team: str, coach_changes: dict, unit_turnover: dict, notable_moves: dict) -> list[str]:
    """Plain-language callouts for one team (spec step 3: "New head
    coach," "Significant turnover on defense -- 6 of 11 starters new")
    -- the display layer's ready-to-render summary, built from the three
    functions above."""
    callouts = []
    change = coach_changes.get(team)
    if change:
        callouts.append(f"New head coach: {change['current_coach']} (previously {change['prior_coach']}).")

    for unit, data in sorted(unit_turnover.get(team, {}).items(), key=lambda kv: -kv[1]["turnover_pct"]):
        if data["turnover_pct"] >= UNIT_TURNOVER_NOTABLE:
            callouts.append(
                f"Significant turnover at {unit} -- {data['departed']} of {data['total']} regular starters "
                f"are new.")

    moves = notable_moves.get(team, {})
    if moves.get("departed"):
        callouts.append(f"Notable departure(s): {', '.join(moves['departed'][:3])}.")
    if moves.get("arrived"):
        callouts.append(f"Notable addition(s): {', '.join(moves['arrived'][:3])}.")
    return callouts
