# WR/TE/RB target and rush-vs-front-7 matchup profiles, narrative only
"""Position-group matchup profiles for the game-preview narrative -- these
numbers never touch the win-probability model (model/train.py's
FEATURE_COLS). Purely used by report/narrative.py to find a genuinely
convincing "this team's WRs vs that defense's pass coverage" storyline,
distinct from the team-wide rolling stats in data/team_stats.py.
"""

import nfl_data_py as nfl
import pandas as pd

POSITION_GROUPS = ["WR", "TE", "RB"]

# Below these counts, a position group's average EPA/target (or a team's
# rush attempts) is too thin a sample to call a "mismatch" with a straight
# face -- single-play EPA has a standard deviation around 1.0-1.7, so a
# handful of plays can swing the average wildly. A full season runs
# ~300/130/95 WR/TE/RB targets and ~465 rush attempts per team, so these
# bars target roughly a quarter of a season as the minimum to trust.
MIN_TARGETS = {"WR": 40, "TE": 20, "RB": 15}
MIN_RUSH_ATTEMPTS = 60


def position_map(seasons: list[int]) -> dict:
    """player_id (GSIS, same id space as pbp's receiver_player_id) ->
    position, from seasonal rosters. A season is one snapshot, so a
    mid-season position change (rare) won't be reflected -- fine for a
    narrative use case, not worth a weekly roster pull."""
    rosters = nfl.import_seasonal_rosters(seasons)
    rosters = rosters[rosters["player_id"] != ""]
    return dict(zip(rosters["player_id"], rosters["position"]))


def build_position_tables(pbp: pd.DataFrame, pos_map: dict) -> dict:
    """team x position tables of EPA/target and target counts (offense
    produced, defense allowed), plus team-level rush EPA/attempt and
    attempt counts. `pbp` should already be scoped to whatever window
    (season-to-date, or a fallback prior season) the caller wants profiled."""
    pbp = pbp.copy()
    pbp["recv_pos"] = pbp["receiver_player_id"].map(pos_map)
    pass_plays = pbp[(pbp["play_type"] == "pass") & pbp["recv_pos"].isin(POSITION_GROUPS)]

    rush_plays = pbp[pbp["play_type"] == "run"]

    return {
        "defense_epa": pass_plays.groupby(["defteam", "recv_pos"])["epa"].mean().unstack(),
        "defense_n": pass_plays.groupby(["defteam", "recv_pos"]).size().unstack(fill_value=0),
        "offense_epa": pass_plays.groupby(["posteam", "recv_pos"])["epa"].mean().unstack(),
        "offense_n": pass_plays.groupby(["posteam", "recv_pos"]).size().unstack(fill_value=0),
        "rush_def_epa": rush_plays.groupby("defteam")["epa"].mean(),
        "rush_def_n": rush_plays.groupby("defteam").size(),
        "rush_off_epa": rush_plays.groupby("posteam")["epa"].mean(),
        "rush_off_n": rush_plays.groupby("posteam").size(),
        # Yards/target and yards/carry allowed, by position -- same shape
        # as the EPA tables above but in raw yardage terms, since Phase 2's
        # prop projections need an actual yardage matchup multiplier, not
        # an abstract EPA one.
        "defense_ypt": pass_plays.groupby(["defteam", "recv_pos"])["receiving_yards"].mean().unstack(),
        "offense_ypt": pass_plays.groupby(["posteam", "recv_pos"])["receiving_yards"].mean().unstack(),
        "rush_def_ypc": rush_plays.groupby("defteam")["rushing_yards"].mean(),
        "rush_off_ypc": rush_plays.groupby("posteam")["rushing_yards"].mean(),
        # TD rate allowed (per attempt/target) by position -- feeds the
        # UI's anytime-TD card reasoning ("Nth-most rushing TDs allowed"),
        # same shape as the yardage tables above.
        "defense_td_rate": pass_plays.groupby(["defteam", "recv_pos"])["pass_touchdown"].mean().unstack(),
        "rush_def_td_rate": rush_plays.groupby("defteam")["rush_touchdown"].mean(),
    }


def _lookup(epa_current, n_current, epa_fallback, n_fallback, key, min_n, col=None):
    """Current-season number if it clears min_n, else the fallback (prior
    season) number -- same current-then-fallback shape as
    model/predict.py's TD-scorer logic, so a team with only a couple of
    games played this season doesn't get a mismatch claim built on 8
    targets."""
    for epa_tbl, n_tbl in ((epa_current, n_current), (epa_fallback, n_fallback)):
        try:
            n = n_tbl.loc[key] if col is None else n_tbl.loc[key, col]
            epa = epa_tbl.loc[key] if col is None else epa_tbl.loc[key, col]
        except KeyError:
            continue
        if pd.notna(n) and n >= min_n and pd.notna(epa):
            return float(epa), int(n)
    return None


def team_position_mismatches(offense_team: str, defense_team: str, current: dict, fallback: dict) -> list[dict]:
    """Every position-group (+ rush) mismatch for offense_team attacking
    defense_team, where both sides clear the minimum sample bar. `score` is
    the offense's own EPA/target plus what the defense has been allowing --
    a good unit (high positive) meeting a bad matchup (defense allowing a
    lot, also positive) pushes the same direction, so adding them directly
    captures "good unit meets bad matchup" in one number, versus a strong
    unit that just happens to be facing a strong defense netting close to
    zero."""
    out = []
    for pos in POSITION_GROUPS:
        off = _lookup(current["offense_epa"], current["offense_n"],
                       fallback["offense_epa"], fallback["offense_n"], offense_team, MIN_TARGETS[pos], pos)
        de = _lookup(current["defense_epa"], current["defense_n"],
                      fallback["defense_epa"], fallback["defense_n"], defense_team, MIN_TARGETS[pos], pos)
        if off is None or de is None:
            continue
        off_epa, off_n = off
        def_epa, def_n = de
        out.append({
            "type": "positional", "group": pos, "team": offense_team, "opponent": defense_team,
            "score": off_epa + def_epa, "off_epa": off_epa, "def_epa": def_epa,
            "off_n": off_n, "def_n": def_n,
        })

    off = _lookup(current["rush_off_epa"], current["rush_off_n"],
                   fallback["rush_off_epa"], fallback["rush_off_n"], offense_team, MIN_RUSH_ATTEMPTS)
    de = _lookup(current["rush_def_epa"], current["rush_def_n"],
                  fallback["rush_def_epa"], fallback["rush_def_n"], defense_team, MIN_RUSH_ATTEMPTS)
    if off is not None and de is not None:
        off_epa, off_n = off
        def_epa, def_n = de
        out.append({
            "type": "positional", "group": "RUSH", "team": offense_team, "opponent": defense_team,
            "score": off_epa + def_epa, "off_epa": off_epa, "def_epa": def_epa,
            "off_n": off_n, "def_n": def_n,
        })
    return out


def game_mismatches(home: str, away: str, current: dict, fallback: dict) -> list[dict]:
    """Every qualifying mismatch in both directions (home attacking away's
    defense, and vice versa) for one game."""
    return (
        team_position_mismatches(home, away, current, fallback)
        + team_position_mismatches(away, home, current, fallback)
    )


def _effective_series(current_tbl, current_n, fallback_tbl, fallback_n, min_n, teams, col=None) -> pd.Series:
    """Each team's current-season-if-enough-else-fallback value, assembled
    into one Series -- so a league rank computed from it puts every team
    on a consistent basis instead of comparing some teams' real
    current-season numbers against others' leftover prior-season ones."""
    values = {}
    for team in teams:
        result = _lookup(current_tbl, current_n, fallback_tbl, fallback_n, team, min_n, col)
        if result is not None:
            values[team] = result[0]
    return pd.Series(values, dtype=float)


def defense_rank(current: dict, fallback: dict, team: str, group: str, stat: str = "yards") -> int | None:
    """1-indexed league rank of how much `team`'s defense allows to
    `group` (WR/TE/RB, or "RUSH"), `stat` = "yards" or "td_rate" -- 1 =
    allows the most, matching how "3rd-most yards allowed" is normally
    phrased. None if the sample's too thin league-wide to rank honestly."""
    if group == "RUSH":
        cur_val, cur_n = current["rush_def_ypc" if stat == "yards" else "rush_def_td_rate"], current["rush_def_n"]
        fb_val, fb_n = fallback["rush_def_ypc" if stat == "yards" else "rush_def_td_rate"], fallback["rush_def_n"]
        teams = set(cur_val.index) | set(fb_val.index)
        series = _effective_series(cur_val, cur_n, fb_val, fb_n, MIN_RUSH_ATTEMPTS, teams)
    else:
        cur_val = current["defense_ypt" if stat == "yards" else "defense_td_rate"]
        fb_val = fallback["defense_ypt" if stat == "yards" else "defense_td_rate"]
        teams = set(current["defense_n"].index) | set(fallback["defense_n"].index)
        series = _effective_series(cur_val, current["defense_n"], fb_val, fallback["defense_n"],
                                    MIN_TARGETS[group], teams, group)
    if team not in series.index or len(series) < 10:
        return None
    return int((series > series[team]).sum()) + 1
