# per-player anytime-TD prop projections
"""Anytime-TD-only player props (Anytime-TD-Only Props Refocus spec --
previously also covered yardage/reception props, removed: narrower scope,
same effort now concentrated on getting the one remaining prop type
right, with real backtesting and calibration behind it -- see
model/td_model.py, model/td_backtest.py, model/td_calibration.py). None
of this feeds the win-probability model; it's a standalone projection
layer over the same underlying play-by-play data, computed fresh for
whatever week is being scored.

QBs are scored on rushing TDs only, never passing -- confirmed against
The Odds API's own player_anytime_td market live (data/fetch_props.py):
its outcomes are the "Yes" side of "does this player score a rush or
receiving TD," the standard real-world "anytime touchdown scorer" market
definition, never passing TDs (which credit the QB who threw it, not
someone who "scored"). A QB's own rushing-TD chance (including "goal-
line vulture" sneaks/keepers) is captured the same way any other rusher's
is, via model/td_model.py's red-zone touch tracking -- no separate
mechanism needed.
"""

import math
import os
import re

import nfl_data_py as nfl
import pandas as pd

from config import CURRENT_SEASON
from data.fetch_injuries import USAGE_MULTIPLIER, fetch_current_player_injury_status
from data.fetch_props import fetch_props_for_week
from data.fetch_week import fetch_week
from data.positional_matchups import position_map
from model.td_model import (
    player_red_zone_touches, positional_baseline_conversion, project_td_probability_live,
    recency_weighted_touch_share, red_zone_defense_rank, season_to_date, team_red_zone_defense,
    team_red_zone_touches_per_game,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBP_PATH = os.path.join(ROOT_DIR, "data", "cache", "pbp.parquet")

STAT_LOG_COLS = {
    "pass_yards": ("passer_player_id", "passer_player_name", "pass_yards"),
    "rush_yards": ("rusher_player_id", "rusher_player_name", "rush_yards"),
    "receptions": ("receiver_player_id", "receiver_player_name", "receptions"),
    "rec_yards": ("receiver_player_id", "receiver_player_name", "rec_yards"),
}


def qb_passing_game_log(pbp: pd.DataFrame) -> pd.DataFrame:
    dropbacks = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna()]
    return dropbacks.groupby(
        ["season", "week", "posteam", "passer_player_id", "passer_player_name"]
    ).agg(pass_yards=("passing_yards", "sum"), pass_tds=("pass_touchdown", "sum"),
          attempts=("passing_yards", "size")).reset_index()


def rb_rushing_game_log(pbp: pd.DataFrame, pos_map: dict) -> pd.DataFrame:
    """Despite the name, this covers every real rusher, not just RBs --
    mobile QBs (scrambles, sneaks, designed runs) and WRs (jet sweeps,
    end-arounds) both show up in real anytime-TD and rushing-yards
    markets, and this table used to be filtered to RB only. That meant
    score_props() had no season data to project against for any of
    them, silently dropping real, currently-live market props even
    though both the market and this codebase's own play-by-play data
    had everything needed. required_lineup()'s RB slot is unaffected --
    it filters candidates by roster position before ever looking a
    player up in this table, so a QB or WR still can't fill an RB
    coverage slot just because they show up here."""
    rushes = pbp[(pbp["play_type"] == "run") & pbp["rusher_player_id"].notna()].copy()
    return rushes.groupby(
        ["season", "week", "posteam", "rusher_player_id", "rusher_player_name"]
    ).agg(rush_yards=("rushing_yards", "sum"), rush_tds=("rush_touchdown", "sum"),
          attempts=("rushing_yards", "size")).reset_index()


def receiving_game_log(pbp: pd.DataFrame, pos_map: dict) -> pd.DataFrame:
    targets = pbp[(pbp["play_type"] == "pass") & pbp["receiver_player_id"].notna()].copy()
    targets["receiver_pos"] = targets["receiver_player_id"].map(pos_map)
    targets = targets[targets["receiver_pos"].isin(["WR", "TE", "RB"])]
    return targets.groupby(
        ["season", "week", "posteam", "receiver_player_id", "receiver_player_name", "receiver_pos"]
    ).agg(
        receptions=("complete_pass", "sum"), rec_yards=("receiving_yards", "sum"),
        rec_tds=("pass_touchdown", "sum"), targets=("receiving_yards", "size"),
    ).reset_index()


def _player_season_avg(game_log: pd.DataFrame, id_col: str, name_col: str, stat_cols: list) -> dict:
    """id -> {"name", "games", stat_avg...} for every player in this game
    log window (current season or a fallback prior season)."""
    totals = game_log.groupby(id_col)[stat_cols].sum()
    games = game_log.groupby(id_col).size()
    names = game_log.groupby(id_col)[name_col].last()
    out = {}
    for pid in totals.index:
        row = {"name": names[pid], "games": int(games[pid])}
        for col in stat_cols:
            row[f"{col}_avg"] = totals.loc[pid, col] / games[pid]
        out[pid] = row
    return out


def _normalize_name(name: str) -> str:
    """Strips cosmetic differences between how a market source spells a
    player's name and how nfl_data_py's roster spells the same real
    person -- periods in initials ("KC Concepcion" vs "K.C. Concepcion"),
    a missing/extra suffix (nfl_data_py itself is inconsistent here:
    "Brian Thomas Jr." keeps the suffix, "Michael Pittman Jr." becomes
    plain "Michael Pittman"). Only ever used to bridge an exact-match
    miss, never in place of an exact match."""
    name = name.replace(".", "")
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip().lower()


def _build_normalized_name_map(names) -> dict:
    """normalized name -> canonical roster spelling, for names whose
    normalized form is unique. A normalized collision (two different
    real players stripping down to the same string -- e.g. two
    "Michael Pittman"s) is left out rather than guessed at; an
    unresolved name falls back to being dropped as unmatched, same as
    today, not silently mapped to the wrong person."""
    grouped: dict[str, set] = {}
    for name in names:
        grouped.setdefault(_normalize_name(name), set()).add(name)
    return {norm: next(iter(canon)) for norm, canon in grouped.items() if len(canon) == 1}


def _lookup_player_avg(current: dict, fallback: dict, player_id: str, min_games: int = 2) -> dict | None:
    for source in (current, fallback):
        row = source.get(player_id)
        if row and row["games"] >= min_games:
            return row
    return None


def apply_injury_usage(mean: float, player_id: str, injury_status: dict) -> float:
    """Discounts a projected mean by the player's own current injury
    status (Questionable/Doubtful reduce expected usage, not just
    efficiency) -- data/fetch_injuries.py's fetch_current_player_injury_status()."""
    record = injury_status.get(player_id)
    if not record:
        return mean
    return mean * USAGE_MULTIPLIER.get(record["status"], 1.0)


def _fallback_season_pbp(season: int) -> pd.DataFrame:
    pbp = pd.read_parquet(PBP_PATH)
    return pbp[pbp["season"] == season - 1]


def _current_season_pbp(season: int, week: int, fallback_pbp: pd.DataFrame) -> pd.DataFrame:
    schedule = nfl.import_schedules([season])
    played = schedule[
        (schedule["game_type"] == "REG") & schedule["home_score"].notna() & (schedule["week"] < week)
    ]
    if played.empty:
        return fallback_pbp.iloc[0:0]
    pbp = nfl.import_pbp_data([season], downcast=True)
    return pbp[pbp["week"] < week]


def _pivot_market_props(props: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, stat, home_team, away_team): the market's
    over probability (already de-vigged in data/fetch_props.py). For
    anytime_td (single-sided -- "Yes" is the only outcome), the implicit
    line is 0.5 (i.e. "at least 1"), same threshold the Poisson projection
    below uses."""
    two_sided = props[props["side"] == "over"][
        ["player", "stat", "line", "price", "prob", "home_team", "away_team"]
    ].rename(columns={"price": "market_price", "prob": "market_over_prob"})

    yes = props[props["side"] == "yes"][
        ["player", "stat", "price", "prob", "home_team", "away_team"]
    ].rename(columns={"price": "market_price", "prob": "market_over_prob"}).copy()
    yes["line"] = 0.5

    return pd.concat([two_sided, yes], ignore_index=True)


# QA spec Section 2: minimum lineup coverage per team per game -- kept
# unchanged by the Anytime-TD-Only Props Refocus spec (every player in
# this lineup now gets an anytime-TD projection instead of a yardage/
# reception one; the lineup composition itself isn't what's changing).
COVERAGE_MINIMUMS = {"QB": 1, "RB": 2, "WR": 5, "TE": 2}
_VOLUME_KEY = {"QB": "attempts_avg", "RB": "attempts_avg", "WR": "targets_avg", "TE": "targets_avg"}


def required_lineup(team: str, rosters: pd.DataFrame,
                     qb_avgs: tuple, rb_avgs: tuple, rec_avgs: tuple) -> dict:
    """Depth-chart-ranked player ids per position for `team`, sized to
    COVERAGE_MINIMUMS -- ranked by each player's own season-to-date (or
    fallback) usage volume, among players *currently on this team's
    roster* (not just whoever touched the ball for this team in old
    game-log data, which could include someone since traded away).
    Players with no recorded games in either window (e.g. a true rookie
    who hasn't played) can't be ranked and are left out -- a real
    limitation of leaning on prior performance data, not a bug.

    Known gap: this trusts nfl_data_py's own current-season team
    assignment at face value, and that source has been observed showing
    an incorrect team for at least one active player during the
    preseason (before the season's own games start correcting it) --
    unlike the market-prop path, there's no independent second source to
    cross-check a fallback-only player against, so a bad upstream team
    assignment here won't get caught the way a stale market attribution
    does. qa/validate_rosters.py's week-over-week diff is the best
    available defense -- a sudden, implausible team change shows up
    there for manual review, even though it can't be auto-corrected."""
    avgs_by_pos = {"QB": qb_avgs, "RB": rb_avgs, "WR": rec_avgs, "TE": rec_avgs}
    out = {}
    for position, n in COVERAGE_MINIMUMS.items():
        current_avg, fallback_avg = avgs_by_pos[position]
        candidates = rosters[(rosters["team"] == team) & (rosters["position"] == position)]["player_id"].unique()
        scored = []
        for pid in candidates:
            rec = _lookup_player_avg(current_avg, fallback_avg, pid, min_games=1)
            if rec:
                scored.append((pid, rec.get(_VOLUME_KEY[position], 0) or 0))
        scored.sort(key=lambda x: -x[1])
        out[position] = [pid for pid, _ in scored[:n]]
    return out


def _project_td(team: str, opponent: str, player_id: str, position: str | None,
                 touch_share: tuple, team_touches: dict, team_def: pd.DataFrame,
                 baseline_rates: dict, league_avg_rz_td_rate: float,
                 injury_status: dict) -> tuple[float, dict | None] | None:
    """Anytime-TD probability for one player, red-zone-share-based
    (model/td_model.py's project_td_probability_live), discounted for
    their own current injury status -- the single projection path used by
    both the market-loop (has_line cards) and the no-line coverage-fill
    fallback below, so a market-priced player and a required-lineup
    fallback player are projected by the exact same math, just with or
    without a market line to compare against. None if this player has no
    red-zone touch history at all (nothing to project from -- a real
    limitation of leaning on prior performance data, not a bug)."""
    result = project_td_probability_live(
        team, opponent, player_id, position, touch_share, team_touches, team_def,
        baseline_rates, league_avg_rz_td_rate)
    if result is None:
        return None
    expected_tds = apply_injury_usage(result["expected_tds"], player_id, injury_status)
    prob = 1 - math.exp(-expected_tds) if expected_tds > 0 else 0.0
    rank = red_zone_defense_rank(team_def, opponent)
    return prob, ({"rank": rank} if rank else None)


def score_props(week: int, season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Every player prop posted for this week, with the model's own
    projection, over-probability, and edge vs. the market's de-vigged
    probability -- plus, per QA spec Section 2, a "no line available"
    model-only card for any required starter (COVERAGE_MINIMUMS) the
    sportsbook hasn't priced, so lineup completeness never depends
    entirely on which players a book happened to post lines for. This
    runs the full computation even when zero market props exist yet
    (common weeks out from kickoff) -- previously that case short-circuited
    to an empty frame, which is exactly the "games missing props entirely"
    problem this section exists to fix."""
    games = fetch_week(week, season)
    props, alt_lines_by_event = fetch_props_for_week(games)
    # Alternate spreads/totals/team totals (Live News & Expanded Odds
    # spec) travel with the returned frame via .attrs rather than
    # changing this function's return type -- real, multi-book-aggregated
    # data (data/fetch_props.py's parse_alt_lines). Re-keyed here from
    # event_id to (home_team, away_team) -- the key every other per-game
    # UI lookup in this codebase already uses -- with each game's kickoff
    # datetime attached, so report/alt_lines.py's display layer can gate
    # on "how close is this to kickoff" without needing its own copy of
    # the event_id join. Most rungs run single-book (n_books=1) this far
    # before kickoff (see parse_alt_lines' docstring) -- report/
    # alt_lines.py only actually renders a ladder once real multi-book
    # coverage has filled in, not just whenever this dict is non-empty.
    alt_lines_by_game = {}
    for _, g in games.iterrows():
        event_id = g.get("event_id")
        if pd.isna(event_id) or event_id not in alt_lines_by_event:
            continue
        kickoff_dt = None
        if pd.notna(g.get("gameday")) and pd.notna(g.get("gametime")):
            try:
                kickoff_dt = pd.Timestamp(f"{g['gameday']} {g['gametime']}")
            except ValueError:
                kickoff_dt = None
        alt_lines_by_game[(g["home_team"], g["away_team"])] = {
            **alt_lines_by_event[event_id], "kickoff": kickoff_dt,
        }
    market = _pivot_market_props(props) if not props.empty else pd.DataFrame(
        columns=["player", "stat", "line", "market_price", "market_over_prob", "home_team", "away_team"])

    rosters = nfl.import_seasonal_rosters([season, season - 1])
    rosters = rosters[rosters["player_id"] != ""]
    name_to_id = dict(zip(rosters["player_name"], rosters["player_id"]))
    name_to_team = dict(zip(rosters["player_name"], rosters["team"]))
    name_to_espn_id = dict(zip(rosters["player_name"], rosters["espn_id"]))
    pos_map = dict(zip(rosters["player_id"], rosters["position"]))
    # A market source's spelling of a name doesn't always exactly match
    # nfl_data_py's own -- falls back to a normalized match (see
    # _normalize_name) only when there's no exact hit, so a real,
    # currently-live market prop doesn't silently get dropped over
    # nothing more than a stray period or suffix.
    normalized_roster_names = _build_normalized_name_map(rosters["player_name"].unique())

    def _resolve_market_name(market_name: str) -> str:
        if market_name in name_to_id:
            return market_name
        normalized_hit = normalized_roster_names.get(_normalize_name(market_name))
        if normalized_hit:
            return normalized_hit
        # Last resort: a source with the two words swapped (seen in the
        # wild as e.g. "James Jordan" for roster's "Jordan James") --
        # only for a plain two-word name, and only accepted if the
        # swapped form itself resolves unambiguously, so this can't
        # match two unrelated real people who happen to share a
        # first/last name pair in reverse.
        parts = market_name.split()
        if len(parts) == 2:
            swapped = f"{parts[1]} {parts[0]}"
            if swapped in name_to_id:
                return swapped
            swapped_hit = normalized_roster_names.get(_normalize_name(swapped))
            if swapped_hit:
                return swapped_hit
        return market_name

    fallback_pbp = _fallback_season_pbp(season)
    current_pbp = _current_season_pbp(season, week, fallback_pbp)

    # Kept for required_lineup()'s own ranking purposes -- who's the
    # depth-chart starter at each position, by real usage volume -- which
    # this spec leaves unchanged ("Keep the current lineup structure").
    # Actual TD projection below uses model/td_model.py's red-zone touch
    # data instead, a different and more predictive signal than raw
    # attempts/targets volume.
    qb_current, qb_fallback = qb_passing_game_log(current_pbp), qb_passing_game_log(fallback_pbp)
    rb_current, rb_fallback = rb_rushing_game_log(current_pbp, pos_map), rb_rushing_game_log(fallback_pbp, pos_map)
    rec_current, rec_fallback = (
        receiving_game_log(current_pbp, pos_map), receiving_game_log(fallback_pbp, pos_map))

    qb_avg_current = _player_season_avg(qb_current, "passer_player_id", "passer_player_name", ["pass_yards", "pass_tds", "attempts"])
    qb_avg_fallback = _player_season_avg(qb_fallback, "passer_player_id", "passer_player_name", ["pass_yards", "pass_tds", "attempts"])
    rb_avg_current = _player_season_avg(rb_current, "rusher_player_id", "rusher_player_name", ["rush_yards", "rush_tds", "attempts"])
    rb_avg_fallback = _player_season_avg(rb_fallback, "rusher_player_id", "rusher_player_name", ["rush_yards", "rush_tds", "attempts"])
    rec_avg_current = _player_season_avg(
        rec_current, "receiver_player_id", "receiver_player_name", ["receptions", "rec_yards", "rec_tds", "targets"])
    rec_avg_fallback = _player_season_avg(
        rec_fallback, "receiver_player_id", "receiver_player_name", ["receptions", "rec_yards", "rec_tds", "targets"])

    # Anytime-TD projection inputs (model/td_model.py). Current-season and
    # fallback (prior season, treated as one complete "week 30" window so
    # every game in it counts) are built separately and resolved per
    # player via project_td_probability_live's own current-if-enough-
    # else-fallback lookup -- same contract as _lookup_player_avg above,
    # just applied to red-zone share instead of season-average stat lines.
    touch_log_current = player_red_zone_touches(current_pbp)
    touch_log_fallback = player_red_zone_touches(fallback_pbp)
    touch_share = (
        recency_weighted_touch_share(touch_log_current, week),
        recency_weighted_touch_share(touch_log_fallback, 30),
    )
    team_touches = {
        **team_red_zone_touches_per_game(touch_log_fallback, 30),
        **team_red_zone_touches_per_game(touch_log_current, week),
    }
    rz_def_current = season_to_date(team_red_zone_defense(current_pbp), ["team"], ["rz_trips_allowed", "rz_tds_allowed"], week)
    rz_def_fallback = season_to_date(team_red_zone_defense(fallback_pbp), ["team"], ["rz_trips_allowed", "rz_tds_allowed"], 30)
    # Team-level red-zone defense: current season's own read on a team if
    # it has any games yet, else that team's fallback (prior season) row
    # -- same current-if-any-else-fallback merge model/predict.py's
    # get_pregame_stats() already uses for team-level rolling stats, a
    # coarser granularity than the strict per-player min-games gate
    # red-zone SHARE needs above.
    team_def = pd.concat([rz_def_fallback[~rz_def_fallback["team"].isin(rz_def_current["team"])], rz_def_current])
    baseline_rates = {
        **positional_baseline_conversion(touch_log_fallback, pos_map, 30),
        **positional_baseline_conversion(touch_log_current, pos_map, week),
    }
    league_avg_rz_td_rate = (
        (team_def["rz_tds_allowed"] / team_def["rz_trips_allowed"]).mean() if not team_def.empty else None)

    try:
        injury_status = fetch_current_player_injury_status([season, season - 1])
    except Exception:
        injury_status = {}

    rows = []
    dropped_stale = []
    covered = set()  # (team, player_id) already represented by a market prop
    for _, m in market.iterrows():
        canonical_name = _resolve_market_name(m["player"])
        player_id = name_to_id.get(canonical_name)
        team = name_to_team.get(canonical_name)
        if player_id is None or team is None:
            continue
        # QA spec Section 1: a market prop's player is only trustworthy if
        # their *current* roster team is actually one of the two teams in
        # this game -- a mismatch means the roster moved them (trade,
        # release, practice-squad churn) since the book last updated its
        # own attribution, and building a card off that would show a
        # player under the wrong team/opponent.
        if team not in (m["home_team"], m["away_team"]):
            dropped_stale.append(f"{m['player']} (prop listed for {m['home_team']}/{m['away_team']}, roster has them at {team})")
            continue
        opponent = m["away_team"] if team == m["home_team"] else m["home_team"]

        position = pos_map.get(player_id)

        # anytime_td is the only market this codebase requests any more
        # (data/fetch_props.py's PROP_MARKETS) -- but market rows come
        # from a live third-party feed, so this stays an explicit check
        # rather than an assumption.
        if m["stat"] != "anytime_td":
            continue
        result = _project_td(team, opponent, player_id, position, touch_share, team_touches, team_def,
                              baseline_rates, league_avg_rz_td_rate, injury_status)
        if result is None:
            continue
        model_prob, reasoning = result

        injury = injury_status.get(player_id)
        espn_id = name_to_espn_id.get(canonical_name)
        covered.add((team, player_id))
        rows.append({
            "player": m["player"], "player_id": player_id, "stat": m["stat"], "team": team, "opponent": opponent,
            "position": position, "espn_id": espn_id, "line": m["line"], "market_price": m["market_price"],
            "market_over_prob": m["market_over_prob"],
            "projection": model_prob, "model_over_prob": model_prob,
            "edge": model_prob - m["market_over_prob"],
            "reasoning": reasoning,
            "injury_status": injury["status"] if injury else None,
            "has_line": True,
        })

    if dropped_stale:
        print(f"score_props: dropped {len(dropped_stale)} prop(s) with a stale team attribution:")
        for entry in dropped_stale:
            print(f"  - {entry}")

    # QA spec Section 2: fill in required-lineup gaps the market didn't
    # cover with a model-only projection (no Vegas line, no P(over) --
    # there's nothing to be over/under without a line).
    id_to_name = dict(zip(rosters["player_id"], rosters["player_name"]))
    id_to_espn_id = dict(zip(rosters["player_id"], rosters["espn_id"]))
    qb_avgs, rb_avgs, rec_avgs = (qb_avg_current, qb_avg_fallback), (rb_avg_current, rb_avg_fallback), (rec_avg_current, rec_avg_fallback)
    coverage_gaps = []

    for _, game in games.iterrows():
        for team, opponent in ((game["home_team"], game["away_team"]), (game["away_team"], game["home_team"])):
            lineup = required_lineup(team, rosters, qb_avgs, rb_avgs, rec_avgs)
            for position, player_ids in lineup.items():
                for player_id in player_ids:
                    if (team, player_id) in covered:
                        continue
                    covered.add((team, player_id))
                    fallback_result = _project_td(team, opponent, player_id, position, touch_share, team_touches,
                                                   team_def, baseline_rates, league_avg_rz_td_rate, injury_status)
                    if fallback_result is None:
                        coverage_gaps.append(f"{id_to_name.get(player_id, player_id)} ({team}, {position}): "
                                              f"no usable season data to project a fallback card")
                        continue
                    model_prob, reasoning = fallback_result
                    injury = injury_status.get(player_id)
                    rows.append({
                        "player": id_to_name.get(player_id, player_id), "stat": "anytime_td", "team": team,
                        "opponent": opponent, "position": position, "espn_id": id_to_espn_id.get(player_id),
                        "line": None, "market_price": None, "market_over_prob": None,
                        "projection": model_prob, "model_over_prob": None, "edge": None,
                        "reasoning": reasoning, "injury_status": injury["status"] if injury else None,
                        "has_line": False,
                    })

    if coverage_gaps:
        print(f"score_props: {len(coverage_gaps)} required-lineup slot(s) couldn't be filled even with a fallback:")
        for entry in coverage_gaps:
            print(f"  - {entry}")

    result = pd.DataFrame(rows)
    result.attrs["alt_lines_by_game"] = alt_lines_by_game
    return result
