# per-player rolling stat lines + over/under prop projections
"""Per-player stat lines and over/under projections for Phase 2's player
props -- extends model/predict.py's anytime-TD-scorer logic (which only
projects who's *most likely* to score) to full stat lines with a real
probability against a specific Vegas line. None of this feeds the win-
probability model; it's a standalone projection layer over the same
underlying play-by-play data, computed fresh for whatever week is being
scored, never backtested per historical week the way the team-level
rolling stats are.
"""

import os
import re

import nfl_data_py as nfl
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from config import CURRENT_SEASON
from data.fetch_injuries import USAGE_MULTIPLIER, fetch_current_player_injury_status
from data.fetch_props import fetch_props_for_week
from data.fetch_week import fetch_week
from data.positional_matchups import POSITION_GROUPS, build_position_tables, defense_rank, position_map

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBP_PATH = os.path.join(ROOT_DIR, "data", "cache", "pbp.parquet")

STAT_LOG_COLS = {
    "pass_yards": ("passer_player_id", "passer_player_name", "pass_yards"),
    "rush_yards": ("rusher_player_id", "rusher_player_name", "rush_yards"),
    "receptions": ("receiver_player_id", "receiver_player_name", "receptions"),
    "rec_yards": ("receiver_player_id", "receiver_player_name", "rec_yards"),
}

# Empirical per-game standard deviations for the normal-distribution
# yardage props, from 2025 play-by-play among players with a real workload
# (>=15 dropbacks / >=8 carries / >=4 targets that game) -- not guessed.
YARDAGE_STD = {"pass_yards": 75.0, "rush_yards": 35.0, "rec_yards": 33.0}

# Caps how far a single opponent matchup can move a projection off the
# player's own baseline rate -- same spirit as model/predict.py's TD-scorer
# def_factor/total_factor clipping, so one small-sample defensive number
# can't swing a projection to something implausible.
MATCHUP_FACTOR_BOUNDS = (0.75, 1.30)


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


def team_pass_yards_allowed(pbp: pd.DataFrame) -> pd.Series:
    """team -> average pass yards allowed per game. Simple team-level
    factor for the QB pass-yards matchup adjustment -- unlike WR/TE/RB,
    there's no separate defense-by-QB-position breakout in
    data/positional_matchups.py (a defense only ever faces one QB per
    game, so there's nothing to break out by)."""
    dropbacks = pbp[(pbp["qb_dropback"] == 1) & pbp["defteam"].notna()]
    per_game = dropbacks.groupby(["season", "week", "defteam"])["passing_yards"].sum().reset_index()
    return per_game.groupby("defteam")["passing_yards"].mean()


def _rank_series(current: pd.Series, fallback: pd.Series, team: str) -> int | None:
    """1-indexed league rank (1 = allows the most) for a flat team ->
    value Series, using each team's current-if-available-else-fallback
    number -- same UI-reasoning purpose as data/positional_matchups.py's
    defense_rank(), just for the flat (non-position-grouped) pass-defense
    case."""
    teams = set(current.index) | set(fallback.index)
    values = {t: current.get(t, fallback.get(t)) for t in teams}
    values = {t: v for t, v in values.items() if v is not None and pd.notna(v)}
    if team not in values or len(values) < 10:
        return None
    series = pd.Series(values)
    return int((series > series[team]).sum()) + 1


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


def _matchup_factor(off_value: float | None, def_value: float | None, league_avg: float | None) -> float:
    """def_value / league_avg, clipped -- how much tougher or softer this
    specific opponent is than average at defending this stat. off_value is
    accepted but unused here (kept in the signature for symmetry/future
    use, e.g. weighting by the offense's own pass-heaviness); the matchup
    signal itself is purely about the defense faced."""
    if def_value is None or league_avg in (None, 0) or pd.isna(def_value) or pd.isna(league_avg):
        return 1.0
    lo, hi = MATCHUP_FACTOR_BOUNDS
    return max(lo, min(hi, def_value / league_avg))


def project_yardage(mean: float, line: float, stat: str) -> dict:
    """Normal-distribution P(over)/P(under) for a yardage prop -- the
    spec's suggested distribution family for continuous, roughly
    symmetric per-game yardage totals."""
    std = YARDAGE_STD[stat]
    over_prob = float(1 - norm.cdf(line, loc=mean, scale=std))
    return {"projection": mean, "over_prob": over_prob, "under_prob": 1 - over_prob}


def project_count(mean: float, line: float) -> dict:
    """Poisson P(over)/P(under) for a count stat (receptions, TDs) -- same
    family the anytime-TD model already uses (model/predict.py's
    get_td_scorer_prediction: 1 - exp(-rate) is exactly poisson.sf(0,
    rate)). poisson.sf(k, mean) = P(X > k); a .5 line (the norm for these
    markets) means "over" is P(X >= ceil(line)) = P(X > line - 0.5)."""
    over_prob = float(poisson.sf(np.floor(line), mean))
    return {"projection": mean, "over_prob": over_prob, "under_prob": 1 - over_prob}


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


# QA spec Section 2: minimum lineup coverage per team per game, and which
# stat each position's "no line available" fallback card projects.
COVERAGE_MINIMUMS = {"QB": 1, "RB": 2, "WR": 5, "TE": 2}
PRIMARY_STAT = {"QB": "pass_yards", "RB": "rush_yards", "WR": "rec_yards", "TE": "rec_yards"}
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


def _project_mean(stat: str, player_id: str, opponent: str, position: str | None,
                   qb_avgs: tuple, rb_avgs: tuple, rec_avgs: tuple,
                   pos_tables_current: dict, pos_tables_fallback: dict,
                   pass_def: tuple, league_pass_def_fallback: float,
                   injury_status: dict) -> tuple[float, dict | None] | None:
    """The same projection math score_props()'s market loop uses, factored
    out for the no-line fallback path below -- one shared source of truth
    for "what does the model actually think this player does," whether or
    not a sportsbook has priced it."""
    if stat == "pass_yards":
        rec = _lookup_player_avg(*qb_avgs, player_id)
        if rec is None:
            return None
        pass_def_current, pass_def_fallback = pass_def
        def_val = pass_def_current.get(opponent, pass_def_fallback.get(opponent))
        factor = _matchup_factor(None, def_val, league_pass_def_fallback)
        mean = apply_injury_usage(rec["pass_yards_avg"] * factor, player_id, injury_status)
        rank = _rank_series(pass_def_current, pass_def_fallback, opponent)
        return mean, ({"kind": "yards", "group": None, "rank": rank} if rank else None)

    if stat == "rush_yards":
        rec = _lookup_player_avg(*rb_avgs, player_id)
        if rec is None:
            return None
        def_val = pos_tables_current["rush_def_ypc"].get(opponent, pos_tables_fallback["rush_def_ypc"].get(opponent))
        league_avg = pos_tables_fallback["rush_def_ypc"].mean()
        factor = _matchup_factor(None, def_val, league_avg)
        mean = apply_injury_usage(rec["rush_yards_avg"] * factor, player_id, injury_status)
        rank = defense_rank(pos_tables_current, pos_tables_fallback, opponent, "RUSH", "yards")
        return mean, ({"kind": "yards", "group": "RUSH", "rank": rank} if rank else None)

    if stat == "rec_yards":
        if position not in POSITION_GROUPS:
            return None
        rec = _lookup_player_avg(*rec_avgs, player_id)
        if rec is None:
            return None
        def_val, league_avg = None, None
        if position in pos_tables_fallback["defense_ypt"].columns:
            fb = pos_tables_fallback["defense_ypt"][position]
            cur = pos_tables_current["defense_ypt"][position] if position in pos_tables_current["defense_ypt"].columns else pd.Series(dtype=float)
            def_val, league_avg = cur.get(opponent, fb.get(opponent)), fb.mean()
        factor = _matchup_factor(None, def_val, league_avg)
        mean = apply_injury_usage(rec["rec_yards_avg"] * factor, player_id, injury_status)
        rank = defense_rank(pos_tables_current, pos_tables_fallback, opponent, position, "yards")
        return mean, ({"kind": "yards", "group": position, "rank": rank} if rank else None)

    return None


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
    props = fetch_props_for_week(games)
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

    pos_tables_current = build_position_tables(current_pbp, pos_map)
    pos_tables_fallback = build_position_tables(fallback_pbp, pos_map)
    pass_def_current = team_pass_yards_allowed(current_pbp)
    pass_def_fallback = team_pass_yards_allowed(fallback_pbp)
    league_pass_def_fallback = pass_def_fallback.mean()

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
        reasoning = None

        if m["stat"] == "pass_yards":
            rec = _lookup_player_avg(qb_avg_current, qb_avg_fallback, player_id)
            if rec is None:
                continue
            def_val = pass_def_current.get(opponent, pass_def_fallback.get(opponent))
            factor = _matchup_factor(None, def_val, league_pass_def_fallback)
            mean = apply_injury_usage(rec["pass_yards_avg"] * factor, player_id, injury_status)
            proj = project_yardage(mean, m["line"], "pass_yards")
            rank = _rank_series(pass_def_current, pass_def_fallback, opponent)
            if rank is not None:
                reasoning = {"kind": "yards", "group": None, "rank": rank}

        elif m["stat"] == "rush_yards":
            rec = _lookup_player_avg(rb_avg_current, rb_avg_fallback, player_id)
            if rec is None:
                continue
            def_val = pos_tables_current["rush_def_ypc"].get(opponent, pos_tables_fallback["rush_def_ypc"].get(opponent))
            league_avg = pos_tables_fallback["rush_def_ypc"].mean()
            factor = _matchup_factor(None, def_val, league_avg)
            mean = apply_injury_usage(rec["rush_yards_avg"] * factor, player_id, injury_status)
            proj = project_yardage(mean, m["line"], "rush_yards")
            rank = defense_rank(pos_tables_current, pos_tables_fallback, opponent, "RUSH", "yards")
            if rank is not None:
                reasoning = {"kind": "yards", "group": "RUSH", "rank": rank}

        elif m["stat"] in ("receptions", "rec_yards"):
            rec = _lookup_player_avg(rec_avg_current, rec_avg_fallback, player_id)
            if rec is None:
                continue
            pos = pos_map.get(player_id)
            def_val = None
            if pos in pos_tables_fallback["defense_ypt"].columns:
                fb = pos_tables_fallback["defense_ypt"][pos]
                cur = pos_tables_current["defense_ypt"][pos] if pos in pos_tables_current["defense_ypt"].columns else pd.Series(dtype=float)
                def_val = cur.get(opponent, fb.get(opponent))
                league_avg = fb.mean()
            else:
                league_avg = None
            factor = _matchup_factor(None, def_val, league_avg)
            if m["stat"] == "receptions":
                mean = apply_injury_usage(rec["receptions_avg"] * factor, player_id, injury_status)
                proj = project_count(mean, m["line"])
            else:
                mean = apply_injury_usage(rec["rec_yards_avg"] * factor, player_id, injury_status)
                proj = project_yardage(mean, m["line"], "rec_yards")
            if pos in POSITION_GROUPS:
                rank = defense_rank(pos_tables_current, pos_tables_fallback, opponent, pos, "yards")
                if rank is not None:
                    reasoning = {"kind": "yards", "group": pos, "rank": rank}

        elif m["stat"] == "anytime_td":
            rb_rec = _lookup_player_avg(rb_avg_current, rb_avg_fallback, player_id)
            wr_rec = _lookup_player_avg(rec_avg_current, rec_avg_fallback, player_id)
            rush_tds = rb_rec["rush_tds_avg"] if rb_rec else 0.0
            rec_tds = wr_rec["rec_tds_avg"] if wr_rec else 0.0
            if rb_rec is None and wr_rec is None:
                continue
            mean = apply_injury_usage(rush_tds + rec_tds, player_id, injury_status)
            proj = project_count(mean, 0.5)
            # Whichever role (ground or through the air) drives more of this
            # player's own TD rate decides which matchup rank the card cites.
            td_group = "RUSH" if rush_tds >= rec_tds else (position if position in POSITION_GROUPS else None)
            if td_group is not None:
                rank = defense_rank(pos_tables_current, pos_tables_fallback, opponent, td_group, "td_rate")
                if rank is not None:
                    reasoning = {"kind": "td_rate", "group": td_group, "rank": rank}

        else:
            continue

        injury = injury_status.get(player_id)
        espn_id = name_to_espn_id.get(canonical_name)
        covered.add((team, player_id))
        rows.append({
            "player": m["player"], "stat": m["stat"], "team": team, "opponent": opponent,
            "position": position, "espn_id": espn_id, "line": m["line"], "market_price": m["market_price"],
            "market_over_prob": m["market_over_prob"],
            "projection": proj["projection"], "model_over_prob": proj["over_prob"],
            "edge": proj["over_prob"] - m["market_over_prob"],
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
    pass_def = (pass_def_current, pass_def_fallback)
    coverage_gaps = []

    for _, game in games.iterrows():
        for team, opponent in ((game["home_team"], game["away_team"]), (game["away_team"], game["home_team"])):
            lineup = required_lineup(team, rosters, qb_avgs, rb_avgs, rec_avgs)
            for position, player_ids in lineup.items():
                for player_id in player_ids:
                    if (team, player_id) in covered:
                        continue
                    covered.add((team, player_id))
                    stat = PRIMARY_STAT[position]
                    result = _project_mean(
                        stat, player_id, opponent, position, qb_avgs, rb_avgs, rec_avgs,
                        pos_tables_current, pos_tables_fallback, pass_def, league_pass_def_fallback, injury_status)
                    if result is None:
                        coverage_gaps.append(f"{id_to_name.get(player_id, player_id)} ({team}, {position}): "
                                              f"no usable season data to project a fallback card")
                        continue
                    mean, reasoning = result
                    injury = injury_status.get(player_id)
                    rows.append({
                        "player": id_to_name.get(player_id, player_id), "stat": stat, "team": team,
                        "opponent": opponent, "position": position, "espn_id": id_to_espn_id.get(player_id),
                        "line": None, "market_price": None, "market_over_prob": None,
                        "projection": mean, "model_over_prob": None, "edge": None,
                        "reasoning": reasoning, "injury_status": injury["status"] if injury else None,
                        "has_line": False,
                    })

    if coverage_gaps:
        print(f"score_props: {len(coverage_gaps)} required-lineup slot(s) couldn't be filled even with a fallback:")
        for entry in coverage_gaps:
            print(f"  - {entry}")

    return pd.DataFrame(rows)
