# the one command that runs the full pipeline
"""The one command from Section 5 of the spec: run_week.py --week 5

Pulls that week's schedule + current rolling stats + the current betting
line, runs the model, prints/saves the weekly report, and appends the week's
predictions to logs/season_results.csv -- grading any previously-logged
predictions whose games have since finished, so accuracy is trackable over
the season (Section 4.4 / Section 5.6).
"""

import argparse
import datetime as dt
import os

import nfl_data_py as nfl
import pandas as pd
import requests

from config import CURRENT_SEASON
from data.fetch_news import fetch_news
from data.pbp_loader import load_season_pbp
from model.player_stats import score_props
from model.predict import score_week
from model.prediction_log import game_has_kicked_off, log_predictions
from report.build_report import build_report

LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "season_results.csv")

LOG_COLS = [
    "season", "week", "away_team", "home_team",
    "predicted_winner", "home_win_prob", "vegas_spread", "model_spread", "edge",
    "actual_away_score", "actual_home_score", "actual_winner",
    "correct", "model_beat_market",
]

PROPS_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "props_results.csv")

# Only has_line=True props are logged -- a "no line available" fallback
# card has no market probability to grade against, and no over/under side
# was ever claimed for it, so there's nothing to score as right or wrong.
PROPS_LOG_COLS = [
    "season", "week", "player", "player_id", "team", "opponent", "stat", "line",
    "market_over_prob", "model_over_prob", "edge",
    "actual_value", "actual_over", "model_correct", "market_correct", "model_beat_market",
]


# Columns that hold mixed text/bool/NA by design (a winner's team code, a
# True/False verdict, or "not graded yet"). pandas 3 no longer silently upcasts
# on .loc assignment, and read_csv types an all-empty column float64 (and a
# fully-graded True/False one bool), so grading a pending row -- assigning "SEA"
# or pd.NA into either -- raised TypeError and took down the whole run the first
# time there was a previous week to grade.
_LOG_MIXED_COLS = ["actual_winner", "correct", "model_beat_market"]
_LOG_SCORE_COLS = ["actual_away_score", "actual_home_score"]
_PROPS_MIXED_COLS = ["actual_over", "model_correct", "market_correct", "model_beat_market"]


def _coerce_log_dtypes(df: pd.DataFrame, mixed_cols: list[str], float_cols: list[str]) -> pd.DataFrame:
    for col in mixed_cols:
        if col in df.columns:
            df[col] = df[col].astype(object)
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def _load_log() -> pd.DataFrame:
    if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0:
        df = pd.read_csv(LOG_PATH)
    else:
        df = pd.DataFrame(columns=LOG_COLS)
    return _coerce_log_dtypes(df, _LOG_MIXED_COLS, _LOG_SCORE_COLS)


def _grade_pending(log_df: pd.DataFrame) -> pd.DataFrame:
    """Fill in actual results for logged predictions whose games have since
    been played, and score whether the pick (and the model's edge vs. the
    market) was right."""
    pending = log_df[log_df["actual_winner"].isna()]
    if pending.empty:
        return log_df

    schedules = nfl.import_schedules(sorted(pending["season"].unique().tolist()))

    for idx, row in pending.iterrows():
        game = schedules[
            (schedules["season"] == row["season"]) & (schedules["week"] == row["week"])
            & (schedules["home_team"] == row["home_team"]) & (schedules["away_team"] == row["away_team"])
        ]
        if game.empty or pd.isna(game.iloc[0]["home_score"]):
            continue  # not played yet

        home_score, away_score = game.iloc[0]["home_score"], game.iloc[0]["away_score"]
        log_df.loc[idx, "actual_home_score"] = home_score
        log_df.loc[idx, "actual_away_score"] = away_score

        if home_score == away_score:
            # A real NFL tie -- neither team "won," so there's nothing to
            # grade the straight-up pick right or wrong against. Distinct
            # from a push in the ATS block below, which stays unaffected:
            # ATS grades margin against the spread, not who won.
            log_df.loc[idx, "actual_winner"] = "TIE"
            log_df.loc[idx, "correct"] = pd.NA
        else:
            actual_winner = row["home_team"] if home_score > away_score else row["away_team"]
            log_df.loc[idx, "actual_winner"] = actual_winner
            log_df.loc[idx, "correct"] = actual_winner == row["predicted_winner"]

        if pd.notna(row["vegas_spread"]) and pd.notna(row["edge"]) and row["edge"] != 0:
            ats_margin = (home_score - away_score) - row["vegas_spread"]
            if ats_margin != 0:  # exclude pushes
                home_covered = ats_margin > 0
                model_favored_home = row["edge"] > 0
                log_df.loc[idx, "model_beat_market"] = model_favored_home == home_covered

    return log_df


def _write_csv(df: pd.DataFrame, path: str) -> None:
    """Write via a temp file and rename, so a crash mid-write can never leave a truncated
    results log behind (these CSVs hold the season's graded history)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def started_game_pairs(predictions: pd.DataFrame, now: dt.datetime | None = None) -> set[tuple[str, str]]:
    """(home, away) of every game in `predictions` that has already kicked off."""
    return {(g["home_team"], g["away_team"]) for _, g in predictions.iterrows()
            if game_has_kicked_off(g.get("gameday"), g.get("gametime"), now, g.get("home_score"))}


def started_teams(predictions: pd.DataFrame, now: dt.datetime | None = None) -> set[str]:
    return {team for pair in started_game_pairs(predictions, now) for team in pair}


def log_week(predictions: pd.DataFrame, week: int, season: int,
             now: dt.datetime | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (full log, newly-graded-this-run subset) -- the second is
    what report/recap.py's weekly recap is built from, never the full
    season history (that's the Track Record page's job).

    Only games that haven't kicked off get logged. Re-running a week
    replaces its rows, but a game that has already started keeps whatever was
    logged for it before kickoff and gets nothing new -- a row written after the
    fact would be a hindsight "prediction" graded as if it were a real one."""
    started = started_game_pairs(predictions, now)
    log_df = _load_log()
    previously_pending_idx = set(log_df.index[log_df["actual_winner"].isna()])
    try:
        log_df = _grade_pending(log_df)
    except Exception as e:
        # grading fetches final scores; a network hiccup there must not stop this run from
        # logging its own pre-kickoff predictions -- earlier weeks just stay pending until next run
        print(f"Warning: couldn't grade earlier weeks this run ({type(e).__name__}: {e}); they stay pending.")
    newly_graded = log_df.loc[sorted(previously_pending_idx & set(log_df.index[log_df["actual_winner"].notna()]))]

    rows = []
    for _, game in predictions.iterrows():
        if pd.isna(game.get("home_win_prob")):
            continue  # no prediction made for this game (no team history available)
        if (game["home_team"], game["away_team"]) in started:
            print(f"Not logging {game['away_team']} @ {game['home_team']} to the results log: already kicked off.")
            continue
        home_prob = game["home_win_prob"]
        winner = game["home_team"] if home_prob >= 0.5 else game["away_team"]
        rows.append({
            "season": season, "week": week,
            "away_team": game["away_team"], "home_team": game["home_team"],
            "predicted_winner": winner, "home_win_prob": home_prob,
            "vegas_spread": game.get("spread_line"), "model_spread": game.get("implied_spread"),
            "edge": game.get("edge"),
            "actual_away_score": pd.NA, "actual_home_score": pd.NA, "actual_winner": pd.NA,
            "correct": pd.NA, "model_beat_market": pd.NA,
        })

    # Re-running the same week replaces the rows it is re-logging instead of duplicating them.
    # Nothing else is touched: a game that has already started keeps what was logged before
    # kickoff, and so does a game this run happened to produce no prediction for (a line pulled,
    # a team-name mismatch) -- that earlier row was a valid pre-kickoff pick and can't be redone.
    replaced = {(r["home_team"], r["away_team"]) for r in rows}
    this_week = (log_df["season"] == season) & (log_df["week"] == week)
    being_replaced = pd.Series([(h, a) in replaced for h, a in zip(log_df["home_team"], log_df["away_team"])],
                               index=log_df.index, dtype=bool)
    log_df = log_df[~(this_week & being_replaced)]
    log_df = pd.concat([log_df, pd.DataFrame(rows, columns=LOG_COLS)], ignore_index=True)

    _write_csv(log_df, LOG_PATH)
    return log_df, newly_graded


def _load_props_log() -> pd.DataFrame:
    if os.path.exists(PROPS_LOG_PATH) and os.path.getsize(PROPS_LOG_PATH) > 0:
        df = pd.read_csv(PROPS_LOG_PATH)
    else:
        df = pd.DataFrame(columns=PROPS_LOG_COLS)
    return _coerce_log_dtypes(df, _PROPS_MIXED_COLS, ["actual_value"])


def _actual_stat_value(stat: str, player_id, week: int, qb_log: pd.DataFrame,
                        rb_log: pd.DataFrame, rec_log: pd.DataFrame) -> float | None:
    """A player's real total for `stat` in a specific already-played week,
    from the same per-player game-log tables model/player_stats.py builds
    for projection -- None if the player didn't appear in that week's pbp
    at all (e.g. inactive/didn't play), which isn't gradeable as a wrong
    prediction, just not observed."""
    if stat == "pass_yards":
        row = qb_log[(qb_log["week"] == week) & (qb_log["passer_player_id"] == player_id)]
        return float(row["pass_yards"].iloc[0]) if not row.empty else None
    if stat == "rush_yards":
        row = rb_log[(rb_log["week"] == week) & (rb_log["rusher_player_id"] == player_id)]
        return float(row["rush_yards"].iloc[0]) if not row.empty else None
    if stat == "rec_yards":
        row = rec_log[(rec_log["week"] == week) & (rec_log["receiver_player_id"] == player_id)]
        return float(row["rec_yards"].iloc[0]) if not row.empty else None
    if stat == "receptions":
        row = rec_log[(rec_log["week"] == week) & (rec_log["receiver_player_id"] == player_id)]
        return float(row["receptions"].iloc[0]) if not row.empty else None
    if stat == "anytime_td":
        # A player can score on the ground or through the air -- both
        # count toward the same "did they score at all" market, so this
        # sums whichever rows exist rather than picking one table.
        rush_row = rb_log[(rb_log["week"] == week) & (rb_log["rusher_player_id"] == player_id)]
        rec_row = rec_log[(rec_log["week"] == week) & (rec_log["receiver_player_id"] == player_id)]
        if rush_row.empty and rec_row.empty:
            return None
        rush_tds = float(rush_row["rush_tds"].iloc[0]) if not rush_row.empty else 0.0
        rec_tds = float(rec_row["rec_tds"].iloc[0]) if not rec_row.empty else 0.0
        return rush_tds + rec_tds
    return None


def _grade_pending_props(log_df: pd.DataFrame) -> pd.DataFrame:
    """Fills in actual results for logged props whose games have since been
    played, using the exact same game-log builders model/player_stats.py
    uses to project them -- so grading is measuring the same real stat the
    prediction was made against, not an approximation of it."""
    pending = log_df[log_df["actual_value"].isna()]
    if pending.empty:
        return log_df

    from data.positional_matchups import position_map
    from model.player_stats import qb_passing_game_log, rb_rushing_game_log, receiving_game_log

    for season in sorted(pending["season"].unique()):
        season = int(season)
        # nflverse doesn't publish a pbp file for a season until it has at
        # least one played game -- same check model/predict.py's
        # _current_season_pbp() already makes before fetching, needed here
        # too since an all-pending season (nothing played yet) would
        # otherwise 404 on a file that doesn't exist yet.
        schedule = nfl.import_schedules([season])
        if not (schedule["home_score"].notna()).any():
            continue
        pbp = load_season_pbp(season)
        pos_map = position_map([season])
        qb_log = qb_passing_game_log(pbp)
        rb_log = rb_rushing_game_log(pbp, pos_map)
        rec_log = receiving_game_log(pbp, pos_map)
        played_weeks = set(pbp["week"].unique())

        season_pending = pending[pending["season"] == season]
        for idx, row in season_pending.iterrows():
            week = int(row["week"])
            if week not in played_weeks:
                continue  # game hasn't been played yet
            actual = _actual_stat_value(row["stat"], row["player_id"], week, qb_log, rb_log, rec_log)
            if actual is None:
                continue  # player didn't play (inactive/DNP) -- not a gradeable miss

            actual_over = actual > row["line"]
            model_side_over = row["model_over_prob"] >= 0.5
            market_side_over = row["market_over_prob"] >= 0.5

            log_df.loc[idx, "actual_value"] = actual
            log_df.loc[idx, "actual_over"] = actual_over
            log_df.loc[idx, "model_correct"] = model_side_over == actual_over
            log_df.loc[idx, "market_correct"] = market_side_over == actual_over
            if model_side_over != market_side_over:  # only meaningful when they disagreed
                log_df.loc[idx, "model_beat_market"] = model_side_over == actual_over

    return log_df


def log_props_week(props: pd.DataFrame, week: int, season: int,
                   started: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (full log, newly-graded-this-run subset) -- see log_week()'s
    docstring for why the second value matters. `started` is the set of team
    codes whose game has already kicked off: their props get nothing new logged,
    and rows already logged for them are kept -- same reasoning as log_week()."""
    started = started or set()
    log_df = _load_props_log()
    previously_pending_idx = set(log_df.index[log_df["actual_value"].isna()])
    try:
        log_df = _grade_pending_props(log_df)
    except Exception as e:
        # same stance as log_week(): a grading hiccup must not block logging this run's props
        print(f"Warning: couldn't grade earlier props this run ({type(e).__name__}: {e}); they stay pending.")
    newly_graded = log_df.loc[sorted(previously_pending_idx & set(log_df.index[log_df["actual_value"].notna()]))]

    market_backed = props[props["has_line"]] if not props.empty else props
    rows = []
    for _, p in market_backed.iterrows():
        if p["team"] in started:
            continue
        rows.append({
            "season": season, "week": week,
            "player": p["player"], "player_id": p["player_id"], "team": p["team"], "opponent": p["opponent"],
            "stat": p["stat"], "line": p["line"],
            "market_over_prob": p["market_over_prob"], "model_over_prob": p["model_over_prob"], "edge": p["edge"],
            "actual_value": pd.NA, "actual_over": pd.NA,
            "model_correct": pd.NA, "market_correct": pd.NA, "model_beat_market": pd.NA,
        })

    # Re-running the same week replaces only the (team, player, stat) rows it is re-logging.
    # Everything else keeps what was logged: started teams, and any prop this run didn't
    # produce a market line for (e.g. with the Odds API down every prop has has_line=False,
    # which used to wipe the week's whole props log) -- those earlier rows were valid
    # pre-kickoff picks and can't be recreated after the game.
    replaced = {(r["team"], r["player_id"], r["stat"]) for r in rows}
    this_week = (log_df["season"] == season) & (log_df["week"] == week)
    being_replaced = pd.Series([(t, pid, st) in replaced for t, pid, st in
                                zip(log_df["team"], log_df["player_id"], log_df["stat"])],
                               index=log_df.index, dtype=bool)
    log_df = log_df[~(this_week & being_replaced)]
    log_df = pd.concat([log_df, pd.DataFrame(rows, columns=PROPS_LOG_COLS)], ignore_index=True)

    _write_csv(log_df, PROPS_LOG_PATH)
    return log_df, newly_graded


def get_current_week(season: int = CURRENT_SEASON) -> int:
    """The earliest REG-season week that still has at least one game left to
    play -- i.e. this week, or the upcoming one if the whole season's
    schedule is already final. Lets run_week.py be pointed at "now" without
    a --week argument, which is what a weekly automated run needs."""
    schedule = nfl.import_schedules([season])
    reg = schedule[schedule["game_type"] == "REG"]
    unplayed_weeks = reg.loc[reg["home_score"].isna(), "week"]
    if unplayed_weeks.empty:
        return int(reg["week"].max())  # season's done; report the final week
    return int(unplayed_weeks.min())


def run_week(week: int | None = None, season: int = CURRENT_SEASON) -> str:
    if week is None:
        week = get_current_week(season)
    predictions = score_week(week, season)
    props = score_props(week, season)
    try:
        news = fetch_news()
    except requests.RequestException as e:
        print(f"Warning: couldn't fetch live news ({e}); reporting without it.")
        news = None

    # Grade and log BEFORE building the report: the report's Track Record, Past
    # Weeks archive and recap read what these steps write, so building it first
    # (as this used to) left every report one full run behind on results.
    _, newly_graded_games = log_week(predictions, week, season)
    print(f"Logged predictions -> {LOG_PATH}")
    _, newly_graded_props = log_props_week(props, week, season, started=started_teams(predictions))
    print(f"Logged props -> {PROPS_LOG_PATH}")

    # Week 1 Audit & Tuning Plan Phase 6: a separate, append-only ledger
    # alongside the mutable CSV logs above -- see model/prediction_log.py's
    # docstring for why the two coexist rather than one replacing the other.
    ledger_path, n_logged = log_predictions(predictions, week, season)
    print(f"Logged {n_logged} prediction(s) to append-only ledger -> {ledger_path}")

    _generate_recaps(newly_graded_games, newly_graded_props)

    return build_report(predictions, week, season, props, news)


def _generate_recaps(newly_graded_games: pd.DataFrame, newly_graded_props: pd.DataFrame) -> None:
    """One recap per distinct (season, week) that had anything newly
    graded this run -- a bye week or a late catch-up run can grade more
    than one week's games at once, and each gets its own recap rather
    than one blended paragraph. See report/recap.py's docstring for why
    this only ever runs on the just-graded subset, never full history."""
    from report.recap import build_recap, save_recap

    weeks = set(map(tuple, newly_graded_games[["season", "week"]].drop_duplicates().to_numpy())) | \
        set(map(tuple, newly_graded_props[["season", "week"]].drop_duplicates().to_numpy()))
    for season, week in sorted(weeks):
        games_slice = newly_graded_games[(newly_graded_games["season"] == season) & (newly_graded_games["week"] == week)]
        props_slice = newly_graded_props[(newly_graded_props["season"] == season) & (newly_graded_props["week"] == week)]
        text = build_recap(games_slice, props_slice, int(season), int(week))
        if text:
            save_recap(int(season), int(week), text)
            print(f"Generated recap for Week {week}, {season} -> {text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, default=None,
                         help="defaults to the current week if omitted")
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    args = parser.parse_args()

    run_week(args.week, args.season)


if __name__ == "__main__":
    main()
