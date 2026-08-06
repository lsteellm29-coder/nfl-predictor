# the one command that runs the full pipeline
"""The one command from Section 5 of the spec: run_week.py --week 5

Pulls that week's schedule + current rolling stats + the current betting
line, runs the model, prints/saves the weekly report, and appends the week's
predictions to logs/season_results.csv -- grading any previously-logged
predictions whose games have since finished, so accuracy is trackable over
the season (Section 4.4 / Section 5.6).
"""

import argparse
import os

import nfl_data_py as nfl
import pandas as pd

from config import CURRENT_SEASON
from model.player_stats import score_props
from model.predict import score_week
from report.build_report import build_report

LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "season_results.csv")

LOG_COLS = [
    "season", "week", "away_team", "home_team",
    "predicted_winner", "home_win_prob", "vegas_spread", "model_spread", "edge",
    "actual_away_score", "actual_home_score", "actual_winner",
    "correct", "model_beat_market",
]


def _load_log() -> pd.DataFrame:
    if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0:
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=LOG_COLS)


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
        actual_winner = row["home_team"] if home_score > away_score else row["away_team"]

        log_df.loc[idx, "actual_home_score"] = home_score
        log_df.loc[idx, "actual_away_score"] = away_score
        log_df.loc[idx, "actual_winner"] = actual_winner
        log_df.loc[idx, "correct"] = actual_winner == row["predicted_winner"]

        if pd.notna(row["vegas_spread"]) and pd.notna(row["edge"]) and row["edge"] != 0:
            ats_margin = (home_score - away_score) - row["vegas_spread"]
            if ats_margin != 0:  # exclude pushes
                home_covered = ats_margin > 0
                model_favored_home = row["edge"] > 0
                log_df.loc[idx, "model_beat_market"] = model_favored_home == home_covered

    return log_df


def log_week(predictions: pd.DataFrame, week: int, season: int) -> pd.DataFrame:
    log_df = _load_log()
    log_df = _grade_pending(log_df)

    rows = []
    for _, game in predictions.iterrows():
        if pd.isna(game.get("home_win_prob")):
            continue  # no prediction made for this game (no team history available)
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

    # re-running the same week overwrites its rows instead of duplicating them
    log_df = log_df[~((log_df["season"] == season) & (log_df["week"] == week))]
    log_df = pd.concat([log_df, pd.DataFrame(rows, columns=LOG_COLS)], ignore_index=True)

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_df.to_csv(LOG_PATH, index=False)
    return log_df


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
    path = build_report(predictions, week, season, props)
    log_week(predictions, week, season)
    print(f"Logged predictions -> {LOG_PATH}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, default=None,
                         help="defaults to the current week if omitted")
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    args = parser.parse_args()

    run_week(args.week, args.season)


if __name__ == "__main__":
    main()
