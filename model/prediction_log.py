# Week 1 Audit & Tuning Plan Phase 6: append-only prediction ledger
"""An unlogged model is a toy; a logged one is a real track record. This
writes one JSONL line per graded game to logs/predictions.jsonl, opened
in 'a' (append) mode ONLY -- the file is never read back and rewritten,
so nothing already on disk can ever be edited or deleted by a later
call. That's what "never allow overwriting an existing prediction
record" means here in practice: the write path itself is incapable of
it, not just a convention this module promises to follow.

Multiple records for the same game_id across separate calls (e.g. a
Wednesday run vs. a Friday run, if the line has moved) are expected,
not a bug -- "market spread and total at time of prediction" only means
something if the ledger keeps every snapshot instead of collapsing to
whichever run happened most recently. grade.py (project root) is what
joins these snapshots to final scores once each game is over.
"""

import datetime as dt
import hashlib
import json
import os
from zoneinfo import ZoneInfo

import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
PREDICTIONS_LOG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "logs", "predictions.jsonl")
)

# nflverse's schedule `gametime` is documented Eastern Time regardless of
# the game's actual venue -- verified against 2025's Sao Paulo game
# (2025_01_KC_LAC, location="Neutral"), which still carries a plain
# ET-slot time (20:00) exactly like every domestic game, not a
# Brazil-local kickoff hour. Every kickoff below converts through this
# zone, never a venue-local one.
_SCHEDULE_TZ = ZoneInfo("America/New_York")


def model_version_hash(model_path: str = MODEL_PATH, length: int = 12) -> str:
    """sha256 of the deployed model.joblib's raw bytes, truncated for a
    readable but still collision-safe identifier -- a prediction logged
    against one hash is provably tied to one specific trained artifact,
    not just "whatever main() happened to save most recently.\""""
    with open(model_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return digest[:length]


def _kickoff_utc(gameday: str, gametime: str) -> str:
    naive = dt.datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M")
    localized = naive.replace(tzinfo=_SCHEDULE_TZ)
    return localized.astimezone(dt.timezone.utc).isoformat()


def log_predictions(predictions: pd.DataFrame, week: int, season: int,
                     model_path: str = MODEL_PATH) -> tuple[str, int]:
    """Appends one record per game that actually got a prediction (a game
    with no history for one of its teams has home_win_prob=None from
    score_week() and nothing gradeable to log). Returns (log path, count
    of records written this call)."""
    version = model_version_hash(model_path)
    logged_at = dt.datetime.now(dt.timezone.utc).isoformat()

    os.makedirs(os.path.dirname(PREDICTIONS_LOG_PATH), exist_ok=True)
    n_written = 0
    with open(PREDICTIONS_LOG_PATH, "a") as f:
        for _, game in predictions.iterrows():
            if pd.isna(game.get("home_win_prob")):
                continue
            record = {
                "logged_at_utc": logged_at,
                "game_id": game["game_id"],
                "season": season,
                "week": week,
                "kickoff_utc": _kickoff_utc(game["gameday"], game["gametime"]),
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_win_prob": float(game["home_win_prob"]),
                "predicted_spread": float(game["implied_spread"]),
                "market_spread": float(game["spread_line"]) if pd.notna(game.get("spread_line")) else None,
                "market_total": float(game["total_line"]) if pd.notna(game.get("total_line")) else None,
                "model_version": version,
                "top_factors": [[name, float(value)] for name, value in (game.get("top_factors") or [])],
            }
            f.write(json.dumps(record) + "\n")
            n_written += 1
    return PREDICTIONS_LOG_PATH, n_written
