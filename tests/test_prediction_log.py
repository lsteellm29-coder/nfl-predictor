# tests for the append-only prediction ledger (Week 1 Audit & Tuning Plan Phase 6)
import hashlib
import json

import pandas as pd
import pytest

from model.prediction_log import _kickoff_utc, log_predictions, model_version_hash


@pytest.fixture
def isolated_log(tmp_path, monkeypatch):
    """log_predictions() writes to the module-level PREDICTIONS_LOG_PATH
    constant -- pointed at a scratch file per test so runs never touch
    (or depend on) the real logs/predictions.jsonl."""
    path = tmp_path / "predictions.jsonl"
    monkeypatch.setattr("model.prediction_log.PREDICTIONS_LOG_PATH", str(path))
    return path


def _fake_model_file(tmp_path, content: bytes, name: str = "fake_model.joblib") -> str:
    path = tmp_path / name
    path.write_bytes(content)
    return str(path)


def test_model_version_hash_matches_manual_sha256(tmp_path):
    model_path = _fake_model_file(tmp_path, b"some model bytes")
    expected = hashlib.sha256(b"some model bytes").hexdigest()[:12]
    assert model_version_hash(model_path) == expected


def test_model_version_hash_changes_when_the_file_changes(tmp_path):
    a = _fake_model_file(tmp_path, b"version A", name="a.joblib")
    b = _fake_model_file(tmp_path, b"version B", name="b.joblib")
    assert model_version_hash(a) != model_version_hash(b)


def test_kickoff_utc_converts_eastern_afternoon_slot_correctly():
    # 13:00 ET in September (EDT, UTC-4) -> 17:00 UTC.
    result = _kickoff_utc("2025-09-07", "13:00")
    assert result == "2025-09-07T17:00:00+00:00"


def test_kickoff_utc_returns_none_rather_than_raising_when_gametime_missing():
    """A flex-scheduled or not-yet-time-confirmed game still has a real
    spread_line/home_win_prob and must not crash the whole logging call
    over a missing, non-essential schedule field."""
    assert _kickoff_utc("2025-09-07", float("nan")) is None
    assert _kickoff_utc(None, "13:00") is None


def _predictions_df(rows):
    return pd.DataFrame(rows)


def test_log_predictions_writes_one_line_per_gradeable_game(isolated_log, tmp_path):
    model_path = _fake_model_file(tmp_path, b"test model")
    predictions = _predictions_df([
        {
            "game_id": "2026_01_NE_SEA", "gameday": "2026-09-09", "gametime": "20:20",
            "home_team": "SEA", "away_team": "NE", "home_win_prob": 0.58,
            "implied_spread": 2.5, "spread_line": 1.5, "total_line": 44.5,
            "top_factors": [("off_elo_diff", 1.2), ("market_spread", 0.8), ("rest_diff", -0.1)],
        },
        {
            # no prediction made (e.g. no history for one team) -- must be skipped
            "game_id": "2026_01_XX_YY", "gameday": "2026-09-09", "gametime": "13:00",
            "home_team": "YY", "away_team": "XX", "home_win_prob": float("nan"),
            "implied_spread": None, "spread_line": None, "total_line": None, "top_factors": None,
        },
    ])
    path, n_written = log_predictions(predictions, week=1, season=2026, model_path=model_path)
    assert n_written == 1

    with open(path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 1
    record = lines[0]
    assert record["game_id"] == "2026_01_NE_SEA"
    assert record["home_win_prob"] == 0.58
    assert record["market_spread"] == 1.5
    assert record["model_version"] == model_version_hash(model_path)
    assert record["kickoff_utc"] == "2026-09-10T00:20:00+00:00"


def test_log_predictions_appends_never_truncates(isolated_log, tmp_path):
    """The whole point of this ledger: a second call adds to the file,
    it never rewrites or drops what a prior call already wrote."""
    model_path = _fake_model_file(tmp_path, b"test model")
    game = {
        "game_id": "2026_01_NE_SEA", "gameday": "2026-09-09", "gametime": "20:20",
        "home_team": "SEA", "away_team": "NE", "home_win_prob": 0.58,
        "implied_spread": 2.5, "spread_line": 1.5, "total_line": 44.5, "top_factors": [],
    }
    log_predictions(_predictions_df([game]), week=1, season=2026, model_path=model_path)
    log_predictions(_predictions_df([{**game, "spread_line": 2.0}]), week=1, season=2026, model_path=model_path)

    with open(isolated_log) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 2  # both snapshots kept, neither overwrote the other
    assert [r["market_spread"] for r in lines] == [1.5, 2.0]


def test_log_predictions_does_not_crash_on_a_game_with_no_kickoff_time(isolated_log, tmp_path, capsys):
    model_path = _fake_model_file(tmp_path, b"test model")
    game = {
        "game_id": "2026_18_XX_YY", "gameday": "2026-01-04", "gametime": None,
        "home_team": "YY", "away_team": "XX", "home_win_prob": 0.55,
        "implied_spread": 1.0, "spread_line": 1.0, "total_line": 41.0, "top_factors": [],
    }
    path, n_written = log_predictions(_predictions_df([game]), week=18, season=2026, model_path=model_path)
    assert n_written == 1

    with open(path) as f:
        record = json.loads(f.readline())
    assert record["kickoff_utc"] is None
    assert "no confirmed kickoff time" in capsys.readouterr().out
