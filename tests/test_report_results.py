# Report: final scores on cards, pre-game-pick lookup, season-to-date hub frames, lines note.
import json

import numpy as np
import pandas as pd

import data.live_stats as live_stats
import model.prediction_log as prediction_log
import report.build_report as br
from data.live_stats import season_to_date_frames


def _game(**kw):
    base = {"game_id": "2026_02_DET_BUF", "home_team": "BUF", "away_team": "DET",
            "home_score": 41.0, "away_score": 31.0, "pre_game_pick": "BUF"}
    return pd.Series({**base, **kw})


def test_no_result_block_for_an_unplayed_game():
    assert br._final_result_html(_game(home_score=np.nan, away_score=np.nan)) == ""


def test_correct_pick_is_badged_correct():
    html = br._final_result_html(_game())
    assert "DET 31 &ndash; BUF 41" in html and "is-correct" in html and "Model picked BUF: correct" in html


def test_missed_pick_is_badged_missed():
    html = br._final_result_html(_game(pre_game_pick="DET"))
    assert "is-wrong" in html and "Model picked DET: missed" in html


def test_played_game_with_no_pre_game_pick_says_so_instead_of_grading_a_hindsight_number():
    for missing in (None, np.nan):
        html = br._final_result_html(_game(pre_game_pick=missing))
        assert "no pre-game pick on record" in html
        assert "is-correct" not in html and "is-wrong" not in html


def test_tie_is_not_graded():
    html = br._final_result_html(_game(home_score=20.0, away_score=20.0, pre_game_pick="BUF"))
    assert "tie" in html and "is-correct" not in html and "is-wrong" not in html


def _ledger(tmp_path, monkeypatch, records):
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    monkeypatch.setattr(prediction_log, "PREDICTIONS_LOG_PATH", str(path))
    return str(path)


def _rec(game_id, logged, kickoff, p_home, home="BUF", away="DET"):
    return {"game_id": game_id, "logged_at_utc": logged, "kickoff_utc": kickoff, "home_win_prob": p_home,
            "home_team": home, "away_team": away}


def test_pre_game_picks_ignores_records_logged_after_kickoff_and_takes_the_earliest(tmp_path, monkeypatch):
    path = _ledger(tmp_path, monkeypatch, [
        _rec("g1", "2026-09-20T16:00:00+00:00", "2026-09-20T17:00:00+00:00", 0.40),   # pre-kickoff: away
        _rec("g1", "2026-09-20T16:30:00+00:00", "2026-09-20T17:00:00+00:00", 0.90),   # later, still pre-kickoff
        _rec("g2", "2026-09-20T18:00:00+00:00", "2026-09-20T17:00:00+00:00", 0.90),   # AFTER kickoff: not a pick
        _rec("g3", "2026-09-20T16:00:00+00:00", None, 0.90),                          # unknown kickoff: can't verify
    ])
    assert prediction_log.pre_game_picks(path) == {"g1": "DET"}


def test_annotate_results_attaches_the_pick(tmp_path, monkeypatch):
    _ledger(tmp_path, monkeypatch, [_rec("2026_02_DET_BUF", "2026-09-17T12:00:00+00:00",
                                         "2026-09-18T00:15:00+00:00", 0.6)])
    preds = pd.DataFrame([{"game_id": "2026_02_DET_BUF"}, {"game_id": "2026_02_CAR_ATL"}])
    out = br.annotate_results(preds)
    assert out.loc[0, "pre_game_pick"] == "BUF" and pd.isna(out.loc[1, "pre_game_pick"])
    assert "pre_game_pick" not in preds.columns          # input untouched


def test_lines_note_only_appears_for_the_fallback():
    assert br.lines_source_note(pd.DataFrame({"lines_source": ["odds_api", "odds_api"]})) == ""
    assert br.lines_source_note(pd.DataFrame({"x": [1]})) == ""
    assert "nflverse" in br.lines_source_note(pd.DataFrame({"lines_source": ["nflverse"]}))


# ---- team hubs: season-to-date frames --------------------------------------

def _sched(season, played_through):
    rows = [{"season": season, "week": w, "game_type": "REG", "home_team": "A", "away_team": "B",
             "home_score": 20.0 if w <= played_through else np.nan,
             "away_score": 10.0 if w <= played_through else np.nan} for w in range(1, 5)]
    return pd.DataFrame(rows)


def test_season_to_date_frames_extends_the_cached_history(monkeypatch):
    seen = {}

    def fake_rolling_as_of(schedule, pbp, season, week, prior_schedule, prior_pbp):
        seen["week"] = week
        rows = pd.DataFrame({"season": season, "week": [1, 2, 3, 4], "team": "A", "off_epa_per_play_avg": 0.1})
        return rows, schedule

    monkeypatch.setattr(live_stats, "_rolling_as_of", fake_rolling_as_of)
    cached_stats = pd.DataFrame({"season": [2025, 2025], "week": [1, 2], "team": "A", "off_epa_per_play_avg": 0.0})
    cached_sched = _sched(2025, 4)

    stats, sched = season_to_date_frames(2026, cached_stats, cached_sched, schedule=_sched(2026, 2),
                                         pbp=pd.DataFrame(), prior_schedule=pd.DataFrame(), prior_pbp=pd.DataFrame())

    assert seen["week"] == 3                                        # cut at the week after the latest played
    s26 = stats[stats["season"] == 2026]
    assert sorted(s26["week"]) == [1, 2, 3]                          # week 4 row would just repeat -> dropped
    assert (stats["season"] == 2025).sum() == 2                      # history kept
    played26 = sched[(sched["season"] == 2026) & sched["home_score"].notna()]
    assert len(played26) == 2 and (sched["season"] == 2025).sum() == 4


def test_season_to_date_frames_is_a_no_op_before_any_game_is_played():
    cached_stats = pd.DataFrame({"season": [2025], "week": [1], "team": "A"})
    cached_sched = _sched(2025, 4)
    stats, sched = season_to_date_frames(2026, cached_stats, cached_sched, schedule=_sched(2026, 0))
    assert stats is cached_stats and sched is cached_sched
