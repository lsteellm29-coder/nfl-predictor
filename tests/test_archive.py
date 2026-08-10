# tests for report/archive.py's past-weeks archive
import io

import pandas as pd

from report.archive import archive_html

COLS = ["season", "week", "away_team", "home_team", "predicted_winner", "home_win_prob",
        "actual_away_score", "actual_home_score", "actual_winner", "correct"]


def test_empty_log_renders_empty_state_not_a_crash():
    html_out = archive_html(pd.DataFrame(columns=COLS))
    assert "empty-state" in html_out


def test_archive_handles_csv_roundtripped_mixed_dtype():
    """Same class of bug as report/recap.py's: a real logs/season_results.csv
    read via pd.read_csv mixes graded (True/False) and still-pending
    (blank) rows in the same "correct" column, producing object dtype --
    archive_html() must not crash on that shape."""
    csv_text = (
        "season,week,away_team,home_team,predicted_winner,home_win_prob,"
        "actual_away_score,actual_home_score,actual_winner,correct\n"
        "2026,1,NE,SEA,SEA,0.61,17,24,SEA,True\n"
        "2026,1,DAL,NYG,NYG,0.30,14,21,NYG,True\n"
        "2026,2,KC,DEN,KC,0.55,,,,\n"
    )
    df = pd.read_csv(io.StringIO(csv_text))
    html_out = archive_html(df)
    assert "Week 1" in html_out
    assert "Week 2" not in html_out  # still pending, shouldn't appear in the archive
    assert "empty-state" not in html_out


def test_archive_shows_correct_record_per_week():
    df = pd.DataFrame([
        {**dict.fromkeys(COLS), "season": 2026, "week": 1, "away_team": "NE", "home_team": "SEA",
         "predicted_winner": "SEA", "home_win_prob": 0.61, "actual_away_score": 17, "actual_home_score": 24,
         "actual_winner": "SEA", "correct": True},
        {**dict.fromkeys(COLS), "season": 2026, "week": 1, "away_team": "DAL", "home_team": "NYG",
         "predicted_winner": "DAL", "home_win_prob": 0.30, "actual_away_score": 14, "actual_home_score": 21,
         "actual_winner": "NYG", "correct": False},
    ])
    html_out = archive_html(df)
    assert "1-1" in html_out
