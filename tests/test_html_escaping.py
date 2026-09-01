# tests for the HTML-escaping gap found in a security review of report/ + build_artifact.py
"""Player names, coach names, and headlines flow from external data
(nflverse rosters/schedules, ESPN injury feeds, scraped news) into the
final HTML this project publishes (report/build_report.py's plain
report and build_artifact.py's self-contained artifact, now hosted
publicly via GitHub Pages). Several render paths embedded that external
text via plain f-strings/.format() with no html.escape() -- inconsistent
with report/cards.py, charts.py, compare.py, news_section.py, and
team_hub.py, which already escape correctly. This locks in the fix: a
name containing HTML-meaningful characters must never appear unescaped
in any of these render paths.
"""

from report.build_report import _coach_qb_line, _fmt_td_scorer
from report.leaderboard import leaderboard_html
from report.narrative import _phrase_coach_h2h, _phrase_qb_streak, _phrase_rb_streak, _phrase_team_change
from report.recap import _prop_sentences
from report.theme import td_chip_parts

import pandas as pd

PAYLOAD = "<script>alert(1)</script>"
ESCAPED = "&lt;script&gt;alert(1)&lt;/script&gt;"


def test_coach_qb_line_escapes_coach_and_qb_names():
    game = pd.Series({
        "away_team": "NE", "home_team": "SEA",
        "away_coach": PAYLOAD, "home_coach": "Normal Coach",
        "away_qb_name": PAYLOAD, "home_qb_name": "Normal QB",
    })
    result = _coach_qb_line(game)
    assert PAYLOAD not in result
    assert ESCAPED in result


def test_fmt_td_scorer_escapes_player_name():
    pred = {"player": PAYLOAD, "prob": 0.4, "def_factor": 1.0, "total_factor": 1.0, "player_share": 0.3, "games": 10}
    result = _fmt_td_scorer(pred)
    assert PAYLOAD not in result
    assert ESCAPED in result


def test_td_chip_parts_escapes_player_name():
    _, player = td_chip_parts({"player": PAYLOAD, "prob": 0.4})
    assert PAYLOAD not in player
    assert ESCAPED in player


def test_phrase_qb_streak_escapes_player_name():
    s = {"player": PAYLOAD, "team": "NE", "direction": "hot", "recent_avg": 0.2, "season_avg": 0.1}
    result = _phrase_qb_streak(s, lambda abbr: "New England Patriots")
    assert PAYLOAD not in result
    assert ESCAPED in result


def test_phrase_rb_streak_escapes_player_name():
    s = {"player": PAYLOAD, "team": "NE", "direction": "cold", "recent_avg": -0.1, "season_avg": 0.05}
    result = _phrase_rb_streak(s, lambda abbr: "New England Patriots")
    assert PAYLOAD not in result
    assert ESCAPED in result


def test_phrase_coach_h2h_escapes_both_coach_names():
    h = {"coach_a": PAYLOAD, "coach_b": "Normal Coach", "a_wins": 3, "b_wins": 1}
    result = _phrase_coach_h2h(h, PAYLOAD)
    assert PAYLOAD not in result
    assert ESCAPED in result


def test_phrase_team_change_escapes_new_coach_name():
    candidate = {"team": "NE", "coach_change": {"current_coach": PAYLOAD}, "unit_turnover": {}}
    result = _phrase_team_change(candidate, lambda abbr: "New England Patriots")
    assert PAYLOAD not in result
    assert ESCAPED in result


def test_recap_prop_sentences_escapes_player_name():
    props = pd.DataFrame([{
        "player": PAYLOAD, "model_correct": True, "actual_over": True, "edge": 0.4,
    }])
    result = " ".join(_prop_sentences(props, seed=0))
    assert PAYLOAD not in result
    assert ESCAPED in result


def test_leaderboard_html_escapes_prop_player_name():
    props = pd.DataFrame([{"player": PAYLOAD, "team": "NE", "edge": 0.4}])
    predictions = pd.DataFrame(columns=["edge"])
    result = leaderboard_html(predictions, props)
    assert PAYLOAD not in result
    assert ESCAPED in result
