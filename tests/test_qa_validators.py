# tests for the pure-logic pieces of the QA validation layer
"""qa/validate_rosters.py's check_staleness is the one QA check that can
actually halt the pipeline on a real, broken roster pull (weekly_pipeline.sh
hard-fails on it) -- covered here directly against synthetic rosters of
varying sizes, including the exact MIN_ROSTER_SIZE boundary. qa/
validate_coverage.py's _position_counts is the building block its own
hard-fail (a team with zero props) is computed from."""

import pandas as pd

from qa.validate_coverage import _position_counts
from qa.validate_rosters import MIN_ROSTER_SIZE, check_staleness


def _roster(team_sizes: dict) -> pd.DataFrame:
    rows = []
    for team, n in team_sizes.items():
        for i in range(n):
            rows.append({"player_id": f"{team}-{i}", "player_name": f"Player {i}", "team": team, "position": "WR"})
    return pd.DataFrame(rows)


def test_check_staleness_empty_roster_is_a_problem():
    problems = check_staleness(pd.DataFrame(columns=["player_id", "player_name", "team", "position"]))
    assert len(problems) == 1
    assert "zero players" in problems[0]


def test_check_staleness_full_rosters_pass_clean():
    roster = _roster({"SEA": 53, "NE": 55})
    assert check_staleness(roster) == []


def test_check_staleness_flags_team_under_minimum():
    roster = _roster({"SEA": 53, "NE": 12})  # NE looks like a broken partial pull
    problems = check_staleness(roster)
    assert len(problems) == 1
    assert "NE" in problems[0]
    assert "SEA" not in problems[0]


def test_check_staleness_boundary_exactly_at_minimum_is_not_flagged():
    """MIN_ROSTER_SIZE itself (40) should NOT be flagged -- the check is
    strictly less-than, matching `counts < MIN_ROSTER_SIZE` in the
    source. A team at exactly 40 is a real, if thin, roster, not
    evidence of a broken pull."""
    roster = _roster({"SEA": MIN_ROSTER_SIZE})
    assert check_staleness(roster) == []


def test_check_staleness_boundary_one_under_minimum_is_flagged():
    roster = _roster({"SEA": MIN_ROSTER_SIZE - 1})
    problems = check_staleness(roster)
    assert len(problems) == 1
    assert "SEA" in problems[0]


def test_check_staleness_flags_every_team_under_minimum_not_just_the_first():
    roster = _roster({"SEA": 10, "NE": 15, "KC": 53})
    problems = check_staleness(roster)
    assert len(problems) == 2
    joined = " ".join(problems)
    assert "SEA" in joined and "NE" in joined and "KC" not in joined


def test_position_counts_dedupes_by_player_not_by_row():
    """A player who somehow appears twice for the same team (e.g. a
    market-matched row and a fallback row both keyed to them) must only
    count once toward coverage -- score_props() itself doesn't guarantee
    row-level uniqueness the way this QA check needs, so the dedup lives
    here."""
    props = pd.DataFrame([
        {"team": "SEA", "player": "Player A", "position": "WR"},
        {"team": "SEA", "player": "Player A", "position": "WR"},  # duplicate row, same player
        {"team": "SEA", "player": "Player B", "position": "RB"},
    ])
    counts = _position_counts(props, "SEA")
    assert counts["WR"] == 1
    assert counts["RB"] == 1


def test_position_counts_only_counts_the_requested_team():
    props = pd.DataFrame([
        {"team": "SEA", "player": "Player A", "position": "WR"},
        {"team": "NE", "player": "Player C", "position": "WR"},
    ])
    counts = _position_counts(props, "SEA")
    assert counts["WR"] == 1
    assert sum(counts.values()) == 1
