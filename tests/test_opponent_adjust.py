# tests for data/opponent_adjust.py's iterative DVOA-style adjustment
"""Covers the exact formulas the module's own docstring cites numbers
for: the subtract-and-recenter adjustment, the damped blend, and the
documented n_passes=1 invariant ("reproduces the original single-pass
behavior exactly")."""

import pandas as pd

from data.opponent_adjust import OPPONENT_STAT, _blend, add_opponent_adjusted_columns, iterative_opponent_adjust


def test_blend_is_a_weighted_average():
    # alpha=0.3 -> 30% new, 70% previous, per the module's own damping formula
    new = pd.DataFrame({"season": [2025], "week": [2], "team": ["A"], "off_epa_per_play_adj_avg": [10.0],
                         "def_epa_per_play_adj_avg": [0.0], "off_ypp_adj_avg": [0.0], "def_ypp_adj_avg": [0.0]})
    prev = pd.DataFrame({"season": [2025], "week": [2], "team": ["A"], "off_epa_per_play_adj_avg": [0.0],
                          "def_epa_per_play_adj_avg": [0.0], "off_ypp_adj_avg": [0.0], "def_ypp_adj_avg": [0.0]})
    out = _blend(new, prev, alpha=0.3)
    assert out["off_epa_per_play_adj_avg"].iloc[0] == 3.0


def test_blend_alpha_1_ignores_previous_pass_entirely():
    new = pd.DataFrame({"season": [2025], "week": [2], "team": ["A"], "off_epa_per_play_adj_avg": [5.0],
                         "def_epa_per_play_adj_avg": [-2.0], "off_ypp_adj_avg": [1.0], "def_ypp_adj_avg": [1.0]})
    prev = pd.DataFrame({"season": [2025], "week": [2], "team": ["A"], "off_epa_per_play_adj_avg": [100.0],
                          "def_epa_per_play_adj_avg": [100.0], "off_ypp_adj_avg": [100.0], "def_ypp_adj_avg": [100.0]})
    out = _blend(new, prev, alpha=1.0)
    assert out["off_epa_per_play_adj_avg"].iloc[0] == 5.0
    assert out["def_epa_per_play_adj_avg"].iloc[0] == -2.0


def _game_stats_cols(team, opponent, off_epa, def_epa, off_ypp=5.0, def_ypp=5.0):
    return {"team": team, "opponent": opponent, "off_epa_per_play": off_epa,
            "def_epa_per_play": def_epa, "off_ypp": off_ypp, "def_ypp": def_ypp}


def test_add_opponent_adjusted_columns_matches_hand_computed_formula():
    """adjusted = raw - opponent_pregame_rolling_avg + league_avg, per the
    module's own docstring formula -- verified by hand for team A facing
    team B's known (rolling) defense."""
    games = pd.DataFrame([
        {"season": 2025, "week": 3, **_game_stats_cols("A", "B", off_epa=0.20, def_epa=0.05)},
        {"season": 2025, "week": 3, **_game_stats_cols("B", "A", off_epa=0.00, def_epa=0.10)},
    ])
    rolling = pd.DataFrame([
        {"season": 2025, "week": 3, "team": "A", "off_epa_per_play_avg": 0.10, "def_epa_per_play_avg": -0.10,
         "off_ypp_avg": 5.0, "def_ypp_avg": 5.0},
        {"season": 2025, "week": 3, "team": "B", "off_epa_per_play_avg": -0.05, "def_epa_per_play_avg": 0.05,
         "off_ypp_avg": 5.0, "def_ypp_avg": 5.0},
    ])
    out = add_opponent_adjusted_columns(games, rolling, rolling_col_suffix="_avg")

    league_avg_def = rolling["def_epa_per_play_avg"].mean()  # (-0.10 + 0.05) / 2 = -0.025
    row_a = out[out["team"] == "A"].iloc[0]
    # A's off_epa_per_play (0.20) minus B's def_epa_per_play_avg (0.05), plus the league avg def rate.
    expected_a = 0.20 - 0.05 + league_avg_def
    assert abs(row_a["off_epa_per_play_adj"] - expected_a) < 1e-9


def test_add_opponent_adjusted_columns_propagates_nan_for_missing_opponent_data():
    """A team facing an opponent with no rolling entry yet (that
    opponent's own first game of the season) should propagate NaN rather
    than silently guessing a value -- the module's own stated contract."""
    games = pd.DataFrame([{"season": 2025, "week": 1, **_game_stats_cols("A", "B", off_epa=0.20, def_epa=0.05)}])
    rolling = pd.DataFrame(columns=["season", "week", "team", "off_epa_per_play_avg",
                                     "def_epa_per_play_avg", "off_ypp_avg", "def_ypp_avg"])
    out = add_opponent_adjusted_columns(games, rolling, rolling_col_suffix="_avg")
    assert pd.isna(out["off_epa_per_play_adj"].iloc[0])


def _multi_week_games():
    """Two teams playing each other twice (weeks 1 and 2) -- enough for
    roll_opponent_adjusted's expanding().shift(1) to produce a real
    (non-NaN) value on the second week, needed to exercise
    iterative_opponent_adjust end to end."""
    rows = []
    for week, (a_off, a_def, b_off, b_def) in enumerate(
            [(0.20, 0.05, 0.00, 0.10), (0.15, 0.00, 0.05, 0.08)], start=1):
        rows.append({"season": 2025, "week": week, **_game_stats_cols("A", "B", a_off, a_def)})
        rows.append({"season": 2025, "week": week, **_game_stats_cols("B", "A", b_off, b_def)})
    return pd.DataFrame(rows)


def _multi_week_rolling():
    rows = []
    for week in (1, 2):
        rows.append({"season": 2025, "week": week, "team": "A", "off_epa_per_play_avg": 0.10,
                     "def_epa_per_play_avg": -0.05, "off_ypp_avg": 5.0, "def_ypp_avg": 5.0})
        rows.append({"season": 2025, "week": week, "team": "B", "off_epa_per_play_avg": -0.02,
                     "def_epa_per_play_avg": 0.06, "off_ypp_avg": 5.0, "def_ypp_avg": 5.0})
    return pd.DataFrame(rows)


def test_iterative_opponent_adjust_n_passes_1_matches_manual_single_pass():
    """The module's own docstring states n_passes=1 "reproduces the
    original single-pass behavior exactly" -- verified directly against
    add_opponent_adjusted_columns + roll_opponent_adjusted called by hand
    once, not just trusted from the docstring."""
    from data.opponent_adjust import roll_opponent_adjusted

    games, rolling = _multi_week_games(), _multi_week_rolling()

    manual = roll_opponent_adjusted(add_opponent_adjusted_columns(games, rolling, "_avg"))
    iterative = iterative_opponent_adjust(games, rolling, n_passes=1)

    pd.testing.assert_frame_equal(
        manual.sort_values(["team", "week"]).reset_index(drop=True),
        iterative.sort_values(["team", "week"]).reset_index(drop=True),
    )


def test_iterative_opponent_adjust_multiple_passes_runs_and_stays_finite():
    """3-pass default shouldn't diverge to inf/absurd values on a small,
    well-behaved input -- a coarse guard against the exact undamped-
    divergence failure mode the module's docstring documents finding."""
    games, rolling = _multi_week_games(), _multi_week_rolling()
    out = iterative_opponent_adjust(games, rolling, n_passes=3)
    numeric_cols = [f"{c}_adj_avg" for c in OPPONENT_STAT]
    finite_or_nan = out[numeric_cols].apply(lambda s: s.dropna().abs().max() < 100 if s.notna().any() else True)
    assert finite_or_nan.all()
