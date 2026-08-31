# tests for the training-time prior-season fallback (Week 1 Audit & Tuning Plan Phase 2)
"""model/train.py's _team_stats_with_fallback() exists because every
season's week 1 has no in-season rolling history by construction
(data/team_stats.py's expanding().shift(1) always produces NaN for a
team's first row of a season), and build_feature_frame() had no
fallback for that at all -- unlike model/predict.py's
get_pregame_stats()/_fallback_stats(), which already substitutes a
team's final prior-season row for exactly this case at live scoring
time. Verified empirically against the real 10-season cache: before
this fix, literally zero week-1 games (across all 10 seasons) survived
into training, meaning the model had never seen a real Week 1 game or
the fallback-shaped input it's actually asked to use the moment it's
deployed -- for a project whose entire point is scoring Week 1, this
was the single most consequential finding of the whole audit.
"""

import numpy as np
import pandas as pd

from model.train import _team_stats_with_fallback


def _row(team, season, week, is_home, epa=0.1, games_played=0):
    return {
        "season": season, "week": week, "team": team, "opponent": "OPP",
        "is_home": is_home, "rest_days": 7, "div_game": 0,
        "off_epa_per_play_avg": epa, "games_played": games_played,
    }


def test_week_1_falls_back_to_prior_seasons_final_row():
    team_stats = pd.DataFrame([
        _row("SEA", 2024, 16, True, epa=0.12, games_played=16),   # 2024's final row
        _row("SEA", 2025, 1, True, epa=np.nan, games_played=0),   # 2025 week 1: no history yet
        _row("SEA", 2025, 2, True, epa=0.08, games_played=1),     # 2025 week 2: real value already
    ])
    result = _team_stats_with_fallback(team_stats).set_index(["season", "week"])
    assert result.loc[(2025, 1), "off_epa_per_play_avg"] == 0.12  # borrowed from 2024's final row
    assert result.loc[(2025, 2), "off_epa_per_play_avg"] == 0.08  # untouched, already real


def test_no_prior_season_available_stays_null_not_guessed():
    """The very first cached season has nothing to fall back to -- must
    stay null (and get dropped by build_feature_frame()'s own dropna
    downstream), never a fabricated value."""
    team_stats = pd.DataFrame([_row("SEA", 2016, 1, True, epa=np.nan)])
    result = _team_stats_with_fallback(team_stats)
    assert pd.isna(result["off_epa_per_play_avg"].iloc[0])


def test_games_played_is_never_borrowed_from_the_fallback():
    """games_played has different semantics ("games so far THIS season's
    window") and isn't a model feature at all -- it must read the
    season's own real value (0 at week 1), never last season's count,
    even though every OTHER stat column on that same row does fall back."""
    team_stats = pd.DataFrame([
        _row("SEA", 2024, 16, True, epa=0.12, games_played=16),
        _row("SEA", 2025, 1, True, epa=np.nan, games_played=0),
    ])
    result = _team_stats_with_fallback(team_stats).set_index(["season", "week"])
    assert result.loc[(2025, 1), "off_epa_per_play_avg"] == 0.12  # this DID fall back
    assert result.loc[(2025, 1), "games_played"] == 0  # this did NOT


def test_different_teams_do_not_cross_contaminate():
    team_stats = pd.DataFrame([
        _row("SEA", 2024, 16, True, epa=0.12),
        _row("NE", 2024, 16, True, epa=0.55),
        _row("SEA", 2025, 1, True, epa=np.nan),
        _row("NE", 2025, 1, True, epa=np.nan),
    ])
    result = _team_stats_with_fallback(team_stats).set_index(["team", "season", "week"])
    assert result.loc[("SEA", 2025, 1), "off_epa_per_play_avg"] == 0.12
    assert result.loc[("NE", 2025, 1), "off_epa_per_play_avg"] == 0.55


def test_fallback_applies_per_column_not_all_or_nothing():
    """The real shape of the bug this was refined to catch:
    ats_win_pct_last5 is deliberately grouped by team alone (not
    team+season) in data/team_stats.py, so it's already real at week 1
    on its own -- an "is the WHOLE row null" gate would skip every
    week-1 row entirely just because that one column has a value. Each
    column must be filled independently: an already-real column stays
    untouched, a still-null column on the SAME row still gets its own
    fallback."""
    team_stats = pd.DataFrame([
        {**_row("SEA", 2024, 16, True, epa=0.12), "off_ypp_avg": 5.5},
        {**_row("SEA", 2025, 1, True, epa=0.03), "off_ypp_avg": np.nan},  # epa real, ypp null
    ])
    result = _team_stats_with_fallback(team_stats).set_index(["season", "week"])
    assert result.loc[(2025, 1), "off_epa_per_play_avg"] == 0.03  # already real, untouched
    assert result.loc[(2025, 1), "off_ypp_avg"] == 5.5  # null on this row, filled from 2024
