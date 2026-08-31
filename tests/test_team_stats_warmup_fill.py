# tests for the opponent-adjustment warmup-gap fallback (Week 1 Audit & Tuning Plan Phase 2)
"""data/team_stats.py's _fill_adjusted_warmup_gap() exists because the
3-pass iterative opponent adjustment (data/opponent_adjust.py) compounds
its own expanding().shift(1) warmup requirement across passes -- by
pass 3, a team's *_adj_avg columns don't produce a real number until
roughly week 5 of a season. Verified empirically against the real
10-season cache: every week 1-4 game (about 24% of all training rows)
was silently dropped by model/train.py's `.dropna(subset=FEATURE_COLS)`
before this fix, meaning the model never trained on a single real Week
1 game for a project whose entire deployment target is Week 1.
"""

import numpy as np
import pandas as pd

from data.opponent_adjust import OPPONENT_STAT
from data.team_stats import _fill_adjusted_warmup_gap


def _row(week: int, **overrides) -> dict:
    """A complete one-row record with all 4 OPPONENT_STAT columns
    (raw + adjusted) present, defaulted to a harmless real value/NaN
    pair -- _fill_adjusted_warmup_gap() always operates on all 4, so
    every test fixture needs all 4 present, same as the real
    build_rolling_team_stats() output always has."""
    row = {"team": "SEA", "week": week}
    for stat_col in OPPONENT_STAT:
        row[f"{stat_col}_avg"] = 0.1
        row[f"{stat_col}_adj_avg"] = np.nan
    row.update(overrides)
    return row


def test_fills_null_adjusted_value_with_the_raw_value():
    result = pd.DataFrame([
        _row(1, off_epa_per_play_avg=0.05, off_epa_per_play_adj_avg=np.nan),
        _row(5, off_epa_per_play_avg=0.08, off_epa_per_play_adj_avg=0.11),
    ])
    filled = _fill_adjusted_warmup_gap(result)
    assert filled["off_epa_per_play_adj_avg"].tolist() == [0.05, 0.11]


def test_does_not_touch_a_real_already_present_adjusted_value():
    """The whole point: once the adjustment IS warmed up, its own
    (more-refined) number must survive untouched, never silently
    replaced by the cruder raw figure."""
    result = pd.DataFrame([_row(6, off_epa_per_play_avg=0.05, off_epa_per_play_adj_avg=0.11)])
    filled = _fill_adjusted_warmup_gap(result)
    assert filled["off_epa_per_play_adj_avg"].iloc[0] == 0.11


def test_covers_all_four_opponent_stat_columns():
    row = _row(1)
    for stat_col in OPPONENT_STAT:
        row[f"{stat_col}_avg"] = 0.42
    result = pd.DataFrame([row])
    filled = _fill_adjusted_warmup_gap(result)
    for stat_col in OPPONENT_STAT:
        assert filled[f"{stat_col}_adj_avg"].iloc[0] == 0.42


def test_both_raw_and_adjusted_null_stays_null():
    """A team's genuine first-ever game (no raw rolling average exists
    either) has nothing to fall back to -- must stay null, not become a
    fabricated 0 or any other guessed value."""
    result = pd.DataFrame([_row(1, off_epa_per_play_avg=np.nan, off_epa_per_play_adj_avg=np.nan)])
    filled = _fill_adjusted_warmup_gap(result)
    assert pd.isna(filled["off_epa_per_play_adj_avg"].iloc[0])


def test_does_not_mutate_the_input_dataframe():
    result = pd.DataFrame([_row(1, off_epa_per_play_avg=0.05, off_epa_per_play_adj_avg=np.nan)])
    _fill_adjusted_warmup_gap(result)
    assert pd.isna(result["off_epa_per_play_adj_avg"].iloc[0])  # original untouched
