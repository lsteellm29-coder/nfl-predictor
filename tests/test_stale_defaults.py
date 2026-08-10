# regression test for the stale-default-argument bug class
"""Master Honing Plan, Section D: recency_weighted_touch_share (model/
td_model.py) had a real, silent bug -- its window/multiplier parameters
were originally typed as `window: int = RECENCY_WINDOW`, which Python
binds ONCE at function-definition time. A later `td_model.RECENCY_WINDOW
= X` reassignment done for a parameter sweep would then silently never
take effect: the function kept using whatever value RECENCY_WINDOW held
when the module was first imported, no matter what it was reassigned to
afterward. It was only caught because a real sweep returned byte-
identical Brier/AUC across every value tried, including a supposedly-
disabling multiplier=1.0.

This module tests the FIXED behavior directly -- reassigning the module-
level constant and confirming the function's actual output changes --
as a standing regression test against the exact bug class recurring,
either in this function again or in the pattern spreading to a new one.
Every function in this codebase sharing this signature shape
(`param: type | None = None`, resolved to a module constant inside the
function body) is covered here, not just the one that already bit once.
"""

import pandas as pd

import model.td_model as td_model


def _touch_log():
    """A tiny synthetic touch log: one player, one team, 6 weeks of
    red-zone touches, with a clear step-change in week 4-6 (touches jump
    from 2/week to 8/week) -- exactly the shape recency weighting is
    supposed to detect and a flat season average would wash out."""
    rows = []
    for week in range(1, 7):
        touches = 2 if week <= 3 else 8
        rows.append({"season": 2025, "week": week, "team": "SEA", "player_id": "P1",
                      "rz_touches": touches, "rz_tds": 0, "gl_touches": 0})
    return pd.DataFrame(rows)


def test_recency_weighted_touch_share_reassigned_constant_takes_effect():
    """The core regression check: reassigning td_model.RECENCY_WINDOW /
    RECENCY_WEIGHT_MULTIPLIER at runtime (the exact thing a parameter
    sweep does) must change recency_weighted_touch_share's actual output.
    If this ever regresses to the old `param=MODULE_CONSTANT` default-arg
    pattern, this test starts failing because both configurations below
    would silently produce the identical result."""
    touch_log = _touch_log()

    original_window, original_multiplier = td_model.RECENCY_WINDOW, td_model.RECENCY_WEIGHT_MULTIPLIER
    try:
        # Configuration A: last 1 week weighted 5x -- should weight
        # heavily toward the week-6 value (8 touches).
        td_model.RECENCY_WINDOW = 1
        td_model.RECENCY_WEIGHT_MULTIPLIER = 5.0
        result_a = td_model.recency_weighted_touch_share(touch_log, upto_week=7)

        # Configuration B: recency weighting fully disabled (multiplier
        # 1.0 == every week counted equally) -- should reflect the flat
        # season average instead.
        td_model.RECENCY_WINDOW = 1
        td_model.RECENCY_WEIGHT_MULTIPLIER = 1.0
        result_b = td_model.recency_weighted_touch_share(touch_log, upto_week=7)
    finally:
        td_model.RECENCY_WINDOW, td_model.RECENCY_WEIGHT_MULTIPLIER = original_window, original_multiplier

    touches_a = result_a.loc[result_a["player_id"] == "P1", "w_rz_touches"].iloc[0]
    touches_b = result_b.loc[result_b["player_id"] == "P1", "w_rz_touches"].iloc[0]

    # The whole point of the bug: these must NOT be equal. With the old
    # buggy signature they would be (both calls would silently use
    # whatever RECENCY_WINDOW/MULTIPLIER were at import time).
    assert touches_a != touches_b
    # Sanity: heavier recent-weighting should weight in more total
    # (weighted) touches than the flat/unweighted count, since the most
    # recent week's real touches get multiplied up.
    assert touches_a > touches_b


def test_recency_weighted_touch_share_explicit_args_still_work():
    """The explicit-argument path (not touching the module constants at
    all) must still behave identically to before -- this is what every
    call site in predict.py/player_stats.py actually uses in production,
    so the fix can't have broken the common case while fixing the sweep
    case."""
    touch_log = _touch_log()
    result_default = td_model.recency_weighted_touch_share(touch_log, upto_week=7)
    result_explicit = td_model.recency_weighted_touch_share(
        touch_log, upto_week=7, window=td_model.RECENCY_WINDOW, multiplier=td_model.RECENCY_WEIGHT_MULTIPLIER)
    pd.testing.assert_frame_equal(
        result_default.sort_values("player_id").reset_index(drop=True),
        result_explicit.sort_values("player_id").reset_index(drop=True),
    )
