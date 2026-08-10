# regression test for the edge-is-NaN-not-None sort bug
"""Caught live while testing the sort-by-edge feature (Master Honing Plan
round 2, item #3): a no-line prop card's `edge` field is None in the raw
dict model/player_stats.py builds, but a round trip through
pd.DataFrame(rows) (score_props()'s own return path) coerces that None
to float NaN once the column holds a mix of None and real floats --
`edge is not None` doesn't catch a NaN (`float('nan') is not None` is
True in Python), so the original code computed `abs(nan)` and rendered
the literal string "nan" into the data-edge HTML attribute, which then
broke numeric sorting in the browser (JS parseFloat('nan') is NaN, which
compares false against everything). This test locks in the fix
(pd.isna() instead of an identity check) for both the raw-None and the
DataFrame-round-tripped-NaN case."""

import pandas as pd

from report.cards import confidence_tier, game_pick_card_html, prop_card_html

BASE_ROW = {
    "player": "Test Player", "team": "SEA", "position": "WR", "stat": "anytime_td",
    "espn_id": None, "reasoning": None, "injury_status": None,
    "has_line": True, "model_over_prob": 0.3, "market_over_prob": 0.25,
}


def _extract_data_edge(html_str: str) -> str:
    marker = 'data-edge="'
    start = html_str.index(marker) + len(marker)
    end = html_str.index('"', start)
    return html_str[start:end]


def test_edge_none_renders_sentinel_not_the_word_none_or_nan():
    row = {**BASE_ROW, "has_line": False, "projection": 0.2, "edge": None}
    html_out = prop_card_html(row, "Home Team", "Away Team", "kickoff", "Opponent")
    assert _extract_data_edge(html_out) == "-1"


def test_edge_nan_from_dataframe_roundtrip_renders_sentinel_not_literal_nan():
    """The actual failure mode: build rows the same way score_props() does
    (a mix of None and real-float edge values in one list of dicts), run
    it through pd.DataFrame + to_dict('records') the same way
    report/props.py's props_section_html() does, and confirm the
    resulting NaN still renders as the -1 sentinel, not the string
    "nan"."""
    rows = [
        {**BASE_ROW, "has_line": False, "projection": 0.2, "edge": None},
        {**BASE_ROW, "player": "Other Player", "edge": 0.1},
    ]
    roundtripped = pd.DataFrame(rows).to_dict("records")
    no_line_row = roundtripped[0]
    assert pd.isna(no_line_row["edge"])  # confirms the round-trip really did coerce None -> NaN

    html_out = prop_card_html(no_line_row, "Home Team", "Away Team", "kickoff", "Opponent")
    data_edge = _extract_data_edge(html_out)
    assert data_edge == "-1"
    assert "nan" not in data_edge.lower()


def test_edge_real_float_renders_absolute_value():
    row = {**BASE_ROW, "edge": -0.23}
    html_out = prop_card_html(row, "Home Team", "Away Team", "kickoff", "Opponent")
    assert _extract_data_edge(html_out) == "0.23"


# Combined Build Plan Part 2: confidence_tier()'s uncertain flag and
# game_pick_card_html()'s fallback-caveat rendering.
def test_confidence_tier_uncertain_forces_toss_up_despite_high_prob():
    """The whole point of the flag: a 75% favorite still shows 75% in the
    label (the real number, never hidden), but the color/tier must not
    read as "strong" when the underlying stats are flagged unreliable."""
    tier, label = confidence_tier(0.75, uncertain=True)
    assert tier == "toss-up"
    assert "75%" in label


def test_confidence_tier_not_uncertain_behaves_as_before():
    tier, _ = confidence_tier(0.75, uncertain=False)
    assert tier == "strong"
    tier, _ = confidence_tier(0.75)  # default value must match the pre-existing behavior
    assert tier == "strong"


def test_game_pick_card_renders_caveat_for_flagged_team():
    game = {
        "home_win_prob": 0.7, "home_moneyline": -200, "away_moneyline": 170,
        "edge": 1.2, "spread_line": -3.0, "implied_spread": -4.2, "total_line": 44.0,
        "confidence_uncertain": True,
        "home_fallback_caveat": {"prior_season": 2025, "departed": 7, "total_starters": 20, "turnover_pct": 0.35},
        "away_fallback_caveat": None,
    }
    html_out = game_pick_card_html(game, "Home Team", "Away Team")
    assert "fallback-caveat" in html_out
    assert "2025" in html_out
    assert "Home Team" in html_out
    assert "side-toss-up" in html_out  # tier nudged despite 70% raw probability


def test_game_pick_card_no_caveat_html_when_nothing_flagged():
    game = {
        "home_win_prob": 0.7, "home_moneyline": -200, "away_moneyline": 170,
        "edge": 1.2, "spread_line": -3.0, "implied_spread": -4.2, "total_line": 44.0,
    }
    html_out = game_pick_card_html(game, "Home Team", "Away Team")
    assert "fallback-caveat" not in html_out
    assert "side-over" in html_out  # normal strong-tier color, not toss-up


# Combined Build Plan Part 4: "Cleared to Play" badge on prop cards.
def test_prop_card_shows_returned_badge_when_flagged():
    row = {**BASE_ROW, "edge": 0.1, "returned_from_injury": True}
    html_out = prop_card_html(row, "Home Team", "Away Team", "kickoff", "Opponent")
    assert "returned-badge" in html_out
    assert "card-returned" in html_out
    assert "Cleared to play" in html_out


def test_prop_card_no_returned_badge_by_default():
    row = {**BASE_ROW, "edge": 0.1}
    html_out = prop_card_html(row, "Home Team", "Away Team", "kickoff", "Opponent")
    assert "returned-badge" not in html_out
    assert "card-returned" not in html_out
