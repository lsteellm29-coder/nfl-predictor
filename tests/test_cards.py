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

from report.cards import prop_card_html

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
