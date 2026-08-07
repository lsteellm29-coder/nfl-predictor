# renders the per-game player props dropdown as a card grid
"""Phase 2's "click into a game" player props view, restyled per the UI
spec into an Underdog/DraftKings-style card grid (report/cards.py) --
still a native <details> disclosure per game (no JS needed for the
expand/collapse itself, only for filtering and tooltips), so it works
identically in the plain HTML report and the self-contained Artifact
build.
"""

import pandas as pd

from report.cards import prop_card_html, prop_filter_bar_html

SECTION_TEMPLATE = """<details class="card-section">
  <summary>Player Props</summary>
  {body}
</details>"""


def props_section_html(game_props: pd.DataFrame, home_full: str, away_full: str, kickoff: str,
                        full_name_fn, headshot_url_fn=None, logo_url_fn=None) -> str:
    """game_props: rows from model/player_stats.py's score_props(), already
    filtered to one specific game. Renders "no props posted yet" rather
    than an empty grid when the book hasn't opened player markets for this
    game -- normal for anything more than a few days out. `headshot_url_fn`/
    `logo_url_fn` are passed straight through to report/cards.py's
    prop_card_html for its headshot-then-logo-then-initials fallback chain."""
    if game_props is None or game_props.empty:
        body = '<div class="empty">No player data available for this game.</div>'
        return SECTION_TEMPLATE.format(body=body)

    rows = game_props.to_dict("records")
    # Market-backed cards (with a real edge to sort by) lead; "no line
    # available" fallback cards -- QA spec Section 2's coverage
    # guarantee -- sort after them, grouped by position for readability.
    def _sort_key(r):
        if r.get("has_line", True) and r.get("edge") is not None:
            return (0, -abs(r["edge"]))
        return (1, r.get("position") or "", r["player"])

    cards = []
    for row in sorted(rows, key=_sort_key):
        opponent_full = full_name_fn(row["opponent"])
        cards.append(prop_card_html(row, home_full, away_full, kickoff, opponent_full, headshot_url_fn, logo_url_fn))

    body = prop_filter_bar_html(rows) + f'<div class="card-grid">{"".join(cards)}</div>'
    return SECTION_TEMPLATE.format(body=body)
