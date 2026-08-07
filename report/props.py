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
                        full_name_fn, headshot_url_fn=None) -> str:
    """game_props: rows from model/player_stats.py's score_props(), already
    filtered to one specific game. Renders "no props posted yet" rather
    than an empty grid when the book hasn't opened player markets for this
    game -- normal for anything more than a few days out. `headshot_url_fn`
    is passed straight through to report/cards.py's prop_card_html (None
    in the Artifact build, since external images can't load there)."""
    if game_props is None or game_props.empty:
        body = '<div class="empty">No player props posted yet for this game.</div>'
        return SECTION_TEMPLATE.format(body=body)

    rows = game_props.to_dict("records")
    cards = []
    for row in sorted(rows, key=lambda r: -abs(r["edge"])):
        opponent_full = full_name_fn(row["opponent"])
        cards.append(prop_card_html(row, home_full, away_full, kickoff, opponent_full, headshot_url_fn))

    body = prop_filter_bar_html(rows) + f'<div class="card-grid">{"".join(cards)}</div>'
    return SECTION_TEMPLATE.format(body=body)
