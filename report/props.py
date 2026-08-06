# renders the per-game player props disclosure section
"""Phase 2's "click into a game" player props view -- a native <details>
disclosure per game (no JS, no separate page/routing needed) so it works
identically in the plain HTML report and the self-contained Artifact
build, which can't load external scripts.
"""

import pandas as pd

STAT_LABEL = {
    "pass_yards": "Pass Yds", "rush_yards": "Rush Yds",
    "receptions": "Receptions", "rec_yards": "Rec Yds", "anytime_td": "Anytime TD",
}

PROPS_STYLE = """
.props { margin-top: 16px; }
.props summary { cursor: pointer; font-size: 13.5px; font-weight: 600; color: var(--accent, #A8710F); list-style: none; }
.props summary::-webkit-details-marker { display: none; }
.props summary::after { content: "\\25B8"; display: inline-block; margin-left: 4px; }
.props[open] summary::after { content: "\\25BE"; }
.props table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13.5px; }
.props th, .props td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border, #E1E4EA); }
.props th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted, #666E7D); }
.props td.num { text-align: right; font-variant-numeric: tabular-nums; }
.props td.edge-pos { color: var(--positive, #2A7A4F); font-weight: 600; }
.props td.edge-neg { color: var(--negative, #B23A3A); font-weight: 600; }
.props .empty { font-size: 13px; color: var(--muted, #666E7D); padding: 10px 0; }
"""

ROW_TEMPLATE = """<tr>
  <td>{player}</td><td>{stat}</td>
  <td class="num">{line}</td>
  <td class="num">{projection}</td>
  <td class="num">{model_prob}</td>
  <td class="num">{market_prob}</td>
  <td class="num {edge_class}">{edge}</td>
</tr>"""

SECTION_TEMPLATE = """<details class="props">
  <summary>Player Props</summary>
  {body}
</details>"""


def _fmt_line(stat: str, line) -> str:
    if stat == "anytime_td":
        return "Yes"
    return f"{line:g}"


def _fmt_projection(stat: str, value: float) -> str:
    if stat in ("pass_yards", "rush_yards", "rec_yards"):
        return f"{value:.0f}"
    return f"{value:.1f}"


def props_section_html(game_props: pd.DataFrame) -> str:
    """game_props: rows from model/player_stats.py's score_props(), already
    filtered to one specific game. Renders "no props posted yet" rather
    than an empty table when the book hasn't opened player markets for
    this game -- normal for anything more than a few days out."""
    if game_props is None or game_props.empty:
        body = '<div class="empty">No player props posted yet for this game.</div>'
        return SECTION_TEMPLATE.format(body=body)

    rows = []
    for _, p in game_props.sort_values("edge", key=abs, ascending=False).iterrows():
        edge_class = "edge-pos" if p["edge"] > 0.03 else ("edge-neg" if p["edge"] < -0.03 else "")
        rows.append(ROW_TEMPLATE.format(
            player=p["player"], stat=STAT_LABEL.get(p["stat"], p["stat"]),
            line=_fmt_line(p["stat"], p["line"]),
            projection=_fmt_projection(p["stat"], p["projection"]),
            model_prob=f"{p['model_over_prob'] * 100:.0f}%",
            market_prob=f"{p['market_over_prob'] * 100:.0f}%",
            edge=f"{p['edge'] * 100:+.0f}%", edge_class=edge_class,
        ))

    table = (
        "<table><thead><tr><th>Player</th><th>Prop</th><th>Line</th>"
        "<th>Model Proj.</th><th>Model %</th><th>Market %</th><th>Edge</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    return SECTION_TEMPLATE.format(body=table)
