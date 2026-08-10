# "Biggest Edges This Week" leaderboard + edge-distribution chart
"""Master Honing Plan round 2, items #4 + #12: build_artifact.py's own
hero-game selection already ranks every game by |edge| (the model's own
implied spread vs. the market's) to pick a single featured game -- this
extends that exact same ranking to a full cross-game/cross-prop view
instead of stopping at the top one, plus a histogram showing the shape
of the whole week's edge distribution at a glance.

Games and props are kept as two separate ranked lists, not one blended
table -- a game edge (points) and a prop edge (probability difference)
aren't the same unit, and forcing them onto one shared ranking would
manufacture a false equivalence the rest of this project's "no
fabricated confidence" discipline doesn't allow elsewhere.
"""

import pandas as pd

LEADERBOARD_SECTION = """<div class="section-head" id="leaderboard"><span class="accent">&#9679;</span> Biggest Edges This Week</div>
<div class="intro" style="margin-top:0;">Every game and prop this week the model disagrees with the market on, ranked by how big that disagreement is.</div>
{body}"""

EMPTY_STATE = '<div class="empty-state">No market lines to compare against yet this week -- check back closer to kickoff.</div>'

GAME_ROW = """<a class="edge-row" href="#{anchor}">
  <div class="edge-row-main">
    <img src="{away_logo}" alt="{away_team}"><span class="edge-row-matchup">{away_team} @ {home_team}</span>
  </div>
  <div class="edge-row-value {edge_class}">{edge_str}</div>
</a>"""

PROP_ROW = """<div class="edge-row">
  <div class="edge-row-main">
    <span class="edge-row-matchup">{player} <span class="edge-row-sub">{team} &middot; Anytime TD</span></span>
  </div>
  <div class="edge-row-value {edge_class}">{edge_str}</div>
</div>"""

LEADERBOARD_STYLE = """
.edge-leaderboard { display: grid; gap: 20px; margin: 8px 0 24px; }
.edge-leaderboard .edge-col-label {
  font-family: 'Oswald', sans-serif; font-size: 11.5px; font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 8px;
}
.edge-row {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 14px; margin-bottom: 6px; text-decoration: none; color: inherit;
}
.edge-row-main { display: flex; align-items: center; gap: 10px; min-width: 0; }
.edge-row-main img { height: 22px; width: 22px; object-fit: contain; flex-shrink: 0; }
.edge-row-matchup { font-size: 13.5px; font-weight: 600; color: var(--ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.edge-row-sub { font-weight: 400; color: var(--muted); font-size: 12px; }
.edge-row-value { font-family: 'Oswald', sans-serif; font-weight: 700; font-size: 14px; font-variant-numeric: tabular-nums; flex-shrink: 0; }
.edge-row-value.edge-hi { color: var(--accent); }
"""


def _edge_class(_abs_edge: float) -> str:
    return "edge-hi"


def games_edge_leaderboard(predictions: pd.DataFrame, n: int = 8) -> list[dict]:
    from report.theme import game_anchor
    ranked = predictions[predictions["edge"].notna()].copy()
    if ranked.empty:
        return []
    ranked["abs_edge"] = ranked["edge"].abs()
    ranked = ranked.sort_values("abs_edge", ascending=False).head(n)
    return ranked.to_dict("records")


def props_edge_leaderboard(props: pd.DataFrame, n: int = 8) -> list[dict]:
    if props is None or props.empty or "edge" not in props.columns:
        return []
    ranked = props[props["edge"].notna()].copy()
    if ranked.empty:
        return []
    ranked["abs_edge"] = ranked["edge"].abs()
    ranked = ranked.drop_duplicates("player").sort_values("abs_edge", ascending=False).head(n)
    return ranked.to_dict("records")


def leaderboard_html(predictions: pd.DataFrame, props: pd.DataFrame, embed_logo_fn=None) -> str:
    from report.logos import get_logo_url
    from report.theme import game_anchor
    logo_fn = embed_logo_fn or get_logo_url

    games = games_edge_leaderboard(predictions)
    props_ranked = props_edge_leaderboard(props)
    if not games and not props_ranked:
        return LEADERBOARD_SECTION.format(body=EMPTY_STATE)

    cols = []
    if games:
        rows = []
        for g in games:
            rows.append(GAME_ROW.format(
                anchor=game_anchor(g), away_logo=logo_fn(g["away_team"]),
                away_team=g["away_team"], home_team=g["home_team"],
                edge_class=_edge_class(g["abs_edge"]), edge_str=f"{g['edge']:+.1f} pts",
            ))
        cols.append(f'<div><div class="edge-col-label">Games</div>{"".join(rows)}</div>')

    if props_ranked:
        rows = []
        for p in props_ranked:
            rows.append(PROP_ROW.format(
                player=p["player"], team=p["team"],
                edge_class=_edge_class(p["abs_edge"]), edge_str=f"{p['edge'] * 100:+.0f} pts",
            ))
        cols.append(f'<div><div class="edge-col-label">Player Props (Anytime TD)</div>{"".join(rows)}</div>')

    return LEADERBOARD_SECTION.format(body=f'<div class="edge-leaderboard">{"".join(cols)}</div>')


# ---- edge distribution histogram (item #12) -------------------------------

def edge_distribution_chart(predictions: pd.DataFrame, props: pd.DataFrame, width: int = 560) -> str:
    """A single histogram of |edge| across every game AND prop this week --
    unlike the leaderboard above, a distribution SHAPE doesn't need the
    two to share a unit (each bucket is just "how many things fall in
    this confidence band"), so games and props are combined here into
    one picture of how lopsided or tight the whole week looks."""
    edges = []
    if predictions is not None and not predictions.empty and "edge" in predictions.columns:
        edges += predictions["edge"].dropna().abs().tolist()
    if props is not None and not props.empty and "edge" in props.columns:
        edges += (props.drop_duplicates("player")["edge"].dropna().abs() * 10).tolist()  # scale prob-edge to a comparable 0-10ish band
    if len(edges) < 3:
        return ""

    buckets = [0, 1, 2, 4, 6, 10, float("inf")]
    labels = ["0-1", "1-2", "2-4", "4-6", "6-10", "10+"]
    counts = [0] * (len(buckets) - 1)
    for e in edges:
        for i in range(len(buckets) - 1):
            if buckets[i] <= e < buckets[i + 1]:
                counts[i] += 1
                break

    max_count = max(counts) or 1
    row_h, bar_h = 30, 14
    pad_l, pad_r = 44, 40
    plot_w = width - pad_l - pad_r
    height = len(labels) * row_h + 8

    bars = []
    for i, (label, count) in enumerate(zip(labels, counts)):
        y = 8 + i * row_h
        bar_w = max(2, (count / max_count) * plot_w) if count else 0
        tooltip = f"{count} game/prop{'s' if count != 1 else ''} in the {label}-point edge range"
        bars.append(f"""
  <text class="axis-label" x="{pad_l - 6}" y="{y + bar_h - 2}" text-anchor="end">{label}</text>
  <rect class="bar-track" x="{pad_l}" y="{y}" width="{plot_w}" height="{bar_h}" rx="4" />
  <rect class="bar-fill" x="{pad_l}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="4"
        tabindex="0" role="img" aria-label="{tooltip}" data-tooltip="{tooltip}" />
  <text class="value-label" x="{pad_l + bar_w + 6:.1f}" y="{y + bar_h - 2}">{count}</text>""")

    return f"""<div class="chart-card">
  <div class="chart-title">Edge Distribution This Week</div>
  <div class="chart-wrap">
    <svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Distribution of model-vs-market edge size across this week's games and props">
      {"".join(bars)}
    </svg>
    <div class="chart-tooltip" role="status" aria-live="polite"></div>
  </div>
</div>"""
