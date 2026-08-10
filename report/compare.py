# player comparison tool: two players' TD-prop profiles side by side
"""Round 2 UX, item #59/11: pick any two players with a TD prop on this
week's slate and see their profiles side by side -- this week's model
projection plus report/charts.py's player_td_trend_chart (predicted-vs-
actual over their last several played games).

Scoped to players who already have a prop this week, matching
report/team_hub.py's identical call: every comparable player is already
cross-position-normalized to one 0-1 "TD this game" probability by
model/player_stats.py, so a WR and a RB compare on the same number
honestly, and this avoids building a second historical-player lookup.

Two <select> dropdowns swap which of the SERVER-RENDERED cards (one per
eligible player, hidden in the DOM) is cloned into each of two visible
slots -- same plain-JS-over-already-embedded-data pattern as
report/cards.py's sort/filter and report/charts.py's tooltip layer, no
fetch, no client-side templating.
"""

import html

import pandas as pd

from report.cards import headshot_html
from report.charts import player_td_trend_chart
from report.team_hub import player_td_games

COMPARE_SECTION = """<div class="section-head" id="compare"><span class="accent">&#9679;</span> Compare Players</div>
<div class="intro" style="margin-top:0;">Any two players with a TD prop this week, side by side -- this week's projection and recent scoring history.</div>
<div class="compare-controls">
  <select class="compare-select" data-slot="a" aria-label="First player to compare">{options_a}</select>
  <span class="compare-vs">vs.</span>
  <select class="compare-select" data-slot="b" aria-label="Second player to compare">{options_b}</select>
</div>
<div class="compare-grid">
  <div class="compare-slot" data-slot-target="a"></div>
  <div class="compare-slot" data-slot-target="b"></div>
</div>
<div class="compare-source" hidden>{cards}</div>"""

EMPTY_STATE = '<div class="empty-state">No player props posted yet this week to compare.</div>'

OPTION = '<option value="{player_id}"{selected}>{player} ({team})</option>'

COMPARE_CARD = """<div class="compare-card pcard" data-player-id="{player_id}">
  <div class="pcard-head">
    {headshot}
    <div>
      <div class="pcard-name">{player}</div>
      <div class="pcard-meta">{team} &middot; {position}</div>
    </div>
  </div>
  <div class="pcard-stat">{projection_pct}<span class="compare-stat-label">This week's TD probability</span></div>
  {trend_section}
</div>"""

NO_TREND = '<div class="team-hub-player-empty">No game history yet to chart.</div>'

COMPARE_STYLE = """
.compare-controls { display: flex; align-items: center; gap: 10px; margin: 12px 0; flex-wrap: wrap; }
.compare-select { font-size: 13px; padding: 7px 12px; border-radius: 999px;
  border: 1px solid var(--border); background: var(--surface); color: var(--ink); flex: 1; min-width: 180px; }
.compare-vs { font-family: 'Oswald', sans-serif; font-weight: 700; color: var(--muted); font-size: 12px; text-transform: uppercase; }
.compare-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
.compare-grid .pcard { min-height: 100%; }
.compare-stat-label { display: block; font-family: 'Inter', sans-serif; font-weight: 400; font-size: 11px; color: var(--muted); margin-top: 2px; }
.compare-trend-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); margin-top: 4px; }
"""

COMPARE_SCRIPT = """
(function () {
  document.querySelectorAll('.compare-select').forEach(function (sel) {
    function render() {
      var slot = document.querySelector('.compare-slot[data-slot-target="' + sel.dataset.slot + '"]');
      var src = document.querySelector('.compare-card[data-player-id="' + sel.value + '"]');
      if (!slot) return;
      slot.innerHTML = '';
      if (src) {
        var clone = src.cloneNode(true);
        slot.appendChild(clone);
        // freshly cloned charts start unwired -- report/charts.py's own
        // lazy-init only runs once at page load / <details> toggle, and
        // this node didn't exist yet at either moment.
        clone.querySelectorAll('.chart-wrap').forEach(function (wrap) { wrap.dataset.wired = ''; });
        if (window.__sharplineWireCharts) clone.querySelectorAll('.chart-wrap').forEach(window.__sharplineWireCharts);
      }
    }
    sel.addEventListener('change', render);
    render();
  });
})();
"""


def _esc(s) -> str:
    return html.escape(str(s))


def compare_html(props: pd.DataFrame, td_backtest_df: pd.DataFrame,
                  headshot_url_fn=None, logo_url_fn=None) -> str:
    if props is None or props.empty:
        return ""

    eligible = props.dropna(subset=["player_id"]).drop_duplicates("player_id")
    if len(eligible) < 2:
        return ""

    by_projection = eligible.sort_values("projection", ascending=False)
    default_a = by_projection.iloc[0]["player_id"]
    default_b = by_projection.iloc[1]["player_id"]

    cards, options = [], []
    for _, r in eligible.sort_values("player").iterrows():
        pid, name, team, position = r["player_id"], r["player"], r["team"], r["position"]
        games = player_td_games(td_backtest_df, pid)
        trend_section = (f'<div class="compare-trend-label">Last {len(games)} games played</div>'
                          f'{player_td_trend_chart(games)}') if games else NO_TREND
        headshot = headshot_html(name, team, r.get("espn_id"), headshot_url_fn, logo_url_fn)
        cards.append(COMPARE_CARD.format(
            player_id=_esc(pid), player=_esc(name), team=_esc(team), position=_esc(position),
            headshot=headshot, projection_pct=f"{r['projection'] * 100:.0f}%",
            trend_section=trend_section,
        ))
        selected_a = ' selected' if pid == default_a else ''
        selected_b = ' selected' if pid == default_b else ''
        options.append((OPTION.format(player_id=_esc(pid), player=_esc(name), team=_esc(team), selected=selected_a),
                         OPTION.format(player_id=_esc(pid), player=_esc(name), team=_esc(team), selected=selected_b)))

    options_a = "".join(o[0] for o in options)
    options_b = "".join(o[1] for o in options)
    return COMPARE_SECTION.format(options_a=options_a, options_b=options_b, cards="".join(cards))
