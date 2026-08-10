# lightweight server-rendered SVG charts, no client-side charting library
"""Master Honing Plan round 2, item #1: turns the numbers already sitting in
logs/season_results.csv, logs/props_results.csv, and model/predict.py's
home_stats/away_stats dicts into real charts instead of raw text/tables.
Built as plain server-rendered SVG (no D3/Chart.js/etc.) -- this site is
fully server-rendered everywhere else, and a season's worth of weekly
points or a handful of category bars doesn't need a client-side charting
runtime to draw well. Follows the project's dataviz skill: form picked by
the data's job (a trend -> line, a magnitude comparison -> bars, never a
dual-axis chart), color assigned by role (identity/magnitude) using
SharpLine's own existing tokens rather than a new palette, mark specs
(2px lines, 4px rounded bar ends, 8px+ hit targets, a 2px surface ring on
markers), and a real hover layer -- a chart is interactive by default,
not an optional upgrade.

Every chart degrades to nothing (an empty string, same "no section at
all" contract report/alt_lines.py already uses) rather than rendering an
empty/misleading chart when there's no real data yet -- matches this
whole project's "no fabricated confidence" identity: an early-season
accuracy line with one data point is still real, but a chart with zero
points would just be chrome around nothing.
"""

import html
import json

# ---- shared chart chrome -----------------------------------------------

CHARTS_STYLE = """
.chart-card { margin: 14px 0; }
.chart-title { font-size: 12.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--muted); margin-bottom: 8px; }
.chart-svg { display: block; width: 100%; height: auto; overflow: visible; }
.chart-svg .grid-line { stroke: var(--border); stroke-width: 1; shape-rendering: crispEdges; }
.chart-svg .baseline { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; }
.chart-svg .axis-label { fill: var(--muted); font-size: 10.5px; font-family: 'Inter', sans-serif; }
.chart-svg .value-label { fill: var(--ink); font-size: 11px; font-weight: 600; font-family: 'Inter', sans-serif; }
.chart-svg .line-path { fill: none; stroke: var(--accent); stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.chart-svg .line-point { fill: var(--accent); stroke: var(--surface); stroke-width: 2; cursor: pointer; }
.chart-svg .line-point:hover, .chart-svg .line-point:focus { fill: var(--ink); outline: none; }
.chart-svg .hit-area { fill: transparent; cursor: pointer; }
.chart-svg .bar-track { fill: var(--surface-raised); }
.chart-svg .bar-fill { fill: var(--accent); }
.chart-svg .bar-fill.is-away { fill: var(--series-2); }
.chart-svg .bar-fill:hover, .chart-svg .hit-area:hover + .bar-fill { filter: brightness(1.15); }
.chart-tooltip {
  position: absolute; pointer-events: none; z-index: 5;
  background: var(--ink); color: var(--paper); font-size: 12px; line-height: 1.4;
  padding: 6px 9px; border-radius: 6px; white-space: nowrap; opacity: 0; transition: opacity 120ms ease;
}
.chart-tooltip.is-visible { opacity: 1; }
.chart-tooltip b { font-variant-numeric: tabular-nums; }
.chart-legend { display: flex; gap: 14px; margin-top: 8px; font-size: 11.5px; color: var(--muted); }
.chart-legend-item { display: flex; align-items: center; gap: 5px; }
.chart-legend-swatch { width: 10px; height: 2px; border-radius: 1px; }
.chart-wrap { position: relative; }
"""

CHARTS_SCRIPT = """
(function () {
  // Generic hover layer for every .chart-wrap on the page -- reads each
  // hoverable mark's data-tooltip (plain text, set via textContent, never
  // innerHTML) and floats a single shared tooltip near the pointer.
  document.querySelectorAll('.chart-wrap').forEach(function (wrap) {
    var tooltip = wrap.querySelector('.chart-tooltip');
    if (!tooltip) return;
    var marks = wrap.querySelectorAll('[data-tooltip]');
    marks.forEach(function (mark) {
      function show(e) {
        tooltip.textContent = mark.getAttribute('data-tooltip');
        tooltip.classList.add('is-visible');
        var wrapRect = wrap.getBoundingClientRect();
        var x = (e && e.clientX ? e.clientX : wrapRect.left) - wrapRect.left;
        var y = (e && e.clientY ? e.clientY : wrapRect.top) - wrapRect.top;
        tooltip.style.left = Math.max(4, Math.min(x + 12, wrapRect.width - 140)) + 'px';
        tooltip.style.top = Math.max(0, y - 34) + 'px';
      }
      function hide() { tooltip.classList.remove('is-visible'); }
      mark.addEventListener('pointermove', show);
      mark.addEventListener('pointerenter', show);
      mark.addEventListener('pointerleave', hide);
      mark.addEventListener('focus', function () { show(null); });
      mark.addEventListener('blur', hide);
    });
  });
})();
"""


def _esc(s) -> str:
    return html.escape(str(s))


# ---- chart 1: season accuracy trend line --------------------------------

def accuracy_trend_chart(weekly: list[dict], width: int = 560, height: int = 160) -> str:
    """weekly: report/track_record.py's compute_season_trend()["weekly"]
    output ([{season, week, n, accuracy}, ...], chronological). Empty
    string if fewer than 2 points -- a single dot isn't a trend, and the
    stat tile above it already shows the one number that exists."""
    if not weekly or len(weekly) < 2:
        return ""

    pad_l, pad_r, pad_t, pad_b = 8, 8, 16, 22
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(weekly)

    def x_at(i):
        return pad_l + (i / (n - 1)) * plot_w if n > 1 else pad_l + plot_w / 2

    def y_at(acc):
        return pad_t + (1 - acc) * plot_h

    grid_lines = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = pad_t + (1 - frac) * plot_h
        cls = "baseline" if frac == 0.5 else "grid-line"
        grid_lines.append(f'<line class="{cls}" x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" />')
        grid_lines.append(f'<text class="axis-label" x="{pad_l - 4}" y="{y + 3:.1f}" text-anchor="end">{frac * 100:.0f}%</text>')

    points = [(x_at(i), y_at(w["accuracy"]), w) for i, w in enumerate(weekly)]
    path_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)

    circles = []
    for i, (x, y, w) in enumerate(points):
        tooltip = f"Week {w['week']}: {w['accuracy'] * 100:.0f}% ({w['n']} game{'s' if w['n'] != 1 else ''})"
        circles.append(
            f'<circle class="line-point" cx="{x:.1f}" cy="{y:.1f}" r="4.5" tabindex="0" '
            f'role="img" aria-label="{_esc(tooltip)}" data-tooltip="{_esc(tooltip)}" />'
        )
        # every-other-week x-axis label to avoid crowding on long seasons
        if i == 0 or i == len(points) - 1 or i % 2 == 0:
            circles.append(f'<text class="axis-label" x="{x:.1f}" y="{height - 4}" text-anchor="middle">W{w["week"]}</text>')

    last_x, last_y, last_w = points[-1]
    end_label = f'<text class="value-label" x="{last_x:.1f}" y="{last_y - 10:.1f}" text-anchor="end">{last_w["accuracy"] * 100:.0f}%</text>'

    return f"""<div class="chart-card">
  <div class="chart-title">Weekly Accuracy</div>
  <div class="chart-wrap">
    <svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Model straight-up accuracy by week, {weekly[0]['week']} through {weekly[-1]['week']}">
      {"".join(grid_lines)}
      <path class="line-path" d="{path_d}" />
      {"".join(circles)}
      {end_label}
    </svg>
    <div class="chart-tooltip" role="status" aria-live="polite"></div>
  </div>
</div>"""


# ---- chart 2: props hit rate by category ---------------------------------

def props_hit_rate_chart(by_category: dict, width: int = 560) -> str:
    """by_category: report/track_record.py's compute_props_trend()
    ["by_category"] output ({category: {n, hit_rate}}). Empty string if
    empty -- see this module's docstring on not chart-shaping nothing."""
    if not by_category:
        return ""
    from report.track_record import CATEGORY_LABEL

    row_h, bar_h, gap = 34, 14, 12
    pad_l, pad_r = 4, 46
    plot_w = width - pad_l - pad_r
    rows = sorted(by_category.items())
    height = len(rows) * row_h + 8

    bars = []
    for i, (cat, stats) in enumerate(rows):
        y = 8 + i * row_h
        label = CATEGORY_LABEL.get(cat, cat)
        rate = stats["hit_rate"]
        fill_w = max(2, rate * plot_w)
        tooltip = f"{label}: {rate * 100:.0f}% ({stats['n']} graded picks)"
        bars.append(f"""
  <text class="axis-label" x="{pad_l}" y="{y - 2}">{_esc(label)} &middot; {stats["n"]} graded</text>
  <rect class="bar-track" x="{pad_l}" y="{y}" width="{plot_w}" height="{bar_h}" rx="4" />
  <rect class="bar-fill" x="{pad_l}" y="{y}" width="{fill_w:.1f}" height="{bar_h}" rx="4"
        tabindex="0" role="img" aria-label="{_esc(tooltip)}" data-tooltip="{_esc(tooltip)}" />
  <text class="value-label" x="{pad_l + plot_w + 6}" y="{y + bar_h - 3}">{rate * 100:.0f}%</text>""")

    baseline_x = pad_l + 0.5 * plot_w
    return f"""<div class="chart-card">
  <div class="chart-title">Prop Hit Rate by Category</div>
  <div class="chart-wrap">
    <svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Player prop hit rate by stat category">
      <line class="baseline" x1="{baseline_x:.1f}" y1="4" x2="{baseline_x:.1f}" y2="{height - 4}" />
      {"".join(bars)}
    </svg>
    <div class="chart-tooltip" role="status" aria-live="polite"></div>
  </div>
</div>"""


# ---- chart 3: team stat comparison (small multiples) ---------------------

# (stat key in home_stats/away_stats, label, format string, plausible
# (min, max) domain used ONLY to size the bar -- the real value is always
# the direct label, never implied by bar length alone). Domains are wide
# enough to rarely clip a real value; a value outside the domain still
# renders, just pinned to the bar's full length rather than crashing.
COMPARISON_STATS = [
    ("off_epa_per_play_avg", "Offensive EPA/play", "{:+.2f}", (-0.25, 0.25)),
    ("def_epa_per_play_avg", "Defensive EPA/play allowed", "{:+.2f}", (-0.25, 0.25)),
    ("off_third_down_pct_avg", "Third-down conversion", "{:.0%}", (0.25, 0.55)),
    ("red_zone_td_pct_avg", "Red-zone TD rate", "{:.0%}", (0.35, 0.75)),
    ("ats_win_pct_season", "ATS record this season", "{:.0%}", (0.20, 0.80)),
]


def _norm(value: float, lo: float, hi: float) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def team_comparison_chart(home_stats: dict, away_stats: dict, home_name: str, away_name: str,
                           width: int = 560) -> str:
    """Small multiples, not one shared axis -- these stats have genuinely
    different scales (EPA per play vs. a percentage), so each row gets
    its own independent 0-1 bar length, scaled against COMPARISON_STATS'
    own plausible domain for that stat alone. home_stats/away_stats: the
    same dicts model/predict.py already attaches to each scored game
    (home_win_prob-keyed rolling stats), passed straight through -- empty
    string if neither side has any of these keys yet (early season, no
    rolling history)."""
    rows_data = []
    for key, label, fmt, (lo, hi) in COMPARISON_STATS:
        h_val, a_val = home_stats.get(key), away_stats.get(key)
        if h_val is None and a_val is None:
            continue
        rows_data.append((key, label, fmt, lo, hi, h_val, a_val))
    if not rows_data:
        return ""

    row_h, bar_h = 46, 12
    pad_l, pad_r = 4, 4
    plot_w = width - pad_l - pad_r
    height = len(rows_data) * row_h + 4

    rows = []
    for i, (key, label, fmt, lo, hi, h_val, a_val) in enumerate(rows_data):
        y = 4 + i * row_h
        h_frac = _norm(h_val, lo, hi) if h_val is not None else 0.0
        a_frac = _norm(a_val, lo, hi) if a_val is not None else 0.0
        h_str = fmt.format(h_val) if h_val is not None else "--"
        a_str = fmt.format(a_val) if a_val is not None else "--"
        rows.append(f"""
  <text class="axis-label" x="{pad_l}" y="{y + 8}">{_esc(label)}</text>
  <rect class="bar-track" x="{pad_l}" y="{y + 14}" width="{plot_w:.1f}" height="{bar_h}" rx="4" />
  <rect class="bar-fill" x="{pad_l}" y="{y + 14}" width="{max(2, h_frac * plot_w):.1f}" height="{bar_h}" rx="4"
        tabindex="0" role="img" aria-label="{_esc(home_name)} {_esc(label)}: {_esc(h_str)}"
        data-tooltip="{_esc(home_name)}: {_esc(h_str)}" />
  <text class="value-label" x="{pad_l + max(2, h_frac * plot_w) + 6:.1f}" y="{y + 14 + bar_h - 2}">{_esc(h_str)}</text>
  <rect class="bar-track" x="{pad_l}" y="{y + 30}" width="{plot_w:.1f}" height="{bar_h}" rx="4" />
  <rect class="bar-fill is-away" x="{pad_l}" y="{y + 30}" width="{max(2, a_frac * plot_w):.1f}" height="{bar_h}" rx="4"
        tabindex="0" role="img" aria-label="{_esc(away_name)} {_esc(label)}: {_esc(a_str)}"
        data-tooltip="{_esc(away_name)}: {_esc(a_str)}" />
  <text class="value-label" x="{pad_l + max(2, a_frac * plot_w) + 6:.1f}" y="{y + 30 + bar_h - 2}">{_esc(a_str)}</text>""")

    return f"""<div class="chart-card">
  <div class="chart-title">Team Comparison</div>
  <div class="chart-wrap">
    <svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(home_name)} versus {_esc(away_name)}, key rolling stats">
      {"".join(rows)}
    </svg>
    <div class="chart-tooltip" role="status" aria-live="polite"></div>
  </div>
  <div class="chart-legend">
    <span class="chart-legend-item"><span class="chart-legend-swatch" style="background: var(--accent);"></span>{_esc(home_name)}</span>
    <span class="chart-legend-item"><span class="chart-legend-swatch" style="background: var(--series-2);"></span>{_esc(away_name)}</span>
  </div>
</div>"""
