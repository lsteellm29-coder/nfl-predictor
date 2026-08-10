# public "Model Track Record" section: real graded accuracy over time
"""Master Honing Plan, Section B: a public track-record section built
from the same logs/season_results.csv and logs/props_results.csv
run_week.py already grades every week (see that file's _grade_pending/
_grade_pending_props) -- season accuracy over time, a week-by-week trend,
and props hit rate by stat category, using only genuinely GRADED rows
(a game/prop whose real outcome is already known), never a projection
dressed up as a result. This is the same "no fabricated confidence"
identity the rest of the report already holds itself to, pointed at the
model's own real history instead of this week's picks -- showing the
actual number, including a middling one, is the trust-building move here,
not hiding it.

Renders an honest empty state (not a blank section, not a fabricated
number) for exactly what it is: this season's games haven't started
being graded yet, or every early-season log is still pending -- the exact
situation on the day this section was built.
"""

import pandas as pd

from report.charts import accuracy_trend_chart, props_hit_rate_chart
from report.recap import recap_html

STAT_CATEGORY = {
    "pass_yards": "yardage", "rush_yards": "yardage", "rec_yards": "yardage",
    "receptions": "reception_count", "anytime_td": "td",
}
CATEGORY_LABEL = {"yardage": "Yardage props", "reception_count": "Reception props", "td": "Anytime-TD props"}


def compute_season_trend(log_df: pd.DataFrame) -> dict:
    """{n_graded, overall_accuracy, model_beat_market_rate, weekly: [{season,
    week, n, accuracy}]} from logs/season_results.csv -- weekly is sorted
    chronologically, one row per (season, week) that has at least one
    graded game. Every rate is None (not 0.0 or a fabricated placeholder)
    when there's nothing graded yet to compute it from."""
    if log_df.empty or "correct" not in log_df.columns:
        return {"n_graded": 0, "overall_accuracy": None, "model_beat_market_rate": None, "weekly": []}
    graded = log_df[log_df["correct"].notna()].copy()
    if graded.empty:
        return {"n_graded": 0, "overall_accuracy": None, "model_beat_market_rate": None, "weekly": []}

    graded["correct"] = graded["correct"].astype(bool)
    beat_market = graded["model_beat_market"].dropna()

    weekly = (
        graded.groupby(["season", "week"]).agg(n=("correct", "size"), accuracy=("correct", "mean"))
        .reset_index().sort_values(["season", "week"])
    )
    return {
        "n_graded": len(graded),
        "overall_accuracy": float(graded["correct"].mean()),
        "model_beat_market_rate": float(beat_market.mean()) if not beat_market.empty else None,
        "weekly": weekly.to_dict("records"),
    }


def compute_props_trend(props_df: pd.DataFrame) -> dict:
    """{n_graded, overall_hit_rate, model_beat_market_rate, by_category:
    {category: {n, hit_rate}}} from logs/props_results.csv -- only rows
    with a real market line ever get logged in the first place (see
    run_week.py's log_props_week), so every graded row here had an actual
    over/under call to be right or wrong about."""
    if props_df.empty or "model_correct" not in props_df.columns:
        return {"n_graded": 0, "overall_hit_rate": None, "model_beat_market_rate": None, "by_category": {}}
    graded = props_df[props_df["model_correct"].notna()].copy()
    if graded.empty:
        return {"n_graded": 0, "overall_hit_rate": None, "model_beat_market_rate": None, "by_category": {}}

    graded["model_correct"] = graded["model_correct"].astype(bool)
    graded["category"] = graded["stat"].map(STAT_CATEGORY).fillna(graded["stat"])
    beat_market = graded["model_beat_market"].dropna()

    by_category = {}
    for category, group in graded.groupby("category"):
        by_category[category] = {"n": len(group), "hit_rate": float(group["model_correct"].mean())}

    return {
        "n_graded": len(graded),
        "overall_hit_rate": float(graded["model_correct"].mean()),
        "model_beat_market_rate": float(beat_market.mean()) if not beat_market.empty else None,
        "by_category": by_category,
    }


EMPTY_STATE = ('<div class="empty-state">No graded results yet this season -- check back once games '
               'start finishing. Every number on this page is a real graded outcome, never a projection, '
               'so there\'s nothing honest to show here until the first games are in the books.</div>')

TRACK_RECORD_SECTION = """<div class="section-head" id="track-record"><span class="accent">&#9679;</span> Model Track Record</div>
<div class="intro" style="margin-top:0;">Every number below is a real graded outcome from logs this project has kept since the season started -- not a projection, not cherry-picked. {empty_note}</div>
{body}"""

STATLINE_BLOCK = """<div class="statline">
  <div class="stat"><div class="num">{win_accuracy}</div><div class="label">Straight-up accuracy, {win_n} graded games</div></div>
  <div class="stat"><div class="num">{win_beat_market}</div><div class="label">Beat the market's own spread</div></div>
  <div class="stat"><div class="num">{props_hit_rate}</div><div class="label">Prop hit rate, {props_n} graded picks</div></div>
  <div class="stat"><div class="num">{props_beat_market}</div><div class="label">Props: beat the market when we disagreed</div></div>
</div>"""

WEEKLY_TABLE = """<div class="table-wrap">
<table>
<thead><tr><th>Week</th><th class="num">Games graded</th><th class="num">Accuracy</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</div>"""

CATEGORY_TABLE = """<div class="table-wrap">
<table>
<thead><tr><th>Category</th><th class="num">Graded picks</th><th class="num">Hit rate</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</div>"""

# Minimal, self-contained styles for the two tables above -- reuses
# report/theme.py's tokens (var(--...)) the same way every other
# component in this codebase does, so it inherits the SharpLine palette
# automatically rather than hardcoding colors.
TRACK_RECORD_STYLE = """
.table-wrap { overflow-x: auto; margin: 14px 0 4px; border: 1px solid var(--border); border-radius: 8px; }
.table-wrap table { border-collapse: collapse; width: 100%; min-width: 320px; font-size: 13.5px; background: var(--surface); }
.table-wrap th, .table-wrap td { text-align: left; padding: 8px 14px; border-bottom: 1px solid var(--border); }
.table-wrap thead th { font-family: 'Oswald', sans-serif; font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--muted); background: var(--surface-raised); font-weight: 700; }
.table-wrap tbody tr:last-child td { border-bottom: none; }
.table-wrap td.num, .table-wrap th.num { font-variant-numeric: tabular-nums; text-align: right; }
"""


def _pct(x: float | None) -> str:
    return f"{x * 100:.0f}%" if x is not None else "--"


def track_record_html(log_df: pd.DataFrame, props_df: pd.DataFrame) -> str:
    season = compute_season_trend(log_df)
    props = compute_props_trend(props_df)

    if season["n_graded"] == 0 and props["n_graded"] == 0:
        return TRACK_RECORD_SECTION.format(empty_note="", body=EMPTY_STATE)

    statline = STATLINE_BLOCK.format(
        win_accuracy=_pct(season["overall_accuracy"]), win_n=season["n_graded"],
        win_beat_market=_pct(season["model_beat_market_rate"]),
        props_hit_rate=_pct(props["overall_hit_rate"]), props_n=props["n_graded"],
        props_beat_market=_pct(props["model_beat_market_rate"]),
    )

    body_parts = [statline]

    recap = recap_html()
    if recap:
        body_parts.append(recap)

    if season["weekly"]:
        # Chart first (the at-a-glance read), the exact same numbers in a
        # table right after (dataviz skill: "a table view exists" -- also
        # just genuinely useful for anyone who wants the precise per-week
        # figures, not just the shape of the trend).
        body_parts.append(accuracy_trend_chart(season["weekly"]))
        rows = "\n".join(
            f'<tr><td>Week {w["week"]} &middot; {w["season"]}</td><td class="num">{w["n"]}</td>'
            f'<td class="num">{_pct(w["accuracy"])}</td></tr>'
            for w in season["weekly"]
        )
        body_parts.append(WEEKLY_TABLE.format(rows=rows))

    if props["by_category"]:
        body_parts.append(props_hit_rate_chart(props["by_category"]))
        rows = "\n".join(
            f'<tr><td>{CATEGORY_LABEL.get(cat, cat)}</td><td class="num">{stats["n"]}</td>'
            f'<td class="num">{_pct(stats["hit_rate"])}</td></tr>'
            for cat, stats in sorted(props["by_category"].items())
        )
        body_parts.append(CATEGORY_TABLE.format(rows=rows))

    return TRACK_RECORD_SECTION.format(empty_note="", body="\n".join(body_parts))
