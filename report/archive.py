# past-weeks archive: browse any prior week's full slate against actual results
"""Master Honing Plan round 2, item #10: the site is otherwise oriented
entirely around the current week -- this reads logs/season_results.csv
(already there, already graded by run_week.py's own _grade_pending) and
gives a visitor a way to browse any past week's picks against what
actually happened, game by game. Purely a display layer over data this
project already logs every week; no new data collection.

Only weeks with at least one GRADED game are shown -- a still-pending
week belongs on the live slate, not an archive of "what already
happened."
"""

import pandas as pd

ARCHIVE_SECTION = """<div class="section-head" id="archive"><span class="accent">&#9679;</span> Past Weeks</div>
<div class="intro" style="margin-top:0;">Every graded week this season, picks against what actually happened.</div>
{body}"""

EMPTY_STATE = ('<div class="empty-state">No weeks fully graded yet this season -- the archive fills in as '
               'games finish.</div>')

WEEK_BLOCK = """<details class="card-section archive-week">
  <summary>Week {week} &middot; {season} <span class="archive-week-record">{correct}-{incorrect}</span></summary>
  <div class="props-panel">
    <div class="table-wrap">
    <table>
      <thead><tr><th>Matchup</th><th>Predicted</th><th class="num">Win %</th><th>Actual</th><th>Result</th></tr></thead>
      <tbody>
      {rows}
      </tbody>
    </table>
    </div>
  </div>
</details>"""

ROW = """<tr>
  <td>{away_team} @ {home_team}</td>
  <td>{predicted_winner}</td>
  <td class="num">{win_pct}</td>
  <td>{actual_line}</td>
  <td>{result_badge}</td>
</tr>"""

ARCHIVE_STYLE = """
.archive-week-record { font-family: 'Inter', sans-serif; font-weight: 400; font-size: 12px; color: var(--muted);
  text-transform: none; letter-spacing: normal; margin-left: 8px; }
.result-badge { display: inline-flex; align-items: center; gap: 4px; font-size: 12px; font-weight: 700;
  padding: 2px 8px; border-radius: 999px; }
.result-badge.is-correct { background: var(--positive-soft); color: var(--positive); }
.result-badge.is-wrong { background: var(--negative-soft); color: var(--negative); }
"""


def _result_badge(correct) -> str:
    if pd.isna(correct):
        return ""
    is_correct = bool(correct)
    cls = "is-correct" if is_correct else "is-wrong"
    label = "Correct" if is_correct else "Missed"
    return f'<span class="result-badge {cls}">{label}</span>'


def archive_html(log_df: pd.DataFrame) -> str:
    if log_df.empty or "correct" not in log_df.columns:
        return ARCHIVE_SECTION.format(body=EMPTY_STATE)

    graded = log_df[log_df["correct"].notna()].copy()
    if graded.empty:
        return ARCHIVE_SECTION.format(body=EMPTY_STATE)
    graded["correct"] = graded["correct"].astype(bool)  # see report/recap.py's comment on why this cast matters

    blocks = []
    for (season, week), group in sorted(graded.groupby(["season", "week"]), key=lambda kv: kv[0], reverse=True):
        rows = []
        for _, g in group.sort_values("home_team").iterrows():
            actual_line = (f"{g['home_team']} {g['actual_home_score']:.0f} -- "
                            f"{g['away_team']} {g['actual_away_score']:.0f}") if pd.notna(g.get("actual_home_score")) else "--"
            rows.append(ROW.format(
                away_team=g["away_team"], home_team=g["home_team"],
                predicted_winner=g["predicted_winner"], win_pct=f"{g['home_win_prob'] * 100:.0f}%"
                if g["predicted_winner"] == g["home_team"] else f"{(1 - g['home_win_prob']) * 100:.0f}%",
                actual_line=actual_line, result_badge=_result_badge(g["correct"]),
            ))
        n_correct = int(group["correct"].sum())
        blocks.append(WEEK_BLOCK.format(
            week=int(week), season=int(season), correct=n_correct, incorrect=len(group) - n_correct,
            rows="\n".join(rows),
        ))

    return ARCHIVE_SECTION.format(body="\n".join(blocks))
