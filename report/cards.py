# Underdog/DraftKings-style card system for player props + game picks
"""Presentation-layer only -- every number rendered here comes straight
from model/player_stats.py's score_props() or model/predict.py's
score_week(), unchanged. No new predictions get made in this file; it
only decides how to *show* predictions that already exist. "Higher"/
"Lower" (for props) and each team's own button (for game picks) get
colored based on which side the model's own probability actually favors --
never inflated, never flipped for visual effect. A thin edge gets a
"Slight lean" label and a barely-there color, not a confident-looking
card it hasn't earned.
"""

import html

import pandas as pd

STAT_LABEL = {
    "pass_yards": "Passing Yards", "rush_yards": "Rushing Yards",
    "receptions": "Receptions", "rec_yards": "Receiving Yards", "anytime_td": "Anytime TD",
}
GROUP_LABEL = {"WR": "wide receivers", "TE": "tight ends", "RB": "running backs", "RUSH": "the ground game"}

# Reuses model/calibration.py's own confidence-bucket boundaries (0.55,
# 0.65) rather than inventing new ones, so "Strong lean" here means the
# same thing it means everywhere else this model talks about confidence.
def confidence_label(prob: float) -> str:
    edge = abs(prob - 0.5)
    if edge >= 0.15:
        return "Strong lean"
    if edge >= 0.05:
        return "Lean"
    return "Slight lean"


# Real definitions, not simplified ones -- matches how each term is
# actually computed elsewhere in this codebase (model/train.py,
# model/elo.py, data/opponent_adjust.py), not a plain-English gloss.
TERM_DEFINITIONS = {
    "edge": "The model's own implied point spread minus the market's posted line. Positive means the model's number favors the home team more than the market's line does.",
    "epa": "Expected Points Added -- the change in expected scoring value from before a play to after it, based on down, distance, field position, and time remaining. A model output, not a box-score stat.",
    "elo": "A rating that updates after every game based on the result, the margin of victory, and how surprising it was given the pre-game gap between the two ratings. This model tracks separate offensive and defensive Elo per team (each starts at 1500), not one blended number.",
    "cpoe": "Completion % Over Expected -- a passer's actual completion rate minus the rate expected on those same throws given depth of target and other pre-throw context. Isolates the passer's own accuracy from receiver separation or scheme.",
    "success rate": "The share of plays that gained enough yardage to be “successful” relative to down and distance -- roughly 40% of yards-to-go on 1st down, 60% on 2nd, 100% on 3rd/4th (nflfastR's own definition, not this project's invention).",
    "market spread": "The current sportsbook point spread, fed directly into the model as one of its own inputs (not just compared against the model's output afterward).",
    "opponent-adjusted": "A team's raw per-play production, adjusted for the strength of the specific opponents they've actually faced on the other side of the ball, so a good game against a bad defense doesn't read identically to the same game against a good one.",
    "model projection": "The model's own point estimate for this stat -- the mean of the distribution it's using to price the Over/Under, not a rounded or adjusted display number.",
}


def info_icon(term: str) -> str:
    key = term.lower()
    definition = TERM_DEFINITIONS.get(key)
    if not definition:
        return html.escape(term)
    tip_id = f"tip-{abs(hash(key))}"
    return (
        f'<span class="term-wrap">{html.escape(term)}'
        f'<button type="button" class="info-icon" data-tooltip="{tip_id}" aria-label="What is {html.escape(term)}?">ⓘ</button>'
        f'<span class="info-tooltip" id="{tip_id}" hidden>{html.escape(definition)}</span>'
        f'</span>'
    )


CARDS_STYLE = """
.card-section { margin-top: 18px; }
.card-section > summary { cursor: pointer; font-size: 13.5px; font-weight: 700; color: var(--accent, #A8710F);
  list-style: none; text-transform: uppercase; letter-spacing: 0.04em; }
.card-section > summary::-webkit-details-marker { display: none; }
.card-section > summary::after { content: "\\25B8"; display: inline-block; margin-left: 4px; }
.card-section[open] > summary::after { content: "\\25BE"; }

.filter-bar { display: flex; gap: 6px; flex-wrap: wrap; margin: 12px 0; }
.filter-btn { font-size: 12px; font-weight: 600; padding: 5px 11px; border-radius: 999px;
  border: 1px solid var(--border, #E1E4EA); background: var(--surface, #fff); color: var(--muted, #666E7D); cursor: pointer; }
.filter-btn.is-active { background: var(--accent, #A8710F); border-color: var(--accent, #A8710F); color: #fff; }

.card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; margin-top: 8px; }

.pcard, .gcard { background: var(--surface, #fff); border: 1px solid var(--border, #E1E4EA); border-radius: 10px;
  padding: 14px; display: flex; flex-direction: column; gap: 10px; }
.pcard-head { display: flex; align-items: center; gap: 10px; }
.pcard-photo { width: 40px; height: 40px; border-radius: 50%; object-fit: cover; background: var(--border, #E1E4EA); flex-shrink: 0; }
.pcard-photo.is-fallback { display: flex; align-items: center; justify-content: center; font-size: 15px; font-weight: 700; color: var(--muted, #666E7D); }
.pcard-name { font-weight: 700; font-size: 14.5px; color: var(--ink, #171A21); line-height: 1.25; }
.pcard-meta { font-size: 12px; color: var(--muted, #666E7D); }
.pcard-stat { font-family: 'Oswald', sans-serif; font-weight: 700; font-size: 15px; color: var(--ink, #171A21); }
.pcard-matchup { font-size: 11.5px; color: var(--muted, #666E7D); }

.split { display: flex; gap: 6px; }
.split-side { flex: 1; border-radius: 8px; padding: 8px 6px; text-align: center; border: 1px solid var(--border, #E1E4EA);
  background: var(--paper, #F4F5F8); color: var(--muted, #666E7D); }
.split-side .side-label { font-weight: 700; font-size: 13px; }
.split-side .side-conf { display: block; font-size: 10.5px; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.03em; }
.split-side.side-over { background: var(--positive-soft, #E7F5EC); border-color: var(--positive, #2A7A4F); color: var(--positive, #2A7A4F); }
.split-side.side-under { background: var(--negative-soft, #FBEAEA); border-color: var(--negative, #B23A3A); color: var(--negative, #B23A3A); }

.card-detail summary { cursor: pointer; font-size: 12px; font-weight: 600; color: var(--accent, #A8710F); list-style: none; }
.card-detail summary::-webkit-details-marker { display: none; }
.card-detail summary::after { content: " \\25B8"; }
.card-detail[open] summary::after { content: " \\25BE"; }
.card-detail-body { margin-top: 8px; font-size: 12.5px; color: var(--ink, #171A21); line-height: 1.5; }
.vs-row { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted, #666E7D); margin-bottom: 6px;
  padding-bottom: 6px; border-bottom: 1px solid var(--border, #E1E4EA); }
.vs-row b { color: var(--ink, #171A21); }
.card-injury { margin-top: 6px; font-size: 11.5px; color: var(--negative, #B23A3A); }

.term-wrap { position: relative; }
.info-icon { background: none; border: none; color: var(--muted, #666E7D); cursor: pointer; font-size: 12px;
  padding: 0 0 0 3px; vertical-align: middle; }
.info-tooltip { position: absolute; z-index: 5; left: 0; top: 100%; margin-top: 4px; width: 240px;
  background: var(--ink, #171A21); color: var(--paper, #F4F5F8); font-size: 11.5px; line-height: 1.4;
  padding: 8px 10px; border-radius: 6px; font-weight: 400; text-transform: none; letter-spacing: normal; }

.gcard-row { display: flex; gap: 10px; }
"""

def _rank_ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _rank_phrase(rank: int) -> str:
    return "the most" if rank == 1 else f"the {_rank_ordinal(rank)}-most"


def reasoning_sentence(reasoning: dict | None, opponent_full: str) -> str | None:
    """Phrases a model/player_stats.py reasoning dict into the exact style
    the spec asks for -- "Facing a defense allowing the 3rd-most rushing
    yards to RBs" -- using a real league rank (data/positional_matchups.py's
    defense_rank), not a vague qualifier."""
    if not reasoning:
        return None
    phrase = _rank_phrase(reasoning["rank"])
    group = reasoning.get("group")
    if reasoning["kind"] == "yards":
        label = "passing yards allowed" if group is None else (
            "rushing yards allowed to backs" if group == "RUSH" else f"yards allowed to {GROUP_LABEL[group]}")
    else:
        label = "rushing TD rate allowed" if group == "RUSH" else f"TD rate allowed to {GROUP_LABEL[group]}"
    return f"Facing {opponent_full}, allowing {phrase} {label} in the league this season."


def _initials(name: str) -> str:
    parts = [p for p in name.split() if p]
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper() if name else "??"


def _headshot_html(player: str, espn_id, headshot_url_fn) -> str:
    url = headshot_url_fn(espn_id) if headshot_url_fn and espn_id else None
    if url:
        return f'<img class="pcard-photo" src="{html.escape(url)}" alt="{html.escape(player)}" loading="lazy">'
    return f'<div class="pcard-photo is-fallback">{html.escape(_initials(player))}</div>'


def espn_headshot_url(espn_id) -> str:
    """Live ESPN CDN URL, predictable from the id alone -- only usable in
    the plain HTML report, not the self-contained Artifact build, which
    can't load external images."""
    return f"https://a.espncdn.com/i/headshots/nfl/players/full/{espn_id}.png"


def _fmt_line(stat: str, line) -> str:
    return "Yes" if stat == "anytime_td" else f"{line:g}"


def _fmt_projection(stat: str, value: float) -> str:
    return f"{value:.0f}" if stat in ("pass_yards", "rush_yards", "rec_yards") else f"{value:.1f}"


def prop_card_html(row: dict, home_full: str, away_full: str, kickoff: str,
                    opponent_full: str, headshot_url_fn=None) -> str:
    """One player-prop card. `row` is a dict from model/player_stats.py's
    score_props() (plus `home_full`/`away_full`/`kickoff`/`opponent_full`
    resolved by the caller, since score_props() only carries abbreviations).
    `headshot_url_fn` resolves an espn_id to an image URL -- pass None (the
    self-contained Artifact build) to always fall back to an initials
    avatar, since external images won't load there."""
    is_td = row["stat"] == "anytime_td"
    over_label, under_label = ("Yes", "No") if is_td else ("Higher", "Lower")
    model_prob = row["model_over_prob"]
    is_over = model_prob >= 0.5
    conf = confidence_label(model_prob if is_over else 1 - model_prob)

    pct = f"{(model_prob if is_over else 1 - model_prob) * 100:.0f}%"
    over_class = "side-over" if is_over else ""
    under_class = "side-under" if not is_over else ""
    over_conf = f'<span class="side-conf">{pct} &middot; {conf}</span>' if is_over else ""
    under_conf = f'<span class="side-conf">{pct} &middot; {conf}</span>' if not is_over else ""

    reasoning = reasoning_sentence(row.get("reasoning"), opponent_full)
    injury_html = ""
    if row.get("injury_status") and str(row["injury_status"]) != "nan":
        injury_html = f'<div class="card-injury">{html.escape(row["player"])} is listed as {html.escape(str(row["injury_status"]))} -- usage projection adjusted accordingly.</div>'

    detail_lines = []
    if reasoning:
        detail_lines.append(f"<p>{html.escape(reasoning)}</p>")
    detail_lines.append(injury_html)

    return f"""<div class="pcard" data-team="{row['team']}" data-pos="{row.get('position') or ''}" data-stat="{row['stat']}">
  <div class="pcard-head">
    {_headshot_html(row['player'], row.get('espn_id'), headshot_url_fn)}
    <div>
      <div class="pcard-name">{html.escape(row['player'])}</div>
      <div class="pcard-meta">{row['team']} &middot; {row.get('position') or '--'}</div>
    </div>
  </div>
  <div class="pcard-stat">{_fmt_line(row['stat'], row['line'])} {STAT_LABEL.get(row['stat'], row['stat'])}</div>
  <div class="pcard-matchup">{away_full} @ {home_full} &middot; {kickoff}</div>
  <div class="split">
    <div class="split-side {over_class}"><span class="side-label">{over_label}</span>{over_conf}</div>
    <div class="split-side {under_class}"><span class="side-label">{under_label}</span>{under_conf}</div>
  </div>
  <details class="card-detail">
    <summary>Model reasoning</summary>
    <div class="card-detail-body">
      <div class="vs-row"><span>{info_icon('Model Projection')}: <b>{_fmt_projection(row['stat'], row['projection'])}</b></span><span>Vegas: <b>{_fmt_line(row['stat'], row['line'])}</b></span></div>
      {''.join(detail_lines)}
    </div>
  </details>
</div>"""


def prop_filter_bar_html(rows: list[dict]) -> str:
    teams = sorted({r["team"] for r in rows})
    positions = sorted({r.get("position") for r in rows if r.get("position")})
    stats = sorted({r["stat"] for r in rows})
    btns = []
    for t in teams:
        btns.append(f'<button type="button" class="filter-btn" data-filter-type="team" data-filter-value="{t}">{t}</button>')
    for p in positions:
        btns.append(f'<button type="button" class="filter-btn" data-filter-type="pos" data-filter-value="{p}">{p}</button>')
    for s in stats:
        btns.append(f'<button type="button" class="filter-btn" data-filter-type="stat" data-filter-value="{s}">{STAT_LABEL.get(s, s)}</button>')
    return f'<div class="filter-bar">{"".join(btns)}</div>'


def _fmt_ml(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:+.0f}"


def game_pick_card_html(game: dict, home_full: str, away_full: str) -> str:
    """The main game-level pick, styled the same as the prop cards --
    each team gets its own button, only the model's actual favored side
    colored in. total_line is shown as plain informational text, not a
    toggleable pick: this model has no total-points prediction to back
    one, and faking a "lean" there would be exactly the confidence
    inflation the spec says not to do."""
    home_prob = game.get("home_win_prob")
    if home_prob is None or pd.isna(home_prob):
        return ""
    home_favored = home_prob >= 0.5
    conf = confidence_label(home_prob if home_favored else 1 - home_prob)
    pct = f"{(home_prob if home_favored else 1 - home_prob) * 100:.0f}%"
    home_class = "side-over" if home_favored else ""
    away_class = "side-under" if not home_favored else ""
    home_conf = f'<span class="side-conf">{pct} &middot; {conf}</span>' if home_favored else ""
    away_conf = f'<span class="side-conf">{pct} &middot; {conf}</span>' if not home_favored else ""

    home_ml, away_ml = _fmt_ml(game.get("home_moneyline")), _fmt_ml(game.get("away_moneyline"))
    edge = game.get("edge")
    edge_row = ""
    if edge is not None and pd.notna(edge):
        edge_row = f'<div class="vs-row"><span>{info_icon("Edge")}</span><span><b>{edge:+.1f}</b></span></div>'

    spread_line = game.get("spread_line")
    implied_spread = game.get("implied_spread")
    total_line = game.get("total_line")
    spread_str = f"{spread_line:+.1f}" if pd.notna(spread_line) else "--"
    model_str = f"{implied_spread:+.1f}" if pd.notna(implied_spread) else "--"
    total_str = f"{total_line:.1f}" if pd.notna(total_line) else "--"

    return f"""<div class="gcard">
  <div class="gcard-row">
    <div class="split-side {home_class}"><span class="side-label">{html.escape(home_full)}{(' ' + home_ml) if home_ml else ''}</span>{home_conf}</div>
    <div class="split-side {away_class}"><span class="side-label">{html.escape(away_full)}{(' ' + away_ml) if away_ml else ''}</span>{away_conf}</div>
  </div>
  {edge_row}
  <div class="pcard-matchup">{info_icon('Market Spread')}: {spread_str} &middot; {info_icon('Model Projection')}: {model_str} &middot; Total (Vegas -- not modeled): {total_str}</div>
</div>"""


CARDS_SCRIPT = """
document.addEventListener('click', function (e) {
  var icon = e.target.closest('.info-icon');
  document.querySelectorAll('.info-tooltip:not([hidden])').forEach(function (t) {
    if (!icon || t.id !== icon.dataset.tooltip) t.hidden = true;
  });
  if (icon) {
    var tip = document.getElementById(icon.dataset.tooltip);
    if (tip) tip.hidden = !tip.hidden;
    e.stopPropagation();
    return;
  }
  var btn = e.target.closest('.filter-btn');
  if (btn) {
    var bar = btn.closest('.filter-bar');
    var type = btn.dataset.filterType;
    var wasActive = btn.classList.contains('is-active');
    bar.querySelectorAll('.filter-btn[data-filter-type="' + type + '"]').forEach(function (b) {
      b.classList.remove('is-active');
    });
    if (!wasActive) btn.classList.add('is-active');

    var grid = bar.parentElement.querySelector('.card-grid');
    var active = {};
    bar.querySelectorAll('.filter-btn.is-active').forEach(function (b) { active[b.dataset.filterType] = b.dataset.filterValue; });
    grid.querySelectorAll('.pcard').forEach(function (card) {
      var ok = (!active.team || card.dataset.team === active.team)
        && (!active.pos || card.dataset.pos === active.pos)
        && (!active.stat || card.dataset.stat === active.stat);
      card.style.display = ok ? '' : 'none';
    });
  }
});
"""
