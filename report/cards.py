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

STAT_LABEL = {"anytime_td": "Anytime TD"}

# Three-tier confidence system (QA spec): 50-59% on the favored side is
# a toss-up regardless of which way it leans -- yellow, not green/red --
# since a coin-flip-ish edge dressed up in a decisive color is exactly
# the confidence inflation this whole card system is built to avoid.
# 60%+ is treated as the model's real, actionable call.
CONFIDENCE_THRESHOLD = 0.60


def confidence_tier(prob: float) -> tuple[str, str]:
    """prob: the model's probability for whichever side is being colored
    (always >=0.5 by construction -- confidence is only ever shown on the
    side the model actually favors). Returns (css class suffix, label).
    The label always carries the real percentage, toss-up included --
    only the color (via the css class) signals the tier; the number
    itself shouldn't disappear just because it's a weak lean."""
    label = f"Model favors {prob * 100:.0f}%"
    if prob >= CONFIDENCE_THRESHOLD:
        return "strong", label
    return "toss-up", label


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
.pcard-photo.is-logo-fallback { object-fit: contain; padding: 6px; background: var(--paper, #F4F5F8); border: 1px solid var(--border, #E1E4EA); }
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
.split-side.side-toss-up { background: var(--warning-soft, #FFF6D9); border-color: var(--warning, #8A6D00); color: var(--warning, #8A6D00); }
.no-line-badge { font-size: 12px; color: var(--muted, #666E7D); background: var(--paper, #F4F5F8); border: 1px dashed var(--border, #E1E4EA); border-radius: 8px; padding: 8px 10px; text-align: center; }

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
    """Phrases a model/player_stats.py reasoning dict -- {"rank": N}, the
    opponent's league rank for red-zone touchdown rate allowed
    (model/td_model.py's team_red_zone_defense) -- into the exact style
    the spec asks for: "Facing a defense allowing the 3rd-most..." using
    a real league rank, not a vague qualifier."""
    if not reasoning:
        return None
    phrase = _rank_phrase(reasoning["rank"])
    # "whose defense allows" (not "allowing... allowed") -- avoids
    # doubling up on "allow" once the rest of the sentence already ends
    # in "allowed," which reads as a grammar error, not just informal.
    # `phrase` already starts with "the" (_rank_phrase), so it isn't
    # repeated here.
    return f"Facing {opponent_full}, whose defense allows {phrase} red-zone touchdown rate in the league this season."


def _initials(name: str) -> str:
    parts = [p for p in name.split() if p]
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper() if name else "??"


def _headshot_html(player: str, team: str, espn_id, headshot_url_fn, logo_url_fn=None) -> str:
    """Fallback chain (QA spec Section 3): real headshot -> team logo
    placeholder -> initials, so a card is never blank or visibly broken.
    The headshot-to-logo step happens via the image's own onerror handler
    rather than a pre-flight network check per card -- lets the browser
    itself find out in real time whether ESPN's CDN actually has a photo,
    with zero added latency in the common case where it does. Proactively
    flagging *known* gaps before publishing is qa/validate_headshots.py's
    job, a separate concern from this runtime fallback."""
    headshot_url = headshot_url_fn(espn_id) if headshot_url_fn and espn_id else None
    logo_url = logo_url_fn(team) if logo_url_fn else None

    if headshot_url and logo_url:
        onerror = f"this.onerror=null;this.src='{html.escape(logo_url)}';this.classList.add('is-logo-fallback');"
        return (f'<img class="pcard-photo" src="{html.escape(headshot_url)}" alt="{html.escape(player)}" '
                f'loading="lazy" onerror="{onerror}">')
    if logo_url:
        return f'<img class="pcard-photo is-logo-fallback" src="{html.escape(logo_url)}" alt="{html.escape(player)}" loading="lazy">'
    if headshot_url:
        return f'<img class="pcard-photo" src="{html.escape(headshot_url)}" alt="{html.escape(player)}" loading="lazy">'
    return f'<div class="pcard-photo is-fallback">{html.escape(_initials(player))}</div>'


def espn_headshot_url(espn_id) -> str:
    """Live ESPN CDN URL, predictable from the id alone -- only usable in
    the plain HTML report, not the self-contained Artifact build, which
    can't load external images."""
    return f"https://a.espncdn.com/i/headshots/nfl/players/full/{espn_id}.png"


def prop_card_html(row: dict, home_full: str, away_full: str, kickoff: str,
                    opponent_full: str, headshot_url_fn=None, logo_url_fn=None) -> str:
    """One player-prop card -- anytime-TD only (Anytime-TD-Only Props
    Refocus spec). `row` is a dict from model/player_stats.py's
    score_props() (plus `home_full`/`away_full`/`kickoff`/`opponent_full`
    resolved by the caller, since score_props() only carries abbreviations).
    `headshot_url_fn` resolves an espn_id to an image URL (None in the
    Artifact build, which can't load external images) -- when it can't
    produce one, or the resulting image 404s in the browser,
    `logo_url_fn` (team abbr -> logo URL/data-URI) is the next fallback
    before initials."""
    has_line = row.get("has_line", True)

    if has_line:
        model_prob = row["model_over_prob"]
        is_over = model_prob >= 0.5
        tier, conf = confidence_tier(model_prob if is_over else 1 - model_prob)
        # "strong" keeps the direction-specific color (green for Yes, red
        # for No); "toss-up" is the same yellow regardless of which way
        # it barely leans, since below 60% isn't a real enough call to
        # earn a directional color.
        over_class = "side-toss-up" if tier == "toss-up" else ("side-over" if is_over else "")
        under_class = "side-toss-up" if tier == "toss-up" else ("side-under" if not is_over else "")
        over_conf = f'<span class="side-conf">{conf}</span>' if is_over else ""
        under_conf = f'<span class="side-conf">{conf}</span>' if not is_over else ""
        stat_line = f"Yes {STAT_LABEL['anytime_td']}"
        split_html = (
            '<div class="split">'
            f'<div class="split-side {over_class}"><span class="side-label">Yes</span>{over_conf}</div>'
            f'<div class="split-side {under_class}"><span class="side-label">No</span>{under_conf}</div>'
            '</div>'
        )
        # No real over/under number for anytime-TD -- `line` is an
        # internal 0.5 placeholder, not something a sportsbook posts --
        # so this shows the market's own implied probability instead, the
        # same number `edge` is computed from.
        vs_row = (f'<div class="vs-row"><span>{info_icon("Model Projection")}: '
                  f'<b>{row["model_over_prob"] * 100:.0f}%</b></span>'
                  f'<span>Vegas: <b>{row["market_over_prob"] * 100:.0f}%</b></span></div>')
    else:
        # QA spec Section 2: the sportsbook hasn't priced this player yet,
        # but they're required lineup coverage -- show the model's own
        # number plainly instead of omitting them, with no Yes/No toggle
        # (there's no line to be over/under) and no confidence color,
        # since there's nothing from the market to measure an edge
        # against.
        stat_line = f"{STAT_LABEL['anytime_td']} -- No line available"
        split_html = '<div class="no-line-badge">No line available -- showing model projection only</div>'
        vs_row = (f'<div class="vs-row"><span>{info_icon("Model Projection")}: '
                  f'<b>{row["projection"] * 100:.0f}%</b></span></div>')

    reasoning = reasoning_sentence(row.get("reasoning"), opponent_full)
    injury_html = ""
    if row.get("injury_status") and str(row["injury_status"]) != "nan":
        injury_html = (f'<div class="card-injury">{html.escape(row["player"])} is listed as '
                        f'{html.escape(str(row["injury_status"]))} -- the projection above already '
                        f'accounts for reduced usage, so it\'s not just a raw season average.</div>')

    detail_lines = []
    if reasoning:
        detail_lines.append(f"<p>{html.escape(reasoning)}</p>")
    detail_lines.append(injury_html)

    return f"""<div class="pcard" data-team="{row['team']}" data-pos="{row.get('position') or ''}" data-stat="{row['stat']}">
  <div class="pcard-head">
    {_headshot_html(row['player'], row['team'], row.get('espn_id'), headshot_url_fn, logo_url_fn)}
    <div>
      <div class="pcard-name">{html.escape(row['player'])}</div>
      <div class="pcard-meta">{row['team']} &middot; {row.get('position') or '--'}</div>
    </div>
  </div>
  <div class="pcard-stat">{stat_line}</div>
  <div class="pcard-matchup">{away_full} @ {home_full} &middot; {kickoff}</div>
  {split_html}
  <details class="card-detail">
    <summary>Model reasoning</summary>
    <div class="card-detail-body">
      {vs_row}
      {''.join(detail_lines)}
    </div>
  </details>
</div>"""


def prop_filter_bar_html(rows: list[dict]) -> str:
    """Team + position filters only -- a "stat" filter dimension no
    longer makes sense now that anytime-TD is the only prop type every
    card shares (Anytime-TD-Only Props Refocus spec)."""
    teams = sorted({r["team"] for r in rows})
    positions = sorted({r.get("position") for r in rows if r.get("position")})
    btns = []
    for t in teams:
        btns.append(f'<button type="button" class="filter-btn" data-filter-type="team" data-filter-value="{t}">{t}</button>')
    for p in positions:
        btns.append(f'<button type="button" class="filter-btn" data-filter-type="pos" data-filter-value="{p}">{p}</button>')
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
    tier, conf = confidence_tier(home_prob if home_favored else 1 - home_prob)
    home_class = "side-toss-up" if tier == "toss-up" else ("side-over" if home_favored else "")
    away_class = "side-toss-up" if tier == "toss-up" else ("side-under" if not home_favored else "")
    home_conf = f'<span class="side-conf">{conf}</span>' if home_favored else ""
    away_conf = f'<span class="side-conf">{conf}</span>' if not home_favored else ""

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
