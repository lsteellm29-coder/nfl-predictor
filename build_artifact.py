"""Renders the weekly report as a fully self-contained HTML page (all logos +
the display font embedded as data URIs) for publishing as a shareable
Artifact -- Artifacts block remote network requests, so external image/font
URLs would just show as broken. Defaults to the current week; re-run weekly
(see the scheduled task set up alongside this) to keep an already-published
Artifact URL current.
"""

import argparse
import base64
import os
import sys

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import requests

from config import CURRENT_SEASON
from data.fetch_news import fetch_news
from model.player_stats import score_props
from model.predict import score_week
from report.build_report import _by_day, _day_header, _model_metrics, _row_data
from report.cards import CARDS_SCRIPT, CARDS_STYLE
from report.logos import THROWBACK_LOGOS, current_logo_urls, get_logo_url
from report.news_section import NEWS_STYLE, news_section_html
from run_week import get_current_week

ASSETS_DIR = os.path.join(ROOT_DIR, "report", "assets")
OUTPUT_DIR = os.path.join(ROOT_DIR, "report", "output")

with open(os.path.join(ASSETS_DIR, "fonts", "oswald-700.woff2"), "rb") as f:
    OSWALD_B64 = base64.b64encode(f.read()).decode()


def _load_logo_data_uris() -> dict:
    """abbr -> embeddable data URI, built from the checked-in logo assets
    (report/assets/logos/), not from any URL fetched at build time -- an
    automated weekly run shouldn't depend on the network for something this
    static, and Artifacts couldn't load a remote image anyway."""
    abbrs = sorted(set(current_logo_urls()) | set(THROWBACK_LOGOS))
    mapping = {}
    for abbr in abbrs:
        for ext, mime in [("png", "image/png"), ("svg", "image/svg+xml")]:
            path = os.path.join(ASSETS_DIR, "logos", f"{abbr}.{ext}")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    mapping[abbr] = f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
                break
    return mapping


LOGO_DATA_URIS_BY_ABBR = _load_logo_data_uris()


def embed(abbr: str) -> str:
    if abbr in LOGO_DATA_URIS_BY_ABBR:
        return LOGO_DATA_URIS_BY_ABBR[abbr]
    return get_logo_url(abbr)  # fallback: remote URL (won't load in an Artifact, but keeps this from crashing)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL Week {week} Picks -- {season}</title>
<style>
@font-face {{
  font-family: 'Oswald';
  font-style: normal;
  font-weight: 700;
  src: url(data:font/woff2;base64,{oswald_b64}) format('woff2');
}}

:root {{
  --paper: #F4F5F8;
  --surface: #FFFFFF;
  --ink: #171A21;
  --muted: #666E7D;
  --border: #E1E4EA;
  --accent: #A8710F;
  --accent-soft: #FBF0DC;
  --positive: #2A7A4F;
  --positive-soft: #E7F5EC;
  --negative: #B23A3A;
  --negative-soft: #FBEAEA;
  --warning: #8A6D00;
  --warning-soft: #FFF6D9;
  --edge-glow: transparent;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --paper: #0A0D13;
    --surface: #12151C;
    --ink: #E9EBF0;
    --muted: #8A93A6;
    --border: #232733;
    --accent: #E3A63F;
    --accent-soft: #2A2214;
    --positive: #4FBE81;
    --positive-soft: #12261B;
    --negative: #E0716D;
    --negative-soft: #2A1414;
    --warning: #E8C547;
    --warning-soft: #2E2810;
  }}
}}
:root[data-theme="dark"] {{
  --paper: #0A0D13;
  --surface: #12151C;
  --ink: #E9EBF0;
  --muted: #8A93A6;
  --border: #232733;
  --accent: #E3A63F;
  --accent-soft: #2A2214;
  --positive: #4FBE81;
  --positive-soft: #12261B;
  --negative: #E0716D;
  --negative-soft: #2A1414;
  --warning: #E8C547;
  --warning-soft: #2E2810;
}}
:root[data-theme="light"] {{
  --paper: #F4F5F8;
  --surface: #FFFFFF;
  --ink: #171A21;
  --muted: #666E7D;
  --border: #E1E4EA;
  --accent: #A8710F;
  --accent-soft: #FBF0DC;
  --positive: #2A7A4F;
  --positive-soft: #E7F5EC;
  --negative: #B23A3A;
  --negative-soft: #FBEAEA;
  --warning: #8A6D00;
  --warning-soft: #FFF6D9;
}}

* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{
  background: var(--paper);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  font-size: 16.5px;
  line-height: 1.6;
}}
.wrap {{ max-width: 780px; margin: 0 auto; padding: 40px 20px 80px; }}

.masthead {{ margin-bottom: 8px; }}
.eyebrow {{
  font-family: 'Oswald', sans-serif;
  font-size: 13px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
  font-weight: 700;
}}
h1 {{
  font-family: 'Oswald', sans-serif;
  font-weight: 700;
  font-size: clamp(34px, 7vw, 52px);
  letter-spacing: 0.01em;
  text-transform: uppercase;
  margin: 4px 0 18px;
  text-wrap: balance;
}}

.statline {{
  display: flex;
  gap: 0;
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
  margin-bottom: 36px;
}}
.statline .stat {{
  flex: 1;
  padding: 14px 16px;
  border-right: 1px solid var(--border);
}}
.statline .stat:last-child {{ border-right: none; }}
.statline .stat .num {{
  font-family: 'Oswald', sans-serif;
  font-weight: 700;
  font-size: 26px;
  font-variant-numeric: tabular-nums;
}}
.statline .stat .label {{
  font-size: 12px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-top: 2px;
}}

.caveat {{
  font-size: 13.5px;
  color: var(--muted);
  max-width: 62ch;
  line-height: 1.55;
  margin-bottom: 44px;
}}

.day {{
  font-family: 'Oswald', sans-serif;
  font-weight: 700;
  font-size: 14px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 44px 0 16px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}}
.day:first-of-type {{ margin-top: 0; }}

.game {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 22px 24px;
  margin-bottom: 18px;
}}

.matchup-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  flex-wrap: wrap;
}}
.teams {{
  display: flex;
  align-items: center;
  gap: 12px;
}}
.teams img {{ height: 46px; width: 46px; object-fit: contain; flex-shrink: 0; }}
.teams .names {{
  font-family: 'Oswald', sans-serif;
  font-weight: 700;
  font-size: 19px;
  letter-spacing: 0.01em;
}}
.kickoff {{
  font-size: 13px;
  color: var(--muted);
  text-align: right;
  white-space: nowrap;
}}
.subline {{
  font-size: 13px;
  color: var(--muted);
  margin-top: 6px;
}}

.confidence {{
  margin-top: 16px;
}}
.confidence .track {{
  height: 7px;
  border-radius: 4px;
  background: var(--border);
  overflow: hidden;
}}
.confidence .fill {{
  height: 100%;
  background: var(--accent);
  border-radius: 4px;
}}
.confidence .caption {{
  display: flex;
  justify-content: space-between;
  font-size: 12.5px;
  color: var(--muted);
  margin-top: 5px;
  font-variant-numeric: tabular-nums;
}}
.confidence .caption b {{ color: var(--ink); font-weight: 600; }}

.tiles {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(96px, 1fr));
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: 4px;
  margin-top: 16px;
  overflow: hidden;
}}
.tile {{
  background: var(--surface);
  padding: 10px 12px;
}}
.tile .label {{
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
}}
.tile .value {{
  font-family: 'Oswald', sans-serif;
  font-weight: 700;
  font-size: 17px;
  font-variant-numeric: tabular-nums;
  margin-top: 2px;
}}
.tile.edge-pos {{ background: var(--positive-soft); }}
.tile.edge-pos .value {{ color: var(--positive); }}
.tile.edge-neg {{ background: var(--negative-soft); }}
.tile.edge-neg .value {{ color: var(--negative); }}

.tdwatch {{
  margin-top: 16px;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}}
.chip {{
  display: flex;
  align-items: center;
  gap: 8px;
  background: var(--accent-soft);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 6px 12px 6px 8px;
  font-size: 13px;
}}
.chip .pct {{
  font-family: 'Oswald', sans-serif;
  font-weight: 700;
  color: var(--accent);
  font-variant-numeric: tabular-nums;
}}
.chip .team {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }}

.why {{
  margin-top: 16px;
  font-size: 15.5px;
  line-height: 1.68;
  max-width: 68ch;
  color: var(--ink);
}}

footer {{
  margin-top: 56px;
  padding-top: 20px;
  border-top: 1px solid var(--border);
  font-size: 12.5px;
  color: var(--muted);
  line-height: 1.6;
}}
{cards_style}
{news_style}</style>
</head>
<body>
<div class="wrap">
  <div class="masthead">
    <div class="eyebrow">NFL &middot; {season} Season</div>
    <h1>Week {week} Picks</h1>
  </div>

  <div class="statline">
    <div class="stat"><div class="num">{test_accuracy}</div><div class="label">Model accuracy, {test_season} backtest</div></div>
    <div class="stat"><div class="num">{baseline_accuracy}</div><div class="label">Always-home baseline</div></div>
    <div class="stat"><div class="num">{n_games}</div><div class="label">Games this week</div></div>
  </div>

  <div class="caveat">{model_type_article} {model_type} model trained on team-level rolling stats (scoring, EPA/play, third-down and red-zone rates, turnover margin, ATS record), Elo ratings, injury reports, and weather, versus the current Vegas line. TD-scorer odds start from each player's own recent scoring rate, then adjust for the opposing defense's TDs-allowed rate and this game's Vegas-implied scoring environment. Informed estimates, not guarantees.</div>

  {news_section}

  {days}

  <footer>
    Built with nfl_data_py + The Odds API. Some team logos are throwback-era marks sourced from Wikipedia and SportsLogos.net for personal/non-commercial display.
  </footer>
</div>
<script>{cards_script}</script>
</body>
</html>
"""

DAY_BLOCK = """<div class="day">{day_header}</div>
{games}"""

GAME_BLOCK = """<div class="game">
  <div class="matchup-row">
    <div class="teams">
      <img src="{away_logo}" alt="{away_team}">
      <div class="names">{away_full} @ {home_full}</div>
      <img src="{home_logo}" alt="{home_team}">
    </div>
    <div class="kickoff">{kickoff}</div>
  </div>
  <div class="subline">{coaches}</div>

  {game_pick_card}

  <div class="tdwatch">
    <div class="chip"><span class="pct">{away_td_pct}</span><span class="team">{away_team} &middot; {away_td_name}</span></div>
    <div class="chip"><span class="pct">{home_td_pct}</span><span class="team">{home_team} &middot; {home_td_name}</span></div>
  </div>

  <div class="why">{why}</div>
  {props_section}
</div>"""


def td_chip_parts(pred):
    if not pred:
        return "--", "no data"
    return f"{pred['prob'] * 100:.0f}%", pred["player"]


def build(week: int | None = None, season: int = CURRENT_SEASON) -> str:
    if week is None:
        week = get_current_week(season)
    predictions = score_week(week, season)
    props = score_props(week, season)
    n_games = len(predictions)

    try:
        news = fetch_news()
    except requests.RequestException as e:
        print(f"Warning: couldn't fetch live news ({e}); publishing without it.")
        news = None

    days_html = []
    for day, weekday, day_games in _by_day(predictions):
        game_htmls = []
        for _, game in day_games.iterrows():
            # headshot_url_fn=None: external images can't load in a
            # self-contained Artifact. logo_url_fn=embed: the team-logo
            # fallback (QA spec Section 3) uses the same base64-embedded
            # logos already loaded for the game header, so it stays fully
            # self-contained too.
            d = _row_data(game, props, headshot_url_fn=None, logo_url_fn=embed)
            d["away_logo"] = embed(game["away_team"])
            d["home_logo"] = embed(game["home_team"])
            away_td_pct, away_td_name = td_chip_parts(game.get("away_td_scorer"))
            home_td_pct, home_td_name = td_chip_parts(game.get("home_td_scorer"))
            game_htmls.append(GAME_BLOCK.format(
                **d, away_td_pct=away_td_pct, away_td_name=away_td_name,
                home_td_pct=home_td_pct, home_td_name=home_td_name,
            ))
        days_html.append(DAY_BLOCK.format(day_header=_day_header(day, weekday), games="\n".join(game_htmls)))

    metrics = _model_metrics()
    # "xgboost" is read aloud as "ex-gee-boost" (vowel sound) same as
    # "ensemble" -- only "logistic" actually wants "a", not "an".
    model_type_article = "An" if metrics["model_type"] in ("ensemble", "xgboost") else "A"
    html = PAGE.format(
        week=week, season=season, n_games=n_games,
        days="\n".join(days_html), oswald_b64=OSWALD_B64,
        cards_style=CARDS_STYLE, cards_script=CARDS_SCRIPT,
        news_style=NEWS_STYLE, news_section=news_section_html(news),
        model_type=metrics["model_type"], model_type_article=model_type_article,
        test_accuracy=f"{metrics['test_accuracy']:.1%}" if metrics["test_accuracy"] else "N/A",
        baseline_accuracy=f"{metrics['baseline_accuracy']:.1%}" if metrics["baseline_accuracy"] else "N/A",
        test_season=metrics["test_season"] or "recent",
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"artifact_week_{week}_{season}.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Wrote {out_path} ({len(html)} bytes)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, default=None,
                         help="defaults to the current week if omitted")
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    args = parser.parse_args()
    build(args.week, args.season)


if __name__ == "__main__":
    main()
