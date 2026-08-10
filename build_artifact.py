"""Renders the weekly report as a fully self-contained HTML page (all logos +
display fonts embedded as data URIs) for publishing as a shareable Artifact --
Artifacts block remote network requests, so external image/font URLs would
just show as broken. Defaults to the current week; re-run weekly (see the
scheduled task set up alongside this) to keep an already-published Artifact
URL current.

SharpLine Visual Design Overhaul spec: black/gold dark-first design system
(this file + report/build_report.py + report/cards.py + report/news_section.py
all pull from the same CSS custom-property tokens, so a palette change here
is one edit, not per-page restyling), a homepage-style hero + scannable grid
ahead of the existing per-game detail sections, motion/hover polish, and a
mobile-first pass.
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
from report.about import HOW_IT_WORKS_HTML, HOW_IT_WORKS_STYLE
from report.alt_lines import ALT_LINES_STYLE
from report.archive import ARCHIVE_STYLE, archive_html
from report.build_report import _by_day, _day_header, _model_metrics, _row_data
from report.cards import CARDS_SCRIPT, CARDS_STYLE, confidence_tier
from report.charts import CHARTS_SCRIPT, CHARTS_STYLE
from report.leaderboard import LEADERBOARD_STYLE, edge_distribution_chart, leaderboard_html
from report.logos import THROWBACK_LOGOS, current_logo_urls, get_logo_url
from report.news_section import NEWS_STYLE, news_section_html
from report.compare import COMPARE_SCRIPT, COMPARE_STYLE, compare_html
from report.recap import RECAP_STYLE
from report.team_hub import TEAM_HUB_STYLE, team_hubs_html
from report.theme import DAY_BLOCK, GAME_BLOCK, PRINT_STYLE, THEME_STYLE
from report.track_record import TRACK_RECORD_STYLE, track_record_html
from run_week import LOG_PATH, PROPS_LOG_PATH, get_current_week
from data.team_stats import SCHEDULES_PATH, TEAM_STATS_PATH
from model.td_ensemble import BACKTEST_PATH

ASSETS_DIR = os.path.join(ROOT_DIR, "report", "assets")
OUTPUT_DIR = os.path.join(ROOT_DIR, "report", "output")


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
<title>SharpLine -- NFL Week {week} Picks</title>
<style>
{theme_style}

/* ---- hero: the week's biggest edge ---- */
.hero {{
  background: linear-gradient(165deg, var(--surface-raised), var(--surface));
  border: 1px solid var(--accent);
  border-radius: 10px;
  padding: 26px 26px 22px;
  margin-bottom: 12px;
  box-shadow: var(--shadow);
}}
.hero-tag {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: 'Oswald', sans-serif;
  font-size: 11.5px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--paper);
  background: var(--accent);
  border-radius: 999px;
  padding: 4px 11px;
  margin-bottom: 14px;
}}
.hero .matchup-row {{ margin-top: 4px; }}
.hero .teams img {{ height: 56px; width: 56px; }}
.hero .teams .names {{ font-size: 23px; }}

/* ---- quick-scan grid (homepage overview) ---- */
.grid-overview {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
  gap: 10px;
  margin-bottom: 8px;
}}
.grid-item {{
  display: block;
  text-decoration: none;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px;
  box-shadow: var(--shadow);
}}
.grid-item-teams {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: 'Oswald', sans-serif;
  font-weight: 700;
  font-size: 14.5px;
}}
.grid-item-teams img {{ height: 26px; width: 26px; object-fit: contain; flex-shrink: 0; }}
.grid-item-at {{ color: var(--muted); font-weight: 400; font-size: 12px; }}
.grid-item-kickoff {{ font-size: 11.5px; color: var(--muted); margin-top: 6px; }}
.grid-item-pick {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-weight: 600;
  padding: 4px 9px;
  border-radius: 999px;
  margin-top: 10px;
  border: 1px solid var(--border);
}}
.grid-item-pick.pick-strong-home, .grid-item-pick.pick-strong-away {{ background: var(--positive-soft); border-color: var(--positive); color: var(--positive); }}
.grid-item-pick.pick-toss-up {{ background: var(--warning-soft); border-color: var(--warning); color: var(--warning); }}
.grid-item-pick .dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}

/* ---- Artifact-only mobile overrides: the homepage grid/hero don't exist
   in report/build_report.py's plain report, so these stay local rather
   than living in THEME_STYLE's shared mobile block. ---- */
@media (max-width: 640px) {{
  .grid-overview {{ grid-template-columns: 1fr; }}
  .hero .teams img {{ height: 42px; width: 42px; }}
  .hero .teams .names {{ font-size: 18px; }}
}}

.bottom-nav {{
  position: fixed;
  left: 0; right: 0; bottom: 0;
  display: none;
  justify-content: space-around;
  align-items: center;
  background: var(--surface);
  border-top: 1px solid var(--border);
  padding: 10px max(10px, env(safe-area-inset-left)) calc(10px + env(safe-area-inset-bottom));
  z-index: 10;
  box-shadow: 0 -4px 16px rgba(0,0,0,0.35);
}}
.bottom-nav a {{
  font-family: 'Oswald', sans-serif;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  text-decoration: none;
  color: var(--muted);
  text-align: center;
  padding: 6px 10px;
  min-width: 44px;
}}
.bottom-nav a.is-accent {{ color: var(--accent); }}
@media (max-width: 640px) {{
  .bottom-nav {{ display: flex; }}
  .wrap {{ padding-bottom: 96px; }}
}}
{cards_style}
{news_style}
{track_record_style}
{alt_lines_style}
{charts_style}
{recap_style}
{how_it_works_style}
{leaderboard_style}
{archive_style}
{team_hub_style}
{compare_style}
{print_style}</style>
</head>
<body>
<div class="wrap">
  <div class="masthead">
    <div class="wordmark">SharpLine</div>
    <div class="eyebrow">NFL &middot; Week {week} &middot; {season} Season</div>
  </div>
  <div class="intro">Accuracy-first NFL predictions -- real win probabilities, real edges against the market, and player props built from the same model, never dressed up for effect. No betting advice, just the numbers.</div>

  <div class="statline">
    <div class="stat"><div class="num">{test_accuracy}</div><div class="label">Model accuracy, {test_season} backtest</div></div>
    <div class="stat"><div class="num">{baseline_accuracy}</div><div class="label">Always-home baseline</div></div>
    <div class="stat"><div class="num">{n_games}</div><div class="label">Games this week</div></div>
  </div>

  <div class="caveat">{model_type_article} {model_type} model trained on team-level rolling stats (scoring, EPA/play, third-down and red-zone rates, turnover margin, ATS record), Elo ratings, injury reports, and weather, versus the current Vegas line. TD-scorer odds start from each player's own recent scoring rate, then adjust for the opposing defense's TDs-allowed rate and this game's Vegas-implied scoring environment. Informed estimates, not guarantees.</div>

  {news_section}

  {hero_section}

  <div class="section-head" id="slate"><span class="accent">&#9679;</span> This Week's Slate</div>
  <div class="grid-overview">
    {grid_items}
  </div>

  {leaderboard_section}
  {edge_distribution_section}

  <div class="section-head" id="games">Full Breakdown, Game by Game</div>
  {days}

  {team_hubs_section}

  {compare_section}

  {track_record_section}

  {archive_section}

  {how_it_works_section}

  <footer>
    Built with nfl_data_py + The Odds API. Some team logos are throwback-era marks sourced from Wikipedia and SportsLogos.net for personal/non-commercial display.
  </footer>
</div>
<nav class="bottom-nav">
  <a href="#top">Top</a>
  <a href="#slate" class="is-accent">Slate</a>
  <a href="#games">Games</a>
  <a href="#track-record">Record</a>
  <a href="#how-it-works">About</a>
</nav>
<script>{cards_script}
{charts_script}
{compare_script}</script>
</body>
</html>
"""

HERO_BLOCK = """<div class="hero" id="hero">
  <span class="hero-tag">&#9733; Sharpest Edge This Week</span>
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
  {team_comparison_section}
  {props_section}
  {alt_lines_section}
  <div class="subline" style="margin-top:14px;"><a href="#games" style="color: var(--accent);">Jump to the full slate &#8595;</a></div>
</div>"""

GRID_ITEM = """<a class="grid-item" href="#{anchor}">
  <div class="grid-item-teams">
    <img src="{away_logo}" alt="{away_team}"><span>{away_team}</span>
    <span class="grid-item-at">@</span>
    <img src="{home_logo}" alt="{home_team}"><span>{home_team}</span>
  </div>
  <div class="grid-item-kickoff">{kickoff_short}</div>
  <div class="grid-item-pick {pick_class}"><span class="dot"></span>{pick_label}</div>
</a>"""


def _pick_grid_class_and_label(game) -> tuple[str, str]:
    home_prob = game.get("home_win_prob")
    if home_prob is None or pd.isna(home_prob):
        return "pick-toss-up", "No line yet"
    home_favored = home_prob >= 0.5
    prob_for_favored = home_prob if home_favored else 1 - home_prob
    tier, _ = confidence_tier(prob_for_favored)
    team = game["home_team"] if home_favored else game["away_team"]
    css_class = "pick-toss-up" if tier == "toss-up" else f"pick-strong-{'home' if home_favored else 'away'}"
    return css_class, f"{team} {prob_for_favored * 100:.0f}%"


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

    def row_dict(game) -> dict:
        # headshot_url_fn=None: external images can't load in a self-
        # contained Artifact. logo_url_fn=embed: the team-logo fallback
        # (QA spec Section 3) uses the same base64-embedded logos already
        # loaded for the game header, so it stays fully self-contained too.
        # anchor/TD-chip fields come straight from _row_data() now (shared
        # with report/build_report.py's identical GAME_BLOCK rendering);
        # only the logo URLs need overriding here to the embedded data URIs.
        d = _row_data(game, props, headshot_url_fn=None, logo_url_fn=embed)
        d["away_logo"] = embed(game["away_team"])
        d["home_logo"] = embed(game["home_team"])
        return d

    # Hero game logic: whichever game has the biggest edge (the model's
    # own implied spread vs. the market's) -- falling back to raw win-
    # probability confidence for the handful of games with no market
    # line posted yet, since |edge| is undefined without one. This is
    # the "accuracy first" identity from the spec: the hero is whichever
    # game the model disagrees with the market on the most, not
    # whichever matchup has the biggest names.
    def hero_score(game) -> float:
        edge = game.get("edge")
        if edge is not None and pd.notna(edge):
            return abs(edge)
        prob = game.get("home_win_prob")
        return abs(prob - 0.5) if prob is not None and pd.notna(prob) else -1.0

    hero_game = None
    if not predictions.empty:
        scores = predictions.apply(hero_score, axis=1)
        if scores.max() > -1.0:
            hero_game = predictions.loc[scores.idxmax()]

    hero_section = ""
    if hero_game is not None:
        hero_section = HERO_BLOCK.format(**row_dict(hero_game))

    grid_items = []
    days_html = []
    for day, weekday, day_games in _by_day(predictions):
        game_htmls = []
        for _, game in day_games.iterrows():
            d = row_dict(game)
            game_htmls.append(GAME_BLOCK.format(**d))
            pick_class, pick_label = _pick_grid_class_and_label(game)
            grid_items.append(GRID_ITEM.format(
                anchor=d["anchor"], away_logo=d["away_logo"], home_logo=d["home_logo"],
                away_team=game["away_team"], home_team=game["home_team"],
                kickoff_short=d["kickoff"].split("-- ")[-1] if "-- " in d["kickoff"] else d["kickoff"],
                pick_class=pick_class, pick_label=pick_label,
            ))
        days_html.append(DAY_BLOCK.format(day_header=_day_header(day, weekday), games="\n".join(game_htmls)))

    if not grid_items:
        grid_items_html = '<div class="empty-state">No games on the board yet -- check back closer to the week.</div>'
    else:
        grid_items_html = "\n".join(grid_items)

    # Master Honing Plan, Section B: the public track record, built from
    # the same graded logs run_week.py already maintains -- read directly
    # rather than through run_week.py's own loader functions, since those
    # also run the (expensive, live-network) grading pass; this section
    # only ever needs to read what's already been graded and logged, not
    # trigger a fresh grading pass itself.
    log_df = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame()
    props_log_df = pd.read_csv(PROPS_LOG_PATH) if os.path.exists(PROPS_LOG_PATH) else pd.DataFrame()
    track_record_section = track_record_html(log_df, props_log_df)
    archive_section = archive_html(log_df)

    # Master Honing Plan round 2, items #4 + #12: cross-game/prop edge
    # leaderboard + distribution histogram, embedded logos to match the
    # rest of this self-contained Artifact build.
    leaderboard_section = leaderboard_html(predictions, props, embed_logo_fn=embed)
    edge_distribution_section = edge_distribution_chart(predictions, props)

    # Round 2 UX, item #58: same cached parquet files data/team_stats.py
    # and model/td_ensemble.py already maintain, no live fetch.
    team_stats_df = pd.read_parquet(TEAM_STATS_PATH) if os.path.exists(TEAM_STATS_PATH) else pd.DataFrame()
    schedules_df = pd.read_parquet(SCHEDULES_PATH) if os.path.exists(SCHEDULES_PATH) else pd.DataFrame()
    td_backtest_df = pd.read_parquet(BACKTEST_PATH) if os.path.exists(BACKTEST_PATH) else pd.DataFrame()
    team_hubs_section = team_hubs_html(
        predictions, props, team_stats_df, schedules_df, td_backtest_df, row_data_fn=row_dict)
    # headshot_url_fn=None: same reason row_dict() above passes None --
    # external images can't load in a self-contained Artifact.
    compare_section = compare_html(props, td_backtest_df, headshot_url_fn=None, logo_url_fn=embed)

    metrics = _model_metrics()
    # "xgboost" is read aloud as "ex-gee-boost" (vowel sound) same as
    # "ensemble" -- only "logistic" actually wants "a", not "an".
    model_type_article = "An" if metrics["model_type"] in ("ensemble", "xgboost") else "A"
    html = PAGE.format(
        week=week, season=season, n_games=n_games,
        days="\n".join(days_html), theme_style=THEME_STYLE, print_style=PRINT_STYLE,
        cards_style=CARDS_STYLE, cards_script=CARDS_SCRIPT,
        news_style=NEWS_STYLE, news_section=news_section_html(news),
        track_record_style=TRACK_RECORD_STYLE, track_record_section=track_record_section,
        archive_style=ARCHIVE_STYLE, archive_section=archive_section,
        team_hub_style=TEAM_HUB_STYLE, team_hubs_section=team_hubs_section,
        compare_style=COMPARE_STYLE, compare_section=compare_section, compare_script=COMPARE_SCRIPT,
        alt_lines_style=ALT_LINES_STYLE, charts_style=CHARTS_STYLE, charts_script=CHARTS_SCRIPT,
        recap_style=RECAP_STYLE, how_it_works_style=HOW_IT_WORKS_STYLE, how_it_works_section=HOW_IT_WORKS_HTML,
        leaderboard_style=LEADERBOARD_STYLE, leaderboard_section=leaderboard_section,
        edge_distribution_section=edge_distribution_section,
        hero_section=hero_section, grid_items=grid_items_html,
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
