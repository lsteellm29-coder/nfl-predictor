# shared SharpLine design-system CSS + fonts + game-card markup
"""SharpLine Visual Design Overhaul spec, Section 1 (design system first):
the black/gold, dark-first palette, the two embedded display/body fonts, and
the page-shell CSS + per-game card markup that every HTML output --
build_artifact.py's self-contained Artifact and report/build_report.py's
plain hostable report -- renders through. A palette or type change is one
edit here, not per-page restyling; report/cards.py and report/news_section.py
already follow the same var(--token, fallback) pattern for their own
components, so this module is the other half of that same system (the page
shell + tokens those components render inside of).

Only the two DAY_BLOCK/GAME_BLOCK markup templates and the CSS/fonts are
shared here -- HERO_BLOCK/GRID_ITEM/the bottom-nav are Artifact-only
homepage concepts (build_artifact.py's single-document "homepage -> game
page" pattern) and stay local to that file, since report/build_report.py's
plain per-week report has no anchor-based single-page navigation to
support.
"""

import base64
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

with open(os.path.join(ASSETS_DIR, "fonts", "oswald-700.woff2"), "rb") as f:
    OSWALD_B64 = base64.b64encode(f.read()).decode()
with open(os.path.join(ASSETS_DIR, "fonts", "inter-var.woff2"), "rb") as f:
    INTER_B64 = base64.b64encode(f.read()).decode()

# Built with plain (unescaped) CSS braces and substituted via .replace(),
# not .format() -- this string itself never goes through str.format(), so
# a page template that embeds it via {theme_style} doesn't need every CSS
# rule's braces doubled the way a literal-in-template block would.
_THEME_TEMPLATE = """
@font-face {
  font-family: 'Oswald';
  font-style: normal;
  font-weight: 700;
  src: url(data:font/woff2;base64,__OSWALD_B64__) format('woff2');
}
@font-face {
  font-family: 'Inter';
  font-style: normal;
  font-weight: 400 700;
  src: url(data:font/woff2;base64,__INTER_B64__) format('woff2');
}

/* SharpLine design system -- black/gold, dark-first. A light override
   still exists ([data-theme="light"]) for a viewer's own toggle, but the
   bare :root (no override, no matter what prefers-color-scheme says) is
   the dark palette -- "dark mode as default" per spec, not just a media-
   query fallback. Confidence colors (positive/negative/warning) are
   untouched from their already-dark-tuned values; only the base surface
   and accent tokens change. */
:root {
  --paper: #0D0D0D;
  --surface: #1A1A1A;
  --surface-raised: #242424;
  --ink: #F2F0E6;
  --muted: #9C9585;
  --border: #2C2A24;
  --accent: #D4AF37;
  --accent-soft: #2E2611;
  --positive: #4FBE81;
  --positive-soft: #12261B;
  --negative: #E0716D;
  --negative-soft: #2A1414;
  --warning: #E8C547;
  --warning-soft: #2E2810;
  /* Second chart-series color (dataviz skill, Master Honing Plan round 2
     item #1) -- validated against --accent via validate_palette.js:
     gold+muted failed the normal-vision separation floor (14.9, just
     under the 15 gate) even though CVD separation itself cleared; this
     steel-blue passes both cleanly (CVD ΔE 20.9-24.4, normal-vision
     ΔE 24.4) while still reading as a deliberate, brand-appropriate
     second color rather than an arbitrary categorical hue. */
  --series-2: #6E88A8;
  --shadow: 0 1px 2px rgba(0,0,0,0.4);
  --shadow-lift: 0 8px 24px rgba(0,0,0,0.5);
}
:root[data-theme="light"] {
  --paper: #F7F5EF;
  --surface: #FFFFFF;
  --surface-raised: #FBF9F3;
  --ink: #191714;
  --muted: #6B6558;
  --border: #E6E1D3;
  --accent: #A8791F;
  --accent-soft: #FBF0DC;
  --positive: #2A7A4F;
  --positive-soft: #E7F5EC;
  --negative: #B23A3A;
  --negative-soft: #FBEAEA;
  --warning: #8A6D00;
  --warning-soft: #FFF6D9;
  --series-2: #4E6485;
  --shadow: 0 1px 2px rgba(25,23,20,0.06);
  --shadow-lift: 0 8px 24px rgba(25,23,20,0.12);
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  background: var(--paper);
  color: var(--ink);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  font-size: 16.5px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 820px; margin: 0 auto; padding: 32px 20px 100px; }
a { color: inherit; }

/* ---- motion: one shared timing for everything that moves ---- */
.game, .pcard, .gcard, .grid-item, .hero, .filter-btn, .split-side,
.card-detail summary::after, .card-section > summary::after {
  transition: transform 200ms ease, box-shadow 200ms ease, background 200ms ease, border-color 200ms ease;
}
@media (hover: hover) {
  .game:hover, .grid-item:hover { transform: translateY(-2px); box-shadow: var(--shadow-lift); border-color: var(--accent); }
  .pcard:hover, .gcard:hover { transform: translateY(-2px); box-shadow: var(--shadow-lift); }
}
.card-detail-body, .props-panel, .news-feed { overflow: hidden; }

/* ---- masthead / wordmark ---- */
.masthead { margin-bottom: 4px; }
.wordmark {
  font-family: 'Oswald', sans-serif;
  font-weight: 700;
  font-size: clamp(30px, 6vw, 40px);
  letter-spacing: 0.04em;
  color: var(--accent);
  text-transform: uppercase;
  margin: 0;
}
.eyebrow {
  font-family: 'Oswald', sans-serif;
  font-size: 12.5px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 700;
  margin-top: 2px;
}
.intro {
  font-size: 15px;
  color: var(--muted);
  max-width: 60ch;
  line-height: 1.55;
  margin: 16px 0 24px;
}

/* ---- top-level Predictions/News tabs (Combined Build Plan Part 5) ----
   Same underlying idea as .filter-btn (a pill of buttons that toggles
   which content shows) but styled as primary navigation, not a
   secondary filter -- an underline on the active tab rather than a
   filled pill, since this switches the whole page's content, not just
   narrows one grid. */
.top-tabs { display: flex; gap: 4px; margin: 20px 0 4px; border-bottom: 1px solid var(--border); }
.top-tab {
  font-family: 'Oswald', sans-serif; font-size: 14px; font-weight: 700; letter-spacing: 0.03em;
  text-transform: uppercase; padding: 10px 16px; border: none; border-bottom: 2px solid transparent;
  background: none; color: var(--muted); cursor: pointer; margin-bottom: -1px;
}
.top-tab:hover { color: var(--ink); }
.top-tab.is-active { color: var(--accent); border-bottom-color: var(--accent); }
.top-tab:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.tab-panel[hidden] { display: none; }

.statline {
  display: flex;
  gap: 0;
  border: 1px solid var(--border);
  border-radius: 6px;
  margin-bottom: 28px;
  overflow: hidden;
}
.statline .stat {
  flex: 1;
  padding: 14px 16px;
  border-right: 1px solid var(--border);
  background: var(--surface);
}
.statline .stat:last-child { border-right: none; }
.statline .stat .num {
  font-family: 'Oswald', sans-serif;
  font-weight: 700;
  font-size: 26px;
  font-variant-numeric: tabular-nums;
  color: var(--accent);
}
.statline .stat .label {
  font-size: 11.5px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-top: 2px;
}

.caveat {
  font-size: 13px;
  color: var(--muted);
  max-width: 62ch;
  line-height: 1.55;
  margin-bottom: 36px;
}

/* ---- section headings (shared by hero, grid, full slate) ---- */
.section-head {
  font-family: 'Oswald', sans-serif;
  font-weight: 700;
  font-size: 13px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 40px 0 14px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}
.section-head:first-of-type { margin-top: 0; }
.section-head .accent { color: var(--accent); }

.day {
  font-family: 'Oswald', sans-serif;
  font-weight: 700;
  font-size: 13px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 36px 0 14px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}

.game {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 22px 24px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
  scroll-margin-top: 16px;
}

.matchup-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  flex-wrap: wrap;
}
.teams {
  display: flex;
  align-items: center;
  gap: 12px;
}
.teams img { height: 46px; width: 46px; object-fit: contain; flex-shrink: 0; }
.teams .names {
  font-family: 'Oswald', sans-serif;
  font-weight: 700;
  font-size: 19px;
  letter-spacing: 0.01em;
}
.kickoff {
  font-size: 13px;
  color: var(--muted);
  text-align: right;
  white-space: nowrap;
}
.matchup-row-right {
  display: flex;
  align-items: center;
  gap: 10px;
}
/* Client-side favorites/watchlist (Master Honing Plan round 2, item
   #13) -- browser-storage only, no login/account system, matching this
   site's currently accountless architecture. Persists in localStorage
   across visits and across a regenerated weekly page (the storage is
   keyed by game anchor, e.g. "g-NE-SEA", not by any per-build content,
   so a favorite survives this file being rebuilt next week). */
.fav-star {
  background: none; border: none; cursor: pointer; padding: 2px;
  font-size: 19px; line-height: 1; color: var(--border);
  transition: color 150ms ease, transform 150ms ease;
}
.fav-star:hover { color: var(--muted); }
.fav-star.is-favorited { color: var(--accent); }
.fav-star:active { transform: scale(1.15); }
.subline {
  font-size: 13px;
  color: var(--muted);
  margin-top: 6px;
}

.tdwatch {
  margin-top: 16px;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
.chip {
  display: flex;
  align-items: center;
  gap: 8px;
  background: var(--accent-soft);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 6px 12px 6px 8px;
  font-size: 13px;
}
.chip .pct {
  font-family: 'Oswald', sans-serif;
  font-weight: 700;
  color: var(--accent);
  font-variant-numeric: tabular-nums;
}
.chip .team { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }

/* preview/narrative text gets a pull-quote treatment, distinct from the
   raw stat tiles/chips around it -- a left accent rule + slightly larger,
   more relaxed type, so the "why" reads as editorial voice, not another
   data row. */
.why {
  margin-top: 18px;
  font-size: 16px;
  line-height: 1.7;
  max-width: 68ch;
  color: var(--ink);
  padding-left: 16px;
  border-left: 3px solid var(--accent);
}

.empty-state {
  font-size: 13.5px;
  color: var(--muted);
  background: var(--surface-raised);
  border: 1px dashed var(--border);
  border-radius: 8px;
  padding: 14px 16px;
  text-align: center;
}

footer {
  margin-top: 48px;
  padding-top: 20px;
  border-top: 1px solid var(--border);
  font-size: 12px;
  color: var(--muted);
  line-height: 1.6;
}

/* ---- mobile-first: single/2-column reflow, bigger tap targets ---- */
@media (max-width: 640px) {
  .wrap { padding: 20px 14px 88px; }
  .card-grid { grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)) !important; }
  .card-section > summary, .card-detail summary, .filter-btn { padding: 10px 4px; min-height: 44px; display: flex; align-items: center; }
  .filter-btn { padding: 8px 14px; }
}
"""

THEME_STYLE = _THEME_TEMPLATE.replace("__OSWALD_B64__", OSWALD_B64).replace("__INTER_B64__", INTER_B64)

# Round 2 UX item #15: a print/export-friendly view. Redefining the design
# tokens inside @media print is enough to re-theme the whole site for paper
# -- every component in this codebase already reads color through
# var(--token) (theme.py's own header comment), so nothing here needs a
# per-component override, just a light, ink-economical palette swapped in
# at the same custom-property layer everything else already reads from.
PRINT_STYLE = """
@media print {
  @page { margin: 0.6in; }

  :root, :root[data-theme="light"], :root[data-theme="dark"] {
    --paper: #FFFFFF;
    --surface: #FFFFFF;
    --surface-raised: #F4F4F4;
    --ink: #000000;
    --muted: #444444;
    --border: #BBBBBB;
    --accent: #7A5C00;
    --accent-soft: #FFFFFF;
    --positive: #1F6B3D;
    --positive-soft: #FFFFFF;
    --negative: #8A2A2A;
    --negative-soft: #FFFFFF;
    --warning: #6B5400;
    --warning-soft: #FFFFFF;
    --series-2: #33506E;
    --shadow: none;
    --shadow-lift: none;
  }
  * { box-shadow: none !important; text-shadow: none !important; animation: none !important; transition: none !important; }
  body { background: #FFFFFF; }

  /* Interactive-only chrome has nothing to do on paper: nothing to click,
     no hover for a tooltip to answer, no localStorage for a star to read
     back. Hiding it isn't losing information -- the definitions/values
     these controls gate are still printed, just not gated. */
  .bottom-nav, .print-page-btn, .fav-star, .card-share-btn, .filter-bar,
  .info-icon, .chart-tooltip, .sort-btn, .compare-controls select, .top-tabs {
    display: none !important;
  }

  /* Tabs are a screen-only navigation device -- paper has no "switch tab"
     gesture, so print shows both the Predictions and News content in one
     pass rather than whichever one happened to be selected on screen. */
  .tab-panel[hidden] { display: block !important; }

  /* Every <details> disclosure in this codebase (card-section, card-detail,
     team-hub, archive week) defaults closed -- exactly the state that's
     wrong on paper, where there's no click to open one. This overrides
     the browser's own UA rule that hides a <details>'s body unless
     [open] is present, so every section prints fully expanded regardless
     of whatever state a reader last left it in on screen. */
  details > *:not(summary) { display: block !important; max-height: none !important; }
  details > summary { list-style: none; }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::after { content: "" !important; }

  /* Keep one game/card/section from splitting across a page boundary
     where a printer or PDF export would otherwise cut it mid-card. */
  .game, .pcard, .gcard, .card-section, .compare-slot { break-inside: avoid; }

  a[href^="#"]::after { content: ""; }
  .wrap { max-width: 100%; padding: 0; }
}
"""


def game_anchor(game) -> str:
    return f"g-{game['away_team']}-{game['home_team']}"


def td_chip_parts(pred: dict | None) -> tuple[str, str]:
    if not pred:
        return "--", "no data"
    return f"{pred['prob'] * 100:.0f}%", pred["player"]


DAY_BLOCK = """<div class="day">{day_header}</div>
{games}"""

GAME_BLOCK = """<div class="game" id="{anchor}">
  <div class="matchup-row">
    <div class="teams">
      <img src="{away_logo}" alt="{away_team}">
      <div class="names">{away_full} @ {home_full}</div>
      <img src="{home_logo}" alt="{home_team}">
    </div>
    <div class="matchup-row-right">
      <button type="button" class="fav-star" data-fav-game="{anchor}"
              aria-label="Favorite {away_full} at {home_full}" aria-pressed="false">&#9733;</button>
      <div class="kickoff">{kickoff}</div>
    </div>
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
</div>"""
