#!/bin/bash
# Wednesday automation: run the unit test suite (tests/ -- Master Honing
# Plan, Section D; covers the highest-risk pure-logic functions, cheap
# enough to run first and fail fast before touching any real data),
# refresh data, retrain both the win-probability and anytime-TD models,
# run both calibration audits, validate roster/injury-ID/coverage/
# headshot data quality, score the current week, and publish the
# artifact -- in that order, so a retrain always gets audited before
# its picks get published (Phase 6), and every QA gate from the Player
# Props QA & Data Integrity spec (plus the Audit Fix Plan's Step 2
# injury-ID check) runs before anything gets published. The TD ensemble
# classifier (model/td_ensemble.py) retrains right after its own backtest
# so it's always fit on the walk-forward data currently on disk, never a
# stale prior run's. `set -e` below means any qa.* script that
# sys.exit(1)s halts the whole run -- validate_rosters and
# validate_coverage are the two that actually can; validate_injury_ids
# and validate_headshots never do (both degrade gracefully -- a missing
# espn_id just means no injury discount for that player's prop, a missing
# photo falls back to a team logo -- so both are logged, not blocking).
#
# MCP Integration spec (Firecrawl/Hugging Face/Playwright): two steps
# below now do real extra work with no separate script needed, since both
# are on-by-default parameters of functions this pipeline already calls.
# qa.validate_rosters's run() also cross-checks nfl_data_py's roster
# against ourlads.com's independently-maintained depth charts (32
# Firecrawl calls, rate-limited to 11/min on the free tier -- adds a few
# real minutes here); run_week.py's news fetch also scrapes team-specific
# beat news via Firecrawl (32 more calls, same rate limit). Both degrade
# gracefully to their existing pre-Firecrawl behavior if the API key is
# missing/invalid or the service is down. Hugging Face's zero-shot
# classifier (model/nlp_classifier.py) is opt-in, not called by this
# pipeline by default -- see data/fetch_news.py's classify() docstring
# for why. Playwright (data/fetch_playwright_sources.py) isn't called by
# this pipeline at all -- built and tested, not currently needed by any
# source here (see that module's own docstring).
#
# Combined Build Plan Part 1 step 3: qa.validate_rosters's run() also
# writes data/cache/blocked_players.json -- any player balldontlie AND
# ourlads.com BOTH independently disagree with nfl_data_py about (never
# just one source, to avoid blocking over a single lagging source) gets
# held out of that week's props entirely until manually confirmed.
# model/player_stats.py's score_props() reads that file, so this step
# has to run before run_week.py below, same reason it already ran before
# scoring even prior to this change.
set -euo pipefail

cd "$(dirname "$0")"
source venv/bin/activate

LOG_DIR="logs/automation"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== $(date) : unit tests (tests/ -- hard-fail on any regression) ==="
  python -m pytest tests/ -q

  echo "=== $(date) : refreshing historical data cache ==="
  python -m data.fetch_historical

  echo "=== $(date) : rebuilding team stats ==="
  python -m data.team_stats

  echo "=== $(date) : retraining model ==="
  python -m model.train

  echo "=== $(date) : calibration audit ==="
  python -m model.calibration

  echo "=== $(date) : anytime-TD walk-forward backtest ==="
  python -m model.td_backtest

  echo "=== $(date) : retraining anytime-TD ensemble classifier ==="
  python -m model.td_ensemble

  echo "=== $(date) : anytime-TD calibration audit ==="
  python -m model.td_calibration

  echo "=== $(date) : roster validation (hard-fail on stale/empty roster data) ==="
  python -m qa.validate_rosters

  echo "=== $(date) : injury-ID coverage check (informational -- never blocks) ==="
  python -m qa.validate_injury_ids

  echo "=== $(date) : scoring current week + logging ==="
  python run_week.py

  echo "=== $(date) : props calibration audit (informational -- reports on prior weeks' graded picks) ==="
  python -m model.props_calibration

  echo "=== $(date) : lineup coverage validation (hard-fail only on a team with zero props) ==="
  python -m qa.validate_coverage

  echo "=== $(date) : headshot validation (informational -- never blocks) ==="
  python -m qa.validate_headshots

  echo "=== $(date) : publishing artifact ==="
  python build_artifact.py

  echo "=== $(date) : done ==="
} 2>&1 | tee "$LOG_FILE"
