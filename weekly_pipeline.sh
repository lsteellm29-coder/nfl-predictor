#!/bin/bash
# Wednesday automation: refresh data, retrain, run the calibration audit,
# validate roster/coverage/headshot data quality, score the current week,
# and publish the artifact -- in that order, so a retrain always gets
# audited before its picks get published (Phase 6), and every QA gate
# from the Player Props QA & Data Integrity spec runs before anything
# gets published. `set -e` below means any qa.* script that sys.exit(1)s
# halts the whole run -- validate_rosters and validate_coverage are the
# two that actually can; validate_headshots never does (a missing photo
# degrades gracefully at render time, so it's logged, not blocking).
set -euo pipefail

cd "$(dirname "$0")"
source venv/bin/activate

LOG_DIR="logs/automation"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== $(date) : refreshing historical data cache ==="
  python -m data.fetch_historical

  echo "=== $(date) : rebuilding team stats ==="
  python -m data.team_stats

  echo "=== $(date) : retraining model ==="
  python -m model.train

  echo "=== $(date) : calibration audit ==="
  python -m model.calibration

  echo "=== $(date) : roster validation (hard-fail on stale/empty roster data) ==="
  python -m qa.validate_rosters

  echo "=== $(date) : scoring current week + logging ==="
  python run_week.py

  echo "=== $(date) : lineup coverage validation (hard-fail only on a team with zero props) ==="
  python -m qa.validate_coverage

  echo "=== $(date) : headshot validation (informational -- never blocks) ==="
  python -m qa.validate_headshots

  echo "=== $(date) : publishing artifact ==="
  python build_artifact.py

  echo "=== $(date) : done ==="
} 2>&1 | tee "$LOG_FILE"
