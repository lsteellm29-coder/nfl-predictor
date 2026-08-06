#!/bin/bash
# Wednesday automation: refresh data, retrain, run the calibration audit,
# score the current week, and publish the artifact -- in that order, so a
# retrain always gets audited before its picks get published (Phase 6).
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

  echo "=== $(date) : scoring current week + logging ==="
  python run_week.py

  echo "=== $(date) : publishing artifact ==="
  python build_artifact.py

  echo "=== $(date) : done ==="
} 2>&1 | tee "$LOG_FILE"
