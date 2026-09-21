#!/bin/bash
# Publishes the newest self-contained artifact to GitHub Pages (docs/index.html).
#
#   ./publish_pages.sh           stage docs/index.html + the prediction logs, print what would be committed
#   ./publish_pages.sh --push    ...then commit and push to origin/main
#
# Refuses to publish anything containing a secret from .env (ODDS_API_KEY,
# BALLDONTLIE_API_KEY, FIRECRAWL_API_KEY, ...): docs/ is served publicly and the logs go to a
# public repo. The scan (qa/scan_secrets.py) covers the page AND every log file it stages.
set -euo pipefail
cd "$(dirname "$0")"

PY=venv/bin/python
[ -x "$PY" ] || PY=python3

ARTIFACT=$(ls -t report/output/artifact_week_*_*.html 2>/dev/null | head -1 || true)
if [ -z "$ARTIFACT" ]; then echo "No artifact found -- run build_artifact.py first." >&2; exit 1; fi
WEEK_LABEL=$(basename "$ARTIFACT" .html | sed 's/artifact_//; s/_/ /g')   # e.g. "week 2 2026"

# only files that exist: a single missing path would make `git add` abort and stage nothing
LOG_FILES=()
for f in logs/predictions.jsonl logs/predictions_graded.jsonl logs/season_results.csv \
         logs/props_results.csv logs/recaps.json logs/ledger_annotations.jsonl; do
  [ -e "$f" ] && LOG_FILES+=("$f")
done

"$PY" -m qa.scan_secrets "$ARTIFACT" "${LOG_FILES[@]}"

cp "$ARTIFACT" docs/index.html
echo "Copied $ARTIFACT -> docs/index.html ($(du -h docs/index.html | cut -f1))"

git add docs/index.html "${LOG_FILES[@]}"
git status --short

if [ "${1:-}" = "--push" ]; then
  git commit -m "Publish ${WEEK_LABEL} artifact (GitHub Pages) + prediction logs"
  git push origin HEAD:main
else
  echo "(dry run -- re-run with --push to commit and push)"
fi
