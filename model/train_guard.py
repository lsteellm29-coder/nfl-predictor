# decides whether the weekly pipeline should retrain, and archives the model it would replace
"""weekly_pipeline.sh used to retrain, calibrate and re-save every model every
week. But the training data is fixed: config.HISTORICAL_SEASONS is completed
seasons only, so the weekly refit saw the same rows every time. It still wrote a
brand-new model.joblib -- model.calibration re-serializes the file unconditionally
and XGBoost training isn't byte-reproducible -- so the "model_version" hash the
prediction ledger records changed weekly with no real change behind it, which
defeats the point of logging it, and the artifact a set of predictions came from
was silently overwritten.

This fingerprints what the models are actually trained from (the cached training
tables plus the training/feature code). The pipeline retrains only when that
fingerprint differs from the one recorded at the last training run, or when
FORCE_RETRAIN=1 -- and before it does, copies the deployed model to
model/archive/model_<hash>.joblib, so every hash the ledger has ever mentioned
can be traced back to the exact artifact.

    python -m model.train_guard check      # exit 0 = up to date (skip), 1 = retrain
    python -m model.train_guard archive    # copy the deployed model into model/archive/
    python -m model.train_guard record     # store the current fingerprint (after training)
"""

import hashlib
import os
import shutil
import sys

from model.prediction_log import MODEL_PATH, model_version_hash

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive")
FINGERPRINT_PATH = os.path.join(ROOT, "data", "cache", ".model_trained_on")

# What the trained models depend on: training data, and the code that shapes it.
FINGERPRINT_FILES = [
    "data/cache/pbp.parquet", "data/cache/schedules.parquet", "data/cache/team_stats.parquet",
    "config.py", "model/train.py", "model/elo.py", "model/calibration.py",
    "model/td_model.py", "model/td_backtest.py", "model/td_ensemble.py", "model/td_calibration.py",
    "data/team_stats.py", "data/opponent_adjust.py", "data/situational.py",
    # imported by the training/calibration/TD-backtest code above, so editing them changes what is learned
    "data/fetch_injuries.py", "data/baselines.py", "data/positional_matchups.py",
]


def fingerprint(root: str = ROOT, files: list[str] | None = None) -> str:
    digest = hashlib.sha256()
    for rel in (FINGERPRINT_FILES if files is None else files):
        digest.update(rel.encode())
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            digest.update(b"<missing>")
            continue
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def needs_retrain(root: str = ROOT, fingerprint_path: str = FINGERPRINT_PATH,
                  model_path: str = MODEL_PATH, force: bool = False) -> tuple[bool, str]:
    """(retrain?, reason)."""
    if force:
        return True, "FORCE_RETRAIN set"
    if not os.path.exists(model_path):
        return True, "no deployed model"
    if not os.path.exists(fingerprint_path):
        return True, "no record of what the deployed model was trained from"
    with open(fingerprint_path) as f:
        recorded = f.read().strip()
    if recorded != fingerprint(root):
        return True, "training data or training code changed since the last training run"
    return False, "training inputs unchanged since the last training run"


def archive_deployed_model(model_path: str = MODEL_PATH, archive_dir: str = ARCHIVE_DIR) -> str | None:
    """Copies the deployed model to <archive_dir>/model_<hash>.joblib (never
    overwriting one already there). Returns the archive path, or None if there is
    no deployed model."""
    if not os.path.exists(model_path):
        return None
    os.makedirs(archive_dir, exist_ok=True)
    dest = os.path.join(archive_dir, f"model_{model_version_hash(model_path)}.joblib")
    if not os.path.exists(dest):
        shutil.copy2(model_path, dest)
    return dest


def record_fingerprint(root: str = ROOT, fingerprint_path: str = FINGERPRINT_PATH) -> str:
    os.makedirs(os.path.dirname(fingerprint_path), exist_ok=True)
    fp = fingerprint(root)
    with open(fingerprint_path, "w") as f:
        f.write(fp + "\n")
    return fp


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    command = argv[0] if argv else "check"
    if command == "check":
        retrain, reason = needs_retrain(force=os.environ.get("FORCE_RETRAIN") == "1")
        print(f"{'RETRAIN' if retrain else 'KEEP'}: {reason}")
        return 1 if retrain else 0
    if command == "archive":
        dest = archive_deployed_model()
        print(f"Archived deployed model -> {dest}" if dest else "No deployed model to archive.")
        return 0
    if command == "record":
        print(f"Recorded training fingerprint {record_fingerprint()[:12]}")
        return 0
    print(f"unknown command {command!r} (check | archive | record)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
