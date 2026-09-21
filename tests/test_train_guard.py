# model/train_guard.py: retrain only when the training inputs changed; never lose a deployed model.
import os

from model import train_guard
from model.train_guard import archive_deployed_model, fingerprint, needs_retrain, record_fingerprint

FILES = ["a.txt", "sub/b.txt"]


def _tree(tmp_path, a=b"one", b=b"two"):
    (tmp_path / "sub").mkdir(exist_ok=True)
    (tmp_path / "a.txt").write_bytes(a)
    (tmp_path / "sub" / "b.txt").write_bytes(b)
    return str(tmp_path)


def test_fingerprint_is_stable_and_sensitive_to_content(tmp_path):
    root = _tree(tmp_path)
    assert fingerprint(root, FILES) == fingerprint(root, FILES)
    before = fingerprint(root, FILES)
    (tmp_path / "sub" / "b.txt").write_bytes(b"two!")
    assert fingerprint(root, FILES) != before


def test_fingerprint_notices_a_missing_file(tmp_path):
    root = _tree(tmp_path)
    before = fingerprint(root, FILES)
    os.remove(tmp_path / "a.txt")
    assert fingerprint(root, FILES) != before


def test_retrains_when_nothing_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(train_guard, "FINGERPRINT_FILES", FILES)
    model = tmp_path / "model.joblib"
    model.write_bytes(b"m")
    retrain, reason = needs_retrain(str(tmp_path), str(tmp_path / "fp"), str(model))
    assert retrain and "no record" in reason


def test_keeps_the_model_when_inputs_are_unchanged_and_retrains_when_they_change(tmp_path, monkeypatch):
    monkeypatch.setattr(train_guard, "FINGERPRINT_FILES", FILES)
    root = _tree(tmp_path)
    model = tmp_path / "model.joblib"
    model.write_bytes(b"m")
    fp_path = str(tmp_path / "fp")
    record_fingerprint(root, fp_path)

    assert needs_retrain(root, fp_path, str(model)) == (False, "training inputs unchanged since the last training run")
    (tmp_path / "a.txt").write_bytes(b"new data")
    retrain, reason = needs_retrain(root, fp_path, str(model))
    assert retrain and "changed" in reason


def test_force_and_missing_model_always_retrain(tmp_path, monkeypatch):
    monkeypatch.setattr(train_guard, "FINGERPRINT_FILES", FILES)
    root = _tree(tmp_path)
    fp_path = str(tmp_path / "fp")
    record_fingerprint(root, fp_path)
    model = tmp_path / "model.joblib"
    assert needs_retrain(root, fp_path, str(model))[0]                     # no deployed model
    model.write_bytes(b"m")
    assert not needs_retrain(root, fp_path, str(model))[0]
    assert needs_retrain(root, fp_path, str(model), force=True)[0]


def test_archive_copies_by_hash_and_never_overwrites(tmp_path):
    model = tmp_path / "model.joblib"
    model.write_bytes(b"frozen bytes")
    archive = tmp_path / "archive"
    dest = archive_deployed_model(str(model), str(archive))
    assert os.path.basename(dest).startswith("model_") and open(dest, "rb").read() == b"frozen bytes"

    # a later, different model gets its own file; the first stays intact
    model.write_bytes(b"retrained bytes")
    dest2 = archive_deployed_model(str(model), str(archive))
    assert dest2 != dest and open(dest, "rb").read() == b"frozen bytes"
    assert archive_deployed_model(str(tmp_path / "nope.joblib"), str(archive)) is None


def test_archive_never_overwrites_an_existing_file_for_the_same_hash(tmp_path):
    # the exists() guard is what protects an archived artifact -- prove it by planting a sentinel
    # under the name the deployed model would be archived as; it must survive
    model = tmp_path / "model.joblib"
    model.write_bytes(b"deployed bytes")
    archive = tmp_path / "archive"
    archive.mkdir()
    target = archive / f"model_{train_guard.model_version_hash(str(model))}.joblib"
    target.write_bytes(b"SENTINEL")
    assert archive_deployed_model(str(model), str(archive)) == str(target)
    assert target.read_bytes() == b"SENTINEL"
