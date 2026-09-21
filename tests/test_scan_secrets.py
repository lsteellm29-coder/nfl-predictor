# qa/scan_secrets.py: the pre-publish check must catch secrets however .env spells them.
import pytest

from qa import scan_secrets
from qa.scan_secrets import find_leaks


def _env(tmp_path, text):
    path = tmp_path / ".env"
    path.write_bytes(text.encode())
    return str(path)


def _file(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


@pytest.mark.parametrize("line", [
    "API_KEY=abcd1234efgh5678",
    'API_KEY="abcd1234efgh5678"',
    "API_KEY='abcd1234efgh5678'",
    "export API_KEY=abcd1234efgh5678",
    "API_KEY = abcd1234efgh5678",
    "API_KEY=abcd1234efgh5678  # the live key",
    "API_KEY=abcd1234efgh5678\r",                     # CRLF line ending
])
def test_finds_the_secret_however_the_env_line_is_written(tmp_path, line):
    env = _env(tmp_path, line + "\n")
    page = _file(tmp_path, "index.html", "<p>hello abcd1234efgh5678 world</p>")
    assert find_leaks([page], env) == [("API_KEY", page)]


def test_clean_files_and_short_values_are_not_flagged(tmp_path):
    env = _env(tmp_path, "API_KEY=abcd1234efgh5678\nDEBUG=true\nEMPTY=\n")
    page = _file(tmp_path, "index.html", "true and nothing else")
    assert find_leaks([page], env) == []


def test_reports_every_file_that_leaks(tmp_path):
    env = _env(tmp_path, "A_KEY=secretvalue-one\nB_KEY=secretvalue-two\n")
    p1 = _file(tmp_path, "one.html", "x secretvalue-one x")
    p2 = _file(tmp_path, "two.jsonl", "secretvalue-two")
    p3 = _file(tmp_path, "three.csv", "clean")
    assert sorted(find_leaks([p1, p2, p3], env)) == sorted([("A_KEY", p1), ("B_KEY", p2)])


def test_main_exits_nonzero_names_keys_but_never_prints_the_value(tmp_path, monkeypatch, capsys):
    env = _env(tmp_path, "API_KEY=abcd1234efgh5678\n")
    page = _file(tmp_path, "index.html", "abcd1234efgh5678")
    monkeypatch.setattr(scan_secrets, "ENV_PATH", env)
    assert scan_secrets.main([page]) == 1
    err = capsys.readouterr().err
    assert "API_KEY" in err and "abcd1234efgh5678" not in err


def test_main_says_so_when_there_is_no_env_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(scan_secrets, "ENV_PATH", str(tmp_path / "missing.env"))
    assert scan_secrets.main([]) == 0
    assert "skipped" in capsys.readouterr().out
