"""scripts/scan_secrets.py: committed credentials are caught; examples are not.

Every fake credential is assembled at run time, so the scanner never flags this file.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("scan_secrets", ROOT / "scripts" / "scan_secrets.py")
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)

FAKES = {
    "Anthropic API key": "sk-" + "ant-" + "api03-" + "Q" * 40,
    "OpenAI-style API key": "sk-" + "proj-" + "A1b2C3d4" * 6,
    "Google API key": "AI" + "za" + "S" * 35,
    "AWS access key id": "AK" + "IA" + "Z" * 16,
    "GitHub token": "gh" + "p_" + "x" * 36,
    "private key block": "-----BEGIN " + "RSA PRIVATE KEY-----",
}


@pytest.mark.unit
@pytest.mark.parametrize("kind, secret", FAKES.items())
def test_each_credential_format_is_found_and_its_value_never_printed(tmp_path, capsys, kind, secret):
    file = tmp_path / "config.py"
    file.write_text(f"settings = {{}}\nclient = make('{secret}')\n")
    assert scan.main([str(file)]) == 1
    out = capsys.readouterr().out
    assert f"{file}:2: {kind}" in out
    assert secret not in out


@pytest.mark.unit
@pytest.mark.parametrize("line, flagged", [
    ("OPENAI_API_KEY=" + "a7Fq" * 8, True),
    ("export FRED_API_KEY='" + "9" * 32 + "'", True),
    ("  MY_SERVICE_TOKEN: " + "t0k3n" * 4, True),
    ("OPENAI_API_KEY=", False),
    ("#OPENAI_API_KEY=", False),
    ("export OPENAI_API_KEY=...", False),
    ("OPENAI_API_KEY=your-key-here", False),
    ("ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}", False),
    ("OPENAI_API_KEY=placeholder", False),
    ("TRADINGAGENTS_MAX_TOKENS=4096", False),
])
def test_key_assignments_are_flagged_unless_blank_or_a_placeholder(line, flagged):
    assert bool(scan.scan_line(line)) is flagged


@pytest.mark.unit
def test_a_marked_example_line_is_exempt():
    assert scan.scan_line(FAKES["AWS access key id"] + "  # scan-secrets: allow") == []


@pytest.mark.unit
@pytest.mark.parametrize("name, flagged", [
    (".env", True), (".env.production", True), ("chatgpt-auth.json", True), ("secrets.env", True),
    ("prod.secrets.env", True), ("server.pem", True), ("tls.key", True),
    (".env.example", False), (".env.enterprise.example", False), ("secrets.env.example", False),
    ("pricing.json", False),
])
def test_credential_files_are_refused_by_name(tmp_path, name, flagged):
    file = tmp_path / name
    file.write_text("# nothing secret in the content\n")
    assert bool(scan.scan_file(str(file))) is flagged


@pytest.mark.unit
def test_binary_files_are_skipped(tmp_path):
    file = tmp_path / "logo.png"
    file.write_bytes(b"\x89PNG\x00\x00" + FAKES["AWS access key id"].encode())
    assert scan.scan_file(str(file)) == []


@pytest.mark.unit
def test_the_repository_holds_no_secrets():
    """What CI runs: every tracked file, including all the .example templates."""
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        findings = [f for path in scan.tracked_files() for f in scan.scan_file(path)]
    finally:
        os.chdir(cwd)
    assert findings == []


@pytest.mark.unit
@pytest.mark.parametrize("template", [
    ".env.example", ".env.enterprise.example", "deploy/cloud/cloud.env.example",
    "deploy/cloud/secrets.env.example", "deploy/cloud/pricing.example.json",
])
def test_the_shipped_templates_are_clean(template):
    assert scan.scan_file(str(ROOT / template)) == []


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_history_mode_finds_a_key_that_was_committed_and_later_removed(tmp_path, monkeypatch, capsys):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, env=env, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "app.py").write_text(f"KEY = '{FAKES['OpenAI-style API key']}'\n")
    git("add", "app.py")
    git("commit", "-q", "-m", "add")
    (tmp_path / "app.py").write_text("KEY = os.environ['OPENAI_API_KEY']\n")
    git("commit", "-q", "-am", "remove")

    monkeypatch.chdir(tmp_path)
    assert scan.main([]) == 0                 # the tree is clean now...
    assert scan.main(["--history"]) == 1      # ...but the history is not
    out = capsys.readouterr().out
    assert "app.py: OpenAI-style API key" in out
    assert FAKES["OpenAI-style API key"] not in out
