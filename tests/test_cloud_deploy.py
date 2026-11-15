"""deploy/cloud: keys stay out of Git, the image and output; runs are unattended and capped."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLOUD = ROOT / "deploy" / "cloud"
needs_sh = pytest.mark.skipif(sys.platform == "win32" or shutil.which("sh") is None,
                              reason="the entrypoint is a POSIX shell script")

FAKE_KEY = "sk-" + "proj-" + "Kq7" * 14


def _env_file(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip()
    return values


@pytest.fixture
def fake_cli(tmp_path):
    """A stand-in `tradingagents` that reports what it received, never a value."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "tradingagents"
    script.write_text(
        "#!/bin/sh\n"
        'echo "args: $*"\n'
        f'[ "$OPENAI_API_KEY" = "{FAKE_KEY}" ] && echo "openai key: received"\n'
        '[ "$QUOTED_TOKEN" = "two words" ] && echo "quoted: unwrapped"\n'
        '[ "$CRLF_TOKEN" = "abcdefgh" ] && echo "crlf: stripped"\n'
        '[ -n "${EVIL_KEY+x}" ] && echo "evil: $EVIL_KEY"\n'
        '[ -z "${EMPTY_KEY+x}" ] && echo "empty: skipped"\n'
    )
    script.chmod(0o755)
    return bin_dir


def _run_entrypoint(tmp_path, fake_cli, secrets: str, *args):
    secrets_file = tmp_path / "tradingagents.env"
    secrets_file.write_text(secrets)
    env = {"PATH": f"{fake_cli}{os.pathsep}{os.environ['PATH']}",
           "TRADINGAGENTS_SECRETS_FILE": str(secrets_file), "HOME": str(tmp_path)}
    return subprocess.run(["sh", str(CLOUD / "entrypoint.sh"), *args], env=env, cwd=tmp_path,
                          capture_output=True, text=True, timeout=30)


@pytest.mark.unit
@needs_sh
def test_the_entrypoint_exports_secrets_without_printing_or_evaluating_them(tmp_path, fake_cli):
    secrets = (
        "# a comment\n\n"
        f"OPENAI_API_KEY={FAKE_KEY}\n"
        'QUOTED_TOKEN="two words"\n'
        "CRLF_TOKEN=abcdefgh\r\n"  # scan-secrets: allow (fixture, not a credential)
        f"EVIL_KEY=$(touch {tmp_path}/pwned)\n"
        "EMPTY_KEY=\n"
        "not a valid line\n"
        "BAD-NAME=value\n"
    )
    result = _run_entrypoint(tmp_path, fake_cli, secrets, "--ticker", "NVDA", "--no-show")
    assert result.returncode == 0, result.stderr
    assert "args: --ticker NVDA --no-show" in result.stdout
    assert "openai key: received" in result.stdout
    assert "quoted: unwrapped" in result.stdout and "crlf: stripped" in result.stdout
    assert "empty: skipped" in result.stdout
    assert not (tmp_path / "pwned").exists()               # the value was never run
    assert f"evil: $(touch {tmp_path}/pwned)" in result.stdout
    assert FAKE_KEY not in result.stdout and FAKE_KEY not in result.stderr
    assert result.stderr.count("skipped a secrets line") == 2


@pytest.mark.unit
@needs_sh
def test_the_entrypoint_runs_without_a_secrets_file(tmp_path, fake_cli):
    env = {"PATH": f"{fake_cli}{os.pathsep}{os.environ['PATH']}",
           "TRADINGAGENTS_SECRETS_FILE": str(tmp_path / "absent.env")}
    result = subprocess.run(["sh", str(CLOUD / "entrypoint.sh"), "--help"], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and "args: --help" in result.stdout


@pytest.mark.unit
@needs_sh
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads any file")
def test_an_unreadable_secrets_file_stops_the_container(tmp_path, fake_cli):
    secrets_file = tmp_path / "tradingagents.env"
    secrets_file.write_text(f"OPENAI_API_KEY={FAKE_KEY}\n")
    secrets_file.chmod(0)
    env = {"PATH": f"{fake_cli}{os.pathsep}{os.environ['PATH']}", "TRADINGAGENTS_SECRETS_FILE": str(secrets_file)}
    result = subprocess.run(["sh", str(CLOUD / "entrypoint.sh")], env=env, capture_output=True, text=True)
    assert result.returncode == 1 and "not readable" in result.stderr


@pytest.mark.unit
def test_the_compose_service_is_unattended_hardened_and_keyless():
    compose = yaml.safe_load((CLOUD / "docker-compose.cloud.yml").read_text())
    service = compose["services"]["analysis"]
    assert service["tty"] is False and service["stdin_open"] is False
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]
    assert service["environment"]["TRADINGAGENTS_NO_UPDATE_CHECK"] == "1"
    assert service["restart"] == "no"
    assert "ports" not in service                          # nothing listens
    assert service["secrets"][0]["target"] == "tradingagents.env"
    assert compose["secrets"]["tradingagents_secrets"]["file"] == "./secrets.env"
    # Keys never travel as plain environment entries or build args.
    assert not [k for k in service["environment"] if k.endswith(("_KEY", "_TOKEN", "_SECRET"))]
    assert "args" not in service["build"]
    assert service["env_file"] == ["cloud.env"]


@pytest.mark.unit
def test_the_cloud_settings_carry_the_agreed_limits_and_no_provider_or_key():
    settings = _env_file(CLOUD / "cloud.env.example")
    assert settings["TRADINGAGENTS_MAX_LLM_CALLS"] == "40"
    assert settings["TRADINGAGENTS_MAX_TOKENS"] == "4096"
    assert settings["TRADINGAGENTS_MAX_TOKENS_PER_RUN"] == "200000"
    assert settings["TRADINGAGENTS_NO_UPDATE_CHECK"] == "1"
    for name in ("TRADINGAGENTS_LLM_PROVIDER", "TRADINGAGENTS_QUICK_THINK_LLM", "TRADINGAGENTS_DEEP_THINK_LLM"):
        assert settings[name] == ""                       # configurable, never hard-coded
    assert not [k for k in settings if k.endswith(("_API_KEY", "_TOKEN", "_SECRET"))]


@pytest.mark.unit
def test_the_cloud_settings_turn_the_update_check_off(monkeypatch):
    from cli import updates

    for name, value in _env_file(CLOUD / "cloud.env.example").items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    assert updates.enabled() is False


@pytest.mark.unit
def test_the_cloud_settings_build_a_capped_config(monkeypatch):
    import tradingagents.default_config as default_config

    for name, value in _env_file(CLOUD / "cloud.env.example").items():
        monkeypatch.setenv(name, value)
    config = default_config.build_default_config()
    from tradingagents.llm_clients.budget import RunBudget

    budget = RunBudget.from_config(config, [("openai", "m")])
    assert (budget.max_calls, budget.output_reservation, budget.max_tokens) == (40, 4096, 200_000)
    assert config["llm_provider"] == "openai"   # blank in the file: the built-in default stays
    assert config["max_tool_rounds"] == 8


@pytest.mark.unit
@pytest.mark.parametrize("path", [
    ".env", ".env.enterprise", "data/chatgpt-auth.json", "chatgpt-auth.json", "secrets/key.txt",
    "deploy/cloud/secrets.env", "deploy/cloud/cloud.env", "deploy/cloud/prod.secrets.env",
    "certs/server.pem", "certs/server.key",
])
def test_secret_paths_are_gitignored(path):
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    result = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT)
    assert result.returncode == 0, f"{path} is not ignored"


@pytest.mark.unit
@pytest.mark.parametrize("path", [
    ".env.example", "deploy/cloud/cloud.env.example", "deploy/cloud/secrets.env.example",
    "deploy/cloud/pricing.example.json", "deploy/cloud/entrypoint.sh",
])
def test_the_templates_are_not_ignored(path):
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    assert subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT).returncode == 1


@pytest.mark.unit
def test_secrets_are_excluded_from_the_docker_build_context():
    patterns = {line.strip() for line in (ROOT / ".dockerignore").read_text().splitlines()}
    for pattern in ("**/.env*", "data", "secrets", "**/secrets.env", "**/*.secrets.env",
                    "**/chatgpt-auth.json", "**/*.pem", "**/*.key", "deploy/cloud/cloud.env"):
        assert pattern in patterns, pattern


@pytest.mark.unit
@needs_sh
def test_the_entrypoint_accepts_only_credential_names(tmp_path, fake_cli):
    """Review L3: a secrets file could set PATH, LD_PRELOAD or raise a spending limit."""
    probe = fake_cli / "tradingagents"
    probe.write_text(
        "#!/bin/sh\n"
        'for v in LD_PRELOAD PYTHONPATH TRADINGAGENTS_MAX_LLM_CALLS HOME_DIR_TOKEN AWS_ACCESS_KEY_ID '
        'AWS_BEARER_TOKEN_BEDROCK MY_SERVICE_PASSWORD NOT_A_CREDENTIAL; do\n'
        '  eval "val=\\${$v+set}"; echo "$v=${val:-unset}"\n'
        "done\n"
        'case "$PATH" in /evil*) echo "PATH=hijacked" ;; *) echo "PATH=intact" ;; esac\n'
    )
    secrets = (
        "LD_PRELOAD=/tmp/evil.so\n"
        "PYTHONPATH=/tmp/evil\n"
        "PATH=/evil/bin\n"
        "TRADINGAGENTS_MAX_LLM_CALLS=99999\n"
        "NOT_A_CREDENTIAL=value\n"
        "HOME_DIR_TOKEN=abcdefgh\n"  # scan-secrets: allow (fixture)
        "AWS_ACCESS_KEY_ID=AKIDEXAMPLE\n"
        "AWS_BEARER_TOKEN_BEDROCK=bedrock-token\n"
        "MY_SERVICE_PASSWORD=pw-value\n"  # scan-secrets: allow (fixture)
    )
    result = _run_entrypoint(tmp_path, fake_cli, secrets)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    for refused in ("LD_PRELOAD", "PYTHONPATH", "TRADINGAGENTS_MAX_LLM_CALLS", "NOT_A_CREDENTIAL"):
        assert f"{refused}=unset" in out
    for accepted in ("HOME_DIR_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_BEARER_TOKEN_BEDROCK", "MY_SERVICE_PASSWORD"):
        assert f"{accepted}=set" in out
    assert "PATH=intact" in out
    assert "/tmp/evil" not in result.stderr        # names are reported, values never


@pytest.mark.unit
@pytest.mark.parametrize("path, ignored", [
    ("data/chatgpt-auth.json", True), ("data/memory/log.md", True),
    ("tests/data/fixture.csv", False), ("tradingagents/data/x.py", False),
    ("secrets/key.txt", True), ("deploy/secrets/readme.md", False),
])
def test_data_and_secrets_folders_are_ignored_only_at_the_root(path, ignored):
    """Review L8: data/ used to match at any depth."""
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    result = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT)
    assert (result.returncode == 0) is ignored, path
