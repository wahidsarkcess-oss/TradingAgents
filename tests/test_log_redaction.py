"""Secrets never reach logs, tracebacks or unattended CLI output.

Fake credentials are assembled at run time so this file itself never holds a
string the secret scanner would flag.
"""

from __future__ import annotations

import io
import json
import logging
import sys

import pytest
from typer.testing import CliRunner

import cli.main as m
from tradingagents import logsafe
from tradingagents.llm_clients.budget import LLMBudgetExceeded

FAKE_OPENAI = "sk-" + "proj-" + "A1b2C3d4" * 6
FAKE_ANTHROPIC = "sk-" + "ant-" + "api03-" + "Z9y8X7w6" * 5
FAKE_GOOGLE = "AI" + "za" + "Q" * 35
FAKE_AWS = "AK" + "IA" + "ABCDEFGHIJKLMNOP"
FAKE_JWT = "ey" + "J" + "a" * 20 + "." + "eyJ" + "b" * 20 + "." + "c" * 20
CUSTOM_VALUE = "my-own-relay-credential-" + "9" * 12   # no known shape: only the env var reveals it


@pytest.fixture
def secrets_in_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI)
    monkeypatch.setenv("MY_RELAY_TOKEN", CUSTOM_VALUE)


@pytest.mark.unit
def test_env_secret_values_are_masked_wherever_they_appear(secrets_in_env):
    text = f"request failed for key={FAKE_OPENAI} via {CUSTOM_VALUE}"
    out = logsafe.redact(text)
    assert FAKE_OPENAI not in out and CUSTOM_VALUE not in out
    assert out.count("***") >= 2


@pytest.mark.unit
@pytest.mark.parametrize("secret", [FAKE_OPENAI, FAKE_ANTHROPIC, FAKE_GOOGLE, FAKE_AWS, FAKE_JWT])
def test_known_credential_shapes_are_masked_even_when_not_in_the_environment(secret):
    assert secret not in logsafe.redact(f"Error: invalid credential {secret} supplied")


@pytest.mark.unit
@pytest.mark.parametrize("text, kept", [
    ("Authorization: Bearer abcdefghijklmnop1234", "Bearer ***"),
    ("GET https://www.alphavantage.co/query?function=X&apikey=ABCDEF123456", "&apikey=***"),
    ("https://api.example.com/v1?token=sekret-value-123&x=1", "?token=***&x=1"),
    ('{"refresh_token": "rt_abcdefghijklmnop"}', '"refresh_token": "***'),
])
def test_bearer_tokens_query_keys_and_json_tokens_are_masked(text, kept):
    out = logsafe.redact(text)
    assert kept in out


@pytest.mark.unit
def test_ordinary_text_is_left_alone():
    text = "NVDA closed at 182.50; RSI 61; Rating: Overweight; tokens counted 12,345"
    assert logsafe.redact(text) == text


@pytest.mark.unit
def test_chatgpt_sign_in_tokens_are_masked(tmp_path, monkeypatch):
    auth = tmp_path / "chatgpt-auth.json"
    access, refresh = "opaque-access-" + "1" * 20, "opaque-refresh-" + "2" * 20
    auth.write_text(json.dumps({"tokens": {"access_token": access, "refresh_token": refresh},
                                "account": {"email": "someone@example.com"}}))
    monkeypatch.setattr(logsafe, "CHATGPT_AUTH_FILE", auth)
    monkeypatch.setattr(logsafe, "_auth_cache", None)
    out = logsafe.redact(f"refresh failed with {refresh}; retry with {access}")
    assert access not in out and refresh not in out


@pytest.mark.unit
def test_the_formatter_masks_messages_and_tracebacks(secrets_in_env):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logsafe.RedactingFormatter("%(message)s"))
    log = logging.getLogger("tests.redaction")
    log.addHandler(handler)
    try:
        try:
            raise RuntimeError(f"401 Unauthorized: {FAKE_OPENAI}")
        except RuntimeError:
            log.exception("provider call failed with %s", CUSTOM_VALUE)
    finally:
        log.removeHandler(handler)
    out = stream.getvalue()
    assert "Traceback" in out and "401 Unauthorized" in out
    assert FAKE_OPENAI not in out and CUSTOM_VALUE not in out


@pytest.mark.unit
def test_a_scope_redacts_streams_and_root_logging_then_restores_them(secrets_in_env, monkeypatch):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    with logsafe.redaction_scope(streams=True):
        print(f"using {FAKE_OPENAI}")
        logging.getLogger("tradingagents.graph").warning("Settling failed: %s", CUSTOM_VALUE)
    assert sys.stdout is out and sys.stderr is err
    assert root.handlers == []
    assert FAKE_OPENAI not in out.getvalue() and "using ***" in out.getvalue()
    assert CUSTOM_VALUE not in err.getvalue() and "Settling failed: ***" in err.getvalue()


@pytest.mark.unit
def test_scrubbing_an_exception_keeps_its_type_and_chain(secrets_in_env):
    try:
        try:
            raise ConnectionError(f"https://host/v1?key={CUSTOM_VALUE}")
        except ConnectionError as inner:
            raise ValueError(f"bad key {FAKE_OPENAI}") from inner
    except ValueError as exc:
        scrubbed = logsafe.scrub_exception(exc)
    assert isinstance(scrubbed, ValueError)
    assert FAKE_OPENAI not in str(scrubbed)
    assert CUSTOM_VALUE not in str(scrubbed.__cause__)


@pytest.mark.unit
def test_the_cli_never_prints_traceback_locals():
    assert m.app.pretty_exceptions_show_locals is False


@pytest.mark.unit
def test_an_unattended_failure_propagates_without_the_key(secrets_in_env, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    def leaky(**kwargs):
        print(f"debug: client built with {FAKE_OPENAI}")
        raise RuntimeError(f"Incorrect API key provided: {FAKE_OPENAI}")

    monkeypatch.setattr(m, "run_analysis", leaky)
    result = CliRunner().invoke(m.app, [])
    assert isinstance(result.exception, RuntimeError)     # still propagates, as before
    assert FAKE_OPENAI not in str(result.exception)
    assert FAKE_OPENAI not in result.output
    assert "debug: client built with ***" in result.output


@pytest.mark.unit
def test_a_budget_stop_exits_3_with_a_usage_line(monkeypatch):
    usage = {"calls": {"attempted": 40, "failed": 1, "sdk_retries": 0},
             "tokens": {"counted_total": 1234, "estimated_unreported": 0},
             "cost_usd": {"estimated": None, "provider_reported": None}}

    def stop(**kwargs):
        raise LLMBudgetExceeded("calls", "LLM call limit reached: 40 per analysis", usage)

    monkeypatch.setattr(m, "run_analysis", stop)
    result = CliRunner().invoke(m.app, [])
    assert result.exit_code == m.BUDGET_EXIT_CODE == 3
    assert "spending limit" in result.output and "40 LLM call(s)" in result.output
    assert "cost not estimated" in result.output
