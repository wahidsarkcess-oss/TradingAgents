"""Regression tests for the Phase 1 review findings (H1, M1-M6, L1-L8).

Each test reproduces the failure the review described and shows it no longer
happens. Everything runs offline on scripted models or local fakes; no real
provider is called.
"""

from __future__ import annotations

import copy
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from tests.test_graph_end_to_end import TRADE_DATE, _Client, offline  # noqa: F401
from tests.test_run_budget import (  # noqa: F401
    MODELS,
    MeteredModel,
    _call,
    _config,
    _graph,
    _pricing,
    _result,
    baseline_calls,
)
from tradingagents import logsafe
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph import trading_graph
from tradingagents.graph.checkpointer import clear_all_checkpoints
from tradingagents.llm_clients import budget as budget_module
from tradingagents.llm_clients.budget import (
    COST_LIMIT_PROVIDERS,
    OFFICIAL_HOSTS,
    LLMBudgetExceeded,
    ModelPrice,
    PricingError,
    RunBudget,
    bind_owner,
    endpoint_supported,
    load_pricing,
    note_retry,
    reset_owner,
    settlement_scope,
)
from tradingagents.memory import settlement

ROOT = Path(__file__).resolve().parents[1]


def _settled_prices(*args, **kwargs):
    return 0.02, 0.01, 5, "2026-01-02"


# --- H1: reflections have their own allowance ------------------------------------


@pytest.mark.unit
def test_settlement_calls_draw_on_their_own_allowance():
    budget = RunBudget(max_calls=2, max_reflection_calls=1)
    cb = budget.callback("openai", "quick-model")
    _call(cb)
    _call(cb)                                       # the analysis allowance is now spent
    with settlement_scope():
        _call(cb)                                   # a reflection still goes through
        with pytest.raises(LLMBudgetExceeded) as refused:
            _call(cb)
    assert refused.value.scope == "reflection_calls"
    by = budget.summary()["by_allowance"]
    assert by["analysis"]["calls"]["attempted"] == 2 and by["settlement"]["calls"]["attempted"] == 1
    assert budget.summary(session=True)["calls"]["attempted"] == 3     # both count in the session


@pytest.mark.unit
def test_the_reflection_allowance_defaults_to_a_separate_pool_of_the_analysis_size():
    budget = RunBudget(max_calls=1)
    cb = budget.callback("openai", "quick-model")
    _call(cb)
    with settlement_scope():
        _call(cb)
        with pytest.raises(LLMBudgetExceeded):
            _call(cb)


@pytest.mark.unit
def test_reflection_cost_counts_in_the_daily_ledger(tmp_path):
    cfg = _config(tmp_path, max_cost_per_day_usd=100.0, max_tokens=10, pricing_file=str(_pricing(tmp_path)))
    budget = RunBudget.from_config(cfg, MODELS)
    with settlement_scope():
        _call(budget.callback("openai", "quick-model"), usage=(100, 100))
    ledger = json.loads(next((tmp_path / "cache" / "spend").glob("*.json")).read_text())
    assert ledger["charged_usd"] == pytest.approx(0.3)


@pytest.mark.unit
def test_reflections_never_starve_the_analysis(tmp_path, monkeypatch, offline, baseline_calls):  # noqa: F811
    """Review H1: five due reflections used to eat the analysis's 40 calls."""
    monkeypatch.setattr(settlement, "fetch_returns", _settled_prices)
    model = MeteredModel(structured=True)
    graph = _graph(tmp_path, monkeypatch, model, max_llm_calls_per_run=baseline_calls,
                   max_reflection_calls_per_run=2)
    for day in ("2025-12-01", "2025-12-02", "2025-12-03", "2025-12-04", "2025-12-05"):
        graph.memory_log.store_decision(ticker="AAPL", trade_date=day,
                                        final_trade_decision="**Rating**: Buy", rating="Buy")

    state, signal = graph.propagate("NVDA", TRADE_DATE)

    assert signal == "Overweight"                   # the analysis got its full allowance
    by = graph.usage()["by_allowance"]
    assert by["analysis"]["calls"]["attempted"] == baseline_calls
    assert by["settlement"]["calls"]["attempted"] == 2
    aapl = [e for e in graph.memory_log.load_entries() if e["ticker"] == "AAPL"]
    assert all(not e["pending"] for e in aapl)      # every outcome recorded, so scorable
    assert sum(1 for e in aapl if e["reflection"]) == 2
    assert "settled without a reflection" in state["memory_note"]


@pytest.mark.unit
def test_refused_reflections_are_scored_and_reported_in_a_backtest(tmp_path, monkeypatch, offline,  # noqa: F811
                                                                    baseline_calls):  # noqa: F811
    from tradingagents.backtest import run_backtest, summarize

    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    traded = {"now": False}
    monkeypatch.setattr(settlement, "fetch_returns",
                        lambda *a, **k: _settled_prices() if traded["now"] else (None, None, None, None))
    model = MeteredModel(structured=True)
    monkeypatch.setattr(trading_graph, "create_tier_client",
                        lambda config, tier, **k: _Client(model.model_copy(update={"callbacks": k.get("callbacks")})))
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(results_dir=str(tmp_path / "bt"), data_cache_dir=str(tmp_path / "cache"),
               llm_provider="openai", quick_think_llm="quick-model", deep_think_llm="deep-model",
               max_llm_calls_per_run=baseline_calls, max_reflection_calls_per_run=1)

    def progress(done, total, ticker, date):
        traded["now"] = done == total               # the earlier cells' windows trade at the last one

    result = run_backtest(["NVDA"], dates, cfg, progress=progress)
    summary = summarize(result)

    assert result.cells_run == 4 and not result.failures
    # The last cell's Memory Log step had 3 due, 1 reflection allowed: 2 settled bare.
    assert len(result.unreflected) == 0             # (those were reported by the run itself)
    assert summary.pending == 0 and summary.resolved == 4    # nothing silently left out
    assert summary.unreflected == 2
    assert "settled without a reflection" in summary.render()
    last = next(e for e in result_entries(result) if e["date"] == dates[-1])
    assert last["reflection"]                       # the final pass got a fresh allowance


def result_entries(result):
    from tradingagents.memory import TradingMemoryLog

    return TradingMemoryLog({"memory_log_path": str(result.log_path)}).load_entries()


# --- M1: a resume continues its usage ---------------------------------------------


@pytest.fixture
def free_text_calls(tmp_path, monkeypatch, offline):  # noqa: F811
    model = MeteredModel()
    _graph(tmp_path / "full", monkeypatch, model).propagate("NVDA", TRADE_DATE)
    return len(model.calls)


@pytest.mark.unit
def test_a_resumed_analysis_does_not_get_a_fresh_allowance(tmp_path, monkeypatch, offline, free_text_calls):  # noqa: F811
    n = free_text_calls
    crashing = MeteredModel(fail_at=12)
    graph = _graph(tmp_path, monkeypatch, crashing, checkpoint_enabled=True, max_llm_calls_per_run=n)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        graph.propagate("NVDA", TRADE_DATE)
    assert graph.usage()["calls"]["attempted"] == 12

    # A new process resumes: 12 already spent + the n - 11 calls still to make > n.
    resumed = _graph(tmp_path, monkeypatch, MeteredModel(), checkpoint_enabled=True, max_llm_calls_per_run=n)
    with pytest.raises(LLMBudgetExceeded) as refused:
        resumed.propagate("NVDA", TRADE_DATE)
    assert refused.value.scope == "calls"
    assert refused.value.usage["calls"]["attempted"] == n


@pytest.mark.unit
def test_a_resume_within_the_limit_finishes_and_reports_cumulative_usage(tmp_path, monkeypatch, offline,  # noqa: F811
                                                                         free_text_calls):
    n = free_text_calls
    graph = _graph(tmp_path, monkeypatch, MeteredModel(fail_at=12), checkpoint_enabled=True,
                   max_llm_calls_per_run=n + 1)
    with pytest.raises(RuntimeError):
        graph.propagate("NVDA", TRADE_DATE)
    resumed = _graph(tmp_path, monkeypatch, MeteredModel(), checkpoint_enabled=True, max_llm_calls_per_run=n + 1)

    _, signal = resumed.propagate("NVDA", TRADE_DATE)

    assert signal == "Overweight"
    log = next(tmp_path.glob("results/NVDA/TradingAgentsStrategy_logs/*.json"))
    assert json.loads(log.read_text())["llm_usage"]["calls"]["attempted"] == n + 1
    assert not list((tmp_path / "cache" / "checkpoints").glob("*.usage.json"))   # cleared on success


@pytest.mark.unit
@pytest.mark.parametrize("damage", ["delete", "corrupt"])
def test_a_resume_without_a_trustworthy_usage_record_is_refused(tmp_path, monkeypatch, offline, damage):  # noqa: F811
    graph = _graph(tmp_path, monkeypatch, MeteredModel(fail_at=12), checkpoint_enabled=True,
                   max_llm_calls_per_run=500)
    with pytest.raises(RuntimeError):
        graph.propagate("NVDA", TRADE_DATE)
    record = next((tmp_path / "cache" / "checkpoints").glob("*.usage.json"))
    record.unlink() if damage == "delete" else record.write_text("{not json")
    model = MeteredModel()
    resumed = _graph(tmp_path, monkeypatch, model, checkpoint_enabled=True, max_llm_calls_per_run=500)
    with pytest.raises(LLMBudgetExceeded) as refused:
        resumed.propagate("NVDA", TRADE_DATE)
    assert refused.value.scope == "resume"
    assert model.calls == []


@pytest.mark.unit
def test_without_limits_a_resume_works_even_without_a_record(tmp_path, monkeypatch, offline):  # noqa: F811
    graph = _graph(tmp_path, monkeypatch, MeteredModel(fail_at=12), checkpoint_enabled=True)
    with pytest.raises(RuntimeError):
        graph.propagate("NVDA", TRADE_DATE)
    for record in (tmp_path / "cache" / "checkpoints").glob("*.usage.json"):
        record.unlink()
    _, signal = _graph(tmp_path, monkeypatch, MeteredModel(), checkpoint_enabled=True).propagate("NVDA", TRADE_DATE)
    assert signal == "Overweight"


@pytest.mark.unit
def test_clearing_all_checkpoints_removes_the_usage_records(tmp_path, monkeypatch, offline):  # noqa: F811
    graph = _graph(tmp_path, monkeypatch, MeteredModel(fail_at=12), checkpoint_enabled=True)
    with pytest.raises(RuntimeError):
        graph.propagate("NVDA", TRADE_DATE)
    assert list((tmp_path / "cache" / "checkpoints").glob("*.usage.json"))
    clear_all_checkpoints(tmp_path / "cache")
    assert not list((tmp_path / "cache" / "checkpoints").glob("*.usage.json"))


# --- M2: run ownership -------------------------------------------------------------


@pytest.mark.unit
def test_a_new_run_keeps_in_flight_reservations_and_charges_late_completions_to_their_run():
    budget = RunBudget(max_calls=10)
    cb = budget.callback("openai", "quick-model")
    first = budget.begin_run()
    token = bind_owner(first)
    run_id = uuid.uuid4()
    try:
        cb.on_chat_model_start({}, [[HumanMessage(content="hello")]], run_id=run_id)
    finally:
        reset_owner(token)

    budget.end_run(first)                           # the run failed while this call was in flight
    second = budget.begin_run()                     # the next backtest cell starts
    assert first.ended and not second.ended and run_id in budget._inflight

    cb.on_llm_end(_result(usage=(100, 50)), run_id=run_id)   # the old call completes late
    assert first.scopes["analysis"].reported_input == 100
    assert second.total.attempted == 0
    assert budget.summary(session=True)["tokens"]["provider_reported_input"] == 100

    token = bind_owner(first)
    try:
        with pytest.raises(LLMBudgetExceeded) as refused:
            _call(cb)                               # the old run asks again
    finally:
        reset_owner(token)
    assert refused.value.scope == "run_ended"


# The orphaned-analyst scenario is tested deterministically in test_budget_second_pass.py.


@pytest.mark.unit
def test_every_call_of_a_run_is_owned_by_it(tmp_path, monkeypatch, offline):  # noqa: F811
    model = MeteredModel(structured=True)
    graph = _graph(tmp_path, monkeypatch, model)
    graph.propagate("NVDA", TRADE_DATE)
    account = graph.budget.current
    assert account.ended
    assert account.total.attempted == len(model.calls)


# --- M3: the daily ledger ----------------------------------------------------------


def _reserve_in_process(args):
    """One process making calls that stay in flight (never completed)."""
    ledger_dir, pricing_path, attempts = args
    budget = RunBudget(max_day_cost=1.0, output_reservation=100, output_limited=True,
                       pricing=load_pricing(pricing_path, 90), ledger_dir=ledger_dir,
                       models=(("openai", "quick-model"),))
    cb = budget.callback("openai", "quick-model")
    allowed = 0
    for _ in range(attempts):
        try:
            cb.on_chat_model_start({}, [[HumanMessage(content="x" * 20)]], run_id=uuid.uuid4())
            allowed += 1
        except LLMBudgetExceeded:
            pass
    return allowed


@pytest.mark.unit
@pytest.mark.skipif(sys.platform == "win32", reason="fork-based process pool")
def test_concurrent_processes_cannot_jointly_pass_the_daily_limit(tmp_path):
    pricing = _pricing(tmp_path)          # each call reserves 10 in + 100 out = $0.21; 4 fit in $1
    with multiprocessing.get_context("fork").Pool(4) as pool:
        allowed = pool.map(_reserve_in_process, [(str(tmp_path / "spend"), str(pricing), 5)] * 4)
    assert sum(allowed) == 4
    ledger = json.loads(next((tmp_path / "spend").glob("*.json")).read_text())
    assert len(ledger["reserved"]) == 4 and sum(ledger["reserved"].values()) <= 1.0


@pytest.mark.unit
def test_a_corrupt_ledger_is_never_reset(tmp_path):
    cfg = _config(tmp_path, max_cost_per_day_usd=5.0, max_tokens=10, pricing_file=str(_pricing(tmp_path)))
    budget = RunBudget.from_config(cfg, MODELS)
    cb = budget.callback("openai", "quick-model")
    run_id = uuid.uuid4()
    cb.on_chat_model_start({}, [[HumanMessage(content="hi")]], run_id=run_id)
    ledger = next((tmp_path / "cache" / "spend").glob("*.json"))
    ledger.write_text("garbage")                    # damaged mid-call
    cb.on_llm_end(_result(usage=(10, 10)), run_id=run_id)
    assert ledger.read_text() == "garbage"          # not overwritten with a fresh $0 total
    with pytest.raises(LLMBudgetExceeded) as refused:
        _call(cb)
    assert refused.value.scope == "ledger"


@pytest.mark.unit
def test_spend_is_recorded_whenever_prices_are_known(tmp_path):
    cfg = _config(tmp_path, max_cost_per_run_usd=5.0, max_tokens=10, pricing_file=str(_pricing(tmp_path)))
    _call(RunBudget.from_config(cfg, MODELS).callback("openai", "quick-model"), usage=(100, 100))
    unlimited = _config(tmp_path, pricing_file=str(_pricing(tmp_path)))
    _call(RunBudget.from_config(unlimited, MODELS).callback("openai", "quick-model"), usage=(100, 100))
    ledger = json.loads(next((tmp_path / "cache" / "spend").glob("*.json")).read_text())
    assert ledger["charged_usd"] == pytest.approx(0.6) and ledger["calls"] == 2 and ledger["reserved"] == {}


@pytest.mark.unit
def test_a_failed_ledger_write_is_an_error(tmp_path, monkeypatch, caplog):
    def fail(path, data):
        raise OSError("disk full")

    monkeypatch.setattr(budget_module, "_write_json_atomic", fail)
    limited = RunBudget.from_config(
        _config(tmp_path, max_cost_per_day_usd=5.0, max_tokens=10, pricing_file=str(_pricing(tmp_path))), MODELS)
    with pytest.raises(LLMBudgetExceeded) as refused:
        _call(limited.callback("openai", "quick-model"))
    assert refused.value.scope == "ledger"

    recording = RunBudget.from_config(_config(tmp_path, pricing_file=str(_pricing(tmp_path))), MODELS)
    _call(recording.callback("openai", "quick-model"))          # no cost limit: the call goes ahead
    assert "could not be written" in caplog.text


# --- M4: cost limits only where usage is reliable ----------------------------------


@pytest.mark.unit
def test_a_cost_limit_without_an_output_limit_is_refused(tmp_path):
    with pytest.raises(PricingError, match="per-call output limit"):
        RunBudget.from_config(_config(tmp_path, max_cost_per_run_usd=1.0, pricing_file=str(_pricing(tmp_path))),
                              MODELS)


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["chatgpt", "ollama", "openai_compatible"])
def test_cost_limits_are_refused_for_providers_without_reliable_usage(tmp_path, provider):
    pricing = _pricing(tmp_path, models={f"{provider}:m": {"input_per_mtok": 0, "output_per_mtok": 0}})
    with pytest.raises(PricingError, match="not supported for provider"):
        RunBudget.from_config(_config(tmp_path, max_cost_per_run_usd=1.0, max_tokens=100,
                                      pricing_file=str(pricing)), [(provider, "m")])


@pytest.mark.unit
@pytest.mark.parametrize("url, ok", [
    (None, True), ("https://api.openai.com/v1", True),
    ("https://my-relay.example.com/v1", False), ("http://localhost:8000/v1", False),
])
def test_cost_limits_need_the_official_endpoint(tmp_path, url, ok):
    cfg = _config(tmp_path, max_cost_per_run_usd=1.0, max_tokens=100, pricing_file=str(_pricing(tmp_path)))
    models = [("openai", "quick-model", url), ("openai", "deep-model", url)]
    if ok:
        RunBudget.from_config(cfg, models)
    else:
        with pytest.raises(PricingError, match="custom endpoint"):
            RunBudget.from_config(cfg, models)


@pytest.mark.unit
def test_a_chatgpt_plan_with_a_cost_limit_stops_the_graph_before_any_call(tmp_path, monkeypatch):
    model = MeteredModel()
    pricing = _pricing(tmp_path, models={"chatgpt:quick-model": {"input_per_mtok": 0, "output_per_mtok": 0},
                                         "chatgpt:deep-model": {"input_per_mtok": 0, "output_per_mtok": 0}})
    with pytest.raises(PricingError):
        _graph(tmp_path, monkeypatch, model, llm_provider="chatgpt", max_cost_per_run_usd=1.0,
               max_tokens=100, pricing_file=str(pricing))
    assert model.calls == []


@pytest.mark.unit
def test_the_official_hosts_match_the_provider_registry():
    from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS

    for provider, spec in OPENAI_COMPATIBLE_PROVIDERS.items():
        if provider in COST_LIMIT_PROVIDERS and spec.base_url:
            assert endpoint_supported(provider, spec.base_url), provider
    assert {"chatgpt", "ollama", "openai_compatible"}.isdisjoint(OFFICIAL_HOSTS)


@pytest.mark.unit
def test_prompt_estimates_count_bytes_so_cjk_text_is_not_underestimated():
    budget = RunBudget(max_tokens=10_000, output_reservation=1)
    cb = budget.callback("openai", "quick-model")
    run_id = uuid.uuid4()
    cb.on_chat_model_start({}, [[HumanMessage(content="市场" * 500)]], run_id=run_id)   # 1,000 CJK characters
    assert budget._inflight[run_id].input_estimate >= 1_000


# --- M5: credentials in URLs -------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("text, secret", [
    ("connect to https://admin:hunter2pass99@relay.internal/v1 failed", "hunter2pass99"),  # scan-secrets: allow (fixture)
    ("clone https://ghp_" + "Q" * 30 + "@github.com/x", "ghp_" + "Q" * 30),
    ("postgres://app:" + "Zz9" * 4 + "@db:5432/x", "Zz9" * 4),
])
def test_url_user_info_is_redacted(text, secret):
    out = logsafe.redact(text)
    assert secret not in out and "@" in out


@pytest.mark.unit
def test_a_configured_backend_url_credential_is_masked_everywhere(tmp_path, monkeypatch):
    password = "relay-pass-" + "7" * 10
    _graph(tmp_path, monkeypatch, MeteredModel(),
           backend_url=f"https://user:{password}@relay.example/v1?api_key=qk-{'5' * 12}")
    assert password not in logsafe.redact(f"auth failed for {password}")
    assert "qk-" + "5" * 12 not in logsafe.redact("key qk-" + "5" * 12)


@pytest.mark.unit
def test_credentials_in_endpoint_environment_variables_are_masked(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://u:" + "envpass" + "4321" + "@relay.example/v1")
    assert "envpass4321" not in logsafe.redact("the relay rejected envpass4321")


# --- M6: the scanner ---------------------------------------------------------------


def _scanner():
    import importlib.util

    spec = importlib.util.spec_from_file_location("scan_secrets_r", ROOT / "scripts" / "scan_secrets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
@pytest.mark.parametrize("line", [
    'client = OpenAI(api_key="' + "a1B2" * 8 + '")',
    '{"api_key": "' + "k9Lm" * 6 + '"}',
    "password: '" + "Pw7x" * 5 + "'",
    "DATABASE_URL=postgres://u:" + "p4sS" * 4 + "@db/x",
    "hf_" + "A1" * 17,
    "glpat-" + "x1" * 10,
    "sk_" + "live_" + "z9" * 12,
    "GOCSPX-" + "y8" * 14,
    "npm_" + "n3" * 18,
    "Authorization: Bearer " + "t0kQ" * 6,
])
def test_the_formats_the_review_found_missing_are_now_caught(line):
    assert _scanner().scan_line(line), line


@pytest.mark.unit
@pytest.mark.parametrize("line", [
    'api_key = os.environ["OPENAI_API_KEY"]',
    '{"api_key": "placeholder"}',
    '("legacy google_api_key", {"google_api_key": "legacy-key-456"})',
    '"backend_url": "https://user:secret@relay.example/v1"',
    "# scheme://user:password@host",
])
def test_code_reading_keys_and_obvious_placeholders_are_not_flagged(line):
    assert _scanner().scan_line(line) == []


@pytest.mark.unit
@pytest.mark.parametrize("name", [".npmrc", ".pypirc", ".netrc", "credentials", ".git-credentials",
                                  "service-account-prod.json", "release.jks", "vault.kdbx"])
def test_sensitive_file_names_are_refused(tmp_path, name):
    file = tmp_path / name
    file.write_text("nothing\n")
    assert _scanner().scan_file(str(file))


@pytest.mark.unit
def test_a_key_deep_inside_a_large_file_is_found(tmp_path, capsys):
    """Review M6: files over 2 MB used to be skipped with only a warning."""
    big = tmp_path / "dump.txt"
    big.write_text(("x" * 99 + "\n") * 40_000 + "KEY = '" + "AK" + "IA" + "B" * 16 + "'\n")   # ~4 MB
    assert _scanner().main([str(big)]) == 1
    assert "AWS access key id" in capsys.readouterr().out


def _git(cwd, *args):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_history_finds_a_key_added_in_a_merge_resolution(tmp_path, monkeypatch, capsys):
    secret = "sk-" + "proj-" + "M3rG" * 10
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "app.py").write_text("KEY = None\n")
    _git(tmp_path, "add", "app.py")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "checkout", "-q", "-b", "side")
    (tmp_path / "app.py").write_text("KEY = 'side'\n")
    _git(tmp_path, "commit", "-q", "-am", "side")
    _git(tmp_path, "checkout", "-q", "main")
    (tmp_path / "app.py").write_text("KEY = 'main'\n")
    _git(tmp_path, "commit", "-q", "-am", "main")
    _git(tmp_path, "merge", "-q", "side")            # conflicts
    (tmp_path / "app.py").write_text(f"KEY = '{secret}'\n")
    _git(tmp_path, "commit", "-q", "-am", "merge")
    (tmp_path / "app.py").write_text("KEY = None\n")
    _git(tmp_path, "commit", "-q", "-am", "clean")

    monkeypatch.chdir(tmp_path)
    assert _scanner().main(["--history"]) == 1
    out = capsys.readouterr().out
    assert "app.py: OpenAI-style API key" in out and secret not in out


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_history_warns_about_a_shallow_clone(tmp_path, monkeypatch, capsys):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q")
    for n in range(3):
        (origin / "f.txt").write_text(str(n))
        _git(origin, "add", "f.txt")
        _git(origin, "commit", "-q", "-m", str(n))
    _git(tmp_path, "clone", "-q", "--depth", "1", origin.as_uri(), "shallow")
    monkeypatch.chdir(tmp_path / "shallow")
    _scanner().main(["--history"])
    assert "shallow clone" in capsys.readouterr().err


@pytest.mark.unit
def test_reviewed_history_findings_can_be_allowlisted_by_location(tmp_path, monkeypatch, capsys):
    scan = _scanner()
    file = tmp_path / "f.py"
    file.write_text("token = '" + "AK" + "IA" + "Q" * 16 + "'\n")
    finding = scan.scan_file(str(file))[0]
    (tmp_path / ".scan-secrets-allow").write_text(f"# reviewed\n{finding}\n")
    monkeypatch.chdir(tmp_path)
    assert scan.main([str(file)]) == 0


# --- L1: SDK retries -----------------------------------------------------------------


@pytest.mark.unit
def test_an_sdk_retry_past_the_call_limit_is_refused_before_it_is_sent():
    budget = RunBudget(max_calls=1)
    cb = budget.callback("openai", "quick-model")
    run_id = uuid.uuid4()
    cb.on_chat_model_start({}, [[HumanMessage(content="hello")]], run_id=run_id)
    with pytest.raises(LLMBudgetExceeded, match="retry refused"):
        note_retry()                                # the hooked SDK, about to retry
    assert budget.summary()["calls"]["sdk_retries"] == 0


@pytest.mark.unit
def test_a_real_openai_client_stops_retrying_at_the_limit():
    """The real SDK, with the suite's network refusal standing in for an outage."""
    from langchain_openai import ChatOpenAI

    budget = RunBudget(max_calls=2)
    llm = ChatOpenAI(model="quick-model", base_url="http://127.0.0.1:9/v1", api_key="dummy-local",
                     max_retries=5, callbacks=[budget.callback("openai", "quick-model")])
    with pytest.raises(LLMBudgetExceeded):
        llm.invoke("hello")
    calls = budget.summary()["calls"]
    assert (calls["attempted"], calls["sdk_retries"], calls["failed"]) == (1, 1, 1)   # not 1 + 5 retries


@pytest.mark.unit
def test_retries_are_counted_and_never_refused_without_limits():
    budget = RunBudget()
    cb = budget.callback("openai", "quick-model")
    run_id = uuid.uuid4()
    cb.on_chat_model_start({}, [[HumanMessage(content="hello")]], run_id=run_id)
    for _ in range(5):
        note_retry()
    cb.on_llm_end(_result(), run_id=run_id)
    assert budget.summary()["calls"]["sdk_retries"] == 5


# --- L2: TypeSafe screening ------------------------------------------------------------


@pytest.fixture
def jev(monkeypatch):
    from tradingagents.agents import post_screen

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-" + "local" * 4)
    monkeypatch.setattr(post_screen, "resolve_instrument_identity", lambda t: {"company_name": "NVIDIA"})
    return post_screen


@pytest.mark.unit
def test_typesafe_screening_is_not_made_under_a_cost_limit(tmp_path, jev, monkeypatch):
    monkeypatch.setattr(jev, "_ask_each", lambda *a: pytest.fail("a paid request was sent"))
    budget = RunBudget(max_run_cost=1.0, output_limited=True, pricing=load_pricing(_pricing(tmp_path), 90))
    token = bind_owner(budget.current)
    try:
        keep, note = jev.jev_screen("NVDA")(["post one", "post two"])
    finally:
        reset_owner(token)
    assert keep == [True, True] and "skipped" in note


@pytest.mark.unit
def test_typesafe_requests_are_counted_in_the_run(jev, monkeypatch):
    sent = []
    monkeypatch.setattr(jev.requests, "post", lambda *a, **k: sent.append(1) or _typesafe_ok())
    monkeypatch.setattr(jev, "_answers", lambda response, questions: {
        "about": {"noul": 1.0}, "stance": {"choice": "bullish"}})
    monkeypatch.setattr(jev, "_stance", lambda a: "bullish")
    budget = RunBudget(max_calls=40)
    account = budget.begin_run()
    with budget.bound(account):
        jev.jev_screen("NVDA")(["a", "b", "c"])
    assert len(sent) == 3
    assert account.total.external_requests == 3     # every request sent, counted in its own run
    assert account.total.attempted == 0             # not LLM calls, counted apart


def _typesafe_ok():
    return type("Response", (), {"status_code": 200, "headers": {}})()


# --- L5: exceptions whose message cannot be scrubbed in place ----------------------------


class _Leaky(Exception):
    def __init__(self, key):
        super().__init__("request failed")
        self.key = key

    def __str__(self):
        return f"request failed with {self.key}"


@pytest.mark.unit
def test_an_exception_that_still_leaks_after_scrubbing_is_replaced(monkeypatch):
    key = "sk-" + "proj-" + "L3aK" * 10
    safe = logsafe.safe_exception(_Leaky(key))
    assert isinstance(safe, logsafe.RedactedError)
    assert key not in str(safe) and "_Leaky" in str(safe)
    ordinary = ValueError(f"bad {key}")
    assert logsafe.safe_exception(ordinary) is ordinary and key not in str(ordinary)


@pytest.mark.unit
def test_the_cli_replaces_a_leaky_exception(monkeypatch):
    from typer.testing import CliRunner

    import cli.main as m

    key = "sk-" + "proj-" + "L3aK" * 10
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(m, "run_analysis", lambda **k: (_ for _ in ()).throw(_Leaky(key)))
    result = CliRunner().invoke(m.app, [])
    assert isinstance(result.exception, logsafe.RedactedError)
    assert key not in str(result.exception) and key not in result.output


# --- L6: a fault in the budget itself --------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("limited", [False, True])
def test_a_budget_fault_breaks_nothing_without_limits_and_refuses_with_them(monkeypatch, limited, caplog):
    budget = RunBudget(max_calls=5) if limited else RunBudget()

    def broken(*args, **kwargs):
        raise KeyError("bug")

    monkeypatch.setattr(budget, "_start", broken)
    cb = budget.callback("openai", "quick-model")
    if limited:
        with pytest.raises(LLMBudgetExceeded) as refused:
            cb.on_chat_model_start({}, [[HumanMessage(content="x")]], run_id=uuid.uuid4())
        assert refused.value.scope == "usage_record"
    else:
        cb.on_chat_model_start({}, [[HumanMessage(content="x")]], run_id=uuid.uuid4())
        assert "could not record a call" in caplog.text


# --- L7: Anthropic cache writes ------------------------------------------------------------


@pytest.mark.unit
def test_cache_writes_are_priced_and_never_below_twice_the_input_rate_when_unpriced():
    priced = ModelPrice(input_per_mtok=1.0, output_per_mtok=0.0, cache_write_per_mtok=1.25)
    fallback = ModelPrice(input_per_mtok=1.0, output_per_mtok=0.0)
    # 1M input tokens, of which 400k were cache writes (Anthropic counts them in input_tokens).
    assert priced.cost(1_000_000, 0, cache_write_tokens=400_000) == pytest.approx(0.6 + 0.5)
    assert fallback.cost(1_000_000, 0, cache_write_tokens=400_000) == pytest.approx(0.6 + 0.8)


@pytest.mark.unit
def test_cache_write_tokens_reported_by_the_provider_are_charged(tmp_path):
    cfg = _config(tmp_path, pricing_file=str(_pricing(tmp_path)))
    budget = RunBudget.from_config(cfg, MODELS)
    cb = budget.callback("openai", "quick-model")
    run_id = uuid.uuid4()
    cb.on_chat_model_start({}, [[HumanMessage(content="x")]], run_id=run_id)
    response = _result(usage=(1000, 0))
    response.generations[0][0].message.usage_metadata["input_token_details"] = {"cache_creation": 1000}
    cb.on_llm_end(response, run_id=run_id)
    summary = budget.summary()
    assert summary["tokens"]["provider_reported_cache_write"] == 1000
    assert summary["cost_usd"]["estimated"] == pytest.approx(1000 * 2000 / 1e6)   # 2x the $1000/M input
