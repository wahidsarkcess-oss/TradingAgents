"""Regression tests for the second Phase 1 review (cecb17f).

Concurrency tests synchronise with barriers and events, never with timed sleeps;
where a test must wait for a condition, it waits on that condition with a hard
timeout. The crash-recovery test kills a real process. Provider retry tests run
the real Google and OpenAI clients over scripted transports; Bedrock, whose
optional extra is not installed, runs over scripted stand-ins of its SDK.
"""

from __future__ import annotations

import codecs
import copy
import importlib
import json
import logging
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import types
import uuid
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from tests.test_graph_end_to_end import TRADE_DATE, _Client, offline  # noqa: F401
from tests.test_run_budget import MeteredModel, _call, _config, _graph, _pricing
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph import trading_graph
from tradingagents.graph.checkpointer import usage_path
from tradingagents.llm_clients import budget as budget_module
from tradingagents.llm_clients.budget import (
    SETTLEMENT,
    LLMBudgetExceeded,
    ModelPrice,
    Pricing,
    PricingError,
    Rates,
    RunBudget,
    current_account,
    load_pricing,
    note_retry,
    settlement_scope,
)
from tradingagents.memory import settlement

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 20


def wait_until(condition, timeout=TIMEOUT):
    """Wait for ``condition()`` to hold; fail the test if it does not in time."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail("condition not reached in time")
        time.sleep(0.01)


# --- H1: runs and settlement passes on one graph stay independent ----------------


class StaggeredModel(MeteredModel):
    """Run 0 is held at its first call until run 1 has finished entirely, the case
    that broke the previous design (a finishing run ended the other's account)."""

    run1_done: object = Field(default_factory=threading.Event)
    held: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        account = current_account()
        if account is not None and account.id == RUN0[0] and not self.held:
            self.held.append(1)
            assert self.run1_done.wait(TIMEOUT)
        return super()._generate(messages, stop, run_manager, **kwargs)


RUN0: list = []


def _run_in_thread(graph, outcomes, key, done=None):
    try:
        outcomes[key] = graph.propagate("NVDA", TRADE_DATE)[1]
    except Exception as exc:  # noqa: BLE001
        outcomes[key] = f"{type(exc).__name__}: {exc}"
    finally:
        if done is not None:
            done.set()


@pytest.mark.unit
@pytest.mark.parametrize("limits", [{}, {"max_llm_calls_per_run": "baseline"}], ids=["limits-off", "limits-on"])
def test_two_analyses_on_one_graph_do_not_invalidate_each_other(tmp_path, monkeypatch, offline, limits):  # noqa: F811
    baseline = MeteredModel(structured=True)
    _graph(tmp_path / "base", monkeypatch, baseline).propagate("NVDA", TRADE_DATE)
    config = {k: len(baseline.calls) if v == "baseline" else v for k, v in limits.items()}

    model = StaggeredModel(structured=True)
    graph = _graph(tmp_path, monkeypatch, model, **config)
    started = threading.Event()
    original_begin = graph.budget.begin_run

    def begin_run(*args, **kwargs):              # note which account is run 0's
        account = original_begin(*args, **kwargs)
        if not RUN0:
            RUN0.append(account.id)
            started.set()
        return account

    RUN0.clear()
    monkeypatch.setattr(graph.budget, "begin_run", begin_run)
    outcomes = {}
    run0 = threading.Thread(target=_run_in_thread, args=(graph, outcomes, 0))
    run0.start()
    assert started.wait(TIMEOUT)                  # run 0 has its account before run 1 starts
    run1 = threading.Thread(target=_run_in_thread, args=(graph, outcomes, 1, model.run1_done))
    run1.start()
    run1.join(TIMEOUT)
    run0.join(TIMEOUT)

    assert model.held                             # run 0 really was mid-call while run 1 ran
    assert outcomes == {0: "Overweight", 1: "Overweight"}
    assert graph.budget.summary(session=True)["calls"]["attempted"] == 2 * len(baseline.calls)


class GateModel(MeteredModel):
    """The run's first call signals that it is in flight, then waits to be released."""

    inside: object = Field(default_factory=threading.Event)
    release: object = Field(default_factory=threading.Event)
    run_id: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        account = current_account()
        if account is not None and account.kind != SETTLEMENT and not self.run_id:
            self.run_id.append(account.id)
            self.inside.set()
            assert self.release.wait(TIMEOUT)
        return super()._generate(messages, stop, run_manager, **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("limits", [{}, {"max_llm_calls_per_run": 500}], ids=["limits-off", "limits-on"])
def test_a_settlement_pass_during_an_analysis_does_not_end_it(tmp_path, monkeypatch, offline, limits):  # noqa: F811
    # The decision's window "trades" only for the independent settlement pass, so it
    # (not the run's own Memory Log step) settles it, with a reflection call.
    monkeypatch.setattr(settlement, "fetch_returns", lambda *a, **k: (
        (0.02, 0.01, 5, "2026-01-02") if (current_account() and current_account().kind == SETTLEMENT)
        else (None, None, None, None)))
    model = GateModel(structured=True)
    graph = _graph(tmp_path, monkeypatch, model, **limits)
    graph.memory_log.store_decision(ticker="AAPL", trade_date="2025-12-01",
                                    final_trade_decision="**Rating**: Buy", rating="Buy")
    outcomes = {}
    runner = threading.Thread(target=_run_in_thread, args=(graph, outcomes, "run"))
    runner.start()
    assert model.inside.wait(TIMEOUT)               # the analysis is mid-call

    done = graph.settle_all_pending()               # a scheduler settles meanwhile
    model.release.set()
    runner.join(TIMEOUT)

    assert outcomes == {"run": "Overweight"}
    assert done.settled == [("AAPL", "2025-12-01")] and not done.unreflected   # it reflected, alongside the run


# --- L2: SDK logging is the host's ----------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("limited", [False, True])
def test_sdk_debug_logging_reaches_the_host(limited):
    RunBudget(max_calls=5) if limited else RunBudget()
    sdk = logging.getLogger("openai._base_client")
    assert sdk.level == logging.NOTSET                             # nothing changed on the logger
    assert not [f for f in sdk.filters if type(f).__module__.startswith("tradingagents")]
    records = []
    handler = logging.Handler(level=logging.DEBUG)
    handler.emit = records.append
    parent = logging.getLogger("openai")
    old_level = parent.level
    parent.addHandler(handler)
    parent.setLevel(logging.DEBUG)                                 # what OPENAI_LOG=debug does
    try:
        sdk.debug("Request options: ...")
        sdk.info("HTTP Response: 200")
    finally:
        parent.removeHandler(handler)
        parent.setLevel(old_level)
    assert [r.getMessage() for r in records] == ["Request options: ...", "HTTP Response: 200"]


# --- M1: one checkpoint, one runner ---------------------------------------------------


def _checkpoint_graph(tmp_path, monkeypatch, **limits):
    return _graph(tmp_path, monkeypatch, MeteredModel(), checkpoint_enabled=True, **limits)


@pytest.mark.unit
def test_a_checkpoint_cannot_be_resumed_twice_at_once(tmp_path, monkeypatch, offline):  # noqa: F811
    first = _checkpoint_graph(tmp_path, monkeypatch, max_llm_calls_per_run=100)
    second = _checkpoint_graph(tmp_path, monkeypatch, max_llm_calls_per_run=100)
    first.begin_checkpoint("NVDA", TRADE_DATE)
    try:
        with pytest.raises(LLMBudgetExceeded) as refused:
            second.begin_checkpoint("NVDA", TRADE_DATE)
        assert refused.value.scope == "resume_in_progress"
    finally:
        first.end_checkpoint()
    second.begin_checkpoint("NVDA", TRADE_DATE)       # free again once the first ends
    second.end_checkpoint()


@pytest.mark.unit
def test_two_threads_resuming_one_checkpoint_get_one_allowance(tmp_path, monkeypatch, offline):  # noqa: F811
    graphs = [_checkpoint_graph(tmp_path, monkeypatch, max_llm_calls_per_run=100) for _ in range(2)]
    barrier, results = threading.Barrier(2), {}

    def resume(index):
        barrier.wait(TIMEOUT)
        try:
            graphs[index].begin_checkpoint("NVDA", TRADE_DATE)
            results[index] = "held"
        except LLMBudgetExceeded as exc:
            results[index] = exc.scope
        except Exception as exc:  # noqa: BLE001 — recorded, so the assertion shows it
            results[index] = f"{type(exc).__name__}: {exc}"

    threads = [threading.Thread(target=resume, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT)
    assert sorted(results.values()) == ["held", "resume_in_progress"]
    for graph in graphs:
        graph.end_checkpoint()


@pytest.mark.unit
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process handling")
def test_another_process_holding_the_checkpoint_blocks_a_resume(tmp_path, monkeypatch, offline):  # noqa: F811
    graph = _checkpoint_graph(tmp_path, monkeypatch, max_llm_calls_per_run=100)
    record = usage_path(graph.config["data_cache_dir"], "NVDA", TRADE_DATE, graph._run_signature("stock"))
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent("""
            import sys, time
            from tradingagents.dataflows.files import locked
            cm = locked(sys.argv[1]); cm.__enter__()
            print("LOCKED", flush=True); time.sleep(120)
        """), str(record)],
        stdout=subprocess.PIPE, text=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
    try:
        assert holder.stdout.readline().strip() == "LOCKED"
        with pytest.raises(LLMBudgetExceeded) as refused:
            graph.begin_checkpoint("NVDA", TRADE_DATE)
        assert refused.value.scope == "resume_in_progress"
    finally:
        holder.kill()
        holder.wait(TIMEOUT)
    graph.begin_checkpoint("NVDA", TRADE_DATE)        # the lock died with the process
    graph.end_checkpoint()


@pytest.mark.unit
def test_without_limits_a_concurrent_resume_is_not_refused(tmp_path, monkeypatch, offline):  # noqa: F811
    first, second = _checkpoint_graph(tmp_path, monkeypatch), _checkpoint_graph(tmp_path, monkeypatch)
    first.begin_checkpoint("NVDA", TRADE_DATE)
    second.begin_checkpoint("NVDA", TRADE_DATE)       # the original behaviour
    first.end_checkpoint()
    second.end_checkpoint()


# --- M2: a process killed mid-call -----------------------------------------------------


_RESERVE_AND_HANG = textwrap.dedent("""
    import sys, time, uuid
    from langchain_core.messages import HumanMessage
    from tradingagents.llm_clients.budget import ModelPrice, Pricing, RunBudget
    pricing = Pricing({"openai:m": ModelPrice(1000.0, 2000.0)}, "2026-01-01", "test", "p")
    budget = RunBudget(max_calls=10, max_run_cost=50.0, output_reservation=100, output_limited=True,
                       pricing=pricing, ledger_dir=sys.argv[2])
    account = budget.begin_run(persist_path=sys.argv[1])
    with budget.bound(account):
        budget.callback("openai", "m").on_chat_model_start({}, [[HumanMessage(content="x" * 40)]],
                                                            run_id=uuid.uuid4())
        print("RESERVED", flush=True)
        time.sleep(120)   # the request is "in flight" when the process is killed
""")


@pytest.mark.unit
@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL")
def test_a_reservation_open_when_the_process_is_killed_counts_as_spent(tmp_path):
    record = tmp_path / "usage.json"
    child = subprocess.Popen([sys.executable, "-c", _RESERVE_AND_HANG, str(record), str(tmp_path / "spend")],
                             stdout=subprocess.PIPE, text=True, cwd=ROOT,
                             env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert child.stdout.readline().strip() == "RESERVED"
    os.kill(child.pid, signal.SIGKILL)               # a real crash: no cleanup runs
    child.wait(TIMEOUT)
    assert child.returncode == -signal.SIGKILL

    saved = json.loads(record.read_text())
    assert len(saved["inflight"]) == 1               # on disk before the request was sent
    reserved = next(iter(saved["inflight"].values()))

    budget = RunBudget(max_calls=10, max_run_cost=50.0, output_reservation=100, output_limited=True,
                       pricing=Pricing({"openai:m": ModelPrice(1000.0, 2000.0)}, "2026-01-01", "t", "p"),
                       ledger_dir=tmp_path / "spend")
    resumed = budget.begin_run(persist_path=record, restore=True)
    totals = resumed.scopes["analysis"]
    assert totals.attempted == 1 and totals.unresolved_recovered == 1
    assert totals.charged_cost == pytest.approx(reserved["cost"]) and reserved["cost"] > 0
    assert totals.estimated_unreported == reserved["tokens"]
    ledger = json.loads(next((tmp_path / "spend").glob("*.json")).read_text())
    assert sum(ledger["reserved"].values()) == pytest.approx(reserved["cost"])   # still counted today


# --- L3: restored records must be complete ----------------------------------------------


def _valid_record():
    totals = {f.name: 0 for f in budget_module.fields(budget_module._Totals)}
    return {"version": 2, "account": "a", "scopes": {"analysis": dict(totals), "settlement": dict(totals)},
            "inflight": {}}


@pytest.mark.unit
@pytest.mark.parametrize("damage", [
    lambda r: r.pop("version"),
    lambda r: r.update(version=1),
    lambda r: r.update(scopes={}),
    lambda r: r["scopes"].pop("settlement"),
    lambda r: r["scopes"]["analysis"].pop("attempted"),
    lambda r: r["scopes"]["analysis"].update(attempted=-1),
    lambda r: r["scopes"]["analysis"].update(attempted="3"),
    lambda r: r.pop("inflight"),
    lambda r: r.update(inflight={"k": {"scope": "analysis", "tokens": 5}}),
], ids=["no-version", "old-version", "empty-scopes", "missing-scope", "missing-field", "negative",
        "string", "no-inflight", "bad-inflight"])
def test_an_incomplete_or_malformed_record_is_refused(tmp_path, damage):
    record = tmp_path / "u.json"
    data = _valid_record()
    damage(data)
    record.write_text(json.dumps(data))
    with pytest.raises(LLMBudgetExceeded) as refused:
        RunBudget(max_calls=5).begin_run(persist_path=record, restore=True)
    assert refused.value.scope == "resume"


@pytest.mark.unit
def test_a_complete_record_restores(tmp_path):
    record = tmp_path / "u.json"
    data = _valid_record()
    data["scopes"]["analysis"]["attempted"] = 3
    record.write_text(json.dumps(data))
    assert RunBudget(max_calls=5).begin_run(persist_path=record, restore=True).total.attempted == 3


# --- M3: the endpoint the client really calls ------------------------------------------


def _real_graph(tmp_path, monkeypatch, provider, model, **config):
    pricing = _pricing(tmp_path, models={f"{provider}:{model}": {"input_per_mtok": 1, "output_per_mtok": 1}})
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(results_dir=str(tmp_path / "r"), data_cache_dir=str(tmp_path / "c"),
               memory_log_path=str(tmp_path / "log.md"), llm_provider=provider,
               quick_think_llm=model, deep_think_llm=model, max_cost_per_run_usd=1.0, max_tokens=100,
               pricing_file=str(pricing), **config)
    return trading_graph.TradingAgentsGraph(config=cfg)


@pytest.mark.unit
@pytest.mark.parametrize("provider, model, env, value", [
    ("openai", "gpt-x", "OPENAI_BASE_URL", "http://relay.example.internal/v1"),
    ("anthropic", "claude-x", "ANTHROPIC_BASE_URL", "http://anthropic-relay.example.internal"),
])
def test_an_environment_override_of_the_endpoint_is_refused_under_a_cost_limit(tmp_path, monkeypatch, provider,
                                                                               model, env, value):
    monkeypatch.setenv(env, value)
    with pytest.raises(PricingError, match="relay.example.internal"):
        _real_graph(tmp_path, monkeypatch, provider, model)


@pytest.mark.unit
@pytest.mark.parametrize("provider, model", [("openai", "gpt-x"), ("anthropic", "claude-x"),
                                             ("google", "gemini-x")])
def test_the_official_endpoint_is_accepted(tmp_path, monkeypatch, provider, model):
    for env in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "ANTHROPIC_BASE_URL", "ANTHROPIC_API_URL"):
        monkeypatch.delenv(env, raising=False)
    graph = _real_graph(tmp_path, monkeypatch, provider, model)
    assert all(budget_module.endpoint_supported(p, e) for p, _, e in graph.budget.clients)


@pytest.mark.unit
@pytest.mark.parametrize("endpoint, ok", [
    ("https://my-resource.openai.azure.com/", True),
    ("https://my-gateway.azure-api.net/", False),
])
def test_an_azure_gateway_is_refused(tmp_path, monkeypatch, endpoint, ok):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", endpoint)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "dep")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-10-21")
    if ok:
        _real_graph(tmp_path, monkeypatch, "azure", "gpt-x")
    else:
        with pytest.raises(PricingError, match="azure-api.net"):
            _real_graph(tmp_path, monkeypatch, "azure", "gpt-x")


@pytest.mark.unit
def test_an_endpoint_that_cannot_be_determined_is_refused(tmp_path, monkeypatch):
    class Opaque(MeteredModel):
        @property
        def root_client(self):
            raise AttributeError("no client")

    model = Opaque()
    with pytest.raises(PricingError, match="could not be determined"):
        _graph(tmp_path, monkeypatch, model, max_cost_per_run_usd=1.0, max_tokens=100,
               pricing_file=str(_pricing(tmp_path)))
    assert model.calls == []


# --- M4: Google and Bedrock retries through the budget ------------------------------------


def _google_failing_transport(monkeypatch, failures):
    from google.genai import _api_client, errors

    calls = []

    def request_once(self, http_request, stream=False):
        calls.append(1)
        if len(calls) <= failures:
            raise errors.ServerError(503, {"error": {"code": 503, "message": "overloaded",
                                                     "status": "UNAVAILABLE"}})
        raise errors.ClientError(400, {"error": {"code": 400, "message": "stop here",
                                                 "status": "INVALID_ARGUMENT"}})

    monkeypatch.setattr(_api_client.BaseApiClient, "_request_once", request_once)
    monkeypatch.setattr("tradingagents.llm_clients.retrying.time.sleep", lambda s: None)
    return calls


def _google_llm(budget, retries):
    from tradingagents.llm_clients.google_client import GoogleClient

    return GoogleClient("gemini-x", callbacks=[budget.callback("google", "gemini-x")],
                        budget_controlled_retries=retries, google_api_key="dummy-" + "g" * 30).get_llm()


@pytest.mark.unit
def test_google_retries_go_through_the_budget_one_attempt_each(monkeypatch):
    calls = _google_failing_transport(monkeypatch, failures=2)
    budget = RunBudget(max_calls=10)
    llm = _google_llm(budget, retries=2)
    assert llm.max_retries == 1                      # the SDK itself makes one attempt
    with pytest.raises(Exception, match="stop here"):
        llm.invoke("hello")
    assert len(calls) == 3                           # 1 + 2 retries, each a single SDK attempt
    totals = budget.summary(session=True)["calls"]
    assert (totals["attempted"], totals["sdk_retries"], totals["failed"]) == (1, 2, 1)


@pytest.mark.unit
def test_a_google_retry_past_the_call_limit_is_never_sent(monkeypatch):
    calls = _google_failing_transport(monkeypatch, failures=10)
    budget = RunBudget(max_calls=2)
    with pytest.raises(LLMBudgetExceeded, match="retry refused"):
        _google_llm(budget, retries=6).invoke("hello")
    assert len(calls) == 2                           # the third attempt was refused before sending


@pytest.mark.unit
def test_without_limits_the_google_client_is_built_as_before():
    from tradingagents.llm_clients.google_client import GoogleClient

    llm = GoogleClient("gemini-x", google_api_key="dummy-" + "g" * 30).get_llm()
    assert llm.budget_retries is None and llm.max_retries == 6     # the SDK's own default


@pytest.fixture
def fake_bedrock(monkeypatch):
    """langchain-aws and botocore are an optional extra: scripted stand-ins."""
    attempts = []

    class ClientError(Exception):
        def __init__(self, code):
            super().__init__(code)
            self.response = {"Error": {"Code": code}}

    class ChatBedrockConverse(BaseChatModel):
        model: str = ""
        region_name: str = ""
        config: object = None
        max_tokens: int | None = None
        failures: int = 0

        @property
        def _llm_type(self):
            return "fake-bedrock"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            attempts.append(1)
            if len(attempts) <= self.failures:
                raise ClientError("ThrottlingException")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    class Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    aws = types.ModuleType("langchain_aws")
    aws.ChatBedrockConverse = ChatBedrockConverse
    botocore, botocore_config = types.ModuleType("botocore"), types.ModuleType("botocore.config")
    botocore_config.Config = Config
    monkeypatch.setitem(sys.modules, "langchain_aws", aws)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config)
    from tradingagents.llm_clients import bedrock_client

    monkeypatch.setattr(bedrock_client, "_BEDROCK_CLASS", None)
    monkeypatch.setattr("tradingagents.llm_clients.retrying.time.sleep", lambda s: None)
    return attempts


@pytest.mark.unit
def test_bedrock_retries_go_through_the_budget(fake_bedrock):
    from tradingagents.llm_clients.bedrock_client import BedrockClient

    budget = RunBudget(max_calls=10)
    llm = BedrockClient("m", callbacks=[budget.callback("bedrock", "m")], budget_controlled_retries=2).get_llm()
    assert llm.config.kwargs == {"retries": {"max_attempts": 1, "mode": "standard"}}   # botocore: one attempt
    llm.failures = 2
    assert llm.invoke("hello").content == "ok"
    assert len(fake_bedrock) == 3
    assert budget.summary(session=True)["calls"]["sdk_retries"] == 2


@pytest.mark.unit
def test_a_bedrock_retry_past_the_limit_is_refused(fake_bedrock):
    from tradingagents.llm_clients.bedrock_client import BedrockClient

    budget = RunBudget(max_calls=2)
    llm = BedrockClient("m", callbacks=[budget.callback("bedrock", "m")], budget_controlled_retries=5).get_llm()
    llm.failures = 10
    with pytest.raises(LLMBudgetExceeded):
        llm.invoke("hello")
    assert len(fake_bedrock) == 2


@pytest.mark.unit
def test_bedrock_cannot_carry_a_cost_limit(tmp_path):
    pricing = _pricing(tmp_path, models={"bedrock:m": {"input_per_mtok": 1, "output_per_mtok": 1}})
    with pytest.raises(PricingError, match="not supported for provider 'bedrock'"):
        RunBudget.from_config(_config(tmp_path, max_cost_per_run_usd=1.0, max_tokens=10,
                                      pricing_file=str(pricing)), [("bedrock", "m")])


@pytest.mark.unit
@pytest.mark.parametrize("limited", [False, True])
def test_the_graph_routes_google_retries_only_under_limits(tmp_path, limited):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(results_dir=str(tmp_path / "r"), data_cache_dir=str(tmp_path / "c"),
               memory_log_path=str(tmp_path / "log.md"), llm_provider="google",
               quick_think_llm="gemini-x", deep_think_llm="gemini-x", llm_max_retries=3,
               **({"max_llm_calls_per_run": 40} if limited else {}))
    llm = trading_graph.TradingAgentsGraph(config=cfg).quick_thinking_llm
    assert (llm.budget_retries, llm.max_retries) == ((3, 1) if limited else (None, 3))


# --- M5: one cost ceiling per run --------------------------------------------------------


def _priced_budget(tmp_path, **limits):
    return RunBudget(output_reservation=1, output_limited=True, ledger_dir=tmp_path / "spend",
                     pricing=Pricing({"openai:m": ModelPrice(1000.0, 2000.0)}, "2026-01-01", "t", "p"), **limits)


@pytest.mark.unit
def test_reflections_cannot_double_the_per_run_cost(tmp_path):
    budget = RunBudget(max_run_cost=1.0, output_reservation=100, output_limited=True, ledger_dir=tmp_path / "s",
                       pricing=Pricing({"openai:m": ModelPrice(1000.0, 2000.0)}, "2026-01-01", "t", "p"))
    account = budget.begin_run()
    cb = budget.callback("openai", "m")
    with budget.bound(account):
        _call(cb, prompt="x" * 20, usage=(450, 200))     # $0.85 of analysis
        with settlement_scope(), pytest.raises(LLMBudgetExceeded) as refused:
            _call(cb, prompt="x" * 20)                   # a reflection reserving $0.21 more
    assert refused.value.scope == "run_cost"
    assert account.total.charged_cost <= 1.0


# --- L1: retries meet the session and daily limits -----------------------------------------


@pytest.mark.unit
def test_a_retry_past_the_session_limit_is_refused(tmp_path):
    budget = _priced_budget(tmp_path)
    budget.max_session_cost = 0.013                  # the call reserves $0.012; a retry adds $0.010
    cb = budget.callback("openai", "m")
    cb.on_chat_model_start({}, [[HumanMessage(content="x" * 20)]], run_id=uuid.uuid4())
    with pytest.raises(LLMBudgetExceeded) as refused:
        note_retry()
    assert refused.value.scope == "session_cost"


@pytest.mark.unit
def test_a_retry_past_the_daily_limit_is_refused(tmp_path):
    budget = _priced_budget(tmp_path, max_day_cost=0.015)
    cb = budget.callback("openai", "m")
    cb.on_chat_model_start({}, [[HumanMessage(content="x" * 20)]], run_id=uuid.uuid4())   # reserves $0.012
    with pytest.raises(LLMBudgetExceeded) as refused:
        note_retry()
    assert refused.value.scope == "day_cost"
    ledger = json.loads(next((tmp_path / "spend").glob("*.json")).read_text())
    assert ledger["charged_usd"] == 0                # the refused retry charged nothing


# --- L7: pricing tiers ----------------------------------------------------------------------


LONG = ModelPrice(1.0, 2.0, long_context_above=1000, long_context=Rates(4.0, 8.0))


@pytest.mark.unit
def test_long_context_rates_apply_to_the_whole_request_above_the_threshold():
    assert LONG.cost(1000, 100) == pytest.approx((1000 * 1.0 + 100 * 2.0) / 1e6)
    assert LONG.cost(1001, 100) == pytest.approx((1001 * 4.0 + 100 * 8.0) / 1e6)


@pytest.mark.unit
def test_a_reservation_uses_the_long_context_rate_whenever_the_prompt_could_exceed_it():
    # 1,500 bytes is at most 1,500 tokens: the long-context tier may apply.
    assert LONG.worst_case_rates(1500, "openai") == (4.0, 8.0)
    # 900 bytes can never exceed 1,000 tokens.
    assert LONG.worst_case_rates(900, "openai") == (1.0, 2.0)


@pytest.mark.unit
def test_cache_writes_are_reserved_at_their_rate_for_providers_that_charge_them():
    explicit = ModelPrice(1.0, 2.0, cache_write_per_mtok=1.25)
    fallback = ModelPrice(1.0, 2.0)
    assert explicit.worst_case_rates(10, "anthropic") == (1.25, 2.0)
    assert fallback.worst_case_rates(10, "anthropic") == (2.0, 2.0)    # unpriced writes: 2x input
    assert fallback.worst_case_rates(10, "openai") == (1.0, 2.0)       # no cache-write charge there


@pytest.mark.unit
def test_long_context_cache_writes_take_the_highest_rate():
    tiered = ModelPrice(1.0, 2.0, cache_write_per_mtok=1.25, long_context_above=100,
                        long_context=Rates(2.0, 3.0, cache_write=2.5))
    assert tiered.worst_case_rates(500, "anthropic") == (2.5, 3.0)


@pytest.mark.unit
def test_the_pricing_file_reads_and_validates_tiers(tmp_path):
    models = {"anthropic:c": {"input_per_mtok": 3, "output_per_mtok": 15, "cache_write_per_mtok": 3.75,
                              "long_context": {"above_input_tokens": 200000, "input_per_mtok": 6,
                                               "output_per_mtok": 22.5}}}
    price = load_pricing(_pricing(tmp_path, models=models), 90).price("anthropic", "c")
    assert price.long_context_above == 200000 and price.long_context.input == 6
    models["anthropic:c"]["long_context"]["above_input_tokens"] = None
    with pytest.raises(PricingError, match="above_input_tokens"):
        load_pricing(_pricing(tmp_path, models=models), 90)


@pytest.mark.unit
def test_a_long_prompt_reserves_at_the_long_context_rate_in_the_budget(tmp_path):
    pricing = Pricing({"openai:m": LONG}, "2026-01-01", "t", "p")
    budget = RunBudget(output_reservation=10, output_limited=True, pricing=pricing, ledger_dir=tmp_path / "s")
    run_id = uuid.uuid4()
    budget.callback("openai", "m").on_chat_model_start({}, [[HumanMessage(content="x" * 2000)]], run_id=run_id)
    assert budget._inflight[run_id].cost == pytest.approx((1000 * 4.0 + 10 * 8.0) / 1e6)


# --- L4: Azure-style key names ------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("text, secret", [
    ("GET https://x.openai.azure.com/d?api-version=1&api-key=a1b2c3d4e5f6a7b8c9", "a1b2c3d4e5f6a7b8c9"),
    ('headers: {"api-key": "z9y8x7w6v5u4t3s2"}', "z9y8x7w6v5u4t3s2"),  # scan-secrets: allow (fixture)
])
def test_azure_api_key_parameters_are_redacted(text, secret):
    from tradingagents.logsafe import redact

    assert secret not in redact(text)


# --- L5: TypeSafe screening ----------------------------------------------------------------


@pytest.fixture
def typesafe(monkeypatch):
    from tradingagents.agents import post_screen

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-" + "local" * 4)
    monkeypatch.setattr(post_screen, "resolve_instrument_identity", lambda t: {"company_name": "NVIDIA"})
    monkeypatch.setattr(post_screen, "_answers", lambda response, questions: {
        "about": {"noul": 1.0}, "stance": {"choice": "bullish"}})
    monkeypatch.setattr(post_screen, "_stance", lambda a: "bullish")
    monkeypatch.setattr(post_screen.time, "sleep", lambda s: None)
    return post_screen


def _response(status):
    return types.SimpleNamespace(status_code=status, headers={})


@pytest.mark.unit
def test_every_typesafe_attempt_is_counted(typesafe, monkeypatch):
    replies = iter([_response(429), _response(200)])        # rate limited, then answered
    sent = []
    monkeypatch.setattr(typesafe.requests, "post", lambda *a, **k: sent.append(1) or next(replies))
    budget = RunBudget(max_calls=40)
    account = budget.begin_run()
    with budget.bound(account):
        typesafe.jev_screen("NVDA")(["one post"])
    assert len(sent) == 2 and account.total.external_requests == 2      # the retry counted too


@pytest.mark.unit
def test_typesafe_stops_at_its_request_cap(typesafe, monkeypatch):
    sent = []
    monkeypatch.setattr(typesafe.requests, "post", lambda *a, **k: sent.append(1) or _response(200))
    budget = RunBudget(max_external_requests=2)
    account = budget.begin_run()
    with budget.bound(account):
        keep, note = typesafe.jev_screen("NVDA")(["a", "b", "c", "d"])
    assert len(sent) <= 2 and account.total.external_requests <= 2
    assert keep == [True] * 4 and "stopped" in note


@pytest.mark.unit
def test_typesafe_sends_nothing_in_a_cost_limited_analysis(tmp_path, monkeypatch, offline, typesafe):  # noqa: F811
    from tradingagents.agents.analysts import sentiment_analyst

    screened = []
    monkeypatch.setattr(typesafe.requests, "post", lambda *a, **k: pytest.fail("a paid TypeSafe request"))
    monkeypatch.setattr(sentiment_analyst, "fetch_stocktwits_messages",
                        lambda *a, screen=None, **k: screened.append(screen(["a post"])[1]) or "posts")
    model = MeteredModel(structured=True, usage=(10, 5))
    graph = _graph(tmp_path, monkeypatch, model, max_cost_per_run_usd=1000.0, max_tokens=100,
                   pricing_file=str(_pricing(tmp_path)))
    _, signal = graph.propagate("NVDA", TRADE_DATE)
    assert signal == "Overweight"
    assert screened and "skipped" in screened[0]


# --- L6: reservations whose request never started ------------------------------------------


@pytest.mark.unit
def test_a_completed_run_releases_reservations_that_never_reached_the_provider(tmp_path):
    budget = _priced_budget(tmp_path, max_day_cost=10.0)
    account = budget.begin_run()
    with budget.bound(account):
        budget.callback("openai", "m").on_chat_model_start({}, [[HumanMessage(content="x")]], run_id=uuid.uuid4())
    budget.end_run(account, completed=True)          # the graph returned: that call never ran
    assert not budget._inflight and account.total.attempted == 0
    ledger = json.loads(next((tmp_path / "spend").glob("*.json")).read_text())
    assert ledger["reserved"] == {} and ledger["charged_usd"] == 0


@pytest.mark.unit
def test_a_failed_run_keeps_its_open_reservations(tmp_path):
    budget = _priced_budget(tmp_path, max_day_cost=10.0)
    account = budget.begin_run()
    with budget.bound(account):
        budget.callback("openai", "m").on_chat_model_start({}, [[HumanMessage(content="x")]], run_id=uuid.uuid4())
    budget.end_run(account, completed=False)         # an orphan may still be sending it
    assert budget._inflight


@pytest.mark.unit
def test_the_budget_callback_is_the_last_one(tmp_path, monkeypatch):
    model = MeteredModel()
    graph = _graph(tmp_path, monkeypatch, model)
    assert isinstance(graph.quick_thinking_llm.callbacks[-1], budget_module.BudgetCallback)


# --- L8: one settlement account for the end of a backtest ---------------------------------


@pytest.mark.unit
def test_the_final_backtest_settlement_shares_one_reflection_allowance(tmp_path, monkeypatch, offline):  # noqa: F811
    from tradingagents.backtest import run_backtest

    # Prices "trade" only in the final settlement pass, so both tickers settle there.
    monkeypatch.setattr(settlement, "fetch_returns", lambda *a, **k: (
        (0.02, 0.01, 5, "2026-01-02") if (current_account() and current_account().kind == SETTLEMENT)
        else (None, None, None, None)))
    model = MeteredModel(structured=True)
    monkeypatch.setattr(trading_graph, "create_tier_client",
                        lambda config, tier, **k: _Client(model.model_copy(update={"callbacks": k.get("callbacks")})))
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(results_dir=str(tmp_path / "bt"), data_cache_dir=str(tmp_path / "cache"), llm_provider="openai",
               quick_think_llm="quick-model", deep_think_llm="deep-model", max_reflection_calls_per_run=1)
    result = run_backtest(["NVDA", "AAPL"], ["2026-01-05"], cfg)
    assert len(result.unreflected) == 1              # one allowance for the pass, not one per ticker


# --- L9: the orphaned-analyst scenario, deterministically -----------------------------------


class OrphanModel(MeteredModel):
    """One analyst's first call fails at once; the others are held mid-call until released."""

    release: object = Field(default_factory=threading.Event)
    failed: list = Field(default_factory=list)
    held: list = Field(default_factory=list)
    lock: object = Field(default_factory=threading.Lock)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        with self.lock:
            fail = not self.failed
            self.failed.append(1) if fail else self.held.append(1)
        if fail:
            wait_until(lambda: self.held)       # another analyst is mid-call...
            self.release.set()                   # ...and carries on while this one fails
            raise RuntimeError("provider unavailable")
        assert self.release.wait(TIMEOUT)
        return super()._generate(messages, stop, run_manager, **kwargs)


@pytest.mark.unit
def test_analysts_left_running_after_a_failure_cannot_escape_accounting(tmp_path, monkeypatch, offline):  # noqa: F811
    model = OrphanModel()
    graph = _graph(tmp_path, monkeypatch, model, max_llm_calls_per_run=100)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        graph.propagate("NVDA", TRADE_DATE)
    failed_run = graph.budget._last
    assert failed_run.ended and model.held
    next_run = graph.budget.begin_run()              # the next cell starts
    # Branches still running when the run failed finish (this LangGraph waits for
    # them; another may not) and are charged to the failed run, never the next.
    wait_until(lambda: not graph.budget._inflight)

    assert next_run.total.attempted == 0             # nothing charged to the next run
    session = graph.budget.summary(session=True)["calls"]["attempted"]
    assert session == failed_run.total.attempted == len(model.calls) + 1   # +1: the failed call


# --- M6: the scanner --------------------------------------------------------------------------


def _scanner():
    spec = importlib.util.spec_from_file_location("scan_secrets_2", ROOT / "scripts" / "scan_secrets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAKE_KEY = "sk-" + "proj-" + "Q1w2" * 10


@pytest.mark.unit
def test_an_allowlisted_finding_does_not_hide_a_changed_value(tmp_path, monkeypatch, capsys):
    scan = _scanner()
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "cfg.py"
    source.write_text(f"client = make('{FAKE_KEY}')\n")
    (tmp_path / ".scan-secrets-allow").write_text(f"{scan.scan_file(str(source))[0]}\n")
    assert scan.main([str(source)]) == 0
    assert "exempt (reviewed" in capsys.readouterr().out        # the exemption is printed

    source.write_text(f"client = make('{FAKE_KEY[:-4]}Z9Z9')\n")   # a different key, same place
    assert scan.main([str(source)]) == 1


@pytest.mark.unit
def test_an_allowlist_entry_without_a_fingerprint_is_ignored(tmp_path, monkeypatch, capsys):
    scan = _scanner()
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "cfg.py"
    source.write_text(f"client = make('{FAKE_KEY}')\n")
    (tmp_path / ".scan-secrets-allow").write_text(f"{source}:1: OpenAI-style API key\n")
    assert scan.main([str(source)]) == 1
    assert "no [fp:...] fingerprint" in capsys.readouterr().err


@pytest.mark.unit
def test_the_inline_marker_counts_only_under_tests(tmp_path, monkeypatch, capsys):
    scan = _scanner()
    (tmp_path / "tests").mkdir()
    line = f"client = make('{FAKE_KEY}')  # scan-secrets: allow\n"
    fixture, code = tmp_path / "tests" / "test_x.py", tmp_path / "app.py"
    fixture.write_text(line)
    code.write_text(line)
    monkeypatch.chdir(tmp_path)
    assert scan.main(["tests/test_x.py"]) == 0
    assert "exempt (inline marker)" in capsys.readouterr().out   # printed for review
    assert scan.main(["app.py"]) == 1
    assert "marker ignored" in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.parametrize("encoding, bom", [
    ("utf-16-le", codecs.BOM_UTF16_LE), ("utf-16-be", codecs.BOM_UTF16_BE),
    ("utf-16-le", b""), ("utf-16-be", b""), ("utf-8", codecs.BOM_UTF8),
    ("utf-32-le", codecs.BOM_UTF32_LE),
])
def test_utf16_utf32_and_bom_files_are_decoded_and_scanned(tmp_path, encoding, bom):
    file = tmp_path / "config.json"
    file.write_bytes(bom + ('{"OPENAI_API_KEY": "' + FAKE_KEY + '"}\n').encode(encoding))
    assert any("OpenAI-style API key" in f.kind for f in _scanner().scan_file(str(file)))


@pytest.mark.unit
def test_a_real_binary_is_reported_as_skipped(tmp_path, capsys):
    scan = _scanner()
    file = tmp_path / "blob.bin"
    file.write_bytes(bytes(range(256)) * 4)
    assert scan.main([str(file)]) == 0
    assert "binary file, not scanned" in capsys.readouterr().err


# --- compatibility: every limit off ------------------------------------------------------------


@pytest.mark.unit
def test_with_limits_off_nothing_is_hooked_or_changed(tmp_path, monkeypatch, offline):  # noqa: F811
    budget = RunBudget.from_config(_config(tmp_path), [("openai", "m")])
    assert not budget.limited
    model = MeteredModel(structured=True)
    graph = _graph(tmp_path, monkeypatch, model)
    _, signal = graph.propagate("NVDA", TRADE_DATE)
    assert signal == "Overweight"
    account = graph.budget._last
    assert account.ended and account.total.attempted == len(model.calls)
    # A late call from an ended run is accepted with limits off (the original behaviour).
    with graph.budget.bound(account):
        _call(graph.budget.callback("openai", "quick-model"))
    assert account.total.attempted == len(model.calls) + 1
