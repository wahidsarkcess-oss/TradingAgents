"""Regression tests for the targeted fixes after the review of 9d779ea.

M1: a total-cost limit added after construction (backtest --max-cost) reroutes
Google retries through the budget, or fails closed before any request.
M2: a second checkpointed run on one graph object cannot release the first's lock.
L1: a retry whose charge cannot be saved is refused before it is sent.
Plus: inline scanner exemptions only under the top-level tests/, stale checkpoint
lock files removed safely, and a refused resume no longer leaves a live account.
All offline: Google and OpenAI run their real SDKs over scripted transports.
"""

from __future__ import annotations

import copy
import gc
import importlib.util
import json
import uuid
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import HumanMessage

from tests.test_graph_end_to_end import TRADE_DATE, offline  # noqa: F401
from tests.test_run_budget import MeteredModel, _graph, _pricing
from tradingagents.dataflows.files import locked
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph import trading_graph
from tradingagents.graph.checkpointer import clear_all_checkpoints, remove_stale_lock
from tradingagents.llm_clients import budget as budget_module
from tradingagents.llm_clients.budget import (
    LLMBudgetExceeded,
    ModelPrice,
    Pricing,
    PricingError,
    RunBudget,
    current_account,
    note_retry,
)

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_GOOGLE = "https://generativelanguage.googleapis.com/"


# --- helpers ---------------------------------------------------------------------------


def _google_config(tmp_path, **extra):
    pricing = _pricing(tmp_path, models={"google:gemini-x": {"input_per_mtok": 1, "output_per_mtok": 1}})
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(results_dir=str(tmp_path / "results"), data_cache_dir=str(tmp_path / "cache"),
               memory_log_path=str(tmp_path / "log.md"), llm_provider="google",
               quick_think_llm="gemini-x", deep_think_llm="gemini-x", max_tokens=100,
               llm_max_retries=3, pricing_file=str(pricing), **extra)
    return cfg


@pytest.fixture
def google_transport(monkeypatch):
    """The real google-genai client, its HTTP attempts scripted: every one fails with 503."""
    from google.genai import _api_client, errors

    attempts = []

    def request_once(self, http_request, stream=False):
        attempts.append(1)
        raise errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})

    monkeypatch.setattr(_api_client.BaseApiClient, "_request_once", request_once)
    monkeypatch.setattr("tradingagents.llm_clients.retrying.time.sleep", lambda s: None)
    return attempts


# --- M1: a total-cost limit added after construction ---------------------------------------


@pytest.mark.unit
def test_a_session_limit_added_later_routes_google_retries_through_the_budget(tmp_path, google_transport):
    graph = trading_graph.TradingAgentsGraph(config=_google_config(tmp_path))    # no per-run limits
    assert graph.quick_thinking_llm.budget_retries is None     # built as before: SDK retries

    graph.budget.set_session_cost_limit(5.0)                    # what backtest --max-cost does

    for llm in (graph.quick_thinking_llm, graph.deep_thinking_llm):
        assert (llm.budget_retries, llm.max_retries) == (2, 1)  # same 3 attempts, now via the budget
    with pytest.raises(Exception, match="overloaded"):
        graph.quick_thinking_llm.invoke("hello")
    calls = graph.budget.summary(session=True)["calls"]
    assert len(google_transport) == 3                           # every attempt...
    assert calls["attempted"] + calls["sdk_retries"] == 3       # ...seen by the budget


@pytest.mark.unit
def test_the_backtest_max_cost_command_accounts_for_every_google_attempt(tmp_path, monkeypatch, offline,  # noqa: F811
                                                                          google_transport):
    """Review M1 exactly: `tradingagents backtest ... --max-cost` on a Gemini config."""
    from typer.testing import CliRunner

    import cli.main as m

    monkeypatch.setattr(m, "DEFAULT_CONFIG", _google_config(tmp_path))
    results = []
    real_run_backtest = m.run_backtest
    monkeypatch.setattr(m, "run_backtest", lambda *a, **k: results.append(real_run_backtest(*a, **k)) or results[-1])

    CliRunner().invoke(m.app, ["backtest", "NVDA", "--start", TRADE_DATE, "--end", TRADE_DATE,
                               "--analysts", "market", "--max-cost", "5"])

    [result] = results
    assert result.failures                                      # the provider "failed" every attempt
    calls = result.usage["calls"]
    assert google_transport and calls["sdk_retries"] > 0
    assert calls["attempted"] + calls["sdk_retries"] == len(google_transport)


@pytest.mark.unit
def test_a_google_client_that_cannot_be_rerouted_fails_closed_before_any_request(tmp_path, google_transport):
    class Unswitchable:
        """A Gemini client whose retry setting cannot be changed at runtime."""
        client = type("C", (), {"_api_client": type("A", (), {
            "_http_options": type("H", (), {"base_url": OFFICIAL_GOOGLE})()})()})()

        @property
        def max_retries(self):
            return 6

    budget = RunBudget(output_reservation=100, output_limited=True, ledger_dir=tmp_path / "s",
                       pricing=Pricing({"google:gemini-x": ModelPrice(1.0, 1.0)}, "2026-01-01", "t", "p"),
                       models=(("google", "gemini-x"),))
    budget.register_clients([("google", "gemini-x", Unswitchable())])
    with pytest.raises(PricingError, match="route google:gemini-x's retries"):
        budget.set_session_cost_limit(5.0)
    assert budget.max_session_cost is None                      # the limit is not claimed active
    assert google_transport == []


# --- M2: one checkpoint lock per graph object --------------------------------------------


def _checkpoint_graph(tmp_path, monkeypatch, **limits):
    return _graph(tmp_path, monkeypatch, MeteredModel(), checkpoint_enabled=True, **limits)


@pytest.mark.unit
def test_a_second_checkpointed_run_on_one_graph_cannot_release_the_first_lock(tmp_path, monkeypatch, offline):  # noqa: F811
    one = _checkpoint_graph(tmp_path, monkeypatch, max_llm_calls_per_run=10)
    one.begin_checkpoint("NVDA", TRADE_DATE)
    with pytest.raises(LLMBudgetExceeded) as refused:
        one.begin_checkpoint("AAPL", TRADE_DATE)                # e.g. a second thread on the same graph
    assert refused.value.scope == "checkpoint_in_progress"
    gc.collect()                                                # nothing got dropped and closed

    other = _checkpoint_graph(tmp_path, monkeypatch, max_llm_calls_per_run=10)
    with pytest.raises(LLMBudgetExceeded) as blocked:
        other.begin_checkpoint("NVDA", TRADE_DATE)              # NVDA's owner is still active
    assert blocked.value.scope == "resume_in_progress"

    one.end_checkpoint()
    other.begin_checkpoint("NVDA", TRADE_DATE)                  # released only by its owner
    other.end_checkpoint()


@pytest.mark.unit
def test_with_limits_off_a_second_run_on_one_graph_keeps_the_first_lock(tmp_path, monkeypatch, offline):  # noqa: F811
    one = _checkpoint_graph(tmp_path, monkeypatch)              # limits off
    one.begin_checkpoint("NVDA", TRADE_DATE)
    one.begin_checkpoint("AAPL", TRADE_DATE)                    # allowed, as originally
    gc.collect()
    guarded = _checkpoint_graph(tmp_path, monkeypatch, max_llm_calls_per_run=10)
    with pytest.raises(LLMBudgetExceeded):
        guarded.begin_checkpoint("NVDA", TRADE_DATE)            # NVDA's lock was not lost
    one.end_checkpoint()


# --- L1: a retry whose charge cannot be saved ----------------------------------------------


@pytest.mark.unit
def test_google_retry_transport_is_never_called_after_a_persistence_failure(tmp_path, monkeypatch,
                                                                             google_transport):
    from tradingagents.llm_clients.google_client import GoogleClient

    real_write = budget_module._write_json_atomic
    failing = {"on": False}
    monkeypatch.setattr(budget_module, "_write_json_atomic",
                        lambda p, d: (_ for _ in ()).throw(OSError("disk full")) if failing["on"] else real_write(p, d))
    from google.genai import _api_client, errors

    def request_once(self, http_request, stream=False):
        google_transport.append(1)
        failing["on"] = True                                    # the disk fills during the request
        raise errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})

    monkeypatch.setattr(_api_client.BaseApiClient, "_request_once", request_once)
    budget = RunBudget(max_calls=10)
    llm = GoogleClient("gemini-x", callbacks=[budget.callback("google", "gemini-x")],
                       budget_controlled_retries=3, google_api_key="dummy-" + "g" * 30).get_llm()
    account = budget.begin_run(persist_path=tmp_path / "usage.json")
    with budget.bound(account), pytest.raises(LLMBudgetExceeded) as refused:
        llm.invoke("hello")
    assert refused.value.scope == "usage_record"
    assert len(google_transport) == 1                           # no retry was sent


@pytest.mark.unit
def test_openai_retry_transport_is_never_called_after_a_persistence_failure(tmp_path, monkeypatch):
    from langchain_openai import ChatOpenAI

    real_write = budget_module._write_json_atomic
    failing = {"on": False}
    monkeypatch.setattr(budget_module, "_write_json_atomic",
                        lambda p, d: (_ for _ in ()).throw(OSError("disk full")) if failing["on"] else real_write(p, d))
    sent = []

    def handler(request):
        sent.append(1)
        failing["on"] = True
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    budget = RunBudget(max_calls=10)                             # limits on: the SDK hook is installed
    llm = ChatOpenAI(model="gpt-x", api_key="dummy-local", max_retries=3,
                     http_client=httpx.Client(transport=httpx.MockTransport(handler)),
                     callbacks=[budget.callback("openai", "gpt-x")])
    account = budget.begin_run(persist_path=tmp_path / "usage.json")
    with budget.bound(account), pytest.raises(LLMBudgetExceeded):
        llm.invoke("hello")
    assert len(sent) == 1                                       # the SDK's retry was refused


@pytest.mark.unit
def test_with_limits_off_a_persistence_failure_does_not_stop_a_retry(tmp_path, monkeypatch):
    budget = RunBudget()
    account = budget.begin_run(persist_path=tmp_path / "usage.json")
    with budget.bound(account):
        budget.callback("openai", "m").on_chat_model_start({}, [[HumanMessage(content="x")]], run_id=uuid.uuid4())
        monkeypatch.setattr(budget_module, "_write_json_atomic",
                            lambda p, d: (_ for _ in ()).throw(OSError("disk full")))
        note_retry()                                            # counted, not refused
    assert account.total.sdk_retries == 1


# --- scanner: inline exemptions only in the top-level tests/ -------------------------------


def _scanner():
    spec = importlib.util.spec_from_file_location("scan_secrets_t", ROOT / "scripts" / "scan_secrets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
@pytest.mark.parametrize("path, exempt", [
    ("tests/test_a.py", True),
    ("tests/fixtures/keys.py", True),
    ("tradingagents/tests/creds.py", False),
    ("deploy/tests/prod_settings.py", False),
    ("tests", False),
])
def test_the_inline_marker_counts_only_in_the_top_level_tests_folder(path, exempt):
    line = "client = make('" + "sk-" + "proj-" + "Q1w2" * 10 + "')  # scan-secrets: allow"
    assert (_scanner()._findings_on(line, path, 1)[0].exempt == "inline marker") is exempt


@pytest.mark.unit
def test_absolute_paths_count_only_inside_this_repositorys_tests(tmp_path, monkeypatch):
    scan = _scanner()
    (tmp_path / "repo" / "tests").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "repo")
    assert scan.in_tests(str(tmp_path / "repo" / "tests" / "test_x.py"))
    assert not scan.in_tests(str(tmp_path / "elsewhere" / "tests" / "test_x.py"))


# --- stale lock files --------------------------------------------------------------------


@pytest.mark.unit
def test_a_successful_checkpointed_run_leaves_no_lock_file(tmp_path, monkeypatch, offline):  # noqa: F811
    graph = _checkpoint_graph(tmp_path, monkeypatch, max_llm_calls_per_run=500)
    graph.propagate("NVDA", TRADE_DATE)
    assert sorted(p.name for p in (tmp_path / "cache" / "checkpoints").iterdir()) == ["NVDA.db"]


@pytest.mark.unit
def test_clearing_checkpoints_removes_only_unheld_lock_files(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    stale = checkpoints / "AAPL.aaaa.usage.json.lock"
    active = checkpoints / "NVDA.bbbb.usage.json.lock"
    stale.write_text("")
    holder = locked(str(active)[: -len(".lock")], wait=False)   # an active run's lock
    assert holder.__enter__()
    try:
        clear_all_checkpoints(tmp_path)
        assert not stale.exists() and active.exists()           # the active one is untouched
        assert remove_stale_lock(active) is False
    finally:
        holder.__exit__(None, None, None)
    assert remove_stale_lock(active) is True


@pytest.mark.unit
def test_a_lock_file_cleared_away_is_retaken_fresh(tmp_path, monkeypatch, offline):  # noqa: F811
    graph = _checkpoint_graph(tmp_path, monkeypatch, max_llm_calls_per_run=10)
    graph.begin_checkpoint("NVDA", TRADE_DATE)
    graph.end_checkpoint()
    clear_all_checkpoints(tmp_path / "cache")                   # removes the now-unheld lock file
    graph.begin_checkpoint("NVDA", TRADE_DATE)
    assert graph._usage_lock_file.exists()                      # held on a file that still exists
    graph.end_checkpoint()


# --- a refused resume leaves nothing live ----------------------------------------------------


@pytest.mark.unit
def test_a_refused_resume_ends_the_account_started_for_it(tmp_path, monkeypatch, offline):  # noqa: F811
    holder = _checkpoint_graph(tmp_path, monkeypatch, max_llm_calls_per_run=10)
    holder.begin_checkpoint("NVDA", TRADE_DATE)
    try:
        cli_style = _checkpoint_graph(tmp_path, monkeypatch, max_llm_calls_per_run=10)
        cli_style.create_run_state("NVDA", TRADE_DATE)          # the CLI starts the run first...
        started = current_account()
        with pytest.raises(LLMBudgetExceeded):
            cli_style.begin_checkpoint("NVDA", TRADE_DATE)      # ...then its checkpoint is refused
        assert started.ended and started not in cli_style.budget._live
        cli_style.end_checkpoint()
    finally:
        holder.end_checkpoint()


@pytest.mark.unit
def test_documentation_states_the_records_are_not_tamper_resistant():
    assert "not tamper-resistant" in (ROOT / "deploy" / "cloud" / "README.md").read_text().lower()
    assert "not tamper-resistant" in (ROOT / "README.md").read_text().lower()
    assert "not tamper-resistant" in json.dumps(budget_module.__doc__).lower()
