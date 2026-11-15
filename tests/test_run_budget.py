"""The shared LLM budget: call, token and cost limits enforced before each request.

Unit tests drive ``RunBudget`` through its callback the way LangChain does; the
graph tests run the whole pipeline on scripted models with no network, so a limit
is shown to stop a real run at exactly the call it names.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.runnables import RunnableLambda

from tests.test_graph_end_to_end import STRUCTURED, TRADE_DATE, ScriptedModel, _Client, offline  # noqa: F401
from tradingagents.agents.structured import invoke_structured
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph import trading_graph
from tradingagents.llm_clients.budget import (
    BYTES_PER_TOKEN,
    LLMBudgetExceeded,
    PricingError,
    RunBudget,
    _message_bytes,
    describe_usage,
    load_pricing,
)

ROOT = Path(__file__).resolve().parents[1]
TODAY = datetime.now(UTC).date()


# --- helpers ---------------------------------------------------------------------


def _pricing(tmp_path, models=None, as_of=None, **extra) -> Path:
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps({
        "currency": "USD",
        "as_of": (as_of or TODAY).isoformat() if not isinstance(as_of, str) else as_of,
        "source": "test",
        "models": models if models is not None else {
            "openai:quick-model": {"input_per_mtok": 1000.0, "output_per_mtok": 2000.0},
            "openai:deep-model": {"input_per_mtok": 1000.0, "output_per_mtok": 2000.0},
        },
        **extra,
    }), encoding="utf-8")
    return path


def _result(text="ok", usage=None, cost=None) -> LLMResult:
    message = AIMessage(content=text)
    if usage is not None:
        message.usage_metadata = {"input_tokens": usage[0], "output_tokens": usage[1],
                                  "total_tokens": usage[0] + usage[1]}
    llm_output = {"token_usage": {"cost": cost}} if cost is not None else None
    return LLMResult(generations=[[ChatGeneration(message=message)]], llm_output=llm_output)


def _call(callback, prompt="x" * 30, usage=(10, 5), cost=None, fail=False):
    """One request as LangChain reports it: start, then end or error."""
    run_id = uuid.uuid4()
    callback.on_chat_model_start({}, [[HumanMessage(content=prompt)]], run_id=run_id)
    if fail:
        callback.on_llm_error(RuntimeError("boom"), run_id=run_id)
    else:
        callback.on_llm_end(_result(usage=usage, cost=cost), run_id=run_id)
    return run_id


# --- calls -----------------------------------------------------------------------


@pytest.mark.unit
def test_the_call_limit_refuses_the_call_past_it_before_it_is_sent():
    budget = RunBudget(max_calls=3)
    cb = budget.callback("openai", "quick-model")
    for _ in range(3):
        _call(cb)
    with pytest.raises(LLMBudgetExceeded) as refused:
        _call(cb)
    assert refused.value.scope == "calls"
    assert refused.value.usage["calls"]["attempted"] == 3


@pytest.mark.unit
def test_failed_requests_count_towards_the_call_limit():
    budget = RunBudget(max_calls=2)
    cb = budget.callback("openai", "quick-model")
    _call(cb, fail=True)
    _call(cb)
    with pytest.raises(LLMBudgetExceeded):
        _call(cb)
    calls = budget.summary()["calls"]
    assert (calls["attempted"], calls["failed"], calls["succeeded"]) == (2, 1, 1)


@pytest.mark.unit
@pytest.mark.parametrize("sdk_logger, message, args", [
    ("openai._base_client", "Retrying request in %f seconds (retry %i of %s)", (0.5, 1, 2)),
    ("anthropic._base_client", "Retrying request to %s in %f seconds", ("/v1/messages", 0.5)),
])
def test_sdk_retries_are_counted_and_spend_the_call_limit(sdk_logger, message, args):
    budget = RunBudget(max_calls=2)
    cb = budget.callback("openai", "quick-model")
    run_id = uuid.uuid4()
    cb.on_chat_model_start({}, [[HumanMessage(content="hello world")]], run_id=run_id)
    logging.getLogger(sdk_logger).info(message, *args)   # the SDK retrying inside the call
    cb.on_llm_end(_result(usage=(10, 5)), run_id=run_id)

    assert budget.summary()["calls"]["sdk_retries"] == 1
    with pytest.raises(LLMBudgetExceeded, match="call limit"):
        _call(cb)                                        # 1 call + 1 retry = the limit


@pytest.mark.unit
def test_parallel_callers_share_one_thread_safe_budget():
    budget = RunBudget(max_calls=50)
    callbacks = [budget.callback("openai", "quick-model"), budget.callback("openai", "deep-model")]
    allowed, refused = [], []
    lock = threading.Lock()
    start = threading.Barrier(8)

    def worker(index):
        start.wait()
        for _ in range(20):
            try:
                _call(callbacks[index % 2])
                with lock:
                    allowed.append(1)
            except LLMBudgetExceeded:
                with lock:
                    refused.append(1)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(allowed) == 50 and len(refused) == 8 * 20 - 50
    assert budget.summary()["calls"]["attempted"] == 50


# --- tokens ----------------------------------------------------------------------


@pytest.mark.unit
def test_the_token_limit_counts_provider_reported_usage_and_reserves_the_next_call():
    hundred_tokens = "x" * (100 * BYTES_PER_TOKEN)
    budget = RunBudget(max_tokens=1000, output_reservation=100)
    cb = budget.callback("openai", "quick-model")
    _call(cb, prompt=hundred_tokens, usage=(600, 200))  # 800 counted
    assert budget.summary()["tokens"]["provider_reported_input"] == 600
    budget_at_limit = RunBudget(max_tokens=1000, output_reservation=100)
    at_limit = budget_at_limit.callback("openai", "quick-model")
    _call(at_limit, prompt=hundred_tokens, usage=(600, 200))
    _call(at_limit, prompt=hundred_tokens)              # 800 + 100 + 100 = exactly 1000: allowed
    with pytest.raises(LLMBudgetExceeded) as refused:
        _call(cb, prompt=hundred_tokens + "x" * BYTES_PER_TOKEN)   # 800 + 101 + 100 > 1000
    assert refused.value.scope == "tokens"


@pytest.mark.unit
def test_a_response_without_usage_is_counted_at_an_estimate_labelled_as_such():
    budget = RunBudget(max_tokens=10_000)
    cb = budget.callback("openai", "quick-model")
    _call(cb, prompt="x" * (100 * BYTES_PER_TOKEN), usage=None)
    tokens = budget.summary()["tokens"]
    assert tokens["provider_reported_input"] == 0
    assert tokens["estimated_unreported"] >= 100
    assert tokens["calls_without_reported_usage"] == 1


@pytest.mark.unit
def test_begin_run_resets_the_run_but_the_session_keeps_counting():
    budget = RunBudget(max_calls=1)
    cb = budget.callback("openai", "quick-model")
    _call(cb)
    budget.begin_run()
    _call(cb)                                           # a new analysis gets its own allowance
    assert budget.summary()["calls"]["attempted"] == 1
    assert budget.summary(session=True)["calls"]["attempted"] == 2


# --- pricing fails closed --------------------------------------------------------


def _config(tmp_path, **overrides):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(data_cache_dir=str(tmp_path / "cache"), **overrides)
    return cfg


MODELS = [("openai", "quick-model"), ("openai", "deep-model")]


@pytest.mark.unit
def test_a_cost_limit_without_a_pricing_file_fails_closed(tmp_path):
    with pytest.raises(PricingError, match="no pricing file"):
        RunBudget.from_config(_config(tmp_path, max_cost_per_run_usd=1.0), MODELS)


@pytest.mark.unit
@pytest.mark.parametrize("change, message", [
    ({"as_of": (TODAY - timedelta(days=91)).isoformat()}, "older than 90 days"),
    ({"as_of": (TODAY + timedelta(days=1)).isoformat()}, "in the future"),
    ({"as_of": "YYYY-MM-DD"}, "as_of must be"),
    ({"currency": "EUR"}, "currency"),
    ({"models": {"openai:quick-model": {"input_per_mtok": None, "output_per_mtok": 1.0}}}, "non-negative"),
    ({"models": {"openai:quick-model": {"input_per_mtok": -1, "output_per_mtok": 1.0}}}, "non-negative"),
    ({"models": {"openai:quick-model": {"input_per_mtok": 1.0, "output_per_mtok": 1.0}}}, "no price for openai:deep-model"),
])
def test_doubtful_pricing_fails_closed(tmp_path, change, message):
    path = _pricing(tmp_path)
    data = json.loads(path.read_text())
    data.update(change)
    path.write_text(json.dumps(data))
    with pytest.raises(PricingError, match=message):
        RunBudget.from_config(_config(tmp_path, max_cost_per_day_usd=1.0, max_tokens=100,
                                      pricing_file=str(path)), MODELS)


@pytest.mark.unit
def test_the_shipped_pricing_template_is_rejected_until_filled_in(tmp_path):
    with pytest.raises(PricingError):
        load_pricing(ROOT / "deploy" / "cloud" / "pricing.example.json", 90)


@pytest.mark.unit
def test_a_pricing_problem_without_a_cost_limit_only_turns_estimates_off(tmp_path, caplog):
    budget = RunBudget.from_config(_config(tmp_path, pricing_file=str(tmp_path / "missing.json")), MODELS)
    assert budget.pricing is None
    assert budget.summary()["cost_usd"]["estimated"] is None
    assert "Cost estimates are off" in caplog.text


# --- cost ------------------------------------------------------------------------


@pytest.mark.unit
def test_the_run_cost_limit_refuses_a_call_whose_worst_case_would_pass_it(tmp_path):
    cfg = _config(tmp_path, max_cost_per_run_usd=1.0, max_tokens=100, pricing_file=str(_pricing(tmp_path)))
    budget = RunBudget.from_config(cfg, MODELS)
    cb = budget.callback("openai", "quick-model")
    ten_tokens = "x" * (10 * BYTES_PER_TOKEN)
    _call(cb, prompt=ten_tokens, usage=(300, 300))      # 0.3 + 0.6 = $0.90
    assert budget.summary()["cost_usd"]["estimated"] == pytest.approx(0.9)
    with pytest.raises(LLMBudgetExceeded) as refused:   # reserves 10 in + 100 out = $0.21
        _call(cb, prompt=ten_tokens)
    assert refused.value.scope == "run_cost"


@pytest.mark.unit
def test_estimated_and_provider_reported_costs_are_kept_apart(tmp_path):
    cfg = _config(tmp_path, pricing_file=str(_pricing(tmp_path)))
    budget = RunBudget.from_config(cfg, MODELS)
    cb = budget.callback("openai", "quick-model")
    _call(cb, usage=(100, 100), cost=0.5)               # estimated $0.30, provider says $0.50
    _call(cb, usage=(100, 100))                         # estimated $0.30, nothing reported
    cost = budget.summary()["cost_usd"]
    assert cost["estimated"] == pytest.approx(0.6)
    assert cost["provider_reported"] == pytest.approx(0.5)
    assert cost["provider_reported_calls"] == 1
    assert cost["charged_against_limits"] == pytest.approx(0.8)   # the larger of the two, per call
    line = describe_usage(budget.summary())
    assert "estimated" in line and "reported by the provider over 1 call(s)" in line


@pytest.mark.unit
def test_missing_usage_under_a_cost_limit_stops_every_later_call(tmp_path):
    cfg = _config(tmp_path, max_cost_per_run_usd=100.0, max_tokens=100, pricing_file=str(_pricing(tmp_path)))
    budget = RunBudget.from_config(cfg, MODELS)
    cb = budget.callback("openai", "quick-model")
    _call(cb, usage=None)
    with pytest.raises(LLMBudgetExceeded) as refused:
        _call(cb)
    assert refused.value.scope == "usage_unreported"
    budget.begin_run()
    with pytest.raises(LLMBudgetExceeded):              # sticky across runs: spend is unbounded
        _call(cb)


@pytest.mark.unit
def test_failed_requests_are_charged_their_input_estimate(tmp_path):
    cfg = _config(tmp_path, pricing_file=str(_pricing(tmp_path)))
    budget = RunBudget.from_config(cfg, MODELS)
    _call(budget.callback("openai", "quick-model"), prompt="x" * (100 * BYTES_PER_TOKEN), fail=True)
    summary = budget.summary()
    assert summary["cost_usd"]["charged_against_limits"] == pytest.approx(100 * 1000 / 1e6)
    assert summary["tokens"]["estimated_unreported"] == 100


@pytest.mark.unit
def test_the_daily_limit_holds_across_runs_through_its_ledger(tmp_path):
    cfg = _config(tmp_path, max_cost_per_day_usd=1.0, max_tokens=10, pricing_file=str(_pricing(tmp_path)))
    first = RunBudget.from_config(cfg, MODELS)
    _call(first.callback("openai", "quick-model"), usage=(400, 400))   # $1.20 > the day's $1.00
    ledger = next((tmp_path / "cache" / "spend").glob("*.json"))
    assert json.loads(ledger.read_text())["charged_usd"] == pytest.approx(1.2)

    second = RunBudget.from_config(cfg, MODELS)          # another run, same day
    with pytest.raises(LLMBudgetExceeded) as refused:
        _call(second.callback("openai", "quick-model"))
    assert refused.value.scope == "day_cost"


@pytest.mark.unit
def test_an_unreadable_ledger_fails_closed(tmp_path):
    cfg = _config(tmp_path, max_cost_per_day_usd=1.0, max_tokens=100, pricing_file=str(_pricing(tmp_path)))
    budget = RunBudget.from_config(cfg, MODELS)
    ledger_dir = tmp_path / "cache" / "spend"
    ledger_dir.mkdir(parents=True)
    (ledger_dir / f"{TODAY.isoformat()}.json").write_text("not json")
    with pytest.raises(LLMBudgetExceeded, match="cannot be read"):
        _call(budget.callback("openai", "quick-model"))


@pytest.mark.unit
@pytest.mark.parametrize("key, value", [
    ("max_llm_calls_per_run", "abc"), ("max_llm_calls_per_run", "0"),
    ("max_tokens_per_run", "-5"), ("max_cost_per_run_usd", "nan"), ("max_llm_calls_per_run", True),
])
def test_invalid_limits_fail_at_startup(tmp_path, key, value):
    with pytest.raises(ValueError, match=key):
        RunBudget.from_config(_config(tmp_path, **{key: value}), MODELS)


@pytest.mark.unit
def test_limits_from_the_environment_arrive_as_strings_and_are_parsed(tmp_path):
    budget = RunBudget.from_config(_config(tmp_path, max_llm_calls_per_run="40", max_tokens_per_run="200000",
                                           max_tokens="4096"), MODELS)
    assert (budget.max_calls, budget.max_tokens, budget.output_reservation) == (40, 200_000, 4096)


@pytest.mark.unit
def test_the_env_variables_reach_the_config(monkeypatch):
    import tradingagents.default_config as default_config_module

    monkeypatch.setenv("TRADINGAGENTS_MAX_LLM_CALLS", "40")
    monkeypatch.setenv("TRADINGAGENTS_MAX_TOKENS_PER_RUN", "200000")
    monkeypatch.setenv("TRADINGAGENTS_MAX_COST_PER_RUN", "0.5")
    monkeypatch.setenv("TRADINGAGENTS_PRICING_FILE", "/tmp/prices.json")
    config = default_config_module.build_default_config()
    assert config["max_llm_calls_per_run"] == "40"
    assert config["max_tokens_per_run"] == "200000"
    assert config["max_cost_per_run_usd"] == "0.5"
    assert config["pricing_file"] == "/tmp/prices.json"
    assert config["max_pricing_age_days"] == 90


@pytest.mark.unit
def test_every_limit_is_off_by_default():
    for key in ("max_llm_calls_per_run", "max_tokens_per_run", "max_cost_per_run_usd",
                "max_cost_per_day_usd", "max_reflection_calls_per_run", "max_reflection_tokens_per_run",
                "pricing_file"):
        assert DEFAULT_CONFIG[key] is None


# --- structured output cannot bypass a limit -------------------------------------


@pytest.mark.unit
def test_a_refused_structured_call_is_not_retried_as_free_text(caplog):
    def refused(_):
        raise LLMBudgetExceeded("calls", "LLM call limit reached")

    with pytest.raises(LLMBudgetExceeded):
        invoke_structured(RunnableLambda(refused), "prompt", "Portfolio Manager")
    assert "retrying once as free text" not in caplog.text


# --- the whole graph -------------------------------------------------------------


class MeteredModel(ScriptedModel):
    """ScriptedModel whose structured calls go through the model, as a provider's do,
    so the budget sees them; optionally reports token usage like a provider."""

    usage: tuple | None = None

    def with_structured_output(self, schema, **kwargs):
        if not self.structured:
            raise NotImplementedError

        def call(prompt):
            self.invoke(prompt)
            return STRUCTURED[schema]

        return RunnableLambda(call)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        result = super()._generate(messages, stop, run_manager, **kwargs)
        if self.usage:
            tokens_in, tokens_out = self.usage
            if tokens_in == "prompt":   # exactly what the budget estimated for the prompt
                tokens_in = math.ceil(sum(_message_bytes(m) for m in messages) / BYTES_PER_TOKEN)
            result.generations[0].message.usage_metadata = {
                "input_tokens": tokens_in, "output_tokens": tokens_out,
                "total_tokens": tokens_in + tokens_out}
        return result


def _graph(tmp_path, monkeypatch, model, **config):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update({"results_dir": str(tmp_path / "results"), "data_cache_dir": str(tmp_path / "cache"),
                "memory_log_path": str(tmp_path / "log.md"), "llm_provider": "openai",
                "quick_think_llm": "quick-model", "deep_think_llm": "deep-model", **config})
    # Each tier gets its own copy carrying the callbacks the graph attaches, as a
    # real client would; the copies share one call list.
    monkeypatch.setattr(trading_graph, "create_tier_client",
                        lambda config, tier, **k: _Client(model.model_copy(update={"callbacks": k.get("callbacks")})))
    return trading_graph.TradingAgentsGraph(config=cfg)


@pytest.fixture
def baseline_calls(tmp_path, monkeypatch, offline):  # noqa: F811
    model = MeteredModel(structured=True)
    _graph(tmp_path / "baseline", monkeypatch, model).propagate("NVDA", TRADE_DATE)
    return len(model.calls)


@pytest.mark.unit
def test_without_limits_a_run_is_unchanged_and_records_its_usage(tmp_path, monkeypatch, offline):  # noqa: F811
    model = MeteredModel(structured=True)
    graph = _graph(tmp_path, monkeypatch, model)

    state, signal = graph.propagate("NVDA", TRADE_DATE)

    assert signal == "Overweight"
    assert graph.usage()["calls"]["attempted"] == len(model.calls)   # both tiers, every analyst
    log = next(tmp_path.glob("results/NVDA/TradingAgentsStrategy_logs/*.json"))
    usage = json.loads(log.read_text())["llm_usage"]
    assert usage["calls"]["attempted"] == len(model.calls)
    assert usage["cost_usd"]["estimated"] is None                    # no pricing, so no made-up figure


@pytest.mark.unit
@pytest.mark.parametrize("limit", [1, 5, 9])
def test_the_call_limit_stops_a_run_at_exactly_that_many_calls(tmp_path, monkeypatch, offline, limit):  # noqa: F811
    model = MeteredModel(structured=True)
    graph = _graph(tmp_path, monkeypatch, model, max_llm_calls_per_run=limit)

    with pytest.raises(LLMBudgetExceeded) as refused:
        graph.propagate("NVDA", TRADE_DATE)

    assert refused.value.scope == "calls"
    assert len(model.calls) == limit                    # parallel analysts never overshoot
    assert graph.memory_log.load_entries() == []        # no decision recorded


@pytest.mark.unit
def test_a_limit_hit_at_the_last_structured_call_sends_no_free_text_call(
        tmp_path, monkeypatch, offline, baseline_calls, caplog):  # noqa: F811
    model = MeteredModel(structured=True)
    graph = _graph(tmp_path, monkeypatch, model, max_llm_calls_per_run=baseline_calls - 1)

    with pytest.raises(LLMBudgetExceeded):
        graph.propagate("NVDA", TRADE_DATE)             # the Portfolio Manager's call is refused

    assert len(model.calls) == baseline_calls - 1
    assert "retrying once as free text" not in caplog.text


@pytest.mark.unit
def test_a_failing_provider_call_is_counted(tmp_path, monkeypatch, offline):  # noqa: F811
    # The third call fails; whichever agent made it recovers (a structured call
    # retries as free text, as before), and the failure still counts.
    model = MeteredModel(structured=True, fail_at=3)
    graph = _graph(tmp_path, monkeypatch, model)
    with contextlib.suppress(RuntimeError):
        graph.propagate("NVDA", TRADE_DATE)
    calls = graph.usage()["calls"]
    assert calls["failed"] == 1
    assert calls["attempted"] == len(model.calls) == calls["succeeded"] + calls["failed"]


@pytest.mark.unit
def test_the_token_limit_stops_a_run(tmp_path, monkeypatch, offline):  # noqa: F811
    # Real usage within each call's reservation (output under max_tokens), so the
    # limit holds exactly even with analysts in flight together.
    model = MeteredModel(structured=True, usage=(500, 200))
    graph = _graph(tmp_path, monkeypatch, model, max_tokens_per_run=10_000, max_tokens=1_000)
    with pytest.raises(LLMBudgetExceeded) as refused:
        graph.propagate("NVDA", TRADE_DATE)
    assert refused.value.scope == "tokens"
    assert refused.value.usage["tokens"]["counted_total"] <= 10_000


@pytest.mark.unit
def test_a_cost_limit_without_valid_prices_stops_the_graph_before_any_call(tmp_path, monkeypatch):
    model = MeteredModel(structured=True)
    with pytest.raises(ValueError, match="pricing"):
        _graph(tmp_path, monkeypatch, model, max_cost_per_run_usd=1.0)
    assert model.calls == []


@pytest.mark.unit
def test_the_run_cost_limit_stops_a_run(tmp_path, monkeypatch, offline):  # noqa: F811
    model = MeteredModel(structured=True, usage=(1_000, 500))
    graph = _graph(tmp_path, monkeypatch, model, max_cost_per_run_usd=10.0, max_tokens=500,
                   pricing_file=str(_pricing(tmp_path)))   # $2 per call
    with pytest.raises(LLMBudgetExceeded) as refused:
        graph.propagate("NVDA", TRADE_DATE)
    assert refused.value.scope == "run_cost"
    assert refused.value.usage["cost_usd"]["charged_against_limits"] <= 10.0


@pytest.mark.unit
def test_a_backtest_stops_at_its_total_cost_and_keeps_the_cells_it_ran(tmp_path, monkeypatch, offline):  # noqa: F811
    from tradingagents.backtest import run_backtest

    pricing = _pricing(tmp_path)
    # Reported input equal to the budget's estimate, output at max_tokens: each
    # call is charged exactly what it reserved.
    model = MeteredModel(structured=True, usage=("prompt", 50))
    graph = _graph(tmp_path / "one", monkeypatch, model, max_tokens=50, pricing_file=str(pricing))
    graph.propagate("NVDA", "2026-01-08")
    one_run = graph.usage()["cost_usd"]["charged_against_limits"]

    cfg = copy.deepcopy(graph.config)
    cfg.update(results_dir=str(tmp_path / "bt"), memory_log_path=str(tmp_path / "bt" / "log.md"))
    result = run_backtest(["NVDA"], ["2026-01-08", "2026-01-09", "2026-01-12"], cfg,
                          max_total_cost_usd=one_run * 1.5)

    assert result.cells_run == 1
    assert result.stopped_reason and "total estimated cost limit" in result.stopped_reason
    assert result.usage["cost_usd"]["charged_against_limits"] <= one_run * 1.5


@pytest.mark.unit
def test_a_backtest_total_cost_without_prices_fails_before_any_cell(tmp_path, monkeypatch):
    from tradingagents.backtest import run_backtest

    model = MeteredModel(structured=True)
    monkeypatch.setattr(trading_graph, "create_tier_client", lambda config, tier, **k: _Client(model))
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(results_dir=str(tmp_path), data_cache_dir=str(tmp_path / "cache"))
    with pytest.raises(ValueError, match="pricing file"):
        run_backtest(["NVDA"], ["2026-01-08"], cfg, max_total_cost_usd=5.0)
    assert model.calls == []


@pytest.mark.unit
def test_changing_a_limit_does_not_invalidate_a_checkpoint(tmp_path, monkeypatch):
    model = MeteredModel(structured=True)
    loose = _graph(tmp_path, monkeypatch, model)
    strict = _graph(tmp_path, monkeypatch, model, max_llm_calls_per_run=40, max_tokens_per_run=200_000)
    assert loose._run_signature("stock") == strict._run_signature("stock")
