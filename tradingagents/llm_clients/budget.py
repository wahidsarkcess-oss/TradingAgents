"""Limits on LLM calls, tokens and estimated spend, shared by every model a run uses.

One ``RunBudget`` belongs to a ``TradingAgentsGraph``. Each model tier gets a
``BudgetCallback`` bound to that budget, so the quick and deep models and the
analysts running in parallel all draw from one thread-safe count. Every check
happens *before* a request is sent (``on_chat_model_start``), and a refused call
raises ``LLMBudgetExceeded`` out of the model's ``invoke``. No request leaves the
process once a limit is reached.

What is counted, per run (``begin_run``) and per graph session:

- **Calls:** every attempt the model client starts, whether it succeeds or fails,
  plus the retries the OpenAI and Anthropic SDKs make internally (read from their
  "Retrying request" log records). A retry cannot be stopped once the SDK has
  started it; it counts towards the next call's check.
- **Tokens:** the input and output tokens the provider reports for each response.
  A response without usage, a failed request and an SDK retry are counted at an
  estimate (prompt characters / 2 plus the reply) and reported separately as
  ``estimated_unreported``.
- **Cost:** never reported by most providers. ``estimated`` is the counted tokens
  priced from the pricing file you supply. ``provider_reported`` is the cost a
  provider returned with its response (OpenRouter does), with the number of calls
  it covers. The amount charged against the limits is the larger of the two for
  each call; failed requests and retries are charged their input estimate. All
  of this is an upper-bound estimate, not a bill.

Cost limits fail closed. When ``max_cost_per_run_usd`` or ``max_cost_per_day_usd``
is set, the graph refuses to start unless a valid, recent pricing file prices
every model the run uses. A run also stops if a provider returns a response with
no token usage, because its cost can no longer be bounded.

Before each call the budget reserves that call's worst case: its estimated
prompt tokens plus the output limit (``max_tokens``, else 8192 tokens), so
concurrent calls cannot jointly overshoot a token or cost limit. The call limit
is exact. Token and cost limits are exact while each call's real usage stays
within its reservation; the prompt estimate (2 characters per token) overstates
English text about twice, but very dense text (long numeric tables, CJK) can
tokenize tighter, and a provider that ignores ``max_tokens`` (a ChatGPT plan)
can answer longer. Then the calls already in flight can overshoot by the
difference, and every call after them is refused.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from tradingagents.dataflows.files import locked, replace_file

logger = logging.getLogger(__name__)

# The config keys this module reads. None (the default) disables each limit.
LIMIT_KEYS = (
    "max_llm_calls_per_run",
    "max_tokens_per_run",
    "max_cost_per_run_usd",
    "max_cost_per_day_usd",
)
BUDGET_CONFIG_KEYS = (*LIMIT_KEYS, "pricing_file", "max_pricing_age_days")

# Output reserved for a call when no per-call ``max_tokens`` limit is set.
DEFAULT_OUTPUT_RESERVATION = 8192
# Characters per token for estimates. English averages about 4 and numeric tables
# about 2; 2 errs towards refusing a call rather than overshooting a limit.
CHARS_PER_TOKEN = 2

# Loggers whose "Retrying request" records mark an SDK-internal retry.
RETRY_LOGGERS = ("openai._base_client", "anthropic._base_client")


class LLMBudgetExceeded(RuntimeError):
    """A limit refused an LLM call before it was sent.

    ``scope`` names the limit: ``calls``, ``tokens``, ``run_cost``, ``day_cost``,
    ``session_cost``, ``pricing`` or ``usage_unreported``. ``usage`` is the run's
    usage summary when it was refused. The message never contains credentials.
    """

    def __init__(self, scope: str, message: str, usage: dict | None = None):
        super().__init__(message)
        self.scope = scope
        self.usage = usage or {}


class PricingError(ValueError):
    """The pricing file is missing, malformed, stale, or does not price a model."""


# --- pricing -----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens. Cached input falls back to the full input rate."""

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None

    def cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        cached = min(max(cached_tokens, 0), max(input_tokens, 0))
        cached_rate = self.input_per_mtok if self.cached_input_per_mtok is None else self.cached_input_per_mtok
        return ((max(input_tokens, 0) - cached) * self.input_per_mtok
                + cached * cached_rate
                + max(output_tokens, 0) * self.output_per_mtok) / 1_000_000


@dataclass(frozen=True)
class Pricing:
    prices: dict[str, ModelPrice]
    as_of: str
    source: str
    path: str

    def price(self, provider: str, model: str) -> ModelPrice | None:
        """The price of ``provider:model``, else of the bare model id."""
        return self.prices.get(f"{provider.lower()}:{model}") or self.prices.get(model)


def _price_field(entry: dict, name: str, key: str, required: bool) -> float | None:
    value = entry.get(name)
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise PricingError(f"pricing for {key!r}: {name} must be a non-negative number, got {value!r}")
    return float(value)


def load_pricing(path: str | Path, max_age_days: int | None, today: date | None = None) -> Pricing:
    """Read and validate a pricing file; raise ``PricingError`` on anything doubtful.

    Expected shape::

        {"currency": "USD", "as_of": "2026-11-01", "source": "https://...",
         "models": {"openai:gpt-6-luna": {"input_per_mtok": 0.0, "output_per_mtok": 0.0,
                                          "cached_input_per_mtok": 0.0}}}

    ``as_of`` is the day the prices were copied from the provider; a file older
    than ``max_age_days`` (or dated in the future) is treated as unreliable.
    """
    path = Path(path).expanduser()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PricingError(f"pricing file {path} cannot be read ({type(exc).__name__})") from None
    except ValueError:
        raise PricingError(f"pricing file {path} is not valid JSON") from None
    if not isinstance(data, dict):
        raise PricingError(f"pricing file {path} must hold a JSON object")
    if data.get("currency") != "USD":
        raise PricingError(f"pricing file {path}: currency must be \"USD\"")
    as_of = data.get("as_of")
    try:
        as_of_date = datetime.strptime(str(as_of), "%Y-%m-%d").date()
    except ValueError:
        raise PricingError(f"pricing file {path}: as_of must be the YYYY-MM-DD the prices were checked") from None
    today = today or datetime.now(UTC).date()
    if as_of_date > today:
        raise PricingError(f"pricing file {path}: as_of {as_of} is in the future")
    if max_age_days is not None and (today - as_of_date).days > max_age_days:
        raise PricingError(
            f"pricing file {path}: prices dated {as_of} are older than {max_age_days} days; "
            "re-check them against the provider's pricing page and update as_of"
        )
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise PricingError(f"pricing file {path}: models must map model ids to prices")
    prices = {}
    for key, entry in models.items():
        if not isinstance(entry, dict):
            raise PricingError(f"pricing for {key!r} must be an object")
        prices[str(key)] = ModelPrice(
            input_per_mtok=_price_field(entry, "input_per_mtok", key, True),
            output_per_mtok=_price_field(entry, "output_per_mtok", key, True),
            cached_input_per_mtok=_price_field(entry, "cached_input_per_mtok", key, False),
        )
    return Pricing(prices=prices, as_of=str(as_of), source=str(data.get("source") or ""), path=str(path))


# --- config ------------------------------------------------------------------------


def _limit(config: dict, key: str, kind: type) -> int | float | None:
    """A limit from config (env values arrive as strings); None or "" means off."""
    value = config.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number, not a boolean: {value!r}")
    try:
        number = kind(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a {'whole ' if kind is int else ''}number, got {value!r}") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{key} must be greater than 0, got {value!r}")
    return number


# --- counting ----------------------------------------------------------------------


@dataclass
class _Totals:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    sdk_retries: int = 0
    reported_input: int = 0
    reported_output: int = 0
    reported_cached_input: int = 0
    estimated_unreported: int = 0
    calls_without_usage: int = 0
    estimated_cost: float = 0.0
    unpriced_calls: int = 0
    provider_reported_cost: float = 0.0
    provider_reported_cost_calls: int = 0
    charged_cost: float = 0.0

    @property
    def tokens(self) -> int:
        return self.reported_input + self.reported_output + self.estimated_unreported


@dataclass
class _Reservation:
    provider: str
    model: str
    price: ModelPrice | None
    input_estimate: int
    tokens: int
    cost: float


_active = threading.local()   # (budget, run_id) of the call this thread is making


class _RetryCounter(logging.Handler):
    """Counts SDK-internal retries for the call the logging thread is making."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not str(record.msg).startswith("Retrying request"):
                return
            current = getattr(_active, "call", None)
            if current is not None:
                budget, run_id = current
                budget._retry(run_id)
        except Exception:  # noqa: BLE001 — counting must never break a request
            pass


_retry_counter = _RetryCounter(level=logging.INFO)
_retry_counter_lock = threading.Lock()
_retry_counter_installed = False


def _install_retry_counter() -> None:
    global _retry_counter_installed
    with _retry_counter_lock:
        if _retry_counter_installed:
            return
        for name in RETRY_LOGGERS:
            sdk_logger = logging.getLogger(name)
            sdk_logger.addHandler(_retry_counter)
            # The SDKs log retries at INFO; the record must be created to be seen.
            if sdk_logger.getEffectiveLevel() > logging.INFO:
                sdk_logger.setLevel(logging.INFO)
        _retry_counter_installed = True


def _message_chars(message: Any) -> int:
    chars = len(str(getattr(message, "content", message) or ""))
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        chars += len(json.dumps(tool_calls, default=str))
    return chars


def _usage_of(response: Any) -> tuple[int, int, int] | None:
    """(input, output, cached input) tokens the provider reported, or None."""
    try:
        message = response.generations[0][0].message
        usage = getattr(message, "usage_metadata", None)
    except (AttributeError, IndexError, TypeError):
        usage = None
    if usage and (usage.get("input_tokens") is not None or usage.get("output_tokens") is not None):
        cached = (usage.get("input_token_details") or {}).get("cache_read") or 0
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0), int(cached)
    token_usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
    if token_usage.get("prompt_tokens") is not None or token_usage.get("completion_tokens") is not None:
        return int(token_usage.get("prompt_tokens") or 0), int(token_usage.get("completion_tokens") or 0), 0
    return None


def _reported_cost_of(response: Any) -> float | None:
    """A cost the provider itself returned with the response (e.g. OpenRouter's usage.cost)."""
    candidates = [((getattr(response, "llm_output", None) or {}).get("token_usage") or {}).get("cost")]
    try:
        metadata = response.generations[0][0].message.response_metadata or {}
        candidates.append((metadata.get("token_usage") or {}).get("cost"))
        candidates.append((metadata.get("usage") or {}).get("cost"))
    except (AttributeError, IndexError, TypeError):
        pass
    for value in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            return float(value)
    return None


def _output_chars(response: Any) -> int:
    try:
        return sum(_message_chars(g.message) if hasattr(g, "message") else len(g.text or "")
                   for batch in response.generations for g in batch)
    except (AttributeError, TypeError):
        return 0


class RunBudget:
    """Thread-safe limits shared by every LLM call of one graph."""

    def __init__(
        self,
        *,
        max_calls: int | None = None,
        max_tokens: int | None = None,
        max_run_cost: float | None = None,
        max_day_cost: float | None = None,
        output_reservation: int = DEFAULT_OUTPUT_RESERVATION,
        pricing: Pricing | None = None,
        ledger_dir: str | Path | None = None,
        models: tuple[tuple[str, str], ...] = (),
    ):
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.max_run_cost = max_run_cost
        self.max_day_cost = max_day_cost
        self.max_session_cost: float | None = None
        self.output_reservation = output_reservation
        self.pricing = pricing
        self.ledger_dir = Path(ledger_dir).expanduser() if ledger_dir else None
        self.models = tuple(models)
        self._lock = threading.RLock()
        self._run = _Totals()
        self._session = _Totals()
        self._inflight: dict[Any, _Reservation] = {}
        # Sticky: once a provider omits usage while a cost limit is on, spend can
        # no longer be bounded, so every later call is refused.
        self._usage_unreported = False
        if self.active:
            _install_retry_counter()

    @classmethod
    def from_config(cls, config: dict, models: list[tuple[str, str]] | tuple = ()) -> RunBudget:
        """Build from a TradingAgents config; fail closed on doubtful cost limits.

        ``models`` are the ``(provider, model)`` pairs the run will call. When a
        cost limit is set, each must be priced by a valid pricing file, or this
        raises ``PricingError`` before any model is called.
        """
        max_calls = _limit(config, "max_llm_calls_per_run", int)
        max_tokens = _limit(config, "max_tokens_per_run", int)
        max_run_cost = _limit(config, "max_cost_per_run_usd", float)
        max_day_cost = _limit(config, "max_cost_per_day_usd", float)
        max_age = config.get("max_pricing_age_days")
        max_age = None if max_age in (None, "") else int(max_age)
        per_call_output = _limit(config, "max_tokens", int)
        pricing_file = config.get("pricing_file") or None

        pricing = None
        if max_run_cost is not None or max_day_cost is not None:
            if not pricing_file:
                raise PricingError(
                    "a cost limit is set but no pricing file is: set TRADINGAGENTS_PRICING_FILE "
                    "(config pricing_file) to a file with your provider's current prices"
                )
            pricing = load_pricing(pricing_file, max_age)
            _require_prices(pricing, models)
        elif pricing_file:
            # Cost is only reported, not enforced: a doubtful file means no estimate.
            try:
                pricing = load_pricing(pricing_file, max_age)
            except PricingError as exc:
                logger.warning("Cost estimates are off: %s", exc)

        return cls(
            max_calls=max_calls, max_tokens=max_tokens, max_run_cost=max_run_cost,
            max_day_cost=max_day_cost, output_reservation=per_call_output or DEFAULT_OUTPUT_RESERVATION,
            pricing=pricing, ledger_dir=Path(config.get("data_cache_dir") or ".") / "spend",
            models=tuple((p.lower(), m) for p, m in models),
        )

    # --- configuration -------------------------------------------------------

    @property
    def active(self) -> bool:
        return any(v is not None for v in (self.max_calls, self.max_tokens, self.max_run_cost,
                                           self.max_day_cost, self.max_session_cost, self.pricing))

    @property
    def cost_limited(self) -> bool:
        return any(v is not None for v in (self.max_run_cost, self.max_day_cost, self.max_session_cost))

    def set_session_cost_limit(self, limit_usd: float | None) -> None:
        """Cap the estimated spend of everything this graph runs (a whole backtest)."""
        if limit_usd is None:
            self.max_session_cost = None
            return
        limit = _limit({"max_total_cost_usd": limit_usd}, "max_total_cost_usd", float)
        if self.pricing is None:
            raise PricingError(
                "a total cost limit needs a valid pricing file: set TRADINGAGENTS_PRICING_FILE"
            )
        _require_prices(self.pricing, self.models)
        self.max_session_cost = limit
        _install_retry_counter()

    def callback(self, provider: str, model: str) -> BudgetCallback:
        """The callback handler to attach to the client of ``provider:model``."""
        return BudgetCallback(self, provider, model)

    # --- run lifecycle -------------------------------------------------------

    def begin_run(self) -> None:
        """Start counting a new analysis; session totals carry on."""
        with self._lock:
            self._run = _Totals()
            self._inflight.clear()

    def summary(self, session: bool = False) -> dict:
        """Usage of the current run (or the whole session), with every figure labelled."""
        with self._lock:
            totals = self._session if session else self._run
            priced = self.pricing is not None
            return {
                "limits": {
                    "max_llm_calls_per_run": self.max_calls,
                    "max_tokens_per_run": self.max_tokens,
                    "max_cost_per_run_usd": self.max_run_cost,
                    "max_cost_per_day_usd": self.max_day_cost,
                    "max_total_cost_usd": self.max_session_cost,
                    "per_call_output_reservation": self.output_reservation,
                },
                "calls": {
                    "attempted": totals.attempted,
                    "succeeded": totals.succeeded,
                    "failed": totals.failed,
                    "sdk_retries": totals.sdk_retries,
                },
                "tokens": {
                    "provider_reported_input": totals.reported_input,
                    "provider_reported_output": totals.reported_output,
                    "provider_reported_cached_input": totals.reported_cached_input,
                    "estimated_unreported": totals.estimated_unreported,
                    "counted_total": totals.tokens,
                    "calls_without_reported_usage": totals.calls_without_usage,
                },
                "cost_usd": {
                    "estimated": round(totals.estimated_cost, 6) if priced else None,
                    "estimated_basis": (
                        f"counted tokens x prices in {Path(self.pricing.path).name} as of {self.pricing.as_of}"
                        if priced else "no valid pricing file, so no estimate"
                    ),
                    "unpriced_calls": totals.unpriced_calls,
                    "provider_reported": (round(totals.provider_reported_cost, 6)
                                          if totals.provider_reported_cost_calls else None),
                    "provider_reported_calls": totals.provider_reported_cost_calls,
                    "charged_against_limits": round(totals.charged_cost, 6),
                    "note": "Estimates are an upper bound, not a bill; check the provider's billing page.",
                },
            }

    def describe(self, session: bool = False) -> str:
        """One line for a person: calls, tokens, and both kinds of cost."""
        return describe_usage(self.summary(session))

    # --- the callback's side -------------------------------------------------

    def _refuse(self, scope: str, message: str) -> None:
        raise LLMBudgetExceeded(scope, message, self.summary())

    def _start(self, run_id: Any, provider: str, model: str, prompt_chars: int) -> None:
        with self._lock:
            run, session = self._run, self._session
            input_estimate = math.ceil(prompt_chars / CHARS_PER_TOKEN)
            projected_tokens = input_estimate + self.output_reservation
            price = self.pricing.price(provider, model) if self.pricing else None
            projected_cost = price.cost(input_estimate, self.output_reservation) if price else 0.0
            inflight_tokens = sum(r.tokens for r in self._inflight.values())
            inflight_cost = sum(r.cost for r in self._inflight.values())

            if self.max_calls is not None and run.attempted + run.sdk_retries + 1 > self.max_calls:
                self._refuse("calls", f"LLM call limit reached: {self.max_calls} per analysis "
                                      "(TRADINGAGENTS_MAX_LLM_CALLS)")
            if self.max_tokens is not None and run.tokens + inflight_tokens + projected_tokens > self.max_tokens:
                self._refuse("tokens", f"token limit would be exceeded: {self.max_tokens:,} per analysis "
                                       f"(TRADINGAGENTS_MAX_TOKENS_PER_RUN); {run.tokens:,} counted so far, "
                                       f"this call reserves up to {projected_tokens:,}")
            if self.cost_limited:
                if self._usage_unreported:
                    self._refuse("usage_unreported", "a provider response carried no token usage, so spend "
                                                     "can no longer be bounded; cost limits fail closed")
                if price is None:
                    self._refuse("pricing", f"no price for {provider}:{model} in the pricing file; "
                                            "cost limits fail closed")
                if self.max_run_cost is not None and run.charged_cost + inflight_cost + projected_cost > self.max_run_cost:
                    self._refuse("run_cost", f"estimated cost limit would be exceeded: ${self.max_run_cost:.4f} per "
                                             f"analysis (TRADINGAGENTS_MAX_COST_PER_RUN); ${run.charged_cost:.4f} "
                                             f"charged so far, this call reserves up to ${projected_cost:.4f}")
                if (self.max_session_cost is not None
                        and session.charged_cost + inflight_cost + projected_cost > self.max_session_cost):
                    self._refuse("session_cost", f"total estimated cost limit would be exceeded: "
                                                 f"${self.max_session_cost:.4f}; ${session.charged_cost:.4f} charged so far")
                if self.max_day_cost is not None:
                    spent = self._ledger_spent()
                    if spent + inflight_cost + projected_cost > self.max_day_cost:
                        self._refuse("day_cost", f"daily estimated cost limit would be exceeded: "
                                                 f"${self.max_day_cost:.4f} per UTC day "
                                                 f"(TRADINGAGENTS_MAX_COST_PER_DAY); ${spent:.4f} charged today")

            run.attempted += 1
            session.attempted += 1
            self._inflight[run_id] = _Reservation(provider, model, price, input_estimate,
                                                  projected_tokens, projected_cost)
            _active.call = (self, run_id)

    def _end(self, run_id: Any, response: Any) -> None:
        with self._lock:
            reservation = self._inflight.pop(run_id, None)
            self._clear_active(run_id)
            if reservation is None:
                return
            usage = _usage_of(response)
            reported_cost = _reported_cost_of(response)
            price = reservation.price
            for totals in (self._run, self._session):
                totals.succeeded += 1
            if usage is None:
                tokens_in = reservation.input_estimate
                tokens_out = math.ceil(_output_chars(response) / CHARS_PER_TOKEN)
                cached = 0
                for totals in (self._run, self._session):
                    totals.calls_without_usage += 1
                    totals.estimated_unreported += tokens_in + tokens_out
                if self.cost_limited:
                    self._usage_unreported = True
                logger.warning("%s:%s returned no token usage; its tokens are estimated",
                               reservation.provider, reservation.model)
            else:
                tokens_in, tokens_out, cached = usage
                for totals in (self._run, self._session):
                    totals.reported_input += tokens_in
                    totals.reported_output += tokens_out
                    totals.reported_cached_input += cached
            estimated = price.cost(tokens_in, tokens_out, cached) if price else None
            self._charge(estimated, reported_cost)

    def _error(self, run_id: Any) -> None:
        with self._lock:
            reservation = self._inflight.pop(run_id, None)
            self._clear_active(run_id)
            if reservation is None:
                return
            for totals in (self._run, self._session):
                totals.failed += 1
            self._charge_input_estimate(reservation)

    def _retry(self, run_id: Any) -> None:
        with self._lock:
            reservation = self._inflight.get(run_id)
            for totals in (self._run, self._session):
                totals.sdk_retries += 1
            if reservation is not None:
                self._charge_input_estimate(reservation)

    def _charge_input_estimate(self, reservation: _Reservation) -> None:
        """A failed request or a retry: the provider may have read the prompt."""
        for totals in (self._run, self._session):
            totals.estimated_unreported += reservation.input_estimate
        estimated = reservation.price.cost(reservation.input_estimate, 0) if reservation.price else None
        self._charge(estimated, None)

    def _charge(self, estimated: float | None, reported: float | None) -> None:
        charge = max(estimated or 0.0, reported or 0.0)
        for totals in (self._run, self._session):
            if estimated is not None:
                totals.estimated_cost += estimated
            else:
                totals.unpriced_calls += 1
            if reported is not None:
                totals.provider_reported_cost += reported
                totals.provider_reported_cost_calls += 1
            totals.charged_cost += charge
        if self.max_day_cost is not None and charge:
            self._ledger_add(charge)

    @staticmethod
    def _clear_active(run_id: Any) -> None:
        current = getattr(_active, "call", None)
        if current is not None and current[1] == run_id:
            _active.call = None

    # --- the daily ledger ----------------------------------------------------

    def _ledger_path(self) -> Path:
        return (self.ledger_dir or Path(".")) / f"{datetime.now(UTC).date().isoformat()}.json"

    def _ledger_spent(self) -> float:
        path = self._ledger_path()
        if not path.exists():
            return 0.0
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("charged_usd", 0.0)
            spent = float(value)
            if not math.isfinite(spent) or spent < 0:
                raise ValueError
            return spent
        except (OSError, ValueError, AttributeError, TypeError):
            # An unreadable ledger cannot prove today's spend is under the limit.
            self._refuse("day_cost", f"the daily spend ledger {path.name} cannot be read; "
                                     "the daily cost limit fails closed")
            raise  # _refuse always raises; this line is never reached

    def _ledger_add(self, amount: float) -> None:
        path = self._ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with locked(path):
            try:
                data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            except (OSError, ValueError):
                data = {}
            if not isinstance(data, dict):
                data = {}
            data["charged_usd"] = round(float(data.get("charged_usd", 0.0)) + amount, 8)
            data["calls"] = int(data.get("calls", 0)) + 1
            data["basis"] = "estimated upper bound; not a provider bill"

            def write(temp: str) -> None:
                Path(temp).write_text(json.dumps(data, indent=2), encoding="utf-8")

            replace_file(path, write)


def describe_usage(summary: dict) -> str:
    """One line from a ``RunBudget.summary()``: estimated and reported cost kept apart."""
    calls, tokens, cost = summary.get("calls", {}), summary.get("tokens", {}), summary.get("cost_usd", {})
    if not calls:
        return "no LLM usage recorded"
    estimated = (f"${cost['estimated']:.4f} estimated" if cost.get("estimated") is not None
                 else "cost not estimated (no valid pricing file)")
    reported = (f", ${cost['provider_reported']:.4f} reported by the provider over "
                f"{cost['provider_reported_calls']} call(s)" if cost.get("provider_reported") is not None else "")
    return (f"{calls['attempted']} LLM call(s) ({calls['failed']} failed, {calls['sdk_retries']} SDK retries), "
            f"{tokens['counted_total']:,} tokens counted ({tokens['estimated_unreported']:,} estimated), "
            f"{estimated}{reported}")


def _require_prices(pricing: Pricing, models) -> None:
    missing = [f"{p}:{m}" for p, m in models if pricing.price(p, m) is None]
    if missing:
        raise PricingError(
            f"pricing file {pricing.path} has no price for {', '.join(missing)}; "
            "cost limits fail closed until every model in use is priced"
        )


class BudgetCallback(BaseCallbackHandler):
    """Attached to one model client; every call it makes goes through the budget."""

    # Raise out of invoke() instead of LangChain logging and continuing.
    raise_error = True
    run_inline = True

    def __init__(self, budget: RunBudget, provider: str, model: str):
        super().__init__()
        self.budget = budget
        self.provider = (provider or "").lower()
        self.model = model or ""

    def _start(self, run_id, chars: int, kwargs: dict) -> None:
        tools = (kwargs.get("invocation_params") or {}).get("tools")
        if tools:
            chars += len(json.dumps(tools, default=str))
        self.budget._start(run_id, self.provider, self.model, chars)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._start(run_id, sum(_message_chars(m) for batch in messages for m in batch), kwargs)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        self._start(run_id, sum(len(p) for p in prompts), kwargs)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        try:
            self.budget._end(run_id, response)
        except Exception as exc:  # noqa: BLE001 — a counting fault must not fail a finished call
            logger.warning("Could not record LLM usage (%s)", type(exc).__name__)
            if self.budget.cost_limited:
                self.budget._usage_unreported = True

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        try:
            self.budget._error(run_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not record a failed LLM call (%s)", type(exc).__name__)


__all__ = [
    "BUDGET_CONFIG_KEYS", "BudgetCallback", "LLMBudgetExceeded", "ModelPrice", "Pricing",
    "PricingError", "RunBudget", "describe_usage", "load_pricing",
]
