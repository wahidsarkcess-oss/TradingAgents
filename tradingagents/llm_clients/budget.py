"""Limits on LLM calls, tokens and estimated spend, shared by every model a run uses.

One ``RunBudget`` belongs to a ``TradingAgentsGraph``. Each model tier gets a
``BudgetCallback`` bound to that budget, so the quick and deep models and the
analysts running in parallel all draw from one thread-safe count. Every check
happens *before* a request is sent (``on_chat_model_start``, and before each SDK
retry), and a refused call raises ``LLMBudgetExceeded`` out of the model's
``invoke``.

**Runs and allowances.** Each analysis is a ``RunAccount`` (``begin_run``). A call
belongs to the account bound to the context it runs in (LangGraph carries the
context into its worker threads), so a call from an analysis that has already
ended is refused rather than charged to the next one, and a call still in flight
when its analysis ends is charged to that analysis when it completes. An account
has two allowances:

- ``analysis``: the agents' calls, capped by ``max_llm_calls_per_run``,
  ``max_tokens_per_run`` and ``max_cost_per_run_usd``;
- ``settlement``: reflections on past decisions (``settlement_scope()``), capped
  by ``max_reflection_calls_per_run`` / ``max_reflection_tokens_per_run``, which
  default to the analysis values, and ``max_cost_per_run_usd``, each as a separate
  pool. Both count towards session and daily totals.

A checkpointed analysis persists its account to a file beside the checkpoint, so
a resume continues from the usage already spent instead of a fresh allowance.

**What is counted.**

- Calls: every attempt the model client starts, failed ones included, plus the
  retries the OpenAI and Anthropic SDKs make internally. A retry is seen through
  the SDK's "Retrying request" log record, before it is sent, and is refused
  like a call when it would pass a limit.
- Tokens: the input and output tokens the provider reports. A response without
  usage, a failed request and a retry are counted at an estimate (prompt UTF-8
  bytes / 2, plus the reply) and reported apart as ``estimated_unreported``.
- Cost: ``estimated`` is the counted tokens priced from your pricing file;
  ``provider_reported`` is a cost the provider returned itself (OpenRouter does).
  Each call is charged the larger of the two; failures and retries are charged
  their input estimate. An upper-bound estimate, not a bill.
- External paid requests (TypeSafe screening): counted, and not made at all
  while a cost limit is set, since the pricing file cannot price them.

**Cost limits fail closed.** With ``max_cost_per_run_usd``/``max_cost_per_day_usd``
set, the graph refuses to start unless: every model is priced by a valid pricing
file no older than ``max_pricing_age_days``; ``max_tokens`` bounds each reply;
every tier uses a provider whose client honours ``max_tokens`` and reports token
usage (``COST_LIMIT_PROVIDERS``) at its official endpoint. A response without
usage, an unreadable ledger or a failed ledger write stops every later call.

**Daily ledger.** ``<data_cache_dir>/spend/<UTC date>.json``. Whenever prices are
known, each call reserves its worst-case cost in the ledger under a lock shared
by every process using the same cache directory, and settles it on completion;
the daily limit counts settled spend plus every open reservation, so concurrent
processes cannot jointly pass it. A process that dies mid-call leaves its
reservation counted for the rest of the day. The ledger is per cache directory:
hosts or containers without shared storage each keep their own, and file locks
are not reliable on network filesystems.

**Estimates.** Before each call the budget reserves its worst case: estimated
prompt tokens plus the output limit. Limits hold exactly while real usage stays
within the reservation, which ``max_tokens`` guarantees for the output; the prompt
estimate (2 bytes per token) overstates English about fourfold and CJK about
1.5-fold, but unusual text could still tokenize tighter, in which case the calls
in flight can overshoot by the difference and every later call is refused.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import threading
import uuid
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import asdict, dataclass, fields
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_core.callbacks import BaseCallbackHandler

from tradingagents.dataflows.files import locked

logger = logging.getLogger(__name__)

# The config keys this module reads. None (the default) disables each limit.
LIMIT_KEYS = (
    "max_llm_calls_per_run",
    "max_tokens_per_run",
    "max_cost_per_run_usd",
    "max_cost_per_day_usd",
    "max_reflection_calls_per_run",
    "max_reflection_tokens_per_run",
)
BUDGET_CONFIG_KEYS = (*LIMIT_KEYS, "pricing_file", "max_pricing_age_days")

ANALYSIS, SETTLEMENT = "analysis", "settlement"

# Output reserved for a call when no per-call ``max_tokens`` limit is set.
DEFAULT_OUTPUT_RESERVATION = 8192
# UTF-8 bytes per token for estimates: English averages about 4 characters per
# token and a CJK character is 3 bytes for 1-2 tokens, so 2 errs towards refusing.
BYTES_PER_TOKEN = 2

# Loggers whose "Retrying request" records mark an SDK-internal retry.
RETRY_LOGGERS = ("openai._base_client", "anthropic._base_client")

# Providers whose LangChain client honours ``max_tokens`` and returns token usage
# on every (non-streamed) response, at the hosts they officially serve from. Cost
# limits are refused for anything else: a ChatGPT plan drops max_tokens and
# streams, and a local or custom endpoint's usage reporting is unknown.
OFFICIAL_HOSTS: dict[str, tuple[str, ...]] = {
    "openai": ("api.openai.com",),
    "anthropic": ("api.anthropic.com",),
    "google": ("generativelanguage.googleapis.com",),
    "azure": (".openai.azure.com", ".cognitiveservices.azure.com"),
    "bedrock": (),   # no backend_url: the AWS SDK picks the regional endpoint
    "xai": ("api.x.ai",),
    "deepseek": ("api.deepseek.com",),
    "qwen": ("dashscope-intl.aliyuncs.com",),
    "qwen-cn": ("dashscope.aliyuncs.com",),
    "glm": ("api.z.ai",),
    "glm-cn": ("open.bigmodel.cn",),
    "minimax": ("api.minimax.io",),
    "minimax-cn": ("api.minimaxi.com",),
    "openrouter": ("openrouter.ai",),
    "mistral": ("api.mistral.ai",),
    "kimi": ("api.moonshot.ai",),
    "groq": ("api.groq.com",),
    "nvidia": ("integrate.api.nvidia.com",),
}
COST_LIMIT_PROVIDERS = frozenset(OFFICIAL_HOSTS)


class LLMBudgetExceeded(RuntimeError):
    """A limit refused an LLM call (or an SDK retry) before it was sent.

    ``scope`` names the limit: ``calls``, ``tokens``, ``run_cost``,
    ``reflection_calls``, ``reflection_tokens``, ``reflection_cost``,
    ``day_cost``, ``session_cost``, ``pricing``, ``usage_unreported``, ``ledger``,
    ``usage_record``, ``run_ended`` or ``resume``. ``usage`` is the run's usage
    summary when it was refused. The message never contains credentials.
    """

    def __init__(self, scope: str, message: str, usage: dict | None = None):
        super().__init__(message)
        self.scope = scope
        self.usage = usage or {}


class PricingError(ValueError):
    """Cost limits cannot be trusted: pricing, output limit or provider is unsuitable."""


# --- pricing -----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens.

    Cached input falls back to the full input rate. Cache writes (Anthropic) fall
    back to twice the input rate, the highest write multiplier providers charge.
    """

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None

    def cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0,
             cache_write_tokens: int = 0) -> float:
        total_in = max(input_tokens, 0)
        cached = min(max(cached_tokens, 0), total_in)
        written = min(max(cache_write_tokens, 0), total_in - cached)
        cached_rate = self.input_per_mtok if self.cached_input_per_mtok is None else self.cached_input_per_mtok
        write_rate = 2 * self.input_per_mtok if self.cache_write_per_mtok is None else self.cache_write_per_mtok
        return ((total_in - cached - written) * self.input_per_mtok
                + cached * cached_rate
                + written * write_rate
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
                                          "cached_input_per_mtok": 0.0,
                                          "cache_write_per_mtok": 0.0}}}

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
            cache_write_per_mtok=_price_field(entry, "cache_write_per_mtok", key, False),
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


def endpoint_supported(provider: str, base_url: str | None) -> bool:
    """Whether ``provider`` at ``base_url`` reports usage reliably enough for cost limits."""
    provider = (provider or "").lower()
    if provider not in OFFICIAL_HOSTS:
        return False
    if not base_url:
        return True
    host = (urlparse(base_url).hostname or "").lower()
    allowed = OFFICIAL_HOSTS[provider]
    return bool(host) and any(host == a or (a.startswith(".") and host.endswith(a)) for a in allowed)


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
    reported_cache_write: int = 0
    estimated_unreported: int = 0
    calls_without_usage: int = 0
    estimated_cost: float = 0.0
    unpriced_calls: int = 0
    provider_reported_cost: float = 0.0
    provider_reported_cost_calls: int = 0
    charged_cost: float = 0.0
    external_requests: int = 0

    @property
    def tokens(self) -> int:
        return self.reported_input + self.reported_output + self.estimated_unreported

    def add(self, other: _Totals) -> _Totals:
        return _Totals(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    @classmethod
    def from_dict(cls, data: Any) -> _Totals:
        if not isinstance(data, dict):
            raise ValueError("not an object")
        values = {}
        for f in fields(cls):
            value = data.get(f.name, 0)
            kind = float if f.type in ("float", float) else int
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"bad {f.name}")
            values[f.name] = kind(value)
        return cls(**values)


class RunAccount:
    """The usage of one analysis, in its two allowances."""

    def __init__(self, budget: RunBudget, persist_path: Path | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.budget = budget
        self.scopes = {ANALYSIS: _Totals(), SETTLEMENT: _Totals()}
        self.ended = False
        self.persist_path = persist_path
        self.broken: str | None = None   # why the usage record can no longer be trusted

    @property
    def total(self) -> _Totals:
        return self.scopes[ANALYSIS].add(self.scopes[SETTLEMENT])

    def save(self) -> None:
        """Write the usage beside the checkpoint; a failure is recorded, then refused on."""
        if self.persist_path is None:
            return
        try:
            _write_json_atomic(self.persist_path, {
                "version": 1, "account": self.id,
                "scopes": {name: asdict(t) for name, t in self.scopes.items()},
            })
        except OSError as exc:
            self.broken = f"the usage record for this checkpoint could not be saved ({type(exc).__name__})"
            logger.error("LLM budget: %s", self.broken)

    def load(self) -> None:
        data = json.loads(Path(self.persist_path).read_text(encoding="utf-8"))
        scopes = data.get("scopes") if isinstance(data, dict) else None
        if not isinstance(scopes, dict):
            raise ValueError("no scopes")
        self.scopes = {name: _Totals.from_dict(scopes.get(name, {})) for name in (ANALYSIS, SETTLEMENT)}


@dataclass
class _Reservation:
    account: RunAccount
    scope: str
    provider: str
    model: str
    price: ModelPrice | None
    input_estimate: int
    tokens: int
    cost: float
    ledger: tuple[Path, str] | None


# The run that owns a call, and the allowance it draws on, carried by context.
_owner: ContextVar[RunAccount | None] = ContextVar("tradingagents_budget_owner", default=None)
_scope: ContextVar[str] = ContextVar("tradingagents_budget_scope", default=ANALYSIS)
_active = threading.local()   # (budget, run_id) of the call this thread is making


@contextlib.contextmanager
def settlement_scope() -> Iterator[None]:
    """Draw the calls made inside the block on the settlement allowance."""
    token = _scope.set(SETTLEMENT)
    try:
        yield
    finally:
        _scope.reset(token)


def current_account() -> RunAccount | None:
    """The run bound to the calling context, if any."""
    return _owner.get()


def bind_owner(account: RunAccount | None):
    """Bind ``account`` to the calling context; returns the token for ``reset_owner``."""
    return _owner.set(account)


def reset_owner(token) -> None:
    _owner.reset(token)


def meter_external_requests(service: str, requests: int) -> None:
    """Count paid requests made outside the LLM clients (TypeSafe screening).

    Raises ``LLMBudgetExceeded`` when a cost limit is set: the pricing file cannot
    price them, so they are not made. Outside a run this does nothing.
    """
    account = _owner.get()
    if account is not None:
        account.budget._external(account, service, requests)


class _RetryFilter(logging.Filter):
    """Sees each SDK retry before it is sent: counts it, or refuses it by raising.

    The SDK logger's level is lowered to INFO so retry records are created at all;
    records below the level the logger had before are dropped here, so no host
    application sees log output it did not ask for.
    """

    def __init__(self, original_level: int):
        super().__init__()
        self.original_level = original_level

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno == logging.INFO and str(record.msg).startswith("Retrying request"):
            current = getattr(_active, "call", None)
            if current is not None:
                budget, run_id = current
                budget._retry(run_id)   # LLMBudgetExceeded propagates and aborts the retry
        return record.levelno >= self.original_level


_retry_filter_lock = threading.Lock()
_retry_filters: dict[str, _RetryFilter] = {}


def _install_retry_filter() -> None:
    with _retry_filter_lock:
        for name in RETRY_LOGGERS:
            if name in _retry_filters:
                continue
            sdk_logger = logging.getLogger(name)
            retry_filter = _RetryFilter(sdk_logger.getEffectiveLevel())
            if sdk_logger.getEffectiveLevel() > logging.INFO:
                sdk_logger.setLevel(logging.INFO)
            sdk_logger.addFilter(retry_filter)
            _retry_filters[name] = retry_filter


def _message_bytes(message: Any) -> int:
    size = len(str(getattr(message, "content", message) or "").encode("utf-8"))
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        size += len(json.dumps(tool_calls, default=str).encode("utf-8"))
    return size


def estimate_tokens(size_in_bytes: int) -> int:
    return math.ceil(size_in_bytes / BYTES_PER_TOKEN)


def _usage_of(response: Any) -> tuple[int, int, int, int] | None:
    """(input, output, cached input, cache-write) tokens the provider reported, or None."""
    try:
        message = response.generations[0][0].message
        usage = getattr(message, "usage_metadata", None)
    except (AttributeError, IndexError, TypeError):
        usage = None
    if usage and (usage.get("input_tokens") is not None or usage.get("output_tokens") is not None):
        details = usage.get("input_token_details") or {}
        return (int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0),
                int(details.get("cache_read") or 0), int(details.get("cache_creation") or 0))
    token_usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
    if token_usage.get("prompt_tokens") is not None or token_usage.get("completion_tokens") is not None:
        return int(token_usage.get("prompt_tokens") or 0), int(token_usage.get("completion_tokens") or 0), 0, 0
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


def _output_bytes(response: Any) -> int:
    try:
        return sum(_message_bytes(g.message) if hasattr(g, "message") else len((g.text or "").encode())
                   for batch in response.generations for g in batch)
    except (AttributeError, TypeError):
        return 0


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write ``data`` to ``path`` through a temp file and a rename; raise on any failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


class _LedgerError(Exception):
    pass


class RunBudget:
    """Thread-safe limits shared by every LLM call of one graph."""

    def __init__(
        self,
        *,
        max_calls: int | None = None,
        max_tokens: int | None = None,
        max_run_cost: float | None = None,
        max_day_cost: float | None = None,
        max_reflection_calls: int | None = None,
        max_reflection_tokens: int | None = None,
        output_reservation: int = DEFAULT_OUTPUT_RESERVATION,
        output_limited: bool = False,
        pricing: Pricing | None = None,
        ledger_dir: str | Path | None = None,
        models: tuple = (),
    ):
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.max_run_cost = max_run_cost
        self.max_day_cost = max_day_cost
        self.max_reflection_calls = max_reflection_calls
        self.max_reflection_tokens = max_reflection_tokens
        self.max_session_cost: float | None = None
        self.output_reservation = output_reservation
        self.output_limited = output_limited
        self.pricing = pricing
        self.ledger_dir = Path(ledger_dir).expanduser() if ledger_dir else None
        self.models = tuple((m[0].lower(), m[1], m[2] if len(m) > 2 else None) for m in models)
        self._lock = threading.RLock()
        self._current = RunAccount(self)
        self._session = _Totals()
        self._inflight: dict[Any, _Reservation] = {}
        # Sticky: spend can no longer be bounded, so every later call is refused.
        self._usage_unreported = False
        self._ledger_broken: str | None = None
        _install_retry_filter()

    @classmethod
    def from_config(cls, config: dict, models: list | tuple = ()) -> RunBudget:
        """Build from a TradingAgents config; fail closed on doubtful cost limits.

        ``models`` are the ``(provider, model[, base_url])`` the run will call.
        With a cost limit set, this raises ``PricingError`` before any model is
        called unless the pricing, the output limit and every provider and
        endpoint are suitable (see the module docstring).
        """
        max_calls = _limit(config, "max_llm_calls_per_run", int)
        max_tokens = _limit(config, "max_tokens_per_run", int)
        max_run_cost = _limit(config, "max_cost_per_run_usd", float)
        max_day_cost = _limit(config, "max_cost_per_day_usd", float)
        max_reflection_calls = _limit(config, "max_reflection_calls_per_run", int)
        max_reflection_tokens = _limit(config, "max_reflection_tokens_per_run", int)
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
            _require_cost_limit_support(pricing, models, per_call_output)
        elif pricing_file:
            # Cost is only reported, not enforced: a doubtful file means no estimate.
            try:
                pricing = load_pricing(pricing_file, max_age)
            except PricingError as exc:
                logger.warning("Cost estimates are off: %s", exc)

        return cls(
            max_calls=max_calls, max_tokens=max_tokens, max_run_cost=max_run_cost,
            max_day_cost=max_day_cost, max_reflection_calls=max_reflection_calls,
            max_reflection_tokens=max_reflection_tokens,
            output_reservation=per_call_output or DEFAULT_OUTPUT_RESERVATION,
            output_limited=per_call_output is not None,
            pricing=pricing, ledger_dir=Path(config.get("data_cache_dir") or ".") / "spend",
            models=tuple(models),
        )

    # --- configuration -------------------------------------------------------

    @property
    def cost_limited(self) -> bool:
        return any(v is not None for v in (self.max_run_cost, self.max_day_cost, self.max_session_cost))

    @property
    def run_limited(self) -> bool:
        """Whether any per-analysis allowance is capped."""
        return any(v is not None for v in (self.max_calls, self.max_tokens, self.max_run_cost,
                                           self.max_reflection_calls, self.max_reflection_tokens))

    @property
    def limited(self) -> bool:
        return self.run_limited or self.cost_limited

    def allowance(self, scope: str) -> tuple[int | None, int | None, float | None]:
        """(calls, tokens, cost) for an allowance; settlement inherits what it does not set."""
        if scope == SETTLEMENT:
            calls = self.max_reflection_calls if self.max_reflection_calls is not None else self.max_calls
            tokens = self.max_reflection_tokens if self.max_reflection_tokens is not None else self.max_tokens
            return calls, tokens, self.max_run_cost
        return self.max_calls, self.max_tokens, self.max_run_cost

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
        _require_cost_limit_support(self.pricing, self.models,
                                    self.output_reservation if self.output_limited else None)
        self.max_session_cost = limit

    def callback(self, provider: str, model: str) -> BudgetCallback:
        """The callback handler to attach to the client of ``provider:model``."""
        return BudgetCallback(self, provider, model)

    # --- run lifecycle -------------------------------------------------------

    @property
    def current(self) -> RunAccount:
        return self._current

    def begin_run(self, persist_path: str | Path | None = None, restore: bool = False) -> RunAccount:
        """Start a new analysis; the previous one ends, its late calls refused.

        ``persist_path`` keeps the account's usage in a file (beside a checkpoint).
        With ``restore``, the usage already in that file is continued: a missing
        or unreadable record is refused while any per-analysis limit is set, since
        a resumed analysis must not get a fresh allowance.
        """
        with self._lock:
            self._current.ended = True
            account = RunAccount(self, Path(persist_path) if persist_path else None)
            if restore and account.persist_path is not None:
                try:
                    account.load()
                except (OSError, ValueError) as exc:
                    problem = ("missing" if isinstance(exc, FileNotFoundError)
                               else f"unreadable ({type(exc).__name__})")
                    if self.run_limited or self.cost_limited:
                        raise LLMBudgetExceeded(
                            "resume", f"the saved LLM usage for this checkpoint is {problem}, so resuming it "
                            "could exceed its limits; start fresh with --clear-checkpoints", {}) from None
                    logger.warning("Saved LLM usage for this checkpoint is %s; counting from zero", problem)
            account.save()
            self._current = account
            return account

    def end_run(self, account: RunAccount | None = None) -> None:
        """Mark ``account`` (default: the current one) ended; its later calls are refused."""
        with self._lock:
            (account or self._current).ended = True

    def bound(self, account: RunAccount | None = None):
        """Context manager binding ``account`` (default: current) to the calling context."""
        budget = self

        @contextlib.contextmanager
        def scope():
            token = _owner.set(account or budget._current)
            try:
                yield
            finally:
                _owner.reset(token)

        return scope()

    # --- reporting -----------------------------------------------------------

    def summary(self, session: bool = False, account: RunAccount | None = None) -> dict:
        """Usage of a run (default: the current one) or of the whole session."""
        with self._lock:
            if session:
                totals, by_scope = self._session, None
            else:
                account = account or self._current
                totals, by_scope = account.total, account.scopes
            priced = self.pricing is not None
            summary = {
                "limits": {
                    "max_llm_calls_per_run": self.max_calls,
                    "max_tokens_per_run": self.max_tokens,
                    "max_cost_per_run_usd": self.max_run_cost,
                    "max_cost_per_day_usd": self.max_day_cost,
                    "max_reflection_calls_per_run": self.allowance(SETTLEMENT)[0],
                    "max_reflection_tokens_per_run": self.allowance(SETTLEMENT)[1],
                    "max_total_cost_usd": self.max_session_cost,
                    "per_call_output_reservation": self.output_reservation,
                },
                **_totals_summary(totals, priced),
                "external_paid_requests": totals.external_requests,
                "cost_usd": {
                    **_totals_summary(totals, priced)["cost_usd"],
                    "estimated_basis": (
                        f"counted tokens x prices in {Path(self.pricing.path).name} as of {self.pricing.as_of}"
                        if priced else "no valid pricing file, so no estimate"
                    ),
                    "note": "Estimates are an upper bound, not a bill; check the provider's billing page.",
                },
            }
            if by_scope is not None:
                summary["by_allowance"] = {name: _totals_summary(t, priced) for name, t in by_scope.items()}
            return summary

    def describe(self, session: bool = False) -> str:
        """One line for a person: calls, tokens, and both kinds of cost."""
        return describe_usage(self.summary(session))

    # --- the callback's side -------------------------------------------------

    def _refuse(self, scope: str, message: str, account: RunAccount | None = None) -> None:
        raise LLMBudgetExceeded(scope, message, self.summary(account=account))

    def _account_for_call(self) -> RunAccount:
        owner = _owner.get()
        if owner is not None and owner.budget is self:
            if owner.ended:
                self._refuse("run_ended", "a call from an analysis that has already ended was refused",
                             owner)
            return owner
        return self._current

    def _start(self, run_id: Any, provider: str, model: str, prompt_bytes: int) -> None:
        with self._lock:
            account = self._account_for_call()
            scope = _scope.get()
            totals, session = account.scopes[scope], self._session
            max_calls, max_tokens, max_cost = self.allowance(scope)
            prefix = "reflection_" if scope == SETTLEMENT else ""
            label = "reflection" if scope == SETTLEMENT else "analysis"
            input_estimate = estimate_tokens(prompt_bytes)
            projected_tokens = input_estimate + self.output_reservation
            price = self.pricing.price(provider, model) if self.pricing else None
            projected_cost = price.cost(input_estimate, self.output_reservation) if price else 0.0
            same = [r for r in self._inflight.values() if r.account is account and r.scope == scope]

            if account.broken and self.limited:
                self._refuse("usage_record", account.broken, account)
            if max_calls is not None and totals.attempted + totals.sdk_retries + 1 > max_calls:
                self._refuse(f"{prefix}calls", f"LLM call limit reached: {max_calls} per {label} "
                             f"({'TRADINGAGENTS_MAX_REFLECTION_CALLS' if prefix else 'TRADINGAGENTS_MAX_LLM_CALLS'})",
                             account)
            if max_tokens is not None and totals.tokens + sum(r.tokens for r in same) + projected_tokens > max_tokens:
                self._refuse(f"{prefix}tokens", f"token limit would be exceeded: {max_tokens:,} per {label}; "
                             f"{totals.tokens:,} counted so far, this call reserves up to {projected_tokens:,}",
                             account)
            if self.cost_limited:
                if self._usage_unreported:
                    self._refuse("usage_unreported", "a provider response carried no token usage, so spend "
                                 "can no longer be bounded; cost limits fail closed", account)
                if self._ledger_broken:
                    self._refuse("ledger", self._ledger_broken, account)
                if price is None:
                    self._refuse("pricing", f"no price for {provider}:{model} in the pricing file; "
                                 "cost limits fail closed", account)
                if max_cost is not None and totals.charged_cost + sum(r.cost for r in same) + projected_cost > max_cost:
                    self._refuse("reflection_cost" if prefix else "run_cost",
                                 f"estimated cost limit would be exceeded: ${max_cost:.4f} per {label} "
                                 f"(TRADINGAGENTS_MAX_COST_PER_RUN); ${totals.charged_cost:.4f} charged so far, "
                                 f"this call reserves up to ${projected_cost:.4f}", account)
                inflight_cost = sum(r.cost for r in self._inflight.values())
                if (self.max_session_cost is not None
                        and session.charged_cost + inflight_cost + projected_cost > self.max_session_cost):
                    self._refuse("session_cost", f"total estimated cost limit would be exceeded: "
                                 f"${self.max_session_cost:.4f}; ${session.charged_cost:.4f} charged so far",
                                 account)
            ledger = self._ledger_reserve(run_id, projected_cost, account) if price is not None else None

            totals.attempted += 1
            session.attempted += 1
            self._inflight[run_id] = _Reservation(account, scope, provider, model, price, input_estimate,
                                                  projected_tokens, projected_cost, ledger)
            account.save()
            _active.call = (self, run_id)

    def _end(self, run_id: Any, response: Any) -> None:
        with self._lock:
            reservation = self._inflight.pop(run_id, None)
            self._clear_active(run_id)
            if reservation is None:
                return
            totals = reservation.account.scopes[reservation.scope]
            usage = _usage_of(response)
            reported_cost = _reported_cost_of(response)
            for t in (totals, self._session):
                t.succeeded += 1
            if usage is None:
                tokens_in = reservation.input_estimate
                tokens_out = estimate_tokens(_output_bytes(response))
                cached = written = 0
                for t in (totals, self._session):
                    t.calls_without_usage += 1
                    t.estimated_unreported += tokens_in + tokens_out
                if self.cost_limited:
                    self._usage_unreported = True
                logger.warning("%s:%s returned no token usage; its tokens are estimated",
                               reservation.provider, reservation.model)
            else:
                tokens_in, tokens_out, cached, written = usage
                for t in (totals, self._session):
                    t.reported_input += tokens_in
                    t.reported_output += tokens_out
                    t.reported_cached_input += cached
                    t.reported_cache_write += written
            price = reservation.price
            estimated = price.cost(tokens_in, tokens_out, cached, written) if price else None
            charge = self._charge(totals, estimated, reported_cost)
            self._ledger_settle(reservation.ledger, charge)
            reservation.account.save()

    def _error(self, run_id: Any) -> None:
        with self._lock:
            reservation = self._inflight.pop(run_id, None)
            self._clear_active(run_id)
            if reservation is None:
                return
            totals = reservation.account.scopes[reservation.scope]
            for t in (totals, self._session):
                t.failed += 1
            charge = self._charge_input_estimate(totals, reservation)
            self._ledger_settle(reservation.ledger, charge)
            reservation.account.save()

    def _retry(self, run_id: Any) -> None:
        """An SDK is about to retry this call: refuse it, or count and charge it."""
        with self._lock:
            reservation = self._inflight.get(run_id)
            if reservation is None:
                self._session.sdk_retries += 1
                return
            account, scope = reservation.account, reservation.scope
            totals = account.scopes[scope]
            max_calls, max_tokens, max_cost = self.allowance(scope)
            prefix = "reflection_" if scope == SETTLEMENT else ""
            retry_cost = reservation.price.cost(reservation.input_estimate, 0) if reservation.price else 0.0
            if max_calls is not None and totals.attempted + totals.sdk_retries + 1 > max_calls:
                self._refuse(f"{prefix}calls", f"SDK retry refused: the LLM call limit ({max_calls}) is reached",
                             account)
            if max_tokens is not None and totals.tokens + reservation.input_estimate > max_tokens:
                self._refuse(f"{prefix}tokens", f"SDK retry refused: the token limit ({max_tokens:,}) "
                             "would be exceeded", account)
            if self.cost_limited and max_cost is not None and totals.charged_cost + retry_cost > max_cost:
                self._refuse("reflection_cost" if prefix else "run_cost",
                             "SDK retry refused: the estimated cost limit would be exceeded", account)
            for t in (totals, self._session):
                t.sdk_retries += 1
            charge = self._charge_input_estimate(totals, reservation)
            if charge:
                self._ledger_add(charge)
            account.save()

    def _external(self, account: RunAccount, service: str, requests: int) -> None:
        with self._lock:
            if self.cost_limited:
                self._refuse("pricing", f"{service} requests are billed outside the pricing file, so they "
                             "are not made while a cost limit is set", account)
            for t in (account.scopes[ANALYSIS], self._session):
                t.external_requests += requests
            account.save()

    def _charge_input_estimate(self, totals: _Totals, reservation: _Reservation) -> float:
        """A failed request or a retry: the provider may have read the prompt."""
        for t in (totals, self._session):
            t.estimated_unreported += reservation.input_estimate
        estimated = reservation.price.cost(reservation.input_estimate, 0) if reservation.price else None
        return self._charge(totals, estimated, None)

    def _charge(self, totals: _Totals, estimated: float | None, reported: float | None) -> float:
        charge = max(estimated or 0.0, reported or 0.0)
        for t in (totals, self._session):
            if estimated is not None:
                t.estimated_cost += estimated
            else:
                t.unpriced_calls += 1
            if reported is not None:
                t.provider_reported_cost += reported
                t.provider_reported_cost_calls += 1
            t.charged_cost += charge
        return charge

    @staticmethod
    def _clear_active(run_id: Any) -> None:
        current = getattr(_active, "call", None)
        if current is not None and current[1] == run_id:
            _active.call = None

    # --- the daily ledger ----------------------------------------------------

    def _ledger_path(self) -> Path:
        return (self.ledger_dir or Path(".")) / f"{datetime.now(UTC).date().isoformat()}.json"

    @staticmethod
    def _ledger_read(path: Path) -> dict:
        """The ledger, or a fresh one; raise ``_LedgerError`` when it cannot be trusted."""
        if not path.exists():
            return {"charged_usd": 0.0, "calls": 0, "reserved": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError
            charged = float(data.get("charged_usd", 0.0))
            calls = int(data.get("calls", 0))
            reserved = data.get("reserved", {})
            if not isinstance(reserved, dict):
                raise ValueError
            reserved = {str(k): float(v) for k, v in reserved.items()}
            if not all(math.isfinite(v) and v >= 0 for v in (charged, *reserved.values())) or calls < 0:
                raise ValueError
        except (OSError, ValueError, TypeError) as exc:
            raise _LedgerError(f"the daily spend ledger {path.name} cannot be read "
                               f"({type(exc).__name__})") from None
        return {"charged_usd": charged, "calls": calls, "reserved": reserved}

    @staticmethod
    def _ledger_write(path: Path, data: dict) -> None:
        try:
            _write_json_atomic(path, {**data, "basis": "estimated upper bound; not a provider bill"})
        except OSError as exc:
            raise _LedgerError(f"the daily spend ledger {path.name} could not be written "
                               f"({type(exc).__name__})") from None

    def _ledger_failed(self, problem: str, account: RunAccount | None) -> None:
        """A ledger problem: refused under a cost limit, else logged as an error."""
        if self.cost_limited:
            self._ledger_broken = problem
            self._refuse("ledger", f"{problem}; cost limits fail closed", account)
        logger.error("LLM budget: %s; this spend is not recorded", problem)

    def _ledger_reserve(self, run_id: Any, projected_cost: float, account: RunAccount) -> tuple[Path, str] | None:
        """Reserve the call's worst-case cost, checking the daily limit atomically across processes."""
        path, key = self._ledger_path(), f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with locked(path):
                data = self._ledger_read(path)
                spent = data["charged_usd"] + sum(data["reserved"].values())
                if self.max_day_cost is not None and spent + projected_cost > self.max_day_cost:
                    self._refuse("day_cost", f"daily estimated cost limit would be exceeded: "
                                 f"${self.max_day_cost:.4f} per UTC day (TRADINGAGENTS_MAX_COST_PER_DAY); "
                                 f"${spent:.4f} charged or reserved today", account)
                data["reserved"][key] = projected_cost
                self._ledger_write(path, data)
        except _LedgerError as exc:
            self._ledger_failed(str(exc), account)
            return None
        except OSError as exc:
            self._ledger_failed(f"the daily spend ledger lock failed ({type(exc).__name__})", account)
            return None
        return path, key

    def _ledger_settle(self, ledger: tuple[Path, str] | None, charge: float) -> None:
        """Replace a call's reservation with what it was charged."""
        if ledger is None:
            return
        path, key = ledger
        try:
            with locked(path):
                data = self._ledger_read(path)
                data["reserved"].pop(key, None)
                data["charged_usd"] = round(data["charged_usd"] + charge, 8)
                data["calls"] += 1
                self._ledger_write(path, data)
        except (_LedgerError, OSError) as exc:
            # The call has happened; what matters is that the next one is refused.
            problem = str(exc) if isinstance(exc, _LedgerError) else f"ledger lock failed ({type(exc).__name__})"
            if self.cost_limited:
                self._ledger_broken = problem
            logger.error("LLM budget: %s", problem)

    def _ledger_add(self, charge: float) -> None:
        if self.pricing is None:
            return
        path = self._ledger_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with locked(path):
                data = self._ledger_read(path)
                data["charged_usd"] = round(data["charged_usd"] + charge, 8)
                self._ledger_write(path, data)
        except (_LedgerError, OSError) as exc:
            problem = str(exc) if isinstance(exc, _LedgerError) else f"ledger lock failed ({type(exc).__name__})"
            if self.cost_limited:
                self._ledger_broken = problem
            logger.error("LLM budget: %s", problem)


def _totals_summary(totals: _Totals, priced: bool) -> dict:
    return {
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
            "provider_reported_cache_write": totals.reported_cache_write,
            "estimated_unreported": totals.estimated_unreported,
            "counted_total": totals.tokens,
            "calls_without_reported_usage": totals.calls_without_usage,
        },
        "cost_usd": {
            "estimated": round(totals.estimated_cost, 6) if priced else None,
            "unpriced_calls": totals.unpriced_calls,
            "provider_reported": (round(totals.provider_reported_cost, 6)
                                  if totals.provider_reported_cost_calls else None),
            "provider_reported_calls": totals.provider_reported_cost_calls,
            "charged_against_limits": round(totals.charged_cost, 6),
        },
    }


def describe_usage(summary: dict) -> str:
    """One line from a ``RunBudget.summary()``: estimated and reported cost kept apart."""
    calls, tokens, cost = summary.get("calls", {}), summary.get("tokens", {}), summary.get("cost_usd", {})
    if not calls:
        return "no LLM usage recorded"
    estimated = (f"${cost['estimated']:.4f} estimated" if cost.get("estimated") is not None
                 else "cost not estimated (no valid pricing file)")
    reported = (f", ${cost['provider_reported']:.4f} reported by the provider over "
                f"{cost['provider_reported_calls']} call(s)" if cost.get("provider_reported") is not None else "")
    reflections = (summary.get("by_allowance") or {}).get(SETTLEMENT, {}).get("calls", {}).get("attempted")
    of_which = f", {reflections} of them reflections" if reflections else ""
    return (f"{calls['attempted']} LLM call(s){of_which} ({calls['failed']} failed, "
            f"{calls['sdk_retries']} SDK retries), "
            f"{tokens['counted_total']:,} tokens counted ({tokens['estimated_unreported']:,} estimated), "
            f"{estimated}{reported}")


def _require_cost_limit_support(pricing: Pricing, models, per_call_output: int | None) -> None:
    if per_call_output is None:
        raise PricingError(
            "a cost limit needs a per-call output limit (TRADINGAGENTS_MAX_TOKENS / config max_tokens); "
            "without one a reply's cost has no upper bound"
        )
    for entry in models:
        provider, model = entry[0].lower(), entry[1]
        base_url = entry[2] if len(entry) > 2 else None
        if provider not in COST_LIMIT_PROVIDERS:
            raise PricingError(
                f"cost limits are not supported for provider {provider!r}: its client does not reliably "
                "honour max_tokens or report token usage (supported: "
                f"{', '.join(sorted(COST_LIMIT_PROVIDERS))})"
            )
        if not endpoint_supported(provider, base_url):
            raise PricingError(
                f"cost limits are not supported for {provider} at a custom endpoint: its usage "
                "reporting is unknown; use the provider's official endpoint"
            )
        if pricing.price(provider, model) is None:
            raise PricingError(
                f"pricing file {pricing.path} has no price for {provider}:{model}; "
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

    def _start(self, run_id, size: int, kwargs: dict) -> None:
        try:
            tools = (kwargs.get("invocation_params") or {}).get("tools")
            if tools:
                size += len(json.dumps(tools, default=str).encode("utf-8"))
            self.budget._start(run_id, self.provider, self.model, size)
        except LLMBudgetExceeded:
            raise
        except Exception as exc:
            # With limits on, a fault in the budget itself refuses the call (fail
            # closed); with every limit off it must not break a run.
            if self.budget.limited:
                raise LLMBudgetExceeded("usage_record", f"the LLM budget failed ({type(exc).__name__}); "
                                        "the call is refused") from None
            logger.warning("LLM budget could not record a call (%s)", type(exc).__name__)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._start(run_id, sum(_message_bytes(m) for batch in messages for m in batch), kwargs)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        self._start(run_id, sum(len(p.encode("utf-8")) for p in prompts), kwargs)

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
    "BUDGET_CONFIG_KEYS", "BudgetCallback", "COST_LIMIT_PROVIDERS", "LLMBudgetExceeded", "ModelPrice",
    "Pricing", "PricingError", "RunAccount", "RunBudget", "bind_owner", "current_account",
    "describe_usage", "endpoint_supported", "estimate_tokens", "load_pricing",
    "meter_external_requests", "reset_owner", "settlement_scope",
]
