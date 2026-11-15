"""Limits on LLM calls, tokens and estimated spend, shared by every model a run uses.

One ``RunBudget`` belongs to a ``TradingAgentsGraph``. Each model tier gets a
``BudgetCallback`` bound to that budget, so the quick and deep models and the
analysts running in parallel all draw from one thread-safe count. Every check
happens *before* a request is sent (``on_chat_model_start``, and before each SDK
retry), and a refused call raises ``LLMBudgetExceeded`` out of the model's
``invoke``.

**With every limit off** the budget only counts. It refuses nothing, installs no
SDK hooks and changes no client setting, so the application behaves as it did
before budgets existed.

**Runs.** Each analysis, and each settlement pass outside an analysis, is its own
``RunAccount``. A call belongs to the account bound to the context it runs in
(LangGraph carries the context into its worker threads); a call outside any run
goes to a standing default account. Starting an account never ends another: a
run ends only its own account (``end_run``), so concurrent analyses and
settlement passes on one graph never invalidate each other. With limits on, a
call from an account that has already ended (an analyst still running after its
run failed) is refused; a call that was in flight when its run ended is charged
to that run when it completes.

**Allowances.** An account's calls and tokens are capped per allowance:
``analysis`` (``max_llm_calls_per_run``, ``max_tokens_per_run``) and
``settlement``, i.e. reflections on past decisions (``max_reflection_calls_per_run``
/ ``max_reflection_tokens_per_run``, defaulting to the analysis values as a
separate pool). Cost has one ceiling per run: ``max_cost_per_run_usd`` caps the
analysis and its reflections together. Session and daily limits cover everything.

**Persistence.** A checkpointed analysis keeps its usage in a record beside the
checkpoint, including every reservation in flight, written before the request
is sent. A resume continues from that usage, and a reservation still open when
the process died is counted as spent at its worst case.

**What is counted.**

- Calls: every attempt the model client starts, failed ones included, plus
  provider retries. With limits on, retries go through budget-controlled code:
  the OpenAI and Anthropic SDKs' retries are intercepted before each is sent;
  Google's and Bedrock's own retries are switched off and done by the
  TradingAgents client wrappers, which ask the budget first.
- Tokens: the input and output tokens the provider reports. A response without
  usage, a failed request and a retry are counted at an estimate (prompt UTF-8
  bytes / 2, plus the reply) and reported apart as ``estimated_unreported``.
- Cost: ``estimated`` is the counted tokens priced from your pricing file, with
  cache-write and long-context rates where they apply; ``provider_reported`` is
  a cost the provider returned itself (OpenRouter does). Each call is charged the
  larger of the two; failures and retries are charged their input estimate.
- External paid requests (TypeSafe screening): every attempt is counted, capped
  by ``max_external_requests_per_run``, and none is made under a cost limit.

**Cost limits fail closed.** With ``max_cost_per_run_usd``/``max_cost_per_day_usd``
set, the graph refuses to start unless every model is priced by a valid pricing
file no older than ``max_pricing_age_days``; ``max_tokens`` bounds each reply;
and every tier is a provider in ``COST_LIMIT_PROVIDERS`` whose *constructed
client* targets that provider's official endpoint (environment overrides such as
``OPENAI_BASE_URL`` included). A response without usage, an unreadable ledger or
a failed ledger write stops every later call.

**Daily ledger.** ``<data_cache_dir>/spend/<UTC date>.json``. Whenever prices are
known, each call reserves its worst-case cost in the ledger under a lock shared
by every process using the same cache directory, and settles it on completion;
retries are checked and charged the same way. The daily limit counts settled
spend plus every open reservation. A process that dies mid-call leaves its
reservation counted for the rest of the day. The ledger is per cache directory;
file locks are not reliable on network filesystems.

**Estimates.** A reservation is the estimated prompt tokens plus the output limit,
priced at the highest input rate that can apply: the long-context rate when the
prompt could exceed its threshold (a token is never shorter than one UTF-8 byte),
and for providers with prompt-cache writes the cache-write rate.
"""

from __future__ import annotations

import contextlib
import functools
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
    "max_external_requests_per_run",
)
BUDGET_CONFIG_KEYS = (*LIMIT_KEYS, "pricing_file", "max_pricing_age_days")

ANALYSIS, SETTLEMENT = "analysis", "settlement"
RECORD_VERSION = 2

# Output reserved for a call when no per-call ``max_tokens`` limit is set.
DEFAULT_OUTPUT_RESERVATION = 8192
# UTF-8 bytes per token for estimates: English averages about 4 characters per
# token and a CJK character is 3 bytes for 1-2 tokens, so 2 errs towards refusing.
BYTES_PER_TOKEN = 2
# Retries the TradingAgents wrappers make for Google and Bedrock under a budget
# when llm_max_retries is not set.
DEFAULT_BUDGET_RETRIES = 2

# Providers whose LangChain client honours ``max_tokens`` and returns token usage
# on every (non-streamed) response, at the hosts they officially serve from. Cost
# limits are refused for anything else: a ChatGPT plan drops max_tokens and
# streams, a local or custom endpoint's usage reporting is unknown, and Bedrock's
# client could not be verified (langchain-aws is an optional extra).
OFFICIAL_HOSTS: dict[str, tuple[str, ...]] = {
    "openai": ("api.openai.com",),
    "anthropic": ("api.anthropic.com",),
    "google": ("generativelanguage.googleapis.com",),
    "azure": (".openai.azure.com", ".cognitiveservices.azure.com"),
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
# Providers that can charge for writing the prompt cache.
CACHE_WRITE_PROVIDERS = frozenset({"anthropic", "bedrock"})
# Which SDK retries each provider's client.
SDK_OF_PROVIDER = {"anthropic": "anthropic", "google": "google", "bedrock": "bedrock"}


class LLMBudgetExceeded(RuntimeError):
    """A limit refused an LLM call (or a retry, or an external request) before it was sent.

    ``scope`` names the limit: ``calls``, ``tokens``, ``run_cost``,
    ``reflection_calls``, ``reflection_tokens``, ``day_cost``, ``session_cost``,
    ``pricing``, ``usage_unreported``, ``ledger``, ``usage_record``, ``run_ended``,
    ``resume``, ``resume_in_progress`` or ``external_requests``. ``usage`` is the
    run's usage summary when it was refused. The message never contains credentials.
    """

    def __init__(self, scope: str, message: str, usage: dict | None = None):
        super().__init__(message)
        self.scope = scope
        self.usage = usage or {}


class PricingError(ValueError):
    """Cost limits cannot be trusted: pricing, output limit, provider or endpoint is unsuitable."""


# --- pricing -----------------------------------------------------------------------


@dataclass(frozen=True)
class Rates:
    """USD per million tokens for one pricing tier."""

    input: float
    output: float
    cached_input: float | None = None
    cache_write: float | None = None

    @property
    def cached_rate(self) -> float:
        return self.input if self.cached_input is None else self.cached_input

    @property
    def write_rate(self) -> float:
        # Unpriced cache writes are charged at twice the input rate, the highest
        # write multiplier providers charge.
        return 2 * self.input if self.cache_write is None else self.cache_write


@dataclass(frozen=True)
class ModelPrice:
    """A model's prices: a base tier, and optionally a long-context tier that
    applies to a whole request whose input exceeds ``long_context_above`` tokens."""

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None
    long_context_above: int | None = None
    long_context: Rates | None = None

    @property
    def base(self) -> Rates:
        return Rates(self.input_per_mtok, self.output_per_mtok, self.cached_input_per_mtok,
                     self.cache_write_per_mtok)

    def tier(self, input_tokens: int) -> Rates:
        if self.long_context is not None and input_tokens > (self.long_context_above or 0):
            return self.long_context
        return self.base

    def cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0,
             cache_write_tokens: int = 0) -> float:
        """What a completed request cost, from the tokens the provider reported."""
        rates = self.tier(input_tokens)
        total_in = max(input_tokens, 0)
        cached = min(max(cached_tokens, 0), total_in)
        written = min(max(cache_write_tokens, 0), total_in - cached)
        return ((total_in - cached - written) * rates.input
                + cached * rates.cached_rate
                + written * rates.write_rate
                + max(output_tokens, 0) * rates.output) / 1_000_000

    def worst_case_rates(self, prompt_bytes: int, provider: str) -> tuple[float, float]:
        """(input, output) rates to reserve at: the highest that can apply to a prompt
        of ``prompt_bytes`` bytes. A token covers at least one byte, so the prompt
        cannot exceed ``prompt_bytes`` tokens."""
        rates = self.base
        if self.long_context is not None and prompt_bytes > (self.long_context_above or 0):
            rates = Rates(max(rates.input, self.long_context.input), max(rates.output, self.long_context.output),
                          rates.cached_input, max(rates.write_rate, self.long_context.write_rate))
        input_rate = rates.input
        if provider in CACHE_WRITE_PROVIDERS:
            input_rate = max(input_rate, rates.write_rate)
        return input_rate, rates.output

    def reservation(self, prompt_bytes: int, input_tokens: int, output_tokens: int, provider: str) -> float:
        input_rate, output_rate = self.worst_case_rates(prompt_bytes, provider)
        return (max(input_tokens, 0) * input_rate + max(output_tokens, 0) * output_rate) / 1_000_000


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


def _model_price(key: str, entry: Any) -> ModelPrice:
    if not isinstance(entry, dict):
        raise PricingError(f"pricing for {key!r} must be an object")
    long_context, above = None, None
    tier = entry.get("long_context")
    if tier is not None:
        if not isinstance(tier, dict):
            raise PricingError(f"pricing for {key!r}: long_context must be an object")
        above = tier.get("above_input_tokens")
        if isinstance(above, bool) or not isinstance(above, int) or above < 0:
            raise PricingError(f"pricing for {key!r}: long_context.above_input_tokens must be a "
                               "non-negative whole number")
        long_context = Rates(
            _price_field(tier, "input_per_mtok", key, True), _price_field(tier, "output_per_mtok", key, True),
            _price_field(tier, "cached_input_per_mtok", key, False),
            _price_field(tier, "cache_write_per_mtok", key, False),
        )
    return ModelPrice(
        input_per_mtok=_price_field(entry, "input_per_mtok", key, True),
        output_per_mtok=_price_field(entry, "output_per_mtok", key, True),
        cached_input_per_mtok=_price_field(entry, "cached_input_per_mtok", key, False),
        cache_write_per_mtok=_price_field(entry, "cache_write_per_mtok", key, False),
        long_context_above=above, long_context=long_context,
    )


def load_pricing(path: str | Path, max_age_days: int | None, today: date | None = None) -> Pricing:
    """Read and validate a pricing file; raise ``PricingError`` on anything doubtful.

    Expected shape::

        {"currency": "USD", "as_of": "2026-11-01", "source": "https://...",
         "models": {"anthropic:claude-x": {
             "input_per_mtok": 3.0, "output_per_mtok": 15.0,
             "cached_input_per_mtok": 0.3, "cache_write_per_mtok": 3.75,
             "long_context": {"above_input_tokens": 200000,
                              "input_per_mtok": 6.0, "output_per_mtok": 22.5}}}}

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
    prices = {str(key): _model_price(str(key), entry) for key, entry in models.items()}
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
    host = (urlparse(str(base_url)).hostname or "").lower()
    allowed = OFFICIAL_HOSTS[provider]
    return bool(host) and any(host == a or (a.startswith(".") and host.endswith(a)) for a in allowed)


def client_endpoint(llm: Any) -> str | None:
    """The endpoint a constructed LangChain chat model will actually call.

    Read from the SDK client the model built, so environment overrides
    (``OPENAI_BASE_URL``, ``ANTHROPIC_BASE_URL``, ...) and gateways are seen.
    None when it cannot be determined.
    """
    candidates = []
    for path in (("root_client", "base_url"),               # ChatOpenAI and subclasses, Azure
                 ("_client", "base_url"),                   # ChatAnthropic
                 ("client", "_api_client", "_http_options", "base_url"),   # ChatGoogleGenerativeAI
                 ("client", "meta", "endpoint_url")):       # ChatBedrockConverse
        node = llm
        try:
            for attr in path:
                node = getattr(node, attr)
        except Exception:  # noqa: BLE001 — an attribute that cannot be read is no answer
            continue
        if node:
            candidates.append(str(node))
    return candidates[0] if candidates else None


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
    unresolved_recovered: int = 0

    @property
    def tokens(self) -> int:
        return self.reported_input + self.reported_output + self.estimated_unreported

    def add(self, other: _Totals) -> _Totals:
        return _Totals(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    @classmethod
    def from_dict(cls, data: Any) -> _Totals:
        """Strict: every field present, of its type, finite and non-negative."""
        if not isinstance(data, dict):
            raise ValueError("not an object")
        values = {}
        for f in fields(cls):
            if f.name not in data:
                raise ValueError(f"missing {f.name}")
            value = data[f.name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"bad {f.name}")
            values[f.name] = float(value) if f.type in ("float", float) else int(value)
        return cls(**values)


class RunAccount:
    """The usage of one analysis (or one settlement pass), in its two allowances."""

    def __init__(self, budget: RunBudget, persist_path: Path | None = None, kind: str = ANALYSIS):
        self.id = uuid.uuid4().hex[:12]
        self.budget = budget
        self.kind = kind
        self.scopes = {ANALYSIS: _Totals(), SETTLEMENT: _Totals()}
        self.ended = False
        self.persist_path = persist_path
        # Reservations in flight, persisted before each request is sent.
        self.inflight: dict[str, dict] = {}
        self.broken: str | None = None   # why the usage record can no longer be trusted

    @property
    def total(self) -> _Totals:
        return self.scopes[ANALYSIS].add(self.scopes[SETTLEMENT])

    def save(self) -> bool:
        """Write the usage beside the checkpoint; False (and recorded) on failure."""
        if self.persist_path is None:
            return True
        try:
            _write_json_atomic(self.persist_path, {
                "version": RECORD_VERSION, "account": self.id,
                "scopes": {name: asdict(t) for name, t in self.scopes.items()},
                "inflight": self.inflight,
            })
            return True
        except OSError as exc:
            self.broken = f"the usage record for this checkpoint could not be saved ({type(exc).__name__})"
            logger.error("LLM budget: %s", self.broken)
            return False

    def load(self) -> None:
        """Restore a saved record; anything incomplete or malformed raises ValueError."""
        data = json.loads(Path(self.persist_path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != RECORD_VERSION:
            raise ValueError("unknown record version")
        scopes = data.get("scopes")
        if not isinstance(scopes, dict) or set(scopes) != {ANALYSIS, SETTLEMENT}:
            raise ValueError("scopes incomplete")
        restored = {name: _Totals.from_dict(scopes[name]) for name in (ANALYSIS, SETTLEMENT)}
        inflight = data.get("inflight")
        if not isinstance(inflight, dict):
            raise ValueError("inflight missing")
        for entry in inflight.values():
            if (not isinstance(entry, dict) or entry.get("scope") not in (ANALYSIS, SETTLEMENT)
                    or not all(isinstance(entry.get(k), (int, float)) and not isinstance(entry.get(k), bool)
                               and math.isfinite(entry[k]) and entry[k] >= 0 for k in ("tokens", "cost"))):
                raise ValueError("inflight entry malformed")
        self.scopes = restored
        # A reservation still open when the process died: count it as spent, at its
        # worst case, since the request may have been sent and billed.
        for entry in inflight.values():
            totals = self.scopes[entry["scope"]]
            totals.estimated_unreported += int(entry["tokens"])
            totals.charged_cost += float(entry["cost"])
            totals.unresolved_recovered += 1
        self.inflight = {}


@dataclass
class _Reservation:
    account: RunAccount
    scope: str
    provider: str
    model: str
    price: ModelPrice | None
    prompt_bytes: int
    input_estimate: int
    tokens: int
    cost: float
    ledger: tuple[Path, str] | None
    key: str


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


def meter_external_requests(service: str, requests: int = 1) -> None:
    """Count paid requests made outside the LLM clients (TypeSafe screening).

    Raises ``LLMBudgetExceeded`` when a cost limit is set (the pricing file cannot
    price them) or when ``max_external_requests_per_run`` would be passed. Outside
    a run this does nothing.
    """
    account = _owner.get()
    if account is not None:
        account.budget._external(account, service, requests)


def note_retry() -> None:
    """A provider client is about to retry the call this thread is making.

    Counts the retry against the call's run, or raises ``LLMBudgetExceeded`` when
    it would pass a limit, in which case the retry is not sent. Outside a
    budgeted call this does nothing.
    """
    current = getattr(_active, "call", None)
    if current is not None:
        budget, run_id = current
        budget._retry(run_id)


# --- SDK retry hooks (installed only when a budget has limits) ---------------------

_hooks_lock = threading.Lock()
_hooks: dict[str, bool] = {}


def _hook_sdk(name: str) -> bool:
    """Route an SDK's retries through ``note_retry``, before each is sent."""
    try:
        module = __import__(f"{name}._base_client", fromlist=["SyncAPIClient", "AsyncAPIClient"])
        sync_cls, async_cls = module.SyncAPIClient, module.AsyncAPIClient
        sync_original, async_original = sync_cls._sleep_for_retry, async_cls._sleep_for_retry
    except (ImportError, AttributeError):
        return False
    if getattr(sync_original, "_tradingagents_hook", False):
        return True

    @functools.wraps(sync_original)
    def sync_sleep(self, *args, **kwargs):
        note_retry()
        return sync_original(self, *args, **kwargs)

    @functools.wraps(async_original)
    async def async_sleep(self, *args, **kwargs):
        note_retry()
        return await async_original(self, *args, **kwargs)

    sync_sleep._tradingagents_hook = async_sleep._tradingagents_hook = True
    sync_cls._sleep_for_retry, async_cls._sleep_for_retry = sync_sleep, async_sleep
    return True


def install_retry_hooks() -> dict[str, bool]:
    """Hook the OpenAI and Anthropic SDKs' retries; returns which were hooked.

    An SDK that cannot be hooked must have its retries switched off by the caller
    (``max_retries=0``), so no retry can bypass the budget.
    """
    with _hooks_lock:
        for name in ("openai", "anthropic"):
            if not _hooks.get(name):
                _hooks[name] = _hook_sdk(name)
        return dict(_hooks)


def retry_hook_installed(provider: str) -> bool:
    sdk = SDK_OF_PROVIDER.get((provider or "").lower(), "openai")
    return bool(_hooks.get(sdk))


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
        max_external_requests: int | None = None,
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
        self.max_external_requests = max_external_requests
        self.max_session_cost: float | None = None
        self.output_reservation = output_reservation
        self.output_limited = output_limited
        self.pricing = pricing
        self.ledger_dir = Path(ledger_dir).expanduser() if ledger_dir else None
        self.models = tuple((m[0].lower(), m[1], m[2] if len(m) > 2 else None) for m in models)
        # (provider, model, endpoint the constructed client calls), from register_clients.
        self.clients: tuple = ()
        self._lock = threading.RLock()
        # Calls outside any run land here; it is never ended.
        self._default = RunAccount(self)
        self._last = self._default        # the most recent analysis, for usage()
        self._live: set[RunAccount] = set()
        self._session = _Totals()
        self._inflight: dict[Any, _Reservation] = {}
        # Sticky: spend can no longer be bounded, so every later call is refused.
        self._usage_unreported = False
        self._ledger_broken: str | None = None
        if self.limited:
            install_retry_hooks()

    @classmethod
    def from_config(cls, config: dict, models: list | tuple = ()) -> RunBudget:
        """Build from a TradingAgents config; fail closed on doubtful cost limits.

        ``models`` are the ``(provider, model[, base_url])`` the run will call.
        With a cost limit set this raises ``PricingError`` before any model is
        called unless the pricing, the output limit and every provider are
        suitable; ``register_clients`` then checks the endpoints the constructed
        clients actually call.
        """
        max_calls = _limit(config, "max_llm_calls_per_run", int)
        max_tokens = _limit(config, "max_tokens_per_run", int)
        max_run_cost = _limit(config, "max_cost_per_run_usd", float)
        max_day_cost = _limit(config, "max_cost_per_day_usd", float)
        max_reflection_calls = _limit(config, "max_reflection_calls_per_run", int)
        max_reflection_tokens = _limit(config, "max_reflection_tokens_per_run", int)
        max_external = _limit(config, "max_external_requests_per_run", int)
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
            max_reflection_tokens=max_reflection_tokens, max_external_requests=max_external,
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
        """Whether any per-run allowance is capped."""
        return any(v is not None for v in (self.max_calls, self.max_tokens, self.max_run_cost,
                                           self.max_reflection_calls, self.max_reflection_tokens,
                                           self.max_external_requests))

    @property
    def limited(self) -> bool:
        return self.run_limited or self.cost_limited

    def allowance(self, scope: str) -> tuple[int | None, int | None]:
        """(calls, tokens) for an allowance; settlement inherits what it does not set."""
        if scope == SETTLEMENT:
            calls = self.max_reflection_calls if self.max_reflection_calls is not None else self.max_calls
            tokens = self.max_reflection_tokens if self.max_reflection_tokens is not None else self.max_tokens
            return calls, tokens
        return self.max_calls, self.max_tokens

    def register_clients(self, clients) -> None:
        """Record the endpoint each constructed client calls; check it under a cost limit.

        ``clients`` are ``(provider, model, llm)``. Raises ``PricingError`` before
        any call when a cost limit is set and an endpoint is unknown or not the
        provider's official one.
        """
        self.clients = tuple((p.lower(), m, client_endpoint(llm)) for p, m, llm in clients)
        if self.cost_limited:
            _require_actual_endpoints(self.clients)

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
        _require_actual_endpoints(self.clients)
        if any(not retry_hook_installed(p) and SDK_OF_PROVIDER.get(p, "openai") in ("openai", "anthropic")
               for p, _, _ in self.clients):
            install_retry_hooks()
            if any(not retry_hook_installed(p) and SDK_OF_PROVIDER.get(p, "openai") in ("openai", "anthropic")
                   for p, _, _ in self.clients):
                raise PricingError("a total cost limit needs retries routed through the budget, and the "
                                   "SDK in use could not be hooked; set llm_max_retries to 0 at graph creation")
        self.max_session_cost = limit

    def callback(self, provider: str, model: str) -> BudgetCallback:
        """The callback handler to attach to the client of ``provider:model``."""
        return BudgetCallback(self, provider, model)

    # --- run lifecycle -------------------------------------------------------

    @property
    def current(self) -> RunAccount:
        """The account bound to the calling context, else the most recent analysis."""
        owner = _owner.get()
        return owner if owner is not None and owner.budget is self else self._last

    def begin_run(self, persist_path: str | Path | None = None, restore: bool = False,
                  kind: str = ANALYSIS) -> RunAccount:
        """Start an account. Never ends another: concurrent runs stay independent.

        ``persist_path`` keeps the account's usage, and its in-flight reservations,
        in a file (beside a checkpoint). With ``restore``, the usage already in that
        file is continued: a missing, incomplete or malformed record is refused
        while any limit is set, since a resumed analysis must not get a fresh
        allowance.
        """
        with self._lock:
            account = RunAccount(self, Path(persist_path) if persist_path else None, kind)
            if restore and account.persist_path is not None:
                try:
                    account.load()
                except (OSError, ValueError) as exc:
                    problem = ("missing" if isinstance(exc, FileNotFoundError)
                               else f"unreadable or incomplete ({type(exc).__name__}: {exc})")
                    if self.limited:
                        raise LLMBudgetExceeded(
                            "resume", f"the saved LLM usage for this checkpoint is {problem}, so resuming it "
                            "could exceed its limits; start fresh with --clear-checkpoints", {}) from None
                    logger.warning("Saved LLM usage for this checkpoint is %s; counting from zero", problem)
            if not account.save() and self.limited:
                raise LLMBudgetExceeded("usage_record", account.broken or "usage record not saved", {})
            self._live.add(account)
            if kind == ANALYSIS:
                self._last = account
            return account

    def end_run(self, account: RunAccount, completed: bool = False) -> None:
        """End ``account``, and only it.

        With ``completed`` (its graph returned normally, so every call it made has
        ended or failed), a reservation still open never reached the provider: a
        callback before the budget's aborted the call. It is released, not charged.
        """
        with self._lock:
            account.ended = True
            self._live.discard(account)
            if completed:
                for run_id, reservation in list(self._inflight.items()):
                    if reservation.account is account:
                        self._release(run_id, reservation)

    def bound(self, account: RunAccount):
        """Context manager binding ``account`` to the calling context."""

        @contextlib.contextmanager
        def scope():
            token = _owner.set(account)
            try:
                yield account
            finally:
                _owner.reset(token)

        return scope()

    # --- reporting -----------------------------------------------------------

    def summary(self, session: bool = False, account: RunAccount | None = None) -> dict:
        """Usage of a run (default: the bound one, else the most recent) or the session."""
        with self._lock:
            if session:
                totals, by_scope = self._session, None
            else:
                account = account or self.current
                totals, by_scope = account.total, account.scopes
            priced = self.pricing is not None
            body = _totals_summary(totals, priced)
            summary = {
                "limits": {
                    "max_llm_calls_per_run": self.max_calls,
                    "max_tokens_per_run": self.max_tokens,
                    "max_cost_per_run_usd": self.max_run_cost,
                    "max_cost_per_day_usd": self.max_day_cost,
                    "max_reflection_calls_per_run": self.allowance(SETTLEMENT)[0],
                    "max_reflection_tokens_per_run": self.allowance(SETTLEMENT)[1],
                    "max_external_requests_per_run": self.max_external_requests,
                    "max_total_cost_usd": self.max_session_cost,
                    "per_call_output_reservation": self.output_reservation,
                },
                **body,
                "external_paid_requests": totals.external_requests,
                "unresolved_reservations_recovered": totals.unresolved_recovered,
                "cost_usd": {
                    **body["cost_usd"],
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
            if owner.ended and self.limited:
                self._refuse("run_ended", "a call from an analysis that has already ended was refused",
                             owner)
            return owner
        return self._default

    def _start(self, run_id: Any, provider: str, model: str, prompt_bytes: int) -> None:
        with self._lock:
            account = self._account_for_call()
            scope = _scope.get()
            totals, session = account.scopes[scope], self._session
            max_calls, max_tokens = self.allowance(scope)
            prefix = "reflection_" if scope == SETTLEMENT else ""
            label = "reflection" if scope == SETTLEMENT else "analysis"
            input_estimate = estimate_tokens(prompt_bytes)
            projected_tokens = input_estimate + self.output_reservation
            price = self.pricing.price(provider, model) if self.pricing else None
            projected_cost = (price.reservation(prompt_bytes, input_estimate, self.output_reservation, provider)
                              if price else 0.0)
            same_scope = [r for r in self._inflight.values() if r.account is account and r.scope == scope]
            same_run = [r for r in self._inflight.values() if r.account is account]

            if account.broken and self.limited:
                self._refuse("usage_record", account.broken, account)
            if max_calls is not None and totals.attempted + totals.sdk_retries + 1 > max_calls:
                self._refuse(f"{prefix}calls", f"LLM call limit reached: {max_calls} per {label} "
                             f"({'TRADINGAGENTS_MAX_REFLECTION_CALLS' if prefix else 'TRADINGAGENTS_MAX_LLM_CALLS'})",
                             account)
            if (max_tokens is not None
                    and totals.tokens + sum(r.tokens for r in same_scope) + projected_tokens > max_tokens):
                self._refuse(f"{prefix}tokens", f"token limit would be exceeded: {max_tokens:,} per {label}; "
                             f"{totals.tokens:,} counted so far, this call reserves up to {projected_tokens:,}",
                             account)
            if self.cost_limited:
                self._check_cost(account, price, provider, model, projected_cost, same_run)
            ledger = None
            if price is not None:
                ledger = self._ledger_reserve(projected_cost, account)

            key = uuid.uuid4().hex
            totals.attempted += 1
            session.attempted += 1
            reservation = _Reservation(account, scope, provider, model, price, prompt_bytes, input_estimate,
                                       projected_tokens, projected_cost, ledger, key)
            account.inflight[key] = {"scope": scope, "tokens": projected_tokens, "cost": projected_cost}
            # The reservation is on disk before the request is sent, so a crash
            # from here on is recovered as spent.
            if not account.save() and self.limited:
                account.inflight.pop(key, None)
                totals.attempted -= 1
                session.attempted -= 1
                self._ledger_release(ledger)
                self._refuse("usage_record", account.broken or "usage record not saved", account)
            self._inflight[run_id] = reservation
            _active.call = (self, run_id)

    def _check_cost(self, account: RunAccount, price: ModelPrice | None, provider: str, model: str,
                    projected_cost: float, same_run: list) -> None:
        if self._usage_unreported:
            self._refuse("usage_unreported", "a provider response carried no token usage, so spend "
                         "can no longer be bounded; cost limits fail closed", account)
        if self._ledger_broken:
            self._refuse("ledger", self._ledger_broken, account)
        if price is None:
            self._refuse("pricing", f"no price for {provider}:{model} in the pricing file; "
                         "cost limits fail closed", account)
        # One ceiling per run: the analysis and its reflections together.
        run_charged = account.total.charged_cost
        if (self.max_run_cost is not None
                and run_charged + sum(r.cost for r in same_run) + projected_cost > self.max_run_cost):
            self._refuse("run_cost", f"estimated cost limit would be exceeded: ${self.max_run_cost:.4f} per "
                         f"run, analysis and reflections together (TRADINGAGENTS_MAX_COST_PER_RUN); "
                         f"${run_charged:.4f} charged so far, this reserves up to ${projected_cost:.4f}", account)
        inflight_cost = sum(r.cost for r in self._inflight.values())
        if (self.max_session_cost is not None
                and self._session.charged_cost + inflight_cost + projected_cost > self.max_session_cost):
            self._refuse("session_cost", f"total estimated cost limit would be exceeded: "
                         f"${self.max_session_cost:.4f}; ${self._session.charged_cost:.4f} charged so far",
                         account)

    def _end(self, run_id: Any, response: Any) -> None:
        with self._lock:
            reservation = self._inflight.pop(run_id, None)
            self._clear_active(run_id)
            if reservation is None:
                return
            account = reservation.account
            totals = account.scopes[reservation.scope]
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
            account.inflight.pop(reservation.key, None)
            account.save()

    def _error(self, run_id: Any) -> None:
        with self._lock:
            reservation = self._inflight.pop(run_id, None)
            self._clear_active(run_id)
            if reservation is None:
                return
            account = reservation.account
            totals = account.scopes[reservation.scope]
            for t in (totals, self._session):
                t.failed += 1
            charge = self._charge_input_estimate(totals, reservation)
            self._ledger_settle(reservation.ledger, charge)
            account.inflight.pop(reservation.key, None)
            account.save()

    def _release(self, run_id: Any, reservation: _Reservation) -> None:
        """A reservation whose request never started: undo it without charging."""
        self._inflight.pop(run_id, None)
        account = reservation.account
        for t in (account.scopes[reservation.scope], self._session):
            t.attempted = max(0, t.attempted - 1)
        self._ledger_release(reservation.ledger)
        account.inflight.pop(reservation.key, None)
        account.save()

    def _retry(self, run_id: Any) -> None:
        """A client is about to retry this call: refuse it, or count and charge it.

        Checked like a new call: the per-allowance call and token limits, the run's
        cost ceiling, and the session and daily limits.
        """
        with self._lock:
            reservation = self._inflight.get(run_id)
            if reservation is None:
                self._session.sdk_retries += 1
                return
            account, scope = reservation.account, reservation.scope
            totals = account.scopes[scope]
            max_calls, max_tokens = self.allowance(scope)
            prefix = "reflection_" if scope == SETTLEMENT else ""
            retry_cost = (reservation.price.reservation(reservation.prompt_bytes, reservation.input_estimate, 0,
                                                        reservation.provider) if reservation.price else 0.0)
            others = [r for r in self._inflight.values() if r is not reservation]
            same_scope = [r for r in others if r.account is account and r.scope == scope]
            if max_calls is not None and totals.attempted + totals.sdk_retries + 1 > max_calls:
                self._refuse(f"{prefix}calls", f"retry refused: the LLM call limit ({max_calls}) is reached",
                             account)
            if (max_tokens is not None and totals.tokens + sum(r.tokens for r in same_scope)
                    + reservation.tokens + reservation.input_estimate > max_tokens):
                self._refuse(f"{prefix}tokens", f"retry refused: the token limit ({max_tokens:,}) "
                             "would be exceeded", account)
            if self.cost_limited:
                in_flight_run = sum(r.cost for r in self._inflight.values() if r.account is account)
                if (self.max_run_cost is not None
                        and account.total.charged_cost + in_flight_run + retry_cost > self.max_run_cost):
                    self._refuse("run_cost", "retry refused: the run's estimated cost limit would be exceeded",
                                 account)
                in_flight_all = sum(r.cost for r in self._inflight.values())
                if (self.max_session_cost is not None
                        and self._session.charged_cost + in_flight_all + retry_cost > self.max_session_cost):
                    self._refuse("session_cost", "retry refused: the total estimated cost limit would be "
                                 "exceeded", account)
            if reservation.price is not None:
                self._ledger_charge(retry_cost, account, check=True)
            for t in (totals, self._session):
                t.sdk_retries += 1
                t.estimated_unreported += reservation.input_estimate
            estimated = retry_cost if reservation.price else None
            self._charge(totals, estimated, None)
            account.save()

    def _external(self, account: RunAccount, service: str, requests: int) -> None:
        with self._lock:
            if self.cost_limited:
                self._refuse("pricing", f"{service} requests are billed outside the pricing file, so they "
                             "are not made while a cost limit is set", account)
            totals = account.scopes[ANALYSIS]
            if (self.max_external_requests is not None
                    and account.total.external_requests + requests > self.max_external_requests):
                self._refuse("external_requests", f"{service}: the external request limit "
                             f"({self.max_external_requests} per run, TRADINGAGENTS_MAX_EXTERNAL_REQUESTS) "
                             "is reached", account)
            for t in (totals, self._session):
                t.external_requests += requests
            account.save()

    def _charge_input_estimate(self, totals: _Totals, reservation: _Reservation) -> float:
        """A failed request: the provider may have read the prompt."""
        for t in (totals, self._session):
            t.estimated_unreported += reservation.input_estimate
        estimated = (reservation.price.reservation(reservation.prompt_bytes, reservation.input_estimate, 0,
                                                   reservation.provider) if reservation.price else None)
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

    def _ledger_update(self, path: Path, change, account: RunAccount | None) -> bool:
        """Read-modify-write the ledger under its cross-process lock."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with locked(path):
                data = self._ledger_read(path)
                change(data)
                self._ledger_write(path, data)
            return True
        except _LedgerError as exc:
            self._ledger_failed(str(exc), account)
        except OSError as exc:
            self._ledger_failed(f"the daily spend ledger lock failed ({type(exc).__name__})", account)
        return False

    def _check_day(self, data: dict, amount: float, account: RunAccount | None) -> None:
        spent = data["charged_usd"] + sum(data["reserved"].values())
        if self.max_day_cost is not None and spent + amount > self.max_day_cost:
            self._refuse("day_cost", f"daily estimated cost limit would be exceeded: "
                         f"${self.max_day_cost:.4f} per UTC day (TRADINGAGENTS_MAX_COST_PER_DAY); "
                         f"${spent:.4f} charged or reserved today", account)

    def _ledger_reserve(self, projected_cost: float, account: RunAccount) -> tuple[Path, str] | None:
        """Reserve a call's worst-case cost, checking the daily limit atomically across processes."""
        path, key = self._ledger_path(), f"{os.getpid()}-{uuid.uuid4().hex[:12]}"

        def reserve(data):
            self._check_day(data, projected_cost, account)
            data["reserved"][key] = projected_cost

        return (path, key) if self._ledger_update(path, reserve, account) else None

    def _ledger_charge(self, amount: float, account: RunAccount | None, check: bool) -> None:
        """Charge a retry: checked against the daily limit first, like a call."""
        def charge(data):
            if check:
                self._check_day(data, amount, account)
            data["charged_usd"] = round(data["charged_usd"] + amount, 8)

        self._ledger_update(self._ledger_path(), charge, account)

    def _ledger_settle(self, ledger: tuple[Path, str] | None, charge: float) -> None:
        """Replace a call's reservation with what it was charged."""
        if ledger is None:
            return
        path, key = ledger

        def settle(data):
            data["reserved"].pop(key, None)
            data["charged_usd"] = round(data["charged_usd"] + charge, 8)
            data["calls"] += 1

        self._ledger_after_call(path, settle)

    def _ledger_release(self, ledger: tuple[Path, str] | None) -> None:
        if ledger is not None:
            path, key = ledger
            self._ledger_after_call(path, lambda data: data["reserved"].pop(key, None))

    def _ledger_after_call(self, path: Path, change) -> None:
        # The call has happened (or is being undone); a failure must stop the next one.
        try:
            with locked(path):
                data = self._ledger_read(path)
                change(data)
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
            f"{calls['sdk_retries']} retries), "
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
                "honour max_tokens or report token usage, or could not be verified (supported: "
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


def _require_actual_endpoints(clients) -> None:
    for provider, model, endpoint in clients:
        if endpoint is None:
            raise PricingError(
                f"cost limits need to know where {provider}:{model} sends requests, and its client's "
                "endpoint could not be determined; cost limits fail closed"
            )
        if not endpoint_supported(provider, endpoint):
            host = urlparse(endpoint).hostname or "an unknown host"
            raise PricingError(
                f"cost limits are not supported for {provider}:{model}: its client sends requests to "
                f"{host}, not the provider's official endpoint (check *_BASE_URL environment variables)"
            )


class BudgetCallback(BaseCallbackHandler):
    """Attached to one model client; every call it makes goes through the budget.

    It must be the last callback the client holds (``TradingAgentsGraph`` appends
    it after the caller's): a callback after it could abort a call it has already
    reserved for.
    """

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
    "Pricing", "PricingError", "Rates", "RunAccount", "RunBudget", "bind_owner", "client_endpoint",
    "current_account", "describe_usage", "endpoint_supported", "estimate_tokens", "install_retry_hooks",
    "load_pricing", "meter_external_requests", "note_retry", "reset_owner", "retry_hook_installed",
    "settlement_scope",
]
