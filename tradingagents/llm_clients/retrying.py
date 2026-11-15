"""Retries the TradingAgents client wrappers make themselves, under the LLM budget.

With budget limits on, the Google and Bedrock clients' own retries are switched
off (they happen inside their SDKs, where the budget cannot see or stop them) and
the wrappers retry here instead. Before each retry the budget is asked
(``note_retry``): it counts the retry, or refuses it, in which case nothing more
is sent. Only transient failures are retried.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from .budget import LLMBudgetExceeded, note_retry

T = TypeVar("T")

# HTTP statuses worth retrying: timeouts, throttling and server errors.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
# Transport failures, by class name, so neither httpx nor botocore must be imported.
TRANSIENT_ERROR_NAMES = frozenset({
    "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "TimeoutException",
    "RemoteProtocolError", "ReadError", "WriteError",                       # httpx
    "EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError",   # botocore
    "ConnectionClosedError",
})
RETRYABLE_AWS_CODES = frozenset({
    "ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException",
    "InternalServerException", "ModelNotReadyException", "ServiceUnavailable",
})


def _chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def is_transient(exc: BaseException) -> bool:
    """Whether a failure (or one it was raised from) is worth retrying."""
    for error in _chain(exc):
        if isinstance(error, LLMBudgetExceeded):
            return False
        if type(error).__name__ in TRANSIENT_ERROR_NAMES:
            return True
        code = getattr(error, "code", None)            # google.genai.errors.APIError
        if isinstance(code, int) and code in RETRYABLE_STATUS:
            return True
        response = getattr(error, "response", None)    # botocore ClientError
        if isinstance(response, dict):
            aws = (response.get("Error") or {}).get("Code")
            status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
            if aws in RETRYABLE_AWS_CODES or status in RETRYABLE_STATUS:
                return True
    return False


def call_with_budget_retries(call: Callable[[], T], retries: int,
                             sleep: Callable[[float], None] | None = None) -> T:
    """``call()``, retried up to ``retries`` times on transient failures, each retry
    first approved by the budget. A budget refusal propagates at once."""
    attempt = 0
    while True:
        try:
            return call()
        except LLMBudgetExceeded:
            raise
        except Exception as exc:
            if attempt >= retries or not is_transient(exc):
                raise
            note_retry()                       # raises LLMBudgetExceeded to stop here
            (sleep or time.sleep)(min(8.0, 0.5 * 2 ** attempt) * random.uniform(0.8, 1.2))
            attempt += 1
