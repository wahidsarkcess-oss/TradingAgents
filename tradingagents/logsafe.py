"""Keep API keys and OAuth tokens out of logs, tracebacks and unattended output.

Two layers, both applied when text is written rather than where it is produced,
so a key quoted by a third-party SDK's error message is caught too:

- ``redact(text)`` replaces the value of every secret-looking environment
  variable that is set (the provider keys and any ``*_KEY``, ``*_TOKEN``,
  ``*_SECRET``, ``*_PASSWORD`` variable), the tokens in the ChatGPT sign-in file,
  and anything shaped like a known key format, bearer token, JWT or credential
  query parameter, with ``***``.
- ``RedactingFormatter`` and ``RedactingStream`` apply it to log records
  (traceback included) and to ``sys.stdout`` / ``sys.stderr``.

``install_redaction()`` makes the logging layer permanent for a program that
embeds TradingAgents; ``redaction_scope()`` applies both layers for one command
and restores everything afterwards (the CLI uses it).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

from tradingagents.default_config import _TRADINGAGENTS_HOME
from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV

MASK = "***"

# Variables that always hold a secret, beyond the per-provider LLM keys.
SECRET_ENV_NAMES = frozenset(
    {name for name in PROVIDER_API_KEY_ENV.values() if name}
    | {
        "ALPHA_VANTAGE_API_KEY", "FRED_API_KEY", "TYPESAFE_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
        "AZURE_OPENAI_API_KEY", "OPENAI_COMPATIBLE_API_KEY",
    }
)
# Any other variable named like a credential.
_SECRET_NAME = re.compile(r"(?:^|_)(?:API_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)$")
# Shorter values are too likely to be ordinary words to mask everywhere.
MIN_SECRET_LENGTH = 8

CHATGPT_AUTH_FILE = Path(_TRADINGAGENTS_HOME) / "chatgpt-auth.json"

# Known credential shapes. Each pattern's group 1, when present, is kept.
_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),                     # Anthropic
    re.compile(r"sk-(?:proj-|svcacct-|or-v1-)?[A-Za-z0-9_\-]{16,}"),  # OpenAI, OpenRouter, DeepSeek...
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),                       # Google
    re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),                     # AWS access key id
    re.compile(r"xai-[A-Za-z0-9]{20,}"),                          # xAI
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),                          # Groq
    re.compile(r"nvapi-[A-Za-z0-9_\-]{20,}"),                     # NVIDIA
    re.compile(r"(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"),  # JWT
    re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/=\-]{8,}"),
    re.compile(r"(?i)([?&](?:api_?key|apikey|key|token|access_token|refresh_token|client_secret|"
               r"sig|signature)=)[^&\s\"'<>]+"),
    re.compile(r"(?i)([\"']?(?:api_?key|access_token|refresh_token|id_token|client_secret|"
               r"authorization|x-api-key|password)[\"']?\s*[:=]\s*[\"']?)[^\"'\s,}]{8,}"),
]

_auth_cache: tuple[float, frozenset[str]] | None = None
_auth_lock = threading.Lock()


def _auth_file_secrets() -> frozenset[str]:
    """String values under token-like keys in the ChatGPT sign-in file, cached by mtime."""
    global _auth_cache
    try:
        mtime = CHATGPT_AUTH_FILE.stat().st_mtime
    except OSError:
        return frozenset()
    with _auth_lock:
        if _auth_cache is not None and _auth_cache[0] == mtime:
            return _auth_cache[1]
        found: set[str] = set()

        def walk(node, sensitive=False):
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, sensitive or bool(re.search(r"token|secret|key", str(key), re.I)))
            elif isinstance(node, list):
                for value in node:
                    walk(value, sensitive)
            elif sensitive and isinstance(node, str) and len(node) >= MIN_SECRET_LENGTH:
                found.add(node)

        with suppress(OSError, ValueError):
            walk(json.loads(CHATGPT_AUTH_FILE.read_text(encoding="utf-8")))
        _auth_cache = (mtime, frozenset(found))
        return _auth_cache[1]


def secret_values() -> list[str]:
    """Every secret value currently known, longest first so overlaps mask whole."""
    values = set(_auth_file_secrets())
    for name, value in os.environ.items():
        if value and len(value) >= MIN_SECRET_LENGTH and (
                name in SECRET_ENV_NAMES or _SECRET_NAME.search(name.upper())):
            values.add(value)
    return sorted(values, key=len, reverse=True)


def redact(text: str) -> str:
    """``text`` with every known secret value and credential-shaped string masked."""
    if not text:
        return text
    for value in secret_values():
        if value in text:
            text = text.replace(value, MASK)
    for pattern in _PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) or "") + MASK if m.groups() else MASK, text)
    return text


def scrub_exception(exc: BaseException) -> BaseException:
    """Redact the string arguments of ``exc`` and every exception chained to it, in place.

    For an error that will still propagate (and be printed by whoever catches it
    last): its type and traceback are kept, its message no longer carries a secret.
    An exception whose ``__str__`` builds the message from other attributes is
    left as it is.
    """
    pending, seen = [exc], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        with suppress(Exception):  # some exception types refuse new args
            current.args = tuple(redact(a) if isinstance(a, str) else a for a in current.args)
        pending.extend((current.__cause__, current.__context__))
    return exc


class RedactingFormatter(logging.Formatter):
    """A formatter whose whole output, traceback included, passes through ``redact``."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


class RedactingStream:
    """A text stream wrapper that redacts everything written through it.

    Text is redacted per ``write`` call; a secret split across two writes is
    not recognised, which Python's logging and traceback output never do.
    """

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        return self._stream.write(redact(text) if isinstance(text, str) else text)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        return getattr(self._stream, name)


LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"
_installed_marker = "_tradingagents_redaction"


def install_redaction(level: int = logging.WARNING) -> None:
    """Redact every root log handler's output from now on (idempotent).

    With no handler configured, adds one on stderr at ``level``, as Python's
    last-resort handler would print, but redacted.
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(RedactingFormatter(LOG_FORMAT))
        setattr(handler, _installed_marker, True)
        root.addHandler(handler)
        return
    for handler in root.handlers:
        if not isinstance(handler.formatter, RedactingFormatter):
            fmt = getattr(handler.formatter, "_fmt", None) or LOG_FORMAT
            datefmt = getattr(handler.formatter, "datefmt", None)
            handler.setFormatter(RedactingFormatter(fmt, datefmt))


@contextmanager
def redaction_scope(streams: bool) -> Iterator[None]:
    """Redact log output, and with ``streams`` stdout/stderr too, for one command.

    Everything is restored on exit, so a test runner's or host program's own
    streams and handlers are left as they were.
    """
    root = logging.getLogger()
    added = None
    swapped = []
    if not root.handlers:
        added = logging.StreamHandler()
        added.setLevel(logging.WARNING)
        added.setFormatter(RedactingFormatter(LOG_FORMAT))
        root.addHandler(added)
    else:
        for handler in root.handlers:
            if not isinstance(handler.formatter, RedactingFormatter):
                swapped.append((handler, handler.formatter))
                fmt = getattr(handler.formatter, "_fmt", None) or LOG_FORMAT
                handler.setFormatter(RedactingFormatter(fmt, getattr(handler.formatter, "datefmt", None)))
    saved = (sys.stdout, sys.stderr)
    if streams:
        sys.stdout, sys.stderr = RedactingStream(sys.stdout), RedactingStream(sys.stderr)
    try:
        yield
    finally:
        if streams:
            sys.stdout, sys.stderr = saved
        if added is not None:
            root.removeHandler(added)
        for handler, formatter in swapped:
            handler.setFormatter(formatter)


__all__ = ["RedactingFormatter", "RedactingStream", "install_redaction", "redact",
           "redaction_scope", "scrub_exception", "secret_values"]
