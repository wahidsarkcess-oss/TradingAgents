#!/usr/bin/env python3
"""Fail when a file about to be committed (or already tracked) holds a secret.

Standard library only, so it runs before anything is installed: as a pre-commit
hook (``.pre-commit-config.yaml``), in CI, or by hand.

    python scripts/scan_secrets.py                # every file git tracks
    python scripts/scan_secrets.py FILE [FILE...] # these files (pre-commit passes them)
    python scripts/scan_secrets.py --history      # every line ever added, on every branch

Each finding names the file, line and kind, and a fingerprint of the matched
value (the first 16 hex digits of its SHA-256), never the value itself. Files of
any size are scanned, line by line; UTF-8, UTF-16 and UTF-32 (with or without a
byte-order mark) are decoded; other files containing NUL bytes are reported as
skipped binaries.

Exemptions, every one printed in the output so a reviewer sees it:

- the inline marker ``scan-secrets: allow`` on the line, honoured only in files
  under a ``tests/`` directory (test fixtures);
- ``.scan-secrets-allow``: reviewed findings, one per line as printed, each bound
  to its fingerprint, so a different value at the same place is still reported.

``--history`` covers only the commits present locally; in a shallow clone (CI's
default checkout) it warns that older history was not scanned. The scanner is a
heuristic first line, not a guarantee: a credential with no recognisable shape
outside a NAME=value or key: "value" assignment is not found.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import NamedTuple

ALLOW_MARKER = "scan-secrets: allow"
ALLOWLIST_FILE = ".scan-secrets-allow"

# Files that hold credentials and must never be committed, whatever they contain.
FORBIDDEN_NAMES = {
    ".env", ".envrc", ".env.enterprise", ".env.local", "secrets.env",
    "chatgpt-auth.json", "credentials.json", "credentials", "id_rsa", "id_dsa", "id_ecdsa",
    "id_ed25519", ".npmrc", ".pypirc", ".netrc", "_netrc", ".git-credentials", ".htpasswd",
}
FORBIDDEN_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".kdbx", ".secrets.env")
FORBIDDEN_PATTERNS = (re.compile(r"service[-_]?account.*\.json$", re.IGNORECASE),)

# Credential formats. Lengths are set so that short test placeholders do not match.
KEY_PATTERNS = {
    "Anthropic API key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "OpenAI-style API key": re.compile(r"\bsk-(?:proj-|svcacct-|or-v1-)?[A-Za-z0-9_\-]{32,}"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "AWS access key id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "xAI API key": re.compile(r"\bxai-[A-Za-z0-9]{40,}"),
    "Groq API key": re.compile(r"\bgsk_[A-Za-z0-9]{40,}"),
    "NVIDIA API key": re.compile(r"\bnvapi-[A-Za-z0-9_\-]{40,}"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{36,}|\bgithub_pat_[A-Za-z0-9_]{60,}"),
    "Slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{20,}"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "GitLab token": re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}"),
    "Stripe key": re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{20,}"),
    "Google OAuth client secret": re.compile(r"\bGOCSPX-[A-Za-z0-9_\-]{24,}"),
    "npm token": re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"),
    "PyPI token": re.compile(r"\bpypi-AgE[A-Za-z0-9_\-]{50,}"),
    "Perplexity API key": re.compile(r"\bpplx-[A-Za-z0-9]{40,}"),
    "Replicate token": re.compile(r"\br8_[A-Za-z0-9]{30,}"),
    "SendGrid key": re.compile(r"\bSG\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}"),
    "bearer token": re.compile(r"(?i)\bauthorization\b[\"']?\s*[:=]\s*[\"']?bearer\s+[A-Za-z0-9._~+/=\-]{20,}"),
    "private key block": re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    "JSON web token": re.compile(r"\beyJ[A-Za-z0-9_\-]{20,}\.eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}"),
}

# NAME=value lines (env files, shell, YAML) whose name says the value is secret.
ASSIGNMENT = re.compile(
    r"""^\s*(?:export\s+)?(?:-\s+)?["']?(?P<name>[A-Z][A-Z0-9_]*(?:API_?KEY|_KEY|_TOKEN|_SECRET|PASSWORD))["']?"""
    r"""\s*[:=]\s*["']?(?P<value>[^"'\s#]*)"""
)
# key = "value" / "key": "value" in code, JSON and YAML, any case; the value must be a literal.
QUOTED_ASSIGNMENT = re.compile(
    r"""(?i)["']?(?P<name>[a-z0-9_\-]*(?:api[_\-]?key|apikey|secret[_\-]?key|client[_\-]?secret|"""
    r"""access[_\-]?token|auth[_\-]?token|refresh[_\-]?token|password|passwd))["']?\s*[:=]\s*"""
    r"""["'](?P<value>[^"'\s]{16,})["']"""
)
URL_CREDENTIALS = re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s/@:\"'<>]+:(?P<password>[^\s/@\"'<>]{8,})@[^\s/]")
PLACEHOLDER = re.compile(
    r"^(?:|\.\.\.|…|x+|\*+|<.*>|\$\{?.*|\{\{.*|your[-_].*|changeme|change-me|placeholder|dummy|example.*|"
    r"test|fake.*|none|null|redacted|replace[-_]?me|todo|sk-\.\.\..*)$",
    re.IGNORECASE,
)
# Placeholder words inside a quoted literal: test fixtures and documentation.
PLACEHOLDER_WORD = re.compile(
    r"(?i)placeholder|dummy|example|fake|sample|test|your[-_]|xxxx|redacted|changeme|password|secret|\$|\{")


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


class Finding(NamedTuple):
    path: str
    kind: str
    value: str
    line: int | None = None
    commit: str | None = None
    exempt: str | None = None      # why it does not fail the scan, printed regardless

    @property
    def fp(self) -> str:
        return fingerprint(self.value)

    @property
    def key(self) -> tuple:
        """What a reviewed allowlist entry binds to: place, kind and the value's fingerprint."""
        return (self.commit or "", self.path, self.kind, self.fp)

    def __str__(self) -> str:
        where = f"{self.commit} {self.path}" if self.commit else (
            f"{self.path}:{self.line}" if self.line is not None else self.path)
        return f"{where}: {self.kind} [fp:{self.fp}]"


def _looks_like_a_secret(value: str) -> bool:
    """A literal worth flagging: not a placeholder, and mixing letters with digits."""
    return (not PLACEHOLDER.match(value) and not PLACEHOLDER_WORD.search(value)
            and any(c.isdigit() for c in value) and any(c.isalpha() for c in value))


def _forbidden(path: str) -> str | None:
    name = PurePosixPath(path.replace("\\", "/")).name
    if (name in FORBIDDEN_NAMES or name.endswith(FORBIDDEN_SUFFIXES)
            or any(p.search(name) for p in FORBIDDEN_PATTERNS)):
        return f"credential file '{name}' must not be committed"
    if name.startswith(".env.") and not name.endswith(".example"):
        return f"environment file '{name}' must not be committed (commit a .example copy instead)"
    return None


def in_tests(path: str) -> bool:
    """Whether ``path`` lies under a ``tests/`` directory, where inline exemptions count."""
    return "tests" in PurePosixPath(path.replace("\\", "/")).parts[:-1]


def scan_line(line: str) -> list[tuple[str, str]]:
    """(kind, matched value) for each secret on one line, ignoring any marker."""
    found = []
    for kind, pattern in KEY_PATTERNS.items():
        match = pattern.search(line)
        if match:
            found.append((kind, match.group(0)))
    match = ASSIGNMENT.match(line)
    if match and len(match["value"]) >= 8 and not PLACEHOLDER.match(match["value"]):
        found.append((f"value assigned to {match['name']}", match["value"]))
    elif not match:
        quoted = QUOTED_ASSIGNMENT.search(line)
        if quoted and _looks_like_a_secret(quoted["value"]):
            found.append((f"literal assigned to {quoted['name']}", quoted["value"]))
    url = URL_CREDENTIALS.search(line)
    if url and not PLACEHOLDER_WORD.search(url["password"]):
        found.append(("credentials in a URL", url["password"]))
    return found


def _findings_on(line: str, path: str, number: int | None, commit: str | None = None) -> list[Finding]:
    marked = ALLOW_MARKER in line
    out = []
    for kind, value in scan_line(line):
        exempt = None
        if marked:
            exempt = ("inline marker" if in_tests(path)
                      else None)
        finding = Finding(path, kind, value, number, commit, exempt)
        if marked and exempt is None:
            finding = Finding(path, f"{kind} (marker ignored: honoured only under tests/)", value,
                              number, commit)
        out.append(finding)
    return out


_BOMS = ((codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"),
         (codecs.BOM_UTF8, "utf-8-sig"), (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16"))


def detect_encoding(head: bytes) -> str | None:
    """The text encoding of a file from its first bytes, or None for a binary file."""
    for bom, encoding in _BOMS:
        if head.startswith(bom):
            return encoding
    if b"\0" not in head:
        return "utf-8"
    sample = head[:4096]
    even, odd = sample[0::2], sample[1::2]
    if sample and odd.count(0) > 0.3 * len(odd) and even.count(0) < 0.1 * len(even):
        return "utf-16-le"
    if sample and even.count(0) > 0.3 * len(even) and odd.count(0) < 0.1 * len(odd):
        return "utf-16-be"
    return None


def iter_lines(path: str) -> Iterator[str] | None:
    """The file's lines, decoded and streamed (any size), or None for a binary."""
    with open(path, "rb") as handle:
        head = handle.read(8192)
    encoding = detect_encoding(head)
    if encoding is None:
        return None

    def lines():
        with open(path, encoding=encoding, errors="replace", newline="") as handle:
            for line in handle:
                yield line.rstrip("\r\n")

    return lines()


SKIPPED: list[str] = []   # binaries, reported by main()


def scan_file(path: str) -> list[Finding]:
    findings = []
    reason = _forbidden(path)
    if reason:
        findings.append(Finding(path, reason, PurePosixPath(path.replace("\\", "/")).name))
    try:
        if not Path(path).is_file():
            return findings
        lines = iter_lines(path)
    except OSError:
        return findings
    if lines is None:
        SKIPPED.append(path)
        return findings
    for number, line in enumerate(lines, 1):
        findings += _findings_on(line, path, number)
    return findings


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout
    return [p for p in out.decode("utf-8", errors="replace").split("\0") if p]


def is_shallow() -> bool:
    out = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], capture_output=True, text=True)
    return out.stdout.strip() == "true"


def scan_history() -> list[Finding]:
    """Every line added in any commit on any ref (merge resolutions included, via -m),
    and every file name ever committed."""
    log = subprocess.run(
        ["git", "log", "--all", "-p", "-m", "--no-color", "--no-ext-diff", "--format=commit %h"],
        capture_output=True, check=True,
    ).stdout.decode("utf-8", errors="replace")
    findings, commit, path = [], "?", "?"
    for line in log.splitlines():
        if line.startswith("commit "):
            commit = line.split()[1]
        elif line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else line[4:]
            reason = _forbidden(path) if path != "/dev/null" else None
            if reason:
                findings.append(Finding(path, reason, PurePosixPath(path).name, None, commit))
        elif line.startswith("+") and not line.startswith("+++"):
            findings += _findings_on(line[1:], path, None, commit)
    # -m repeats a merge's diff once per parent.
    seen: set = set()
    return [f for f in findings if not (f.key in seen or seen.add(f.key))]


_ALLOW_ENTRY = re.compile(
    r"^(?:(?P<commit>[0-9a-f]{7,40}) )?(?P<path>.+?)(?::(?P<line>\d+))?: (?P<kind>.+) \[fp:(?P<fp>[0-9a-f]{16})\]$")


def load_allowlist(path: str = ALLOWLIST_FILE) -> tuple[set[tuple], list[str]]:
    """Reviewed findings as (commit, path, kind, fingerprint), and entries that bind no value."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return set(), []
    keys, invalid = set(), []
    for line in (raw.strip() for raw in lines):
        if not line or line.startswith("#"):
            continue
        match = _ALLOW_ENTRY.match(line)
        if match:
            keys.add((match["commit"] or "", match["path"], match["kind"], match["fp"]))
        else:
            invalid.append(line)
    return keys, invalid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="files to scan (default: every tracked file)")
    parser.add_argument("--history", action="store_true", help="scan every line ever added instead")
    args = parser.parse_args(argv)

    SKIPPED.clear()
    if args.history:
        if is_shallow():
            print("warning: this is a shallow clone; commits before its depth were not scanned. "
                  "Run --history in a full clone (git fetch --unshallow) for a complete pass.",
                  file=sys.stderr)
        findings = scan_history()
    else:
        findings = [f for path in (args.paths or tracked_files()) for f in scan_file(path)]

    reviewed, invalid = load_allowlist()
    for entry in invalid:
        print(f"warning: {ALLOWLIST_FILE}: ignored an entry with no [fp:...] fingerprint: {entry}",
              file=sys.stderr)
    failing = []
    for finding in findings:
        if finding.exempt:
            print(f"exempt ({finding.exempt}): {finding}")
        elif finding.key in reviewed:
            print(f"exempt (reviewed in {ALLOWLIST_FILE}): {finding}")
        else:
            failing.append(finding)
    for finding in failing:
        print(finding)
    for path in SKIPPED:
        print(f"note: {path}: binary file, not scanned", file=sys.stderr)
    if failing:
        print(f"\n{len(failing)} possible secret(s) found. Remove them (and rotate any real key); a test "
              f"fixture under tests/ may carry '{ALLOW_MARKER}', and a reviewed false positive can be "
              f"listed in {ALLOWLIST_FILE} with its fingerprint.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
