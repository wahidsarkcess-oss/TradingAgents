#!/usr/bin/env python3
"""Fail when a file about to be committed (or already tracked) holds a secret.

Standard library only, so it runs before anything is installed: as a pre-commit
hook (``.pre-commit-config.yaml``), in CI, or by hand.

    python scripts/scan_secrets.py                # every file git tracks
    python scripts/scan_secrets.py FILE [FILE...] # these files (pre-commit passes them)
    python scripts/scan_secrets.py --history      # every line ever added, on every branch

It reports the file, line and kind of each finding, never the value itself.
A line that must contain a credential-shaped string (a documented example, a
test fixture) can be exempted with the marker ``scan-secrets: allow``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ALLOW_MARKER = "scan-secrets: allow"
MAX_BYTES = 2_000_000

# Files that hold credentials and must never be committed, whatever they contain.
FORBIDDEN_NAMES = {
    ".env", ".envrc", ".env.enterprise", ".env.local", "secrets.env",
    "chatgpt-auth.json", "credentials.json", "id_rsa", "id_ed25519",
}
FORBIDDEN_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".secrets.env")

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
    "private key block": re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    "JSON web token": re.compile(r"\beyJ[A-Za-z0-9_\-]{20,}\.eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}"),
}

# NAME=value lines (env files, shell, YAML) whose name says the value is secret.
ASSIGNMENT = re.compile(
    r"""^\s*(?:export\s+)?(?:-\s+)?["']?(?P<name>[A-Z][A-Z0-9_]*(?:API_?KEY|_KEY|_TOKEN|_SECRET|PASSWORD))["']?"""
    r"""\s*[:=]\s*["']?(?P<value>[^"'\s#]*)"""
)
PLACEHOLDER = re.compile(
    r"^(?:|\.\.\.|…|x+|\*+|<.*>|\$\{?.*|your[-_].*|changeme|change-me|placeholder|dummy|example.*|"
    r"test|fake.*|none|null|redacted|replace[-_]?me|todo|sk-\.\.\..*)$",
    re.IGNORECASE,
)


def _forbidden(path: str) -> str | None:
    name = Path(path).name
    if name in FORBIDDEN_NAMES or name.endswith(FORBIDDEN_SUFFIXES):
        return f"credential file '{name}' must not be committed"
    if name.startswith(".env.") and not name.endswith(".example"):
        return f"environment file '{name}' must not be committed (commit a .example copy instead)"
    return None


def scan_line(line: str) -> list[str]:
    """The kinds of secret found on one line (empty when clean or exempted)."""
    if ALLOW_MARKER in line:
        return []
    found = [kind for kind, pattern in KEY_PATTERNS.items() if pattern.search(line)]
    match = ASSIGNMENT.match(line)
    if match and len(match["value"]) >= 8 and not PLACEHOLDER.match(match["value"]):
        found.append(f"value assigned to {match['name']}")
    return found


def scan_text(text: str, label: str) -> list[str]:
    return [f"{label}:{number}: {kind}"
            for number, line in enumerate(text.splitlines(), 1)
            for kind in scan_line(line)]


def scan_file(path: str) -> list[str]:
    findings = []
    reason = _forbidden(path)
    if reason:
        findings.append(f"{path}: {reason}")
    file = Path(path)
    try:
        if not file.is_file() or file.stat().st_size > MAX_BYTES:
            return findings
        data = file.read_bytes()
    except OSError:
        return findings
    if b"\0" in data[:8192]:   # binary
        return findings
    return findings + scan_text(data.decode("utf-8", errors="replace"), path)


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout
    return [p for p in out.decode("utf-8", errors="replace").split("\0") if p]


def scan_history() -> list[str]:
    """Every line added in any commit on any ref, and every file name ever committed."""
    log = subprocess.run(
        ["git", "log", "--all", "-p", "--no-color", "--no-ext-diff", "--format=commit %h"],
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
                findings.append(f"{commit} {path}: {reason}")
        elif line.startswith("+") and not line.startswith("+++"):
            findings += [f"{commit} {path}: {kind}" for kind in scan_line(line[1:])]
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="files to scan (default: every tracked file)")
    parser.add_argument("--history", action="store_true", help="scan every line ever added instead")
    args = parser.parse_args(argv)

    if args.history:
        findings = scan_history()
    else:
        findings = [f for path in (args.paths or tracked_files()) for f in scan_file(path)]
    for finding in findings:
        print(finding)
    if findings:
        print(f"\n{len(findings)} possible secret(s) found. Remove them (and rotate any real key), "
              f"or mark a deliberate example with '{ALLOW_MARKER}'.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
