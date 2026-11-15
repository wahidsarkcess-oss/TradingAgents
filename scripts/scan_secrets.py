#!/usr/bin/env python3
"""Fail when a file about to be committed (or already tracked) holds a secret.

Standard library only, so it runs before anything is installed: as a pre-commit
hook (``.pre-commit-config.yaml``), in CI, or by hand.

    python scripts/scan_secrets.py                # every file git tracks
    python scripts/scan_secrets.py FILE [FILE...] # these files (pre-commit passes them)
    python scripts/scan_secrets.py --history      # every line ever added, on every branch

It reports the file, line and kind of each finding, never the value itself, and
lists the files it skipped as too large to scan. ``--history`` covers only the
commits present locally: in a shallow clone (CI's default checkout) it warns
that older history was not scanned; run it in a full clone for a complete pass.
It is a heuristic first line, not a guarantee: a credential with no recognisable
shape outside a NAME=value or key: "value" assignment is not found.
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
# Reviewed false positives that cannot carry the marker (lines already in history):
# one finding per line, exactly as printed. It holds locations, never values.
ALLOWLIST_FILE = ".scan-secrets-allow"
MAX_BYTES = 2_000_000

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


def _looks_like_a_secret(value: str) -> bool:
    """A literal worth flagging: not a placeholder, and mixing letters with digits."""
    return (not PLACEHOLDER.match(value) and not PLACEHOLDER_WORD.search(value)
            and any(c.isdigit() for c in value) and any(c.isalpha() for c in value))


def _forbidden(path: str) -> str | None:
    name = Path(path).name
    if (name in FORBIDDEN_NAMES or name.endswith(FORBIDDEN_SUFFIXES)
            or any(p.search(name) for p in FORBIDDEN_PATTERNS)):
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
    elif not match:
        quoted = QUOTED_ASSIGNMENT.search(line)
        if quoted and _looks_like_a_secret(quoted["value"]):
            found.append(f"literal assigned to {quoted['name']}")
    url = URL_CREDENTIALS.search(line)
    if url and not PLACEHOLDER_WORD.search(url["password"]):
        found.append("credentials in a URL")
    return found


def scan_text(text: str, label: str) -> list[str]:
    return [f"{label}:{number}: {kind}"
            for number, line in enumerate(text.splitlines(), 1)
            for kind in scan_line(line)]


SKIPPED: list[str] = []   # files too large to scan, reported by main()


def scan_file(path: str) -> list[str]:
    findings = []
    reason = _forbidden(path)
    if reason:
        findings.append(f"{path}: {reason}")
    file = Path(path)
    try:
        if not file.is_file():
            return findings
        if file.stat().st_size > MAX_BYTES:
            SKIPPED.append(path)
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


def is_shallow() -> bool:
    out = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], capture_output=True, text=True)
    return out.stdout.strip() == "true"


def scan_history() -> list[str]:
    """Every line added in any commit on any ref (merge resolutions included, via -m),
    and every file name ever committed."""
    log = subprocess.run(
        ["git", "log", "--all", "-p", "-m", "--no-color", "--no-ext-diff", "--format=commit %h"],
        capture_output=True, check=True,
    ).stdout.decode("utf-8", errors="replace")
    findings, commit, path = [], "?", "?"
    seen: set[str] = set()
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
    # -m repeats a merge's diff once per parent.
    return [f for f in findings if not (f in seen or seen.add(f))]


def _allowlist() -> set[str]:
    try:
        lines = Path(ALLOWLIST_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    return {line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")}


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
    findings = [f for f in findings if f not in _allowlist()]
    for finding in findings:
        print(finding)
    for path in SKIPPED:
        print(f"warning: {path}: skipped, larger than {MAX_BYTES:,} bytes; check it by other means",
              file=sys.stderr)
    if findings:
        print(f"\n{len(findings)} possible secret(s) found. Remove them (and rotate any real key), "
              f"or mark a deliberate example with '{ALLOW_MARKER}'.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
