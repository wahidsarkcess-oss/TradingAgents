"""F3/F4 from the review of b1f75fd.

F3: a lock taken on a lock file that was deleted and replaced between opening and
locking is not the lock: CheckpointLock checks the locked file is still the file
at its path (device and inode) and otherwise retries on the current file. The
exact interleaving from the review is reproduced deterministically. (File locks
remain unreliable on network filesystems; nothing here changes that.)
F4: the scanner normalises paths before deciding a file is under tests/.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tradingagents.graph.checkpointer import CheckpointLock, remove_stale_lock

ROOT = Path(__file__).resolve().parents[1]


def _interleave(lock: CheckpointLock, between_open_and_lock):
    """Run ``between_open_and_lock`` once, after ``lock`` opens its file and before
    it locks it: the window in which the review's race happens."""
    real_lock = CheckpointLock._lock
    fired = []

    def lock_after(handle):
        if not fired:
            fired.append(1)
            between_open_and_lock()
        return real_lock(handle)

    lock._lock = lock_after


@pytest.mark.unit
def test_runner_a_never_shares_the_lock_with_runner_b_after_a_clear(tmp_path):
    """The exact A / clearer / B interleaving from the review."""
    record = tmp_path / "NVDA.k.usage.json"
    Path(f"{record}.lock").write_text("")
    runner_a, runner_b = CheckpointLock(record), CheckpointLock(record)

    def clearer_then_b():
        assert remove_stale_lock(Path(f"{record}.lock"))    # no holder yet: the clearer deletes it
        assert runner_b.acquire()                            # B creates the new file and locks it

    _interleave(runner_a, clearer_then_b)
    assert runner_a.acquire() is False                       # A's lock on the old file is not taken as held
    assert runner_b.held and not runner_a.held
    runner_b.release()
    assert runner_a.acquire()                                # free once B lets go
    runner_a.release()


@pytest.mark.unit
def test_a_file_deleted_between_opening_and_locking_is_retaken_fresh(tmp_path):
    record = tmp_path / "AAPL.k.usage.json"
    lock_file = Path(f"{record}.lock")
    lock_file.write_text("")
    runner = CheckpointLock(record)
    _interleave(runner, lambda: lock_file.unlink())
    assert runner.acquire()                                   # retried on a fresh file
    assert runner._is_current(runner._handle) and lock_file.exists()
    other = CheckpointLock(record)
    assert other.acquire() is False                           # and it really excludes others
    runner.release()


@pytest.mark.unit
def test_a_held_lock_excludes_and_a_released_one_does_not(tmp_path):
    record = tmp_path / "X.k.usage.json"
    first, second = CheckpointLock(record), CheckpointLock(record)
    assert first.acquire() and not second.acquire()
    first.release(remove_file=True)
    assert not Path(f"{record}.lock").exists()
    assert second.acquire()
    second.release()


# --- F4: scanner path normalisation ----------------------------------------------------


def _scanner():
    spec = importlib.util.spec_from_file_location("scan_secrets_f4", ROOT / "scripts" / "scan_secrets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
@pytest.mark.parametrize("path, exempt", [
    ("tests/test_a.py", True),
    ("./tests/test_a.py", True),
    ("tests/sub/fixture.py", True),
    ("tests/../tests/test_a.py", True),
    ("tests/../tradingagents/creds.py", False),
    ("tests/sub/../../tradingagents/creds.py", False),
    ("tradingagents/tests/creds.py", False),
    ("deploy/tests/settings.py", False),
    ("../tests/test_a.py", False),
    ("tests", False),
])
def test_only_the_top_level_tests_folder_is_exempt_after_normalisation(path, exempt):
    line = "client = make('" + "sk-" + "proj-" + "Q1w2" * 10 + "')  # scan-secrets: allow"
    assert (_scanner()._findings_on(line, path, 1)[0].exempt == "inline marker") is exempt


@pytest.mark.unit
def test_absolute_paths_are_normalised_too(tmp_path, monkeypatch):
    scan = _scanner()
    (tmp_path / "repo" / "tests").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "repo")
    assert scan.in_tests(str(tmp_path / "repo" / "tests" / "a.py"))
    assert not scan.in_tests(str(tmp_path / "repo" / "tests" / ".." / "app" / "creds.py"))
