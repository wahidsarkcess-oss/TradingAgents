"""F1/F2 from the review of b1f75fd: checkpoint ownership and Windows-safe cleanup.

F1: a checkpointed run releases only what it acquired. A refused run (through the
public propagate() API or the CLI's run_analysis) must not release another run's
lock or close its checkpoint database.
F2: deleting a lock file the run still holds open fails on Windows; that failure
must never stop the lock and database being released, nor fail a successful run.
Windows deletion semantics are simulated here (a PermissionError, as WinError 32).
These tests use public APIs only, so they also run unchanged against b1f75fd.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import cli.run as cli_run
from tests.test_cli_memory_log import _FakeBuffer, _NullLive
from tests.test_graph_end_to_end import TRADE_DATE, offline  # noqa: F401
from tests.test_run_budget import MeteredModel, _graph
from tradingagents.graph import trading_graph
from tradingagents.graph.checkpointer import clear_all_checkpoints
from tradingagents.llm_clients.budget import LLMBudgetExceeded

ROOT = Path(__file__).resolve().parents[1]
LIMITS = {"max_llm_calls_per_run": 500}


def _checkpoint_graph(tmp_path, monkeypatch, **limits):
    return _graph(tmp_path, monkeypatch, MeteredModel(structured=True), checkpoint_enabled=True, **limits)


def _nvda_is_locked(tmp_path, monkeypatch) -> bool:
    """Whether a fresh graph (another runner) is refused NVDA's checkpoint."""
    probe = _checkpoint_graph(tmp_path, monkeypatch, **LIMITS)
    try:
        probe.begin_checkpoint("NVDA", TRADE_DATE)
    except LLMBudgetExceeded as exc:
        assert exc.scope == "resume_in_progress"
        return True
    probe.end_checkpoint()
    return False


def _lock_files(tmp_path):
    return sorted(p.name for p in (tmp_path / "cache" / "checkpoints").glob("*.lock"))


# --- F1: a refused run releases nothing of another's ------------------------------------


@pytest.mark.unit
def test_a_refused_propagate_leaves_the_active_runs_lock_and_database_working(tmp_path, monkeypatch,
                                                                              offline):  # noqa: F811
    one = _checkpoint_graph(tmp_path, monkeypatch, **LIMITS)
    nvda_thread = one.begin_checkpoint("NVDA", TRADE_DATE)          # NVDA's run is active
    with pytest.raises(LLMBudgetExceeded) as refused:
        one.propagate("AAPL", TRADE_DATE)                            # the public API, same graph
    assert refused.value.scope == "checkpoint_in_progress"

    assert _nvda_is_locked(tmp_path, monkeypatch)                    # NVDA's lock still held
    # NVDA's database is still open and its run can be carried to completion on it.
    state = one.create_run_state("NVDA", TRADE_DATE)
    final = one.graph.invoke(one.checkpoint_input(state),
                             {"configurable": {"thread_id": nvda_thread}, "recursion_limit": 100})
    assert final["final_trade_decision"]
    assert one.graph.checkpointer.get_tuple({"configurable": {"thread_id": nvda_thread}}) is not None

    one.end_checkpoint()                                              # NVDA's run ends...
    assert not _nvda_is_locked(tmp_path, monkeypatch)                # ...and only now is NVDA free


@pytest.mark.unit
def test_a_refused_propagate_on_another_graph_releases_nothing(tmp_path, monkeypatch, offline):  # noqa: F811
    holder = _checkpoint_graph(tmp_path, monkeypatch, **LIMITS)
    holder.begin_checkpoint("NVDA", TRADE_DATE)
    other = _checkpoint_graph(tmp_path, monkeypatch, **LIMITS)
    with pytest.raises(LLMBudgetExceeded) as refused:
        other.propagate("NVDA", TRADE_DATE)                           # same thread, second graph
    assert refused.value.scope == "resume_in_progress"
    assert _nvda_is_locked(tmp_path, monkeypatch)
    assert holder._checkpointer_ctx is not None                      # its database untouched
    holder.end_checkpoint()
    assert not _nvda_is_locked(tmp_path, monkeypatch)


def _run_cli(monkeypatch, tmp_path, graph):
    """Drive the CLI's run_analysis with ``graph`` as the analysis graph."""
    import cli.main as m
    from cli.models import AnalystType

    monkeypatch.setattr(cli_run, "TradingAgentsGraph", lambda *a, **k: graph)
    monkeypatch.setattr(cli_run, "message_buffer", _FakeBuffer())
    monkeypatch.setattr(cli_run, "create_layout", lambda: None)
    monkeypatch.setattr(cli_run, "update_display", lambda *a, **k: None)
    monkeypatch.setattr(cli_run, "Live", _NullLive)
    monkeypatch.setattr(cli_run, "get_user_selections", lambda flags=None: {
        "ticker": "NVDA", "analysis_date": TRADE_DATE, "analysts": [AnalystType.MARKET], "asset_type": "stock",
    })
    monkeypatch.setattr(cli_run, "_build_run_config", lambda selections, checkpoint: graph.config)
    monkeypatch.setattr(m.typer, "prompt", lambda *a, **k: "N")
    cli_run.run_analysis(checkpoint=True)


@pytest.mark.unit
def test_a_refused_cli_run_leaves_the_active_runs_lock_and_database_working(tmp_path, monkeypatch,
                                                                            offline):  # noqa: F811
    holder = _checkpoint_graph(tmp_path, monkeypatch, **LIMITS)
    holder_thread = holder.begin_checkpoint("NVDA", TRADE_DATE)       # another runner has NVDA
    cli_graph = _checkpoint_graph(tmp_path, monkeypatch, **LIMITS)
    with pytest.raises(LLMBudgetExceeded):
        _run_cli(monkeypatch, tmp_path, cli_graph)
    assert _nvda_is_locked(tmp_path, monkeypatch)
    assert holder.graph.checkpointer.get_tuple({"configurable": {"thread_id": holder_thread}}) is None
    holder.end_checkpoint()


@pytest.mark.unit
def test_a_successful_cli_checkpointed_run_releases_everything(tmp_path, monkeypatch, offline):  # noqa: F811
    graph = _checkpoint_graph(tmp_path, monkeypatch, **LIMITS)
    _run_cli(monkeypatch, tmp_path, graph)
    assert _lock_files(tmp_path) == []
    assert graph._checkpointer_ctx is None and not _nvda_is_locked(tmp_path, monkeypatch)


# --- F2: Windows deletion semantics -----------------------------------------------------


@pytest.fixture
def windows_unlink(monkeypatch):
    """Deleting a checkpoint lock file fails as it does on Windows while it is open."""
    real = pathlib.Path.unlink

    def unlink(self, missing_ok=False):
        if self.name.endswith(".usage.json.lock") and self.exists():
            raise PermissionError(32, "The process cannot access the file because it is being used")
        return real(self, missing_ok=missing_ok)

    monkeypatch.setattr(pathlib.Path, "unlink", unlink)


@pytest.mark.unit
@pytest.mark.parametrize("limits", [{}, LIMITS], ids=["limits-off", "limits-on"])
def test_a_successful_checkpointed_run_survives_windows_deletion_semantics(tmp_path, monkeypatch, offline,  # noqa: F811
                                                                           windows_unlink, limits):
    graph = _checkpoint_graph(tmp_path, monkeypatch, **limits)
    _, signal = graph.propagate("NVDA", TRADE_DATE)                   # no PermissionError escapes
    assert signal == "Overweight"
    assert graph._checkpointer_ctx is None                            # database closed
    assert not _nvda_is_locked(tmp_path, monkeypatch)                # lock released
    graph.begin_checkpoint("AAPL", TRADE_DATE)                       # the graph is not stuck "active"
    graph.end_checkpoint()


@pytest.mark.unit
def test_clear_checkpoints_survives_windows_deletion_semantics(tmp_path, windows_unlink):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "X.abc.usage.json.lock").write_text("")
    clear_all_checkpoints(tmp_path)                                   # no PermissionError escapes


@pytest.mark.unit
def test_the_clear_checkpoints_flag_survives_windows_deletion_semantics(tmp_path, monkeypatch, windows_unlink):
    from typer.testing import CliRunner

    import cli.main as m

    checkpoints = tmp_path / "cache" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "X.abc.usage.json.lock").write_text("")
    monkeypatch.setitem(m.DEFAULT_CONFIG, "data_cache_dir", str(tmp_path / "cache"))
    monkeypatch.setattr(m, "run_analysis", lambda **k: None)
    result = CliRunner().invoke(m.app, ["--clear-checkpoints"])
    assert result.exit_code == 0, result.output


@pytest.mark.unit
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process handling")
def test_clearing_never_deletes_a_live_processs_lock(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    lock_file = checkpoints / "NVDA.k.usage.json.lock"
    lock_file.write_text("")
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent("""
            import fcntl, sys, time
            handle = open(sys.argv[1], "a+b"); fcntl.flock(handle, fcntl.LOCK_EX)
            print("LOCKED", flush=True); time.sleep(120)
        """), str(lock_file)], stdout=subprocess.PIPE, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    try:
        assert holder.stdout.readline().strip() == "LOCKED"
        clear_all_checkpoints(tmp_path)
        assert lock_file.exists()
    finally:
        holder.kill()
        holder.wait(20)


@pytest.mark.unit
def test_a_cleanup_error_never_leaves_the_graph_marked_active(tmp_path, monkeypatch, offline):  # noqa: F811
    graph = _checkpoint_graph(tmp_path, monkeypatch, **LIMITS)
    graph.begin_checkpoint("NVDA", TRADE_DATE)
    original = trading_graph.CheckpointLock.release

    def release_then_fail(self, remove_file=False):
        original(self, remove_file)
        raise OSError("cleanup failed")

    monkeypatch.setattr(trading_graph.CheckpointLock, "release", release_then_fail)
    with pytest.raises(OSError, match="cleanup failed"):
        graph.end_checkpoint()
    monkeypatch.setattr(trading_graph.CheckpointLock, "release", original)
    assert graph._checkpointer_ctx is None                            # the database was still closed
    graph.begin_checkpoint("AAPL", TRADE_DATE)                       # not refused as "in progress"
    graph.end_checkpoint()
