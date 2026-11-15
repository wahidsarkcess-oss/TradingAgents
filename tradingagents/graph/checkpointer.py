"""LangGraph checkpoint support for resumable analysis runs.

Per-ticker SQLite databases so concurrent tickers don't contend.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from tradingagents.dataflows.symbols import safe_ticker_component


def _db_path(data_dir: str | Path, ticker: str) -> Path:
    """Return the SQLite checkpoint DB path for a ticker."""
    # Reject ticker values that would escape the checkpoints directory.
    safe = safe_ticker_component(ticker).upper()
    p = Path(data_dir) / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{safe}.db"


def thread_id(ticker: str, date: str, signature: str = "") -> str:
    """Deterministic thread ID for a ticker+date pair.

    ``signature`` folds in graph-shape-affecting run choices so a resume under a
    different graph can't reuse this checkpoint (#1089); omitting it keeps the
    legacy ID.
    """
    base = f"{ticker.upper()}:{date}"
    if signature:
        base = f"{base}:{signature}"
    return hashlib.sha256(base.encode()).hexdigest()[:16]


def usage_path(data_dir: str | Path, ticker: str, date: str, signature: str = "") -> Path:
    """Where a checkpointed analysis keeps its LLM usage, so a resume continues it."""
    safe = safe_ticker_component(ticker).upper()
    p = Path(data_dir) / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{safe}.{thread_id(ticker, date, signature)}.usage.json"


@contextmanager
def get_checkpointer(data_dir: str | Path, ticker: str) -> Generator[SqliteSaver, None, None]:
    """Context manager yielding a SqliteSaver backed by a per-ticker DB."""
    db = _db_path(data_dir, ticker)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
        conn.close()


def checkpoint_step(data_dir: str | Path, ticker: str, date: str, signature: str = "") -> int | None:
    """Return the step number of the latest checkpoint, or None if none exists."""
    db = _db_path(data_dir, ticker)
    if not db.exists():
        return None
    tid = thread_id(ticker, date, signature)
    with get_checkpointer(data_dir, ticker) as saver:
        config = {"configurable": {"thread_id": tid}}
        cp = saver.get_tuple(config)
        if cp is None:
            return None
        return cp.metadata.get("step")


def clear_all_checkpoints(data_dir: str | Path) -> int:
    """Remove all checkpoint databases. Returns the number of databases deleted.

    SQLite keeps committed state in ``-wal`` and ``-shm`` files beside the
    database, so deleting only the ``.db`` leaves a cleared checkpoint with data
    still on disk.
    """
    cp_dir = Path(data_dir) / "checkpoints"
    if not cp_dir.exists():
        return 0
    dbs = list(cp_dir.glob("*.db"))
    for db in dbs:
        for path in (db, *cp_dir.glob(f"{db.name}-*")):
            path.unlink(missing_ok=True)
    # The usage records belong to the checkpoints just removed.
    for record in cp_dir.glob("*.usage.json"):
        record.unlink(missing_ok=True)
    for lock_file in cp_dir.glob("*.usage.json.lock"):
        remove_stale_lock(lock_file)
    return len(dbs)


def remove_stale_lock(lock_file: Path) -> bool:
    """Delete a checkpoint's lock file if no process holds it; True when removed.

    The file is deleted while this process holds its lock, so no other holder
    exists at that moment; a run that opened the old file just before is told by
    its own check (TradingAgentsGraph._lock_usage_record) that it vanished.
    """
    from tradingagents.dataflows.files import locked

    record = str(lock_file)[: -len(".lock")]
    holder = locked(record, wait=False)
    try:
        if not holder.__enter__():
            return False                 # an active run holds it: leave it
        lock_file.unlink(missing_ok=True)
        return True
    finally:
        holder.__exit__(None, None, None)


def clear_checkpoint(data_dir: str | Path, ticker: str, date: str, signature: str = "") -> None:
    """Remove checkpoint for a specific ticker+date by deleting the thread's rows."""
    usage_path(data_dir, ticker, date, signature).unlink(missing_ok=True)
    db = _db_path(data_dir, ticker)
    if not db.exists():
        return
    tid = thread_id(ticker, date, signature)
    conn = sqlite3.connect(str(db))
    try:
        for table in ("writes", "checkpoints"):
            conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (tid,))
        conn.commit()
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()
