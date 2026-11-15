"""LangGraph checkpoint support for resumable analysis runs.

Per-ticker SQLite databases so concurrent tickers don't contend.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from tradingagents.dataflows.symbols import safe_ticker_component

logger = logging.getLogger(__name__)


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


class CheckpointLock:
    """An exclusive, non-blocking lock on a checkpoint's ``<record>.lock`` file.

    Held by one run at a time, across threads, graph objects and processes on
    one host. After locking, the locked file is checked to still be the file at
    the path (same device and inode where the platform reports them): a lock
    taken on a file that was deleted and replaced in between (a concurrent
    ``--clear-checkpoints``) would be a lock no one else sees, so it is released
    and taken again on the current file. Like any file lock, it is not reliable
    on network filesystems.
    """

    attempts = 3

    def __init__(self, record: str | Path):
        self.path = Path(f"{record}.lock")
        self._handle = None
        self._hold = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def _open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return open(self.path, "a+b")

    @staticmethod
    def _lock(handle):
        """Enter a non-blocking hold on ``handle``; (acquired, hold)."""
        import os

        from tradingagents.dataflows.files import _hold_posix, _hold_windows

        hold = (_hold_windows if os.name == "nt" else _hold_posix)(handle, False)
        return hold.__enter__(), hold

    def _is_current(self, handle) -> bool:
        import os

        try:
            path_stat = os.stat(self.path)
        except FileNotFoundError:
            return False
        held = os.fstat(handle.fileno())
        if held.st_ino and path_stat.st_ino:
            return (held.st_dev, held.st_ino) == (path_stat.st_dev, path_stat.st_ino)
        return True     # the platform reports no file identity: existence is all we can check

    def acquire(self) -> bool:
        """Take the lock; False when another holder has it."""
        for _ in range(self.attempts):
            handle = self._open()
            acquired, hold = self._lock(handle)
            if not acquired:
                hold.__exit__(None, None, None)
                handle.close()
                return False
            if self._is_current(handle):
                self._handle, self._hold = handle, hold
                return True
            # Replaced while we took it: let go of the orphan and try the current file.
            try:
                hold.__exit__(None, None, None)
            finally:
                handle.close()
        return False

    def release(self, remove_file: bool = False) -> None:
        """Release the lock; with ``remove_file``, first try to delete the lock file.

        The file is deleted while still held, so no other holder exists at that
        moment. Where the platform refuses to delete an open file (Windows), the
        file is simply left: a leftover lock file is harmless, and the lock is
        released regardless.
        """
        if self._handle is None:
            return
        try:
            if remove_file:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:          # e.g. Windows: the file is open
                    logger.debug("Left checkpoint lock file %s in place (%s)", self.path.name,
                                 type(exc).__name__)
        finally:
            try:
                self._hold.__exit__(None, None, None)
            finally:
                self._handle.close()
                self._handle = self._hold = None


def remove_stale_lock(lock_file: Path) -> bool:
    """Delete a checkpoint's lock file if no process holds it; True when it is gone.

    Never touches a lock a live run holds, and never raises because a file could
    not be deleted.
    """
    lock = CheckpointLock(str(lock_file)[: -len(".lock")])
    try:
        if not lock.acquire():
            return False                 # an active run holds it: leave it
    except OSError as exc:
        logger.debug("Could not inspect checkpoint lock file %s (%s)", lock_file.name, type(exc).__name__)
        return False
    lock.release(remove_file=True)
    return not lock_file.exists()


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
