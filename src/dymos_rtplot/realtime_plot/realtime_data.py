"""Shared recorder and optimizer-history access for realtime dashboard readers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

try:
    from pyoptsparse.pyOpt_history import History
except Exception:  # pragma: no cover
    History = None


_HISTORY_EXCLUDED_KEYS = {
    "funcs",
    "funcsSens",
    "xuser",
    "isMajor",
    "fail",
}


@contextmanager
def readonly_sqlite_connection(path, timeout=1.0):
    """Open an SQLite recorder in read-only URI mode.

    OpenMDAO writes recorder rows while Bokeh reads them. Read-only URI mode avoids
    accidental writer locks and lets refresh callbacks fail quickly when a row is
    still being committed.
    """
    # Use os.path.abspath() instead of Path.resolve() to avoid Windows
    # extended-length path prefix (\\?\) which as_uri() encodes as %3F,
    # producing "file://%3F/C%3A/..." — an invalid URI that SQLite rejects.
    db_uri = Path(os.path.abspath(str(path))).as_uri() + "?mode=ro"
    con = sqlite3.connect(db_uri, uri=True, timeout=timeout)
    try:
        yield con
    finally:
        con.close()


def _retry(operation, attempts=2, delay=0.05):
    last_error = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(delay)
    raise last_error


class CaseRecorderAccess:
    """Read driver cases from a live OpenMDAO SQLite recorder."""

    def __init__(self, case_recorder_filename, case_reader_factory):
        self.case_recorder_filename = str(case_recorder_filename)
        self._case_reader_factory = case_reader_factory
        self._case_reader = None

    def get_case_reader(self):
        """Return a lazily opened OpenMDAO case reader."""
        if self._case_reader is None:
            self._case_reader = _retry(lambda: self._case_reader_factory(self.case_recorder_filename))
        return self._case_reader

    def reset_case_reader(self):
        """Drop the current case reader after a transient read failure."""
        self._case_reader = None

    def is_driver_optimizer(self):
        reader = self.get_case_reader()
        return reader.problem_metadata["driver"]["supports"]["optimization"]["val"]

    def driver_iteration_row(self, counter):
        """Return the raw driver iteration row for a counter, or None."""
        def _read_row():
            with readonly_sqlite_connection(self.case_recorder_filename) as con:
                con.row_factory = sqlite3.Row
                cur = con.cursor()
                cur.execute(
                    "SELECT * FROM driver_iterations WHERE counter=:counter",
                    {"counter": counter},
                )
                return cur.fetchone()

        return _retry(_read_row)

    def get_case_by_counter(self, counter):
        """Return a driver case by counter without treating missing rows as errors."""
        row = self.driver_iteration_row(counter)
        if row is None:
            return None
        try:
            return self.get_case_reader()._driver_cases.get_case(row["iteration_coordinate"])
        except Exception:
            self.reset_case_reader()
            raise


@dataclass
class HistorySnapshot:
    """A point-in-time read of pyOptSparse major-iteration history."""

    entries: list[dict]
    keys: list[str]
    mtime: float | None
    warning: str | None = None


class OptimizerHistoryAccess:
    """Read pyOptSparse history files through a temporary snapshot copy."""

    def __init__(self, hist_file=None, history_factory=None):
        self.hist_file = hist_file
        self._history_factory = History if history_factory is None else history_factory

    def read_if_changed(self, previous_mtime=None):
        """Return a new history snapshot, or None when the file is unchanged."""
        if not self.hist_file or self._history_factory is None:
            return None

        hist_path = Path(self.hist_file)
        if not hist_path.exists():
            return None

        try:
            mtime = hist_path.stat().st_mtime
        except OSError as exc:
            return HistorySnapshot([], [], None, f"Unable to stat optimizer history: {exc}")
        if previous_mtime is not None and mtime == previous_mtime:
            return None

        try:
            return _retry(lambda: self._read_snapshot_copy(hist_path, mtime))
        except Exception as exc:
            return HistorySnapshot([], [], previous_mtime, f"Unable to read optimizer history: {exc}")

    def _read_snapshot_copy(self, hist_path, mtime):
        with tempfile.NamedTemporaryFile(prefix="dymos_rtplot_", suffix=hist_path.suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            # Use SQLite's online backup API instead of shutil.copy2.
            #
            # pyOptSparse 2.x stores history as a SQLite database.  A plain
            # shutil.copy2() of a live SQLite file can capture the main .db
            # file in mid-commit without the companion -journal, producing a
            # copy that raises "database disk image is malformed" when opened.
            # This is especially likely with IPOPT, whose line-search writes a
            # new row on every function evaluation — keeping the writer active
            # most of the time.
            #
            # sqlite3.Connection.backup() uses SQLite's cooperative locking:
            # it acquires a SHARED lock, copies all pages atomically, and retries
            # any pages changed by a concurrent writer.  The result is always a
            # self-consistent snapshot, even while the writer is active.
            #
            # If the file is not a SQLite database (legacy HDF5 .hst from
            # pyOptSparse 1.x), sqlite3.connect() will raise DatabaseError
            # ("file is not a database"), which is caught by the outer _retry()
            # and surfaced as a warning — the same behaviour as before.
            try:
                # NOTE: sqlite3.connect() used as a context manager only manages
                # transactions (commit/rollback); it does NOT close the connection.
                # Explicit close() calls are required to release file handles on
                # Windows — otherwise the file stays locked until GC.
                src_con = sqlite3.connect(str(hist_path), timeout=1.0)
                try:
                    dst_con = sqlite3.connect(str(tmp_path))
                    try:
                        src_con.backup(dst_con, pages=-1)
                    finally:
                        dst_con.close()
                finally:
                    src_con.close()
            except sqlite3.DatabaseError:
                # Not a SQLite database — fall back to a best-effort file copy
                # (preserves compatibility with HDF5-based legacy .hst files).
                shutil.copy2(hist_path, tmp_path)
            hist = self._history_factory(str(tmp_path), flag="r")
            try:
                iter_keys = [
                    key for key in hist.getIterKeys()
                    if key not in _HISTORY_EXCLUDED_KEYS
                ]
                values = hist.getValues(names=iter_keys, major=True, scale=False) if iter_keys else {}
            finally:
                close = getattr(hist, "close", None)
                if close is not None:
                    close()

            entries = []
            num_rows = max((value.shape[0] for value in values.values()), default=0)
            for row_idx in range(num_rows):
                entry = {}
                for key in iter_keys:
                    entry[key] = values[key][row_idx]
                entries.append(entry)
            return HistorySnapshot(entries, iter_keys, mtime)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
