import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from dymos_rtplot.realtime_plot.realtime_broker import LiveDataBroker
from dymos_rtplot.realtime_plot.realtime_data import HistorySnapshot, OptimizerHistoryAccess


class _FakeCase:
    counter = 1
    timestamp = 1.0
    derivatives = None

    def get_objectives(self, scaled=False):
        return {"obj": np.array([10.0 if scaled else 1.0])}

    def get_design_vars(self, scaled=False):
        return {"dv": np.array([20.0 if scaled else 2.0])}

    def get_constraints(self, scaled=False):
        return {"con": np.array([30.0 if scaled else 3.0])}


class _FakeTracker:
    def __init__(self, fail_first=False):
        self.fail_first = fail_first
        self.calls = 0

    def get_case_by_counter(self, counter):
        self.calls += 1
        if self.fail_first:
            self.fail_first = False
            raise RuntimeError("locked")
        if counter == 1:
            return _FakeCase()
        return None

    def is_source_process_running(self):
        return True


class _FakeHistoryAccess:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def read_if_changed(self, previous_mtime=None):
        return self.snapshot


class RealtimeDataTests(unittest.TestCase):
    def test_broker_retries_case_row_read_on_next_poll(self):
        tracker = _FakeTracker(fail_first=True)
        broker = LiveDataBroker(tracker)

        self.assertEqual(broker.poll(), [])
        self.assertEqual(len(broker.poll()), 1)
        self.assertEqual(broker.latest_snapshot().counter, 1)

    def test_broker_uses_injected_history_snapshot(self):
        history = HistorySnapshot(
            entries=[{"iter": np.array([3.0]), "metric": np.array([4.0])}],
            keys=["iter", "metric"],
            mtime=10.0,
        )
        broker = LiveDataBroker(_FakeTracker(), history_access=_FakeHistoryAccess(history))

        broker.poll()

        self.assertEqual(broker.get_history_keys(), ["iter", "metric"])
        self.assertEqual(len(broker.get_history_entries()), 1)
        self.assertEqual(float(broker.latest_snapshot().opt_history["metric"][0]), 4.0)


class _FakeHistoryResult:
    """Minimal stand-in for pyOptSparse History that reads a SQLite .hst copy."""

    def __init__(self, filename, flag="r"):
        self._con = sqlite3.connect(filename)

    def getIterKeys(self):
        cur = self._con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        return [row[0] for row in cur.fetchall() if row[0] != "sqlite_sequence"]

    def getValues(self, names, major=True, scale=False):
        out = {}
        for name in names:
            try:
                cur = self._con.execute(f"SELECT val FROM {name} ORDER BY rowid")  # noqa
                out[name] = np.array([r[0] for r in cur.fetchall()])
            except sqlite3.OperationalError:
                out[name] = np.array([])
        return out

    def close(self):
        self._con.close()


def _make_fake_hst(path, n_rows=5):
    """Create a minimal SQLite .hst file with n_rows of data."""
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE obj (val REAL)")
    for i in range(n_rows):
        con.execute("INSERT INTO obj VALUES (?)", (float(i),))
    con.commit()
    con.close()


class HistoryBackupTests(unittest.TestCase):
    """Tests for the sqlite3.backup() fix in OptimizerHistoryAccess._read_snapshot_copy.

    These tests use a fake history factory so pyoptsparse is not required.
    """

    def test_read_snapshot_sqlite_happy_path(self):
        """backup() successfully copies a valid SQLite .hst and reads it back."""
        with tempfile.TemporaryDirectory() as td:
            hist_path = Path(td) / "test.hst"
            _make_fake_hst(hist_path, n_rows=10)

            access = OptimizerHistoryAccess(
                hist_file=str(hist_path),
                history_factory=_FakeHistoryResult,
            )
            snapshot = access.read_if_changed(previous_mtime=None)

        self.assertIsNotNone(snapshot)
        self.assertIsNone(snapshot.warning)
        self.assertIn("obj", snapshot.keys)
        self.assertEqual(len(snapshot.entries), 10)
        self.assertAlmostEqual(float(snapshot.entries[3]["obj"]), 3.0)

    def test_read_snapshot_non_sqlite_falls_back_to_copy(self):
        """A non-SQLite file triggers the shutil.copy2 fallback (or a warning)."""
        with tempfile.TemporaryDirectory() as td:
            hist_path = Path(td) / "legacy.hst"
            # Write HDF5-style magic bytes (not a SQLite file)
            hist_path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 512)

            access = OptimizerHistoryAccess(
                hist_file=str(hist_path),
                history_factory=_FakeHistoryResult,
            )
            snapshot = access.read_if_changed(previous_mtime=None)

        # Should return a warning snapshot rather than raising
        self.assertIsNotNone(snapshot)
        self.assertIsNotNone(snapshot.warning)

    def test_lock_timeout_on_sqlite_returns_warning_not_corrupt_copy(self):
        """A backup() lock-timeout must propagate as a warning, not silently fall
        back to shutil.copy2() on the live file.

        Root cause: sqlite3.OperationalError IS a subclass of sqlite3.DatabaseError.
        The old code caught DatabaseError and fell back to shutil.copy2(), so a
        transient 'database is locked' timeout would copy a potentially mid-write
        database — producing 'database disk image is malformed'.

        The fix detects SQLite vs non-SQLite by file magic BEFORE entering the
        backup path, so OperationalError can never reach the shutil.copy2 fallback.
        """
        import dymos_rtplot.realtime_plot.realtime_data as _rrd_mod

        with tempfile.TemporaryDirectory() as td:
            hist_path = Path(td) / "locked.hst"
            _make_fake_hst(hist_path, n_rows=5)

            access = OptimizerHistoryAccess(
                hist_file=str(hist_path),
                history_factory=_FakeHistoryResult,
            )

            # sqlite3.Connection is a C type whose methods cannot be patched directly.
            # Instead, patch the sqlite3 module name inside realtime_data so every
            # sqlite3.connect() call returns a mock whose backup() raises
            # OperationalError.  We keep the real exception classes so the
            # propagation path works correctly.
            real_sqlite3 = _rrd_mod.sqlite3

            def _make_locked_con(path, **kw):
                m = MagicMock()
                m.backup.side_effect = real_sqlite3.OperationalError("database is locked")
                return m

            mock_sqlite3 = MagicMock(wraps=real_sqlite3)
            mock_sqlite3.connect.side_effect = _make_locked_con

            with patch.object(_rrd_mod, "sqlite3", mock_sqlite3):
                snapshot = access.read_if_changed(previous_mtime=None)

        # Must surface as a warning — NOT silently fall back to a racy file copy
        self.assertIsNotNone(snapshot)
        self.assertIsNotNone(snapshot.warning)
        self.assertIn("locked", snapshot.warning.lower())

    def test_read_snapshot_concurrent_writer_no_corruption(self):
        """backup() survives concurrent writes without producing a corrupt copy.

        This test verifies the fix is correct in principle: even with a writer
        actively inserting rows, backup() always produces a valid snapshot.
        On fast local SSDs the race may not visibly trigger with shutil.copy2
        either, but backup() is correct by design whereas shutil.copy2 is not.
        """
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            hist_path = Path(td) / "live.hst"
            _make_fake_hst(hist_path, n_rows=1)

            stop = threading.Event()

            def _writer():
                con = sqlite3.connect(str(hist_path))
                i = 100
                while not stop.is_set():
                    con.execute("INSERT INTO obj VALUES (?)", (float(i),))
                    con.commit()
                    i += 1
                    time.sleep(0.001)
                con.close()

            wt = threading.Thread(target=_writer, daemon=True)
            wt.start()
            time.sleep(0.01)  # let the writer start

            errors = []
            access = OptimizerHistoryAccess(
                hist_file=str(hist_path),
                history_factory=_FakeHistoryResult,
            )
            for _ in range(30):
                snapshot = access.read_if_changed(previous_mtime=None)
                if snapshot is not None and snapshot.warning:
                    errors.append(snapshot.warning)
                # Reset mtime so every call re-reads
                access._history_mtime = None
                time.sleep(0.003)

            stop.set()
            wt.join(timeout=2)

        self.assertEqual(
            errors, [],
            msg=f"backup() produced errors during concurrent writes: {errors}",
        )


if __name__ == "__main__":
    unittest.main()
