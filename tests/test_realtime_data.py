import unittest

import numpy as np

from dymos_rtplot.realtime_plot.realtime_broker import LiveDataBroker
from dymos_rtplot.realtime_plot.realtime_data import HistorySnapshot


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


if __name__ == "__main__":
    unittest.main()
