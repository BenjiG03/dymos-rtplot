import unittest
from unittest import mock

import numpy as np

from dymos_rtplot.realtime_plot import realtime_dashboard


class _FakeDoc:
    def __init__(self):
        self.roots = []
        self.callbacks = []
        self.title = None

    def add_root(self, root):
        self.roots.append(root)

    def add_periodic_callback(self, callback, period):
        self.callbacks.append((callback, period))


class _FakeCaseTracker:
    def __init__(self, recorder_filename="cases.sql"):
        self.recorder_filename = recorder_filename

    def get_case_recorder_filename(self):
        return self.recorder_filename

    def set_source_process_pid(self, pid):
        self.pid = pid


class _FakeBroker:
    def __init__(self, *args, **kwargs):
        self.snapshots = [mock.Mock()]
        self.latest = self.snapshots[0]
        self.poll_calls = 0

    def poll(self):
        self.poll_calls += 1
        return self.snapshots

    def latest_snapshot(self):
        return self.latest

    def is_running(self):
        return True


class _FakeTab:
    def __init__(self):
        self.layout = "case-layout"
        self.refresh = mock.Mock()
        self._update_wrapped_in_try = mock.Mock()
        self.panel = mock.Mock(child="child")


class _FakeTabs:
    def __init__(self, tabs, sizing_mode=None):
        self.tabs = tabs
        self.sizing_mode = sizing_mode
        self.active = 0

    def on_change(self, attr, callback):
        self.callback = callback


class DashboardLifecycleTests(unittest.TestCase):
    def test_dashboard_bootstraps_existing_data_for_new_session(self):
        tabs = {
            realtime_dashboard.CASE_PLOTTER_TAB: _FakeTab(),
            realtime_dashboard.TRAJECTORY_TAB: _FakeTab(),
            realtime_dashboard.SERIES_TAB: _FakeTab(),
            realtime_dashboard.JACOBIAN_ENTRIES_TAB: _FakeTab(),
            realtime_dashboard.JACOBIAN_HEATMAP_TAB: _FakeTab(),
        }

        def build_tab(tab_name, *args, **kwargs):
            tab = tabs[tab_name]
            panel = mock.Mock(child=tab.layout, title=tab_name)
            return tab, panel

        with mock.patch("dymos_rtplot.realtime_plot.realtime_dashboard.LiveDataBroker", _FakeBroker), \
             mock.patch("dymos_rtplot.realtime_plot.realtime_dashboard._build_dashboard_tab", side_effect=build_tab), \
             mock.patch("dymos_rtplot.realtime_plot.realtime_dashboard.Tabs", _FakeTabs):
            realtime_dashboard._RealTimeDymosDashboard(
                _FakeCaseTracker(),
                callback_period=300,
                doc=_FakeDoc(),
                pid_of_calling_script=123,
                script="demo.py",
            )

        tabs[realtime_dashboard.CASE_PLOTTER_TAB]._update_wrapped_in_try.assert_called()
        tabs[realtime_dashboard.TRAJECTORY_TAB].refresh.assert_called()
        tabs[realtime_dashboard.SERIES_TAB].refresh.assert_called()

    def test_polynomial_control_uses_recorded_timeseries(self):
        broker = mock.Mock(metadata={"trajectories": []})
        tab = realtime_dashboard._TrajectoryTab(broker)
        case = mock.Mock()
        case.get_val.side_effect = lambda path: {
            "traj.phase0.timeseries.time": np.array([[0.0], [1.0], [2.0]]),
            "traj.phase0.timeseries.controls:pc": np.array([[1.0], [2.0], [3.0]]),
        }[path]
        phase_meta = {
            "promoted_path": "traj.phase0",
            "controls": {
                "pc": {
                    "shape": [1],
                    "lower": None,
                    "upper": None,
                    "control_type": "polynomial",
                    "path": "traj.phase0.polynomial_controls:pc",
                }
            },
            "timeseries_outputs": {
                "controls:pc": {
                    "path": "timeseries.controls:pc",
                    "name": "pc",
                    "units": "m",
                    "category": "control",
                }
            },
            "path_constraints": [],
            "boundary_constraints": {"initial": [], "final": []},
        }

        xvals, yvals, violation, warning = tab._control_trace(case, phase_meta, "pc")

        self.assertIsNone(warning)
        self.assertEqual(list(xvals), [0.0, 1.0, 2.0])
        self.assertEqual(list(yvals), [1.0, 2.0, 3.0])
        self.assertFalse(np.any(violation))


if __name__ == "__main__":
    unittest.main()
