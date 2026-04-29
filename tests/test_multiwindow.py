import unittest
from unittest import mock
import queue as std_queue

from bokeh.models import Div

from dymos_rtplot import rtplot
from dymos_rtplot.realtime_plot import realtime_dashboard, realtime_plot


class _FakeCaseTracker:
    def __init__(self, is_optimizer):
        self._is_optimizer = is_optimizer

    def is_driver_optimizer(self):
        return self._is_optimizer


class _FakeBroker:
    def __init__(self, metadata=None):
        self.metadata = metadata or {}
        self.snapshots = []

    def latest_snapshot(self):
        return self.snapshots[-1] if self.snapshots else None

    def get_history_keys(self):
        return []

    def get_history_warning(self):
        return None

    def get_series(self, group, name):
        return []


class _FakeSnapshot:
    def __init__(self):
        self.major_iteration = 1
        self.counter = 1
        self.scaled_objs = {"obj": 1.0}
        self.scaled_desvars = {"dv": 2.0}
        self.scaled_cons = {"con": 3.0}
        self.derivatives = {("of", "wrt"): [[1.0, 0.0], [0.0, 2.0]]}


class _FakeQueue:
    def __init__(self, messages=None):
        self._messages = list(messages or [])
        self.put_messages = []

    def get(self, timeout=None):
        if self._messages:
            return self._messages.pop(0)
        raise std_queue.Empty

    def put(self, message):
        self.put_messages.append(message)


class _FakeProcess:
    def __init__(self, target=None, args=(), alive_after_start=True):
        self.target = target
        self.args = args
        self.started = False
        self.terminated = False
        self.join_calls = 0
        self._alive = False
        self._alive_after_start = alive_after_start

    def start(self):
        self.started = True
        self._alive = self._alive_after_start

    def join(self, timeout=None):
        self.join_calls += 1
        self._alive = False

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        self._alive = False


class _FakeContext:
    def __init__(self, queue_messages, alive_after_start=True):
        self.queue_obj = _FakeQueue(queue_messages)
        self.processes = []
        self._alive_after_start = alive_after_start

    def Queue(self):
        return self.queue_obj

    def Process(self, target=None, args=()):
        proc = _FakeProcess(target=target, args=args, alive_after_start=self._alive_after_start)
        self.processes.append(proc)
        return proc


class _FakeDoc:
    def __init__(self):
        self.roots = []
        self.callbacks = []
        self.title = None

    def add_root(self, root):
        self.roots.append(root)

    def add_periodic_callback(self, callback, period):
        self.callbacks.append((callback, period))


class _FakeKernel32:
    def __init__(self):
        self.get_current_process_called = False
        self.affinity_calls = []
        self.GetCurrentProcess = mock.Mock(side_effect=self._get_current_process)
        self.SetProcessAffinityMask = mock.Mock(side_effect=self._set_process_affinity_mask)

    def _get_current_process(self):
        self.get_current_process_called = True
        return 1234

    def _set_process_affinity_mask(self, handle, mask_value):
        self.affinity_calls.append((handle, mask_value))
        return 1


class MultiWindowParsingTests(unittest.TestCase):
    def test_main_routes_direct_entrypoint_args_to_rtplot(self):
        with mock.patch("dymos_rtplot.rtplot._rtplot_cmd") as rtplot_cmd, \
             mock.patch("dymos_rtplot.rtplot._realtime_plot_cmd") as realtime_cmd:
            rtplot.main(
                [
                    "--dashboard-mode",
                    "multiwindow",
                    "--tabs",
                    "case-plotter,trajectory",
                    "example.py",
                    "--user-arg",
                ]
            )

        realtime_cmd.assert_not_called()
        rtplot_cmd.assert_called_once()
        args, user_args = rtplot_cmd.call_args[0]
        self.assertEqual(args.file, ["example.py"])
        self.assertEqual(args.dashboard_mode, "multiwindow")
        self.assertEqual(args.tabs, "case-plotter,trajectory")
        self.assertEqual(user_args, ["--user-arg"])

    def test_entrypoint_parser_exposes_dashboard_options(self):
        parser = rtplot._build_entrypoint_parser()
        args = parser.parse_args(
            [
                "--dashboard-mode",
                "multiwindow",
                "--tabs",
                "case-plotter,trajectory",
                "--tab-core",
                "case-plotter=0",
                "--base-port",
                "58000",
                "example.py",
            ]
        )
        self.assertEqual(args.dashboard_mode, "multiwindow")
        self.assertEqual(args.tabs, "case-plotter,trajectory")
        self.assertEqual(args.tab_core, "case-plotter=0")
        self.assertEqual(args.base_port, 58000)
        self.assertEqual(args.file, "example.py")

    def test_default_tab_order_uses_case_plotter(self):
        self.assertEqual(
            realtime_plot._normalize_dashboard_tabs(None),
            [
                "case-plotter",
                "trajectory",
                "series",
                "jacobian-entries",
                "jacobian-heatmap",
            ],
        )
        self.assertEqual(
            realtime_dashboard.DASHBOARD_TAB_TITLES[realtime_dashboard.CASE_PLOTTER_TAB],
            "Case Plotter",
        )
        self.assertNotIn("Current RTPlot", realtime_dashboard.DASHBOARD_TAB_TITLES.values())

    def test_unknown_tab_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown tab"):
            realtime_plot._normalize_dashboard_tabs("case-plotter,unknown")

    def test_tabbed_mode_rejects_multiwindow_only_flags(self):
        tracker = _FakeCaseTracker(is_optimizer=True)
        with self.assertRaisesRegex(ValueError, "--tabs is only supported"):
            realtime_plot._resolve_dashboard_launch_config(
                tracker,
                "tabbed",
                "case-plotter",
                None,
            )

    def test_multiwindow_requires_optimizer_dashboard(self):
        tracker = _FakeCaseTracker(is_optimizer=False)
        with self.assertRaisesRegex(ValueError, "only available for the optimizer dashboard"):
            realtime_plot._resolve_dashboard_launch_config(
                tracker,
                "multiwindow",
                "case-plotter",
                None,
            )

    def test_multiwindow_parses_selected_tabs_and_core_assignments(self):
        tracker = _FakeCaseTracker(is_optimizer=True)
        with mock.patch("dymos_rtplot.realtime_plot.realtime_plot.os.cpu_count", return_value=8):
            tabs, assignments = realtime_plot._resolve_dashboard_launch_config(
                tracker,
                "multiwindow",
                "case-plotter,trajectory",
                "case-plotter=0,trajectory=2",
            )
        self.assertEqual(tabs, ["case-plotter", "trajectory"])
        self.assertEqual(assignments, {"case-plotter": 0, "trajectory": 2})

    def test_port_for_dashboard_tab_uses_stable_offsets(self):
        self.assertEqual(realtime_plot._port_for_dashboard_tab("case-plotter", 57003), 57003)
        self.assertEqual(realtime_plot._port_for_dashboard_tab("trajectory", 57003), 57004)
        self.assertEqual(realtime_plot._port_for_dashboard_tab("series", 57003), 57005)
        self.assertEqual(realtime_plot._port_for_dashboard_tab("jacobian-entries", 57003), 57006)
        self.assertEqual(realtime_plot._port_for_dashboard_tab("jacobian-heatmap", 57003), 57007)

    def test_core_assignment_requires_selected_tab(self):
        tracker = _FakeCaseTracker(is_optimizer=True)
        with mock.patch("dymos_rtplot.realtime_plot.realtime_plot.os.cpu_count", return_value=8):
            with self.assertRaisesRegex(ValueError, "is not included in --tabs"):
                realtime_plot._resolve_dashboard_launch_config(
                    tracker,
                    "multiwindow",
                    "trajectory",
                    "case-plotter=0",
                )

    def test_invalid_core_is_rejected(self):
        tracker = _FakeCaseTracker(is_optimizer=True)
        with mock.patch("dymos_rtplot.realtime_plot.realtime_plot.os.cpu_count", return_value=4):
            with self.assertRaisesRegex(ValueError, "out of range"):
                realtime_plot._resolve_dashboard_launch_config(
                    tracker,
                    "multiwindow",
                    "case-plotter",
                    "case-plotter=4",
                )

    def test_windows_affinity_uses_handle_and_integer_mask(self):
        fake_kernel32 = _FakeKernel32()
        with mock.patch("dymos_rtplot.realtime_plot.realtime_plot.sys.platform", "win32"), \
             mock.patch("dymos_rtplot.realtime_plot.realtime_plot.os.cpu_count", return_value=16), \
             mock.patch("dymos_rtplot.realtime_plot.realtime_plot.ctypes.WinDLL", return_value=fake_kernel32):
            realtime_plot._set_process_cpu_affinity(3)

        self.assertTrue(fake_kernel32.get_current_process_called)
        self.assertEqual(fake_kernel32.affinity_calls, [(1234, 8)])

    def test_build_dashboard_tab_uses_case_plotter_title(self):
        fake_plot = mock.Mock()
        fake_plot.layout = Div(text="layout")
        with mock.patch(
            "dymos_rtplot.realtime_plot.realtime_dashboard._RealTimeOptimizerPlot",
            return_value=fake_plot,
        ):
            plot, panel = realtime_dashboard._build_dashboard_tab(
                realtime_dashboard.CASE_PLOTTER_TAB,
                case_tracker=mock.sentinel.case_tracker,
                callback_period=123,
                doc=mock.sentinel.doc,
                pid_of_calling_script=5,
                script="script.py",
            )
        self.assertIs(plot, fake_plot)
        self.assertEqual(panel.title, "Case Plotter")
        self.assertIs(panel.child, fake_plot.layout)

    def test_standalone_tab_app_adds_root_callback_and_title(self):
        fake_tab = mock.Mock()
        fake_panel = mock.Mock(child="child-root")
        doc = _FakeDoc()
        with mock.patch(
            "dymos_rtplot.realtime_plot.realtime_dashboard._build_dashboard_tab",
            return_value=(fake_tab, fake_panel),
        ):
            app = realtime_dashboard._StandaloneDashboardTabApp(
                realtime_dashboard.TRAJECTORY_TAB,
                case_tracker=mock.sentinel.case_tracker,
                callback_period=250,
                doc=doc,
                pid_of_calling_script=10,
                script="demo.py",
                metadata={"m": 1},
                hist_file="demo.hst",
            )
        self.assertIs(app._tab, fake_tab)
        self.assertEqual(doc.roots, ["child-root"])
        self.assertEqual(len(doc.callbacks), 1)
        self.assertEqual(doc.callbacks[0][1], 250)
        self.assertEqual(doc.title, "Dymos RTPlot - Trajectory")

    def test_trajectory_tab_does_not_rebuild_children_when_order_is_unchanged(self):
        broker = _FakeBroker(
            metadata={
                "trajectories": [
                    {
                        "name": "traj",
                        "phases": [
                            {
                                "name": "phase0",
                                "promoted_path": "traj.phase0",
                                "states": {"x": {}},
                                "controls": {"u": {}},
                                "timeseries_outputs": {"y": {"category": "ode"}},
                            }
                        ],
                    }
                ]
            }
        )
        tab = realtime_dashboard._TrajectoryTab(broker)
        tab._ensure_initialized()
        tab._rebuild_plots()
        original_children = tab._plots_column.children
        tab._rebuild_plots()
        self.assertIs(tab._plots_column.children, original_children)

    def test_series_tab_refresh_keeps_visible_legend(self):
        broker = _FakeBroker()
        snapshot = _FakeSnapshot()
        broker.snapshots = [snapshot]
        broker.get_series = mock.Mock(return_value=[[1.0]])
        tab = realtime_dashboard._SeriesTab(broker)
        tab._var_select.value = ["obj"]
        tab.refresh(force=True)
        self.assertTrue(tab._figure.legend)
        self.assertTrue(tab._figure.legend[0].visible)
        self.assertEqual(tab._figure.legend[0].location, "top_left")
        self.assertEqual(tab._figure.legend[0].click_policy, "hide")

    def test_jacobian_entries_tab_refresh_keeps_visible_legend(self):
        broker = _FakeBroker()
        snapshot = _FakeSnapshot()
        broker.snapshots = [snapshot]
        tab = realtime_dashboard._JacobianEntriesTab(broker)
        tab._selected_block = "of | wrt"
        tab._block_select.value = "of | wrt"
        tab._entry_select.value = ["0,0"]
        tab.refresh(force=True)
        self.assertTrue(tab._figure.legend)
        self.assertTrue(tab._figure.legend[0].visible)
        self.assertEqual(tab._figure.legend[0].location, "top_left")
        self.assertEqual(tab._figure.legend[0].click_policy, "hide")

    def test_launch_multiwindow_dashboard_starts_selected_tabs_and_opens_browser(self):
        messages = [
            {"status": "started", "tab": "case-plotter", "url": "http://127.0.0.1:5001/"},
            {"status": "started", "tab": "trajectory", "url": "http://127.0.0.1:5002/"},
        ]
        fake_context = _FakeContext(messages)
        with mock.patch(
            "dymos_rtplot.realtime_plot.realtime_plot.multiprocessing.get_context",
            return_value=fake_context,
        ), mock.patch(
            "dymos_rtplot.realtime_plot.realtime_plot.webbrowser.open_new_tab"
        ) as open_tab:
            realtime_plot._launch_multiwindow_dashboard(
                case_recorder_filename="cases.sql",
                callback_period=300,
                pid_of_calling_script=111,
                script="demo.py",
                meta_file="cases.sql.rtplot_meta.json",
                hist_file="cases.hst",
                open_browser=True,
                host="127.0.0.1",
                selected_tabs=["case-plotter", "trajectory"],
                core_assignments={"trajectory": 2},
                base_port=58000,
            )

        self.assertEqual(len(fake_context.processes), 2)
        self.assertTrue(all(proc.started for proc in fake_context.processes))
        self.assertEqual(
            [proc.args[1] for proc in fake_context.processes],
            ["case-plotter", "trajectory"],
        )
        self.assertEqual(
            [proc.args[-1] for proc in fake_context.processes],
            [58000, 58001],
        )
        open_tab.assert_any_call("http://127.0.0.1:5001/")
        open_tab.assert_any_call("http://127.0.0.1:5002/")
        self.assertEqual(open_tab.call_count, 2)

    def test_launch_multiwindow_dashboard_raises_if_no_tabs_start(self):
        fake_context = _FakeContext([], alive_after_start=False)
        with mock.patch(
            "dymos_rtplot.realtime_plot.realtime_plot.multiprocessing.get_context",
            return_value=fake_context,
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed to start any multiwindow dashboard tabs"):
                realtime_plot._launch_multiwindow_dashboard(
                    case_recorder_filename="cases.sql",
                    callback_period=300,
                    pid_of_calling_script=111,
                    script="demo.py",
                    meta_file=None,
                    hist_file=None,
                    open_browser=False,
                    host="127.0.0.1",
                    selected_tabs=["case-plotter"],
                    core_assignments={},
                    base_port=58000,
                )


if __name__ == "__main__":
    unittest.main()
