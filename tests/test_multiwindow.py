import unittest
from unittest import mock
import queue as std_queue
import tempfile
from pathlib import Path

import numpy as np
from bokeh.models import Div

from dymos_rtplot import rtplot
from dymos_rtplot.realtime_plot import realtime_broker, realtime_dashboard, realtime_plot


class _FakeCaseTracker:
    def __init__(self, is_optimizer):
        self._is_optimizer = is_optimizer

    def is_driver_optimizer(self):
        return self._is_optimizer


class _FakeBroker:
    def __init__(self, metadata=None):
        self.metadata = metadata or {}
        self.snapshots = []
        self.history_entries = []
        self.history_warning = None

    def latest_snapshot(self):
        return self.snapshots[-1] if self.snapshots else None

    def get_history_keys(self):
        if not self.history_entries:
            return []
        return list(self.history_entries[0].keys())

    def get_history_warning(self):
        return self.history_warning

    def get_history_entries(self):
        return list(self.history_entries)

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


class _FakeCase:
    def __init__(self, counter=1, timestamp=1.0, fail_builds=0):
        self.counter = counter
        self.timestamp = timestamp
        self._fail_builds = fail_builds

    def _maybe_fail(self):
        if self._fail_builds > 0:
            self._fail_builds -= 1
            raise RuntimeError("transient read failure")

    def get_objectives(self, scaled=False):
        self._maybe_fail()
        return {"obj": [1.0] if not scaled else [10.0]}

    def get_design_vars(self, scaled=False):
        self._maybe_fail()
        return {"dv": [2.0] if not scaled else [20.0]}

    def get_constraints(self, scaled=False):
        self._maybe_fail()
        return {"con": [3.0] if not scaled else [30.0]}


class _FakeBrokerCaseTracker:
    def __init__(self, cases=None, running=False):
        self.cases = dict(cases or {})
        self.running = running

    def get_case_by_counter(self, counter):
        return self.cases.get(counter)

    def is_source_process_running(self):
        return self.running


class _FakeCaseValues:
    def __init__(self, values):
        self._values = values

    def get_val(self, path):
        return self._values[path]


class _FakeSession:
    def __init__(self, connection_count=0, destroyed=False):
        self.connection_count = connection_count
        self.destroyed = destroyed


class _FakeIOLoop:
    def __init__(self):
        self.stop_called = False

    def stop(self):
        self.stop_called = True


class _FakeServer:
    def __init__(self, sessions=None):
        self.sessions = list(sessions or [])
        self.io_loop = _FakeIOLoop()

    def get_sessions(self, path):
        return list(self.sessions)


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
                "--idle-shutdown-seconds",
                "45",
                "--disable-jacobian-highlighting",
                "--light-mode",
                "example.py",
            ]
        )
        self.assertEqual(args.dashboard_mode, "multiwindow")
        self.assertEqual(args.tabs, "case-plotter,trajectory")
        self.assertEqual(args.tab_core, "case-plotter=0")
        self.assertEqual(args.base_port, 58000)
        self.assertEqual(args.idle_shutdown_seconds, 45.0)
        self.assertFalse(args.highlight_jacobian_structure)
        self.assertFalse(args.dark_mode)
        self.assertEqual(args.file, "example.py")

    def test_main_routes_clean_subcommand_without_entrypoint_rewrite(self):
        with mock.patch(
            "dymos_rtplot.rtplot.clean_rtplot_artifacts",
            return_value={"root": ".", "dirs": [], "files": []},
        ) as clean_fn:
            rtplot.main(["clean", ".", "--dry-run"])

        clean_fn.assert_called_once_with(".", dry_run=True)

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

    def test_count_active_server_connections_ignores_destroyed_or_disconnected_sessions(self):
        server = _FakeServer(
            sessions=[
                _FakeSession(connection_count=2, destroyed=False),
                _FakeSession(connection_count=1, destroyed=True),
                _FakeSession(connection_count=0, destroyed=False),
            ]
        )
        self.assertEqual(realtime_plot._count_active_server_connections(server), 2)

    def test_child_session_monitor_resets_idle_timer_when_connections_return(self):
        server = _FakeServer(sessions=[_FakeSession(connection_count=0)])
        current_time = {"value": 10.0}
        monitor = realtime_plot._make_child_session_monitor(
            server,
            "trajectory",
            pid_of_calling_script=None,
            idle_shutdown_seconds=15.0,
            now_fn=lambda: current_time["value"],
        )

        monitor()
        current_time["value"] = 20.0
        server.sessions = [_FakeSession(connection_count=1)]
        monitor()
        server.sessions = [_FakeSession(connection_count=0)]
        monitor()
        current_time["value"] = 30.0
        monitor()

        self.assertFalse(server.io_loop.stop_called)

    def test_child_session_monitor_stops_after_idle_timeout(self):
        server = _FakeServer(sessions=[_FakeSession(connection_count=0)])
        current_time = {"value": 100.0}
        monitor = realtime_plot._make_child_session_monitor(
            server,
            "trajectory",
            pid_of_calling_script=None,
            idle_shutdown_seconds=15.0,
            now_fn=lambda: current_time["value"],
        )

        monitor()
        current_time["value"] = 114.0
        monitor()
        self.assertFalse(server.io_loop.stop_called)

        current_time["value"] = 115.0
        monitor()
        self.assertTrue(server.io_loop.stop_called)

    def test_child_session_monitor_stops_when_source_process_ends_with_no_sessions(self):
        server = _FakeServer(sessions=[_FakeSession(connection_count=0)])
        current_time = {"value": 50.0}
        monitor = realtime_plot._make_child_session_monitor(
            server,
            "series",
            pid_of_calling_script=123,
            idle_shutdown_seconds=15.0,
            now_fn=lambda: current_time["value"],
        )

        with mock.patch(
            "dymos_rtplot.realtime_plot.realtime_plot._source_process_running",
            return_value=False,
        ):
            monitor()
            current_time["value"] = 51.0
            monitor()

        self.assertTrue(server.io_loop.stop_called)

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
                                "defect_outputs": {"collocation:x": {"path": "traj.phase0.collocation_constraint.defects:x"}},
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
        self.assertIn("defects", tab._selected_order())

    def test_trajectory_tab_includes_defect_outputs_category(self):
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
                                "defect_outputs": {
                                    "collocation:x": {
                                        "path": "traj.phase0.collocation_constraint.defects:x",
                                        "node_ptau": [-0.5, 0.5],
                                    }
                                },
                                "timeseries_outputs": {},
                            }
                        ],
                    }
                ]
            }
        )
        tab = realtime_dashboard._TrajectoryTab(broker)
        tab._ensure_initialized()
        traj_meta = tab._traj_meta()
        self.assertIn("collocation:x", tab._category_variables(traj_meta, "defects"))

    def test_trajectory_tab_grid_uses_dark_container_styles_in_dark_mode(self):
        realtime_dashboard._set_dark_mode_enabled(True)
        try:
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
                                    "controls": {},
                                    "defect_outputs": {},
                                    "timeseries_outputs": {},
                                }
                            ],
                        }
                    ]
                }
            )
            tab = realtime_dashboard._TrajectoryTab(broker)
            tab._ensure_initialized()
            tab._rebuild_plots()

            grid = tab._plots_column.children[1]
            self.assertEqual(grid.styles.get("background-color"), "#0b1220")
        finally:
            realtime_dashboard._set_dark_mode_enabled(True)

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

    def test_series_tab_uses_history_iteration_axis_when_counts_diverge(self):
        broker = _FakeBroker()
        broker.snapshots = [_FakeSnapshot(), _FakeSnapshot(), _FakeSnapshot()]
        broker.history_entries = [
            {"iter": [10], "metric": [1.5]},
            {"iter": [20], "metric": [2.5]},
        ]
        tab = realtime_dashboard._SeriesTab(broker)
        tab._group_select.value = "Optimizer History"
        tab._var_select.value = ["metric"]

        tab.refresh(force=True)

        self.assertEqual(list(tab._source.data["iteration"]), [10.0, 20.0])
        self.assertEqual(list(tab._source.data["metric"]), [1.5, 2.5])
        self.assertIn("diverged", tab._warning.text)

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

    def test_jacobian_entries_tab_filters_missing_derivative_iterations(self):
        broker = _FakeBroker()
        s1 = _FakeSnapshot()
        s1.major_iteration = 1
        s1.derivatives = None
        s2 = _FakeSnapshot()
        s2.major_iteration = 2
        s2.derivatives = {("of", "wrt"): np.array([[3.0]])}
        s3 = _FakeSnapshot()
        s3.major_iteration = 3
        s3.derivatives = {("of", "wrt"): np.array([[4.0]])}
        broker.snapshots = [s1, s2, s3]
        tab = realtime_dashboard._JacobianEntriesTab(broker)
        tab._selected_block = "of | wrt"
        tab._block_select.value = "of | wrt"
        tab._entry_select.value = ["0,0"]

        tab.refresh(force=True)

        self.assertEqual(list(tab._source.data["iteration"]), [2, 3])
        self.assertEqual(list(tab._source.data["0,0"]), [3.0, 4.0])
        self.assertIn("no derivative block", tab._warning.text)

    def test_state_marker_trace_uses_discrete_state_nodes(self):
        broker = _FakeBroker(metadata={"trajectories": []})
        tab = realtime_dashboard._TrajectoryTab(broker)
        case = _FakeCaseValues(
            {
                "traj.phase0.states:x": [[10.0], [20.0], [30.0]],
                "traj.phase0.t_duration": [100.0],
            }
        )
        phase_meta = {
            "promoted_path": "traj.phase0",
            "state_input_node_ptau": [-1.0, 0.0, 1.0],
        }

        xvals, yvals = tab._state_marker_trace(case, phase_meta, "x")

        self.assertEqual(list(xvals), [0.0, 50.0, 100.0])
        self.assertEqual(list(yvals), [10.0, 20.0, 30.0])

    def test_control_marker_trace_uses_discrete_control_nodes(self):
        broker = _FakeBroker(metadata={"trajectories": []})
        tab = realtime_dashboard._TrajectoryTab(broker)
        case = _FakeCaseValues(
            {
                "traj.phase0.controls:u": [[1.0], [2.0], [3.0], [4.0]],
                "traj.phase0.t_duration": [20.0],
            }
        )
        phase_meta = {
            "promoted_path": "traj.phase0",
            "control_input_node_ptau": [-1.0, -0.5, 0.5, 1.0],
        }

        xvals, yvals = tab._control_marker_trace(case, phase_meta, "u")

        self.assertEqual(list(xvals), [0.0, 5.0, 15.0, 20.0])
        self.assertEqual(list(yvals), [1.0, 2.0, 3.0, 4.0])

    def test_collapse_repeated_samples_drops_duplicate_boundary_points(self):
        xvals, yvals, violation = realtime_dashboard._collapse_repeated_samples(
            [0.0, 1.0, 1.0, 2.0],
            [10.0, 20.0, 20.0, 30.0],
            [False, False, True, False],
        )

        self.assertEqual(list(xvals), [0.0, 1.0, 2.0])
        self.assertEqual(list(yvals), [10.0, 20.0, 30.0])
        self.assertEqual(list(violation), [False, True, False])

    def test_collapse_repeated_samples_keeps_discontinuous_duplicates(self):
        xvals, yvals, violation = realtime_dashboard._collapse_repeated_samples(
            [0.0, 1.0, 1.0, 2.0],
            [10.0, 20.0, 21.0, 30.0],
            [False, False, True, False],
        )

        self.assertEqual(list(xvals), [0.0, 1.0, 1.0, 2.0])
        self.assertEqual(list(yvals), [10.0, 20.0, 21.0, 30.0])
        self.assertEqual(list(violation), [False, False, True, False])

    def test_collapse_repeated_samples_trims_mismatched_violation_length(self):
        xvals, yvals, violation = realtime_dashboard._collapse_repeated_samples(
            [0.0, 1.0, 1.0, 2.0],
            [10.0, 20.0, 20.0, 30.0],
            [False, True, False],
        )

        self.assertEqual(list(xvals), [0.0, 1.0])
        self.assertEqual(list(yvals), [10.0, 20.0])
        self.assertEqual(list(violation), [False, True])

    def test_normalize_trace_arrays_trims_to_common_length(self):
        xvals, yvals, violation = realtime_dashboard._normalize_trace_arrays(
            [0.0, 1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            [False, True],
        )

        self.assertEqual(list(xvals), [0.0, 1.0])
        self.assertEqual(list(yvals), [10.0, 20.0])
        self.assertEqual(list(violation), [False, True])

    def test_matrix_rank_and_condition_reports_infinite_for_singular_matrix(self):
        rank, cond = realtime_dashboard._matrix_rank_and_condition([[1.0, 2.0], [2.0, 4.0]])
        self.assertEqual(rank, 1)
        self.assertEqual(cond, float("inf"))

    def test_dependent_indices_identify_redundant_columns_and_rows(self):
        matrix = np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
        dep_cols = realtime_dashboard._dependent_indices(matrix, "columns")
        dep_rows = realtime_dashboard._dependent_indices(matrix, "rows")
        self.assertEqual(len(dep_cols), 2)
        self.assertEqual(len(dep_rows), 2)
        self.assertTrue(set(dep_cols).issubset({0, 1, 2}))
        self.assertTrue(set(dep_rows).issubset({0, 1, 2}))

    def test_jacobian_heatmap_warns_on_singular_and_dependent_structure(self):
        broker = _FakeBroker()
        snapshot = _FakeSnapshot()
        snapshot.derivatives = {
            ("of", "x"): np.array([[1.0, 2.0], [2.0, 4.0], [0.0, 0.0]]),
        }
        broker.snapshots = [snapshot]
        tab = realtime_dashboard._JacobianHeatmapTab(broker)

        tab.refresh(force=True)

        self.assertIn("condition number is infinite", tab._warning.text)
        self.assertIn("Zero rows:", tab._warning.text)
        self.assertIn("Dependent rows:", tab._warning.text)
        self.assertIn("Dependent columns:", tab._warning.text)
        self.assertIn("cond=inf", tab._stats.text)
        self.assertTrue(tab._row_highlight_source.data["kind"])
        self.assertTrue(tab._col_highlight_source.data["kind"])

    def test_jacobian_heatmap_can_disable_highlighting(self):
        broker = _FakeBroker()
        snapshot = _FakeSnapshot()
        snapshot.derivatives = {
            ("of", "x"): np.array([[1.0, 2.0], [2.0, 4.0], [0.0, 0.0]]),
        }
        broker.snapshots = [snapshot]
        tab = realtime_dashboard._JacobianHeatmapTab(broker, highlight_structure=False)

        tab.refresh(force=True)

        self.assertEqual(tab._row_highlight_source.data["kind"], [])
        self.assertEqual(tab._col_highlight_source.data["kind"], [])
        self.assertIn("highlighting is disabled", tab._warning.text)

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
                idle_shutdown_seconds=22.5,
                highlight_jacobian_structure=False,
                dark_mode=False,
            )

        self.assertEqual(len(fake_context.processes), 2)
        self.assertTrue(all(proc.started for proc in fake_context.processes))
        self.assertEqual(
            [proc.args[1] for proc in fake_context.processes],
            ["case-plotter", "trajectory"],
        )
        self.assertEqual(
            [proc.args[-4] for proc in fake_context.processes],
            [58000, 58001],
        )
        self.assertEqual(
            [proc.args[-3] for proc in fake_context.processes],
            [22.5, 22.5],
        )
        self.assertEqual(
            [proc.args[-2] for proc in fake_context.processes],
            [False, False],
        )
        self.assertEqual(
            [proc.args[-1] for proc in fake_context.processes],
            [False, False],
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
                    idle_shutdown_seconds=15.0,
                    highlight_jacobian_structure=True,
                    dark_mode=True,
                )

    def test_live_data_broker_retries_transient_unreadable_case(self):
        tracker = _FakeBrokerCaseTracker(cases={1: _FakeCase(fail_builds=1)})
        broker = realtime_broker.LiveDataBroker(tracker)

        first = broker.poll()
        second = broker.poll()

        self.assertEqual(first, [])
        self.assertEqual(len(second), 1)
        self.assertEqual(len(broker.snapshots), 1)
        self.assertEqual(broker.latest_snapshot().counter, 1)

    def test_dashboard_update_retries_after_broker_error(self):
        dashboard = object.__new__(realtime_dashboard._RealTimeDymosDashboard)
        dashboard._tab_objects = {
            realtime_dashboard.CASE_PLOTTER_TAB: mock.Mock(),
            realtime_dashboard.TRAJECTORY_TAB: mock.Mock(),
            realtime_dashboard.SERIES_TAB: mock.Mock(),
        }
        dashboard._tab_objects[realtime_dashboard.CASE_PLOTTER_TAB]._update_wrapped_in_try = mock.Mock()
        dashboard._broker = mock.Mock()
        dashboard._broker.poll.side_effect = RuntimeError("transient refresh failure")
        dashboard._tabs = mock.Mock(active=0)
        dashboard._last_active = 0
        dashboard._tab_rendered = {}

        realtime_dashboard._RealTimeDymosDashboard._update(dashboard)

        dashboard._tab_objects[realtime_dashboard.CASE_PLOTTER_TAB]._update_wrapped_in_try.assert_called_once()
        dashboard._broker.poll.assert_called_once()

    def test_clean_rtplot_artifacts_dry_run_finds_expected_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.sqlite").write_text("x", encoding="utf-8")
            (root / "a.sqlite.rtplot_meta.json").write_text("{}", encoding="utf-8")
            (root / "a.hst").write_text("x", encoding="utf-8")
            out_dir = root / "case_out"
            out_dir.mkdir()
            (out_dir / "dymos_solution.db").write_text("x", encoding="utf-8")

            result = realtime_plot.clean_rtplot_artifacts(root, dry_run=True)

            self.assertEqual(len(result["files"]), 3)
            self.assertEqual(len(result["dirs"]), 1)
            self.assertTrue((root / "a.sqlite").exists())
            self.assertTrue(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
