"""
Classes and functions to support the realtime plotting.
"""

import ctypes
from ctypes import wintypes
import multiprocessing
import os
import queue
import sqlite3
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import openmdao.api as om
from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.recorders.case import Case
from openmdao.utils import hooks
from openmdao.utils.file_utils import _load_and_exec, is_python_file
from openmdao.utils.gui_testing_utils import get_free_port
from openmdao.utils.record_util import check_valid_sqlite3_db
from dymos_rtplot.realtime_plot.realtime_analysis_driver_plot \
    import _RealTimeAnalysisDriverPlot
from dymos_rtplot.realtime_plot.realtime_dashboard import (
    CASE_PLOTTER_TAB,
    DASHBOARD_TAB_TITLES,
    JACOBIAN_ENTRIES_TAB,
    JACOBIAN_HEATMAP_TAB,
    SERIES_TAB,
    TRAJECTORY_TAB,
    _RealTimeDymosDashboard,
    _StandaloneDashboardTabApp,
    get_dashboard_tab_names,
)
from dymos_rtplot.realtime_plot.realtime_metadata import (
    load_rtplot_metadata,
    write_rtplot_metadata,
)
from dymos_rtplot.realtime_plot.realtime_optimizer_plot import _RealTimeOptimizerPlot

try:
    from bokeh.server.server import Server
    from bokeh.application.application import Application
    from bokeh.application.handlers import FunctionHandler
    from tornado.ioloop import PeriodicCallback
    from tornado.web import StaticFileHandler

    bokeh_and_dependencies_available = True
except ImportError:
    bokeh_and_dependencies_available = False


# the time between calls to the udpate method
# if this is too small, the GUI interactions get delayed because
# code is busy trying to keep up with the periodic callbacks
_time_between_callbacks_in_ms = 300

# Number of milliseconds for unused session lifetime
_unused_session_lifetime_milliseconds = 1000 * 60 * 10

_DASHBOARD_MODE_TABBED = 'tabbed'
_DASHBOARD_MODE_MULTIWINDOW = 'multiwindow'
_DASHBOARD_MODE_CHOICES = (_DASHBOARD_MODE_TABBED, _DASHBOARD_MODE_MULTIWINDOW)
_MULTIWINDOW_STARTUP_TIMEOUT = 30.0
_DEFAULT_MULTIWINDOW_BASE_PORT = 57003
_CHILD_IDLE_SHUTDOWN_SECONDS = 15.0
_CHILD_IDLE_CHECK_PERIOD_MS = 1000


def _parse_csv_items(value):
    if value is None:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def _normalize_dashboard_tabs(value):
    known_tabs = get_dashboard_tab_names()
    if value is None:
        return list(known_tabs)

    seen = set()
    selected = []
    for item in _parse_csv_items(value):
        if item not in known_tabs:
            valid = ", ".join(known_tabs)
            raise ValueError(f"Unknown tab '{item}'. Expected one of: {valid}.")
        if item not in seen:
            seen.add(item)
            selected.append(item)

    if not selected:
        raise ValueError("At least one dashboard tab must be selected.")
    return selected


def _parse_tab_core_assignments(value):
    assignments = {}
    if value is None:
        return assignments

    for item in _parse_csv_items(value):
        if '=' not in item:
            raise ValueError(
                f"Invalid tab/core assignment '{item}'. Expected format tab=core."
            )
        tab_name, core_text = item.split('=', 1)
        tab_name = tab_name.strip()
        core_text = core_text.strip()
        if not tab_name or not core_text:
            raise ValueError(
                f"Invalid tab/core assignment '{item}'. Expected format tab=core."
            )
        try:
            core = int(core_text)
        except ValueError as err:
            raise ValueError(f"CPU core for '{tab_name}' must be an integer.") from err
        assignments[tab_name] = core
    return assignments


def _validate_core_number(core):
    cpu_count = os.cpu_count()
    if core < 0:
        raise ValueError(f"CPU core index must be non-negative, got {core}.")
    if cpu_count is not None and core >= cpu_count:
        raise ValueError(
            f"CPU core index {core} is out of range for this machine ({cpu_count} cores)."
        )
    if sys.platform.startswith('win'):
        max_bits = ctypes.sizeof(ctypes.c_size_t) * 8
        if core >= max_bits:
            raise ValueError(
                f"CPU core index {core} exceeds Windows affinity mask width ({max_bits} bits)."
            )


def _resolve_dashboard_launch_config(case_tracker, dashboard_mode, tabs_value, tab_core_value):
    if dashboard_mode == _DASHBOARD_MODE_TABBED:
        if tabs_value:
            raise ValueError("--tabs is only supported with --dashboard-mode multiwindow.")
        if tab_core_value:
            raise ValueError("--tab-core is only supported with --dashboard-mode multiwindow.")
        return [], {}

    if not case_tracker.is_driver_optimizer():
        raise ValueError(
            "--dashboard-mode multiwindow is only available for the optimizer dashboard."
        )

    selected_tabs = _normalize_dashboard_tabs(tabs_value)
    core_assignments = _parse_tab_core_assignments(tab_core_value)
    known_tabs = set(get_dashboard_tab_names())
    for tab_name, core in core_assignments.items():
        if tab_name not in known_tabs:
            valid = ", ".join(get_dashboard_tab_names())
            raise ValueError(f"Unknown tab '{tab_name}' in --tab-core. Expected one of: {valid}.")
        if tab_name not in selected_tabs:
            raise ValueError(
                f"Tab '{tab_name}' was assigned a CPU core but is not included in --tabs."
            )
        _validate_core_number(core)
    return selected_tabs, core_assignments


def _port_for_dashboard_tab(tab_name, base_port):
    tab_names = get_dashboard_tab_names()
    if tab_name not in tab_names:
        valid = ", ".join(tab_names)
        raise ValueError(f"Unknown tab '{tab_name}'. Expected one of: {valid}.")
    return base_port + tab_names.index(tab_name)


def _set_process_cpu_affinity(core):
    _validate_core_number(core)
    if sys.platform.startswith('win'):
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
        process_handle = kernel32.GetCurrentProcess()
        mask_value = ctypes.c_size_t(1 << core).value
        if not kernel32.SetProcessAffinityMask(process_handle, mask_value):
            error_code = ctypes.get_last_error()
            raise OSError(
                f"SetProcessAffinityMask failed for CPU core {core} with Win32 error {error_code}."
            )
        return
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {core})
        return
    print(f"CPU affinity is not supported on this platform; ignoring requested core {core}.")


def _source_process_running(pid_of_calling_script):
    if pid_of_calling_script is None:
        return False
    try:
        from openmdao.utils.shell_proc import _is_process_running
        return _is_process_running(pid_of_calling_script)
    except Exception:
        return False


def _count_active_server_connections(server):
    active_connections = 0
    for session in server.get_sessions("/"):
        connection_count = getattr(session, "connection_count", 0)
        destroyed = getattr(session, "destroyed", False)
        if connection_count > 0 and not destroyed:
            active_connections += connection_count
    return active_connections


def _make_child_session_monitor(server, tab_name, pid_of_calling_script,
                                idle_shutdown_seconds=_CHILD_IDLE_SHUTDOWN_SECONDS,
                                now_fn=time.time):
    idle_since = None

    def _check():
        nonlocal idle_since

        active_connections = _count_active_server_connections(server)
        if active_connections > 0:
            idle_since = None
            return

        now = now_fn()
        if idle_since is None:
            idle_since = now
            return

        source_alive = True
        if pid_of_calling_script is not None:
            source_alive = _source_process_running(pid_of_calling_script)

        if not source_alive or (now - idle_since) >= idle_shutdown_seconds:
            if pid_of_calling_script is not None and not source_alive:
                reason = "source process ended and no active browser sessions remain"
            else:
                reason = (
                    f"no active browser sessions for {int(idle_shutdown_seconds)} seconds"
                )
            print(f"Stopping {DASHBOARD_TAB_TITLES[tab_name]} server: {reason}")
            server.io_loop.stop()

    return _check


def _append_dashboard_launch_args(cmd, options):
    if getattr(options, 'open_browser', False):
        cmd.append('--open-browser')
    if getattr(options, 'host', None):
        cmd.extend(['--host', options.host])
    if getattr(options, 'dashboard_mode', None):
        cmd.extend(['--dashboard-mode', options.dashboard_mode])
    if getattr(options, 'tabs', None):
        cmd.extend(['--tabs', options.tabs])
    if getattr(options, 'tab_core', None):
        cmd.extend(['--tab-core', options.tab_core])
    if getattr(options, 'base_port', None) is not None:
        cmd.extend(['--base-port', str(options.base_port)])


def _add_dashboard_cli_arguments(parser):
    parser.add_argument(
        '--open-browser',
        action='store_true',
        help='Attempt to open the dashboard URL in the system browser.',
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host interface to bind the Bokeh server to. Defaults to 127.0.0.1.',
    )
    parser.add_argument(
        '--dashboard-mode',
        choices=_DASHBOARD_MODE_CHOICES,
        default=_DASHBOARD_MODE_TABBED,
        help='Launch the optimizer dashboard in one tabbed window or as separate per-tab windows.',
    )
    parser.add_argument(
        '--tabs',
        type=str,
        default=None,
        help='Comma-separated dashboard tabs to launch in multiwindow mode.',
    )
    parser.add_argument(
        '--tab-core',
        type=str,
        default=None,
        help='Comma-separated CPU affinity assignments in the form tab=core for multiwindow mode.',
    )
    parser.add_argument(
        '--base-port',
        type=int,
        default=_DEFAULT_MULTIWINDOW_BASE_PORT,
        help='Base port for deterministic multiwindow tab URLs. Tab ports are assigned by tab order offset.',
    )


def _realtime_plot_setup_parser(parser):
    """
    Set up the realtime plot subparser for the 'openmdao realtime_plot' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument(
        "case_recorder_filename",
        type=str,
        help="Name of openmdao case recorder filename. It should contain driver cases",
    )

    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Process ID of calling optimization script, "
        "defaults to None if called by the user directly",
    )

    parser.add_argument('--script', type=str, default=None,
                        help='The name of the script that created the case recorder file.')
    parser.add_argument('--meta-file', type=str, default=None,
                        help='Optional metadata sidecar for the richer dashboard.')
    parser.add_argument('--hist-file', type=str, default=None,
                        help='Optional pyOptSparse history file.')
    _add_dashboard_cli_arguments(parser)


def _realtime_plot_cmd(options, user_args):
    """
    Run the realtime_plot command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    if bokeh_and_dependencies_available:
        realtime_plot(
            options.case_recorder_filename,
            _time_between_callbacks_in_ms,
            options.pid,
            options.script,
            options.meta_file,
            options.hist_file,
            options.open_browser,
            options.host,
            options.dashboard_mode,
            options.tabs,
            options.tab_core,
            options.base_port,
        )
    else:
        print(
            "The bokeh library and dependencies are not installed so the realtime "
            "plot is not available. "
        )
        return


def _rtplot_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao rtplot' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    _add_dashboard_cli_arguments(parser)


def _rtplot_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao rtplot'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    if not bokeh_and_dependencies_available:
        print(
            "The bokeh library and dependencies are not installed so the rtplot "
            "command is not available. "
        )
        return

    file_path = options.file[0]
    if is_python_file(file_path):
        script_path = file_path
    else:
        script_path = None

    def _view_realtime_plot_hook(problem):
        driver = problem.driver
        if not driver:
            raise RuntimeError(
                "Unable to run realtime optimization progress plot because no Driver")
        if len(problem.driver._rec_mgr._recorders) == 0:
            raise RuntimeError(
                "Unable to run realtime optimization progress plot "
                    "because no case recorder attached to Driver"
            )

        case_recorder_file = str(problem.driver._rec_mgr._recorders[0]._filepath)
        meta_path = write_rtplot_metadata(problem, case_recorder_file)
        hist_file = driver.options['hist_file'] if 'hist_file' in driver.options else None

        cmd = [sys.executable, '-m', 'dymos_rtplot.rtplot',
               'realtime_plot', '--pid', str(os.getpid()),
               '--meta-file', str(meta_path),
               case_recorder_file]
        if hist_file:
            cmd.insert(-1, '--hist-file')
            cmd.insert(-1, str(Path(hist_file).resolve()))
        if script_path:
            cmd.insert(-1, '--script')
            cmd.insert(-1, script_path)
        _append_dashboard_launch_args(cmd, options)
        cp = subprocess.Popen(cmd)  # nosec: trusted input

        # Do a quick non-blocking check to see if it immediately failed
        # This will catch immediate failures but won't wait for the process to finish
        quick_check = cp.poll()
        if quick_check is not None and quick_check != 0:
            # Process already terminated with an error
            stderr = cp.stderr.read().decode()
            raise RuntimeError(
                f"Failed to start up the realtime plot server with code {quick_check}: {stderr}.")

    def _view_realtime_plot(case_recorder_file):
        meta_path = None
        metadata = load_rtplot_metadata(case_recorder_filename=case_recorder_file)
        if metadata:
            meta_path = str(Path(case_recorder_file).with_suffix(Path(case_recorder_file).suffix + '.rtplot_meta.json'))
        cmd = [
            sys.executable,
            "-m",
            "dymos_rtplot.rtplot",
            "realtime_plot",
            "--pid",
            str(os.getpid()),
            case_recorder_file,
        ]
        if meta_path:
            cmd.insert(-1, '--meta-file')
            cmd.insert(-1, meta_path)
        _append_dashboard_launch_args(cmd, options)

        cp = subprocess.Popen(cmd)  # nosec: trusted input

        # Do a quick non-blocking check to see if it immediately failed
        # This will catch immediate failures but won't wait for the process to finish
        quick_check = cp.poll()
        if quick_check is not None and quick_check != 0:
            # Process already terminated with an error
            stderr = cp.stderr.read().decode()
            raise RuntimeError(
                f"Failed to start up the realtime plot server with code {quick_check}: {stderr}."
            )

    # check to see if options.file is python script, sqlite file or neither
    file_path = options.file[0]
    try:
        check_valid_sqlite3_db(file_path)
        _view_realtime_plot(file_path)
        return
    except IOError:
        pass
    if is_python_file(file_path):
        def _recording_setup_hook(problem):
            driver = problem.driver
            if not driver:
                return
            if len(driver._rec_mgr._recorders) == 0:
                auto_case_path = Path(file_path).resolve().with_name(
                    f"{Path(file_path).stem}_rtplot_auto_{os.getpid()}.sqlite"
                )
                driver.add_recorder(om.SqliteRecorder(str(auto_case_path)))
            driver.recording_options['record_outputs'] = True
            driver.recording_options['record_derivatives'] = True
            driver.recording_options['includes'] = ['*']
            if 'optimizer' in driver.options and driver.options['optimizer'] == 'IPOPT':
                opt_settings = driver.opt_settings
                existing = set(opt_settings.get('save_major_iteration_variables', []))
                existing.update({
                    'alg_mod',
                    'd_norm',
                    'regularization_size',
                    'ls_trials',
                    'g_violation',
                    'grad_lag_x',
                })
                opt_settings['save_major_iteration_variables'] = sorted(existing)

        # register the hook
        hooks._register_hook(
            "_setup_recording", "Problem", pre=_recording_setup_hook, ncalls=1
        )
        hooks._register_hook(
            "_setup_recording", "Problem", post=_view_realtime_plot_hook, ncalls=1
        )
        # run the script
        _load_and_exec(file_path, user_args)
    else:
        raise RuntimeError(
            "The argument to the openmdao rtplot command must be either a "
            "case recorder file or an OpenMDAO python script."
        )


class _CaseRecorderTracker:
    """
    A class that is used to get information from a case recorder.

    This class was created to handle the realtime reading of a case recorder file.
    These features are not provided by the SqliteCaseReader class.
    """

    def __init__(self, case_recorder_filename):
        self._case_recorder_filename = case_recorder_filename
        self._cr = None
        self._initial_case = (
            None  # need the initial case to get info about the variables
        )
        self._next_id_to_read = 1
        self._pid_of_calling_script = None

    def get_case_reader(self):
        return self._cr

    def set_source_process_pid(self, pid):
        self._pid_of_calling_script = pid

    def _open_case_recorder(self):
        if self._cr is None:
            self._cr = SqliteCaseReader(self._case_recorder_filename)

    def get_case_recorder_filename(self):
        return self._case_recorder_filename

    def is_driver_optimizer(self):
        self._open_case_recorder()
        return self._cr.problem_metadata["driver"]["supports"]["optimization"]["val"]

    def _get_case_by_counter(self, counter):
        # use SQL to see if a case with this counter exists
        with sqlite3.connect(self._case_recorder_filename) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute(
                "SELECT * FROM driver_iterations WHERE " "counter=:counter",
                {"counter": counter},
            )
            row = cur.fetchone()

        if row:
            self._open_case_recorder()
            return self._cr._driver_cases.get_case(row["iteration_coordinate"])
        else:
            return None

    def get_case_by_counter(self, counter):
        return self._get_case_by_counter(counter)

    def _get_data_from_case(self, driver_case):
        objs = driver_case.get_objectives(scaled=False)
        design_vars = driver_case.get_design_vars(scaled=False)
        constraints = driver_case.get_constraints(scaled=False)

        new_data = {
            "counter": int(driver_case.counter),
        }

        # get objectives
        objectives = {}
        for name, value in objs.items():
            objectives[name] = value
        new_data["objs"] = objectives

        # get des vars
        desvars = {}
        for name, value in design_vars.items():
            desvars[name] = value
        new_data["desvars"] = desvars

        # get cons
        cons = {}
        for name, value in constraints.items():
            cons[name] = value
        new_data["cons"] = cons

        return new_data

    def _get_new_case(self):
        # get the next unread case from the recorder
        driver_case = self._get_case_by_counter(self._next_id_to_read)
        if driver_case is None:
            return None

        if self._initial_case is None:
            self._initial_case = driver_case

        self._next_id_to_read += 1

        return driver_case

    def is_source_process_running(self):
        if self._pid_of_calling_script is None:
            return False
        try:
            from openmdao.utils.shell_proc import _is_process_running
            return _is_process_running(self._pid_of_calling_script)
        except Exception:
            return False

    def _get_obj_names(self):
        obj_vars = self._initial_case.get_objectives()
        return obj_vars.keys()

    def _get_desvar_names(self):
        design_vars = self._initial_case.get_design_vars()
        return design_vars.keys()

    def _get_cons_names(self):
        cons = self._initial_case.get_constraints()
        return cons.keys()

    def _get_constraint_bounds(self, name):
        cons = self._initial_case.get_constraints()
        var_info = cons._var_info[name]
        return (var_info['lower'], var_info['upper'])

    def _get_desvar_bounds(self, name):
        lower = self._cr.problem_metadata['variables'][name]['lower']
        upper = self._cr.problem_metadata['variables'][name]['upper']
        return lower, upper

    def _get_units(self, name):
        try:
            units = self._initial_case._get_units(name)
        except RuntimeError as err:
            if str(err).startswith("Can't get units for the promoted name"):
                return "Ambiguous"
            raise
        except KeyError:
            return "Unavailable"

        if units is None:
            units = "Unitless"
        return units

    def _get_shape(self, name):
        item = self._initial_case[name]
        return item.shape

    def _get_size(self, name):
        item = self._initial_case[name]
        return item.size


def _serve_dashboard_tab_process(startup_queue, tab_name, core, case_recorder_filename,
                                 callback_period, pid_of_calling_script, script,
                                 meta_file, hist_file, host, port_number):
    server = None
    idle_callback = None
    try:
        if core is not None:
            _set_process_cpu_affinity(core)

        app_url = f"http://{host}:{port_number}/"

        def _make_tab_doc(doc):
            case_tracker = _CaseRecorderTracker(case_recorder_filename)
            case_tracker.set_source_process_pid(pid_of_calling_script)
            metadata = load_rtplot_metadata(
                meta_file=meta_file,
                case_recorder_filename=case_recorder_filename,
            )
            if tab_name == CASE_PLOTTER_TAB:
                _RealTimeOptimizerPlot(
                    case_tracker,
                    callback_period,
                    doc,
                    pid_of_calling_script,
                    script,
                    add_root=True,
                    start_callback=True,
                )
                doc.title = f"Dymos RTPlot - {DASHBOARD_TAB_TITLES[tab_name]}"
            else:
                _StandaloneDashboardTabApp(
                    tab_name,
                    case_tracker,
                    callback_period,
                    doc,
                    pid_of_calling_script,
                    script,
                    metadata=metadata,
                    hist_file=hist_file,
                )

        server = Server(
            {"/": Application(FunctionHandler(_make_tab_doc))},
            address=host,
            port=port_number,
            allow_websocket_origin=[f"{host}:{port_number}"],
            unused_session_lifetime_milliseconds=_unused_session_lifetime_milliseconds,
            extra_patterns=[
                (
                    "/images/(.*)",
                    StaticFileHandler,
                    {"path": os.path.normpath(os.path.dirname(__file__) + "/images/")},
                ),
            ],
        )
        server.start()
        idle_callback = PeriodicCallback(
            _make_child_session_monitor(server, tab_name, pid_of_calling_script),
            _CHILD_IDLE_CHECK_PERIOD_MS,
        )
        idle_callback.start()
        startup_queue.put({
            "status": "started",
            "tab": tab_name,
            "url": app_url,
        })
        print(f"{DASHBOARD_TAB_TITLES[tab_name]} server running on {app_url}")
        server.io_loop.start()
    except KeyboardInterrupt:
        startup_queue.put({
            "status": "stopped",
            "tab": tab_name,
        })
    except Exception as err:
        startup_queue.put({
            "status": "error",
            "tab": tab_name,
            "error": str(err),
        })
        print(f"Error starting {DASHBOARD_TAB_TITLES.get(tab_name, tab_name)} server: {err}")
    finally:
        if idle_callback is not None:
            idle_callback.stop()
        if server is not None:
            server.stop()


def _launch_multiwindow_dashboard(case_recorder_filename, callback_period,
                                  pid_of_calling_script, script, meta_file,
                                  hist_file, open_browser, host, selected_tabs,
                                  core_assignments, base_port):
    context = multiprocessing.get_context('spawn')
    startup_queue = context.Queue()
    processes = {}

    try:
        for tab_name in selected_tabs:
            proc = context.Process(
                target=_serve_dashboard_tab_process,
                args=(
                    startup_queue,
                    tab_name,
                    core_assignments.get(tab_name),
                    case_recorder_filename,
                    callback_period,
                    pid_of_calling_script,
                    script,
                    meta_file,
                    hist_file,
                    host,
                    _port_for_dashboard_tab(tab_name, base_port),
                ),
            )
            proc.start()
            processes[tab_name] = proc

        urls = {}
        pending = set(selected_tabs)
        deadline = time.time() + _MULTIWINDOW_STARTUP_TIMEOUT
        while pending and time.time() < deadline:
            try:
                message = startup_queue.get(timeout=0.5)
            except queue.Empty:
                for tab_name in list(pending):
                    proc = processes[tab_name]
                    if not proc.is_alive():
                        pending.remove(tab_name)
                continue

            tab_name = message.get("tab")
            if tab_name not in pending:
                continue
            if message.get("status") == "started":
                urls[tab_name] = message["url"]
                pending.remove(tab_name)
            else:
                pending.remove(tab_name)
                error = message.get("error")
                if error:
                    print(f"{DASHBOARD_TAB_TITLES.get(tab_name, tab_name)} failed to start: {error}")

        if not urls:
            raise RuntimeError("Failed to start any multiwindow dashboard tabs.")

        for tab_name in selected_tabs:
            if tab_name in urls:
                print(f"{DASHBOARD_TAB_TITLES[tab_name]} URL: {urls[tab_name]}")

        if open_browser:
            for tab_name in selected_tabs:
                if tab_name in urls:
                    webbrowser.open_new_tab(urls[tab_name])
        else:
            print("Open the URLs above manually in your browser.")

        while True:
            alive = False
            for proc in processes.values():
                proc.join(timeout=0.5)
                alive = alive or proc.is_alive()
            if not alive:
                break
    except KeyboardInterrupt:
        print("Stopping multiwindow dashboard processes")
    finally:
        for proc in processes.values():
            if proc.is_alive():
                proc.terminate()
        for proc in processes.values():
            proc.join(timeout=1.0)


def realtime_plot(case_recorder_filename, callback_period,
                  pid_of_calling_script, script, meta_file=None, hist_file=None,
                  open_browser=False, host='127.0.0.1',
                  dashboard_mode=_DASHBOARD_MODE_TABBED, tabs_value=None,
                  tab_core_value=None, base_port=_DEFAULT_MULTIWINDOW_BASE_PORT):
    """
    Visualize the objectives, desvars, and constraints during an optimization or analysis process.

    Parameters
    ----------
    case_recorder_filename : str
        The path to the case recorder file that is the source of the data for the plot.
    callback_period : float
        The time period between when the application calls the update method.
    pid_of_calling_script : int
        The process id of the calling optimization script, if called this way.
    script : str or None
        If not None, the file path of the script that created the case recorder file.
    """
    server = None
    case_tracker = _CaseRecorderTracker(case_recorder_filename)
    case_tracker.set_source_process_pid(pid_of_calling_script)
    is_optimizer = case_tracker.is_driver_optimizer()

    try:
        selected_tabs, core_assignments = _resolve_dashboard_launch_config(
            case_tracker,
            dashboard_mode,
            tabs_value,
            tab_core_value,
        )
    except ValueError as err:
        raise RuntimeError(str(err)) from err

    if dashboard_mode == _DASHBOARD_MODE_MULTIWINDOW:
        _launch_multiwindow_dashboard(
            case_recorder_filename,
            callback_period,
            pid_of_calling_script,
            script,
            meta_file,
            hist_file,
            open_browser,
            host,
            selected_tabs,
            core_assignments,
            base_port,
        )
        return

    def _make_realtime_plot_doc(doc):
        print(f"Creating realtime plot document for {case_recorder_filename}")
        metadata = load_rtplot_metadata(meta_file=meta_file, case_recorder_filename=case_recorder_filename)
        if is_optimizer:
            local_case_tracker = _CaseRecorderTracker(case_recorder_filename)
            local_case_tracker.set_source_process_pid(pid_of_calling_script)
            _RealTimeDymosDashboard(
                local_case_tracker,
                callback_period,
                doc=doc,
                pid_of_calling_script=pid_of_calling_script,
                script=script,
                metadata=metadata,
                hist_file=hist_file,
            )
        else:
            local_case_tracker = _CaseRecorderTracker(case_recorder_filename)
            local_case_tracker.set_source_process_pid(pid_of_calling_script)
            _RealTimeAnalysisDriverPlot(
                local_case_tracker,
                callback_period,
                doc=doc,
                pid_of_calling_script=pid_of_calling_script,
                script=script,
            )

    _port_number = get_free_port()
    app_url = f"http://{host}:{_port_number}/"

    try:
        server = Server(
            {"/": Application(FunctionHandler(_make_realtime_plot_doc))},
            address=host,
            port=_port_number,
            allow_websocket_origin=[f"{host}:{_port_number}"],
            unused_session_lifetime_milliseconds=_unused_session_lifetime_milliseconds,
            extra_patterns=[
                (
                    "/images/(.*)",
                    StaticFileHandler,
                    {"path": os.path.normpath(os.path.dirname(__file__) + "/images/")},
                ),
            ],
        )
        server.start()

        testflo_running = os.environ.pop('TESTFLO_RUNNING', None)

        if not testflo_running:
            if open_browser:
                def _open_browser():
                    try:
                        opened = webbrowser.open_new_tab(app_url)
                    except Exception as err:
                        print(f"Automatic browser launch failed: {err}")
                        opened = False
                    if not opened:
                        print(f"Open this URL manually in a browser: {app_url}")

                server.io_loop.add_callback(_open_browser)
        else:
            # for testing, we are, for now, just testing that the command runs.
            # So can stop the plot process right away
            def update_data():
                raise KeyboardInterrupt("end plotting process when in testing mode")

            periodic_callback = PeriodicCallback(update_data, 1000)  # 1 second
            periodic_callback.start()

        print(
            f"Real-time optimization plot server running on {app_url}"
        )
        if not open_browser and not testflo_running:
            print(f"Open this URL manually in a browser: {app_url}")
        server.io_loop.start()
    except KeyboardInterrupt as e:
        print(
            f"Real-time optimization plot server stopped due to keyboard interrupt: {e}"
        )
    except Exception as e:
        print(f"Error starting real-time optimization plot server: {e}")
    finally:
        print("Stopping real-time optimization plot server")
        if server is not None:
            server.stop()
