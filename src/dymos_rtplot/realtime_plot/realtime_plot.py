"""
Classes and functions to support the realtime plotting.
"""

import os
import sqlite3
import subprocess
import sys
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
from dymos_rtplot.realtime_plot.realtime_dashboard import _RealTimeDymosDashboard
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
               '--host', '127.0.0.1',
               '--meta-file', str(meta_path),
               case_recorder_file]
        if hist_file:
            cmd.insert(-1, '--hist-file')
            cmd.insert(-1, str(Path(hist_file).resolve()))
        if script_path:
            cmd.insert(-1, '--script')
            cmd.insert(-1, script_path)
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
            "--host",
            "127.0.0.1",
            case_recorder_file,
        ]
        if meta_path:
            cmd.insert(-1, '--meta-file')
            cmd.insert(-1, meta_path)

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


def realtime_plot(case_recorder_filename, callback_period,
                  pid_of_calling_script, script, meta_file=None, hist_file=None,
                  open_browser=False, host='127.0.0.1'):
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

    def _make_realtime_plot_doc(doc):
        print(f"Creating realtime plot document for {case_recorder_filename}")
        case_tracker = _CaseRecorderTracker(case_recorder_filename)
        case_tracker.set_source_process_pid(pid_of_calling_script)
        metadata = load_rtplot_metadata(meta_file=meta_file, case_recorder_filename=case_recorder_filename)
        if case_tracker.is_driver_optimizer():
            _RealTimeDymosDashboard(
                case_tracker,
                callback_period,
                doc=doc,
                pid_of_calling_script=pid_of_calling_script,
                script=script,
                metadata=metadata,
                hist_file=hist_file,
            )
        else:
            _RealTimeAnalysisDriverPlot(
                case_tracker,
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
