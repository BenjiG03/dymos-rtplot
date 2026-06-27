"""
Reproducer for the SQL malformed-disk-image / stops-updating bug,
using pyOptSparseDriver + IPOPT so that:

  - The IPOPT line-search generates many more .hst file writes per
    major iteration than SLSQP does, driving OptimizerHistoryAccess
    into frequent contention with the writer.
  - Every pyOptsparse function evaluation (including line-search calls)
    writes to the .hst file; the dashboard's shutil.copy2 snapshot path
    has to race against that stream.

Usage
-----
    python tests/reproduce_ipopt_error.py
"""

import os
import sys
import sqlite3
import threading
import time
import pathlib

# ── ensure local src is importable ────────────────────────────────────────────
repo_root = pathlib.Path(os.path.abspath(__file__)).parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "tests"))

import slow_brachistochrone as sb
import dymos as dm
import openmdao.api as om
from openmdao.recorders.sqlite_reader import SqliteCaseReader
from dymos_rtplot.realtime_plot.realtime_data import (
    CaseRecorderAccess,
    OptimizerHistoryAccess,
)

try:
    from pyoptsparse.pyOpt_history import History as PyOptHistory
    HAS_PYOPTSPARSE = True
except ImportError:
    PyOptHistory = None
    HAS_PYOPTSPARSE = False

RECORDER = str(repo_root / "tests" / "reproduce_ipopt_error.sqlite")
HIST_FILE = str(repo_root / "tests" / "reproduce_ipopt_error.hst")


# ── Optimisation thread ────────────────────────────────────────────────────────

def run_optimization():
    for path in (RECORDER, HIST_FILE):
        if os.path.exists(path):
            os.remove(path)

    p, phase = sb.build_problem(RECORDER)

    # Swap to pyOptSparseDriver + IPOPT if available
    if HAS_PYOPTSPARSE:
        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.options['print_results'] = False
        # Attach history file so dashboard's OptimizerHistoryAccess can poll it
        p.driver.options['hist_file'] = HIST_FILE
        # IPOPT settings: allow plenty of iterations, keep tolerances loose so
        # line-search fires frequently and .hst is written rapidly
        p.driver.opt_settings['max_iter'] = 150
        p.driver.opt_settings['tol'] = 1e-6
        p.driver.opt_settings['acceptable_tol'] = 1e-4
        p.driver.opt_settings['acceptable_iter'] = 5
        # Re-attach the recorder (build_problem already added one to the old driver)
        recorder = om.SqliteRecorder(RECORDER)
        p.driver.add_recorder(recorder)
        p.driver.recording_options['record_outputs']     = True
        p.driver.recording_options['record_derivatives'] = True
        p.driver.recording_options['includes']           = ['*']
    else:
        print('[OPT] pyoptsparse not available – falling back to SLSQP', flush=True)
        p.driver.options['maxiter'] = 150

    p.setup(force_alloc_complex=True)
    p.set_val('traj.phase0.t_initial', 0.0)
    p.set_val('traj.phase0.t_duration', 1.8)
    p.set_val('traj.phase0.states:x',       phase.interp('x',     ys=[0, 10]))
    p.set_val('traj.phase0.states:y',       phase.interp('y',     ys=[0, 5]))
    p.set_val('traj.phase0.states:v',       phase.interp('v',     ys=[0, 9.9]))
    p.set_val('traj.phase0.controls:theta', phase.interp('theta', ys=[5, 100.5]))
    p.set_val('traj.phase0.parameters:g',   9.80665)

    dm.run_problem(p, run_driver=True, simulate=False)
    p.cleanup()
    print('[OPT] Done.', flush=True)


# ── Dashboard-style read loop ──────────────────────────────────────────────────

def dashboard_read_loop(stop_event, opt_thread):
    """
    Replicate what LiveDataBroker + CaseRecorderAccess + OptimizerHistoryAccess
    does every 300 ms.
    """
    db = RECORDER
    # Use os.path.abspath to avoid \\?\ extended-length prefix → %3F URI encoding
    uri = pathlib.Path(os.path.abspath(db)).as_uri() + '?mode=ro'

    access      = None
    hist_access = OptimizerHistoryAccess(
        hist_file=HIST_FILE,
        history_factory=PyOptHistory,  # None if pyoptsparse not installed
    )

    next_counter  = 1
    n_ok          = 0
    sql_errors    = []
    hist_errors   = []
    hist_warnings = []
    idle_since    = None
    prev_mtime    = None

    while not stop_event.is_set():
        # ── Wait for recorder file ────────────────────────────────────────────
        if not os.path.exists(db):
            time.sleep(0.1)
            continue

        # ── Lazy-init case access ─────────────────────────────────────────────
        if access is None:
            try:
                access = CaseRecorderAccess(db, SqliteCaseReader)
            except Exception as exc:
                print(f'[DASH] CaseRecorderAccess init error: {exc}', flush=True)
                time.sleep(0.3)
                continue

        # ── Phase 1: row existence check (readonly URI) ───────────────────────
        try:
            with sqlite3.connect(uri, uri=True, timeout=1.0) as con:
                con.row_factory = sqlite3.Row
                cur = con.cursor()
                cur.execute(
                    'SELECT * FROM driver_iterations WHERE counter=:c',
                    {'c': next_counter},
                )
                row = cur.fetchone()
        except Exception as exc:
            sql_errors.append(('row_check', next_counter, str(exc)))
            print(f'[DASH] ROW-CHECK ERROR counter={next_counter}: {exc}', flush=True)
            time.sleep(0.3)
            continue

        if row is None:
            if not opt_thread.is_alive():
                if idle_since is None:
                    idle_since = time.time()
                elif time.time() - idle_since > 3.0:
                    break
            time.sleep(0.3)
            continue
        idle_since = None

        # ── Phase 2: full case read via cached SqliteCaseReader ───────────────
        try:
            case = access.get_case_reader()._driver_cases.get_case(
                row['iteration_coordinate']
            )
            _ = case.get_objectives(scaled=False)
            _ = case.get_design_vars(scaled=False)
            _ = case.get_constraints(scaled=False)
            n_ok += 1
            if n_ok % 25 == 0:
                print(f'[DASH] {n_ok} cases read (counter={next_counter})', flush=True)
            next_counter += 1
        except Exception as exc:
            sql_errors.append(('case_read', next_counter, str(exc)))
            print(f'[DASH] CASE-READ ERROR counter={next_counter}: {exc}', flush=True)
            access.reset_case_reader()
            time.sleep(0.3)
            continue

        # ── Phase 3: optimizer history read (shutil.copy2 race) ──────────────
        if HAS_PYOPTSPARSE and os.path.exists(HIST_FILE):
            try:
                snapshot = hist_access.read_if_changed(prev_mtime)
                if snapshot is not None:
                    if snapshot.warning:
                        hist_warnings.append((next_counter - 1, snapshot.warning))
                        print(f'[HIST] WARNING at counter={next_counter - 1}: {snapshot.warning}',
                              flush=True)
                    else:
                        prev_mtime = snapshot.mtime
                        if len(snapshot.entries) > 0 and len(snapshot.entries) % 10 == 0:
                            print(f'[HIST] {len(snapshot.entries)} history entries read',
                                  flush=True)
            except Exception as exc:
                hist_errors.append((next_counter - 1, str(exc)))
                print(f'[HIST] ERROR at counter={next_counter - 1}: {exc}', flush=True)

        time.sleep(0.05)  # 50 ms between polls – more aggressive than dashboard's 300 ms

    print(
        f'[DASH] Finished. {n_ok} successful SQL reads, '
        f'{len(sql_errors)} SQL errors, '
        f'{len(hist_warnings)} hist warnings, '
        f'{len(hist_errors)} hist errors.',
        flush=True,
    )
    for kind, c, msg in sql_errors:
        print(f'  [sql:{kind}] counter={c}: {msg}')
    for c, msg in hist_warnings:
        print(f'  [hist:warn] at_counter={c}: {msg}')
    for c, msg in hist_errors:
        print(f'  [hist:err]  at_counter={c}: {msg}')
    return sql_errors, hist_warnings, hist_errors


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if not HAS_PYOPTSPARSE:
        print('WARNING: pyoptsparse not installed; running SLSQP fallback '
              '(history-file race will not be exercised)', flush=True)

    stop_event = threading.Event()
    opt_thread = threading.Thread(target=run_optimization, daemon=True)
    opt_thread.start()

    try:
        sql_errors, hist_warnings, hist_errors = dashboard_read_loop(stop_event, opt_thread)
    except KeyboardInterrupt:
        print('\n[MAIN] Interrupted.', flush=True)
        sql_errors = hist_warnings = hist_errors = []
    finally:
        stop_event.set()

    opt_thread.join(timeout=10)

    # Give Windows a moment to release file handles after the optimizer thread exits.
    time.sleep(1.0)
    for path in (RECORDER, HIST_FILE):
        try:
            if os.path.exists(path):
                os.remove(path)
        except PermissionError as exc:
            print(f'[MAIN] Could not remove {path}: {exc}', flush=True)

    # -- Summary ----------------------------------------------------------------
    corrupt_sql = [e for e in sql_errors if 'malformed' in e[2].lower() or 'corrupt' in e[2].lower()]
    lock_sql    = [e for e in sql_errors if 'locked' in e[2].lower() or 'busy' in e[2].lower()]
    perm_hist   = [e for e in hist_errors if 'permission' in e[1].lower() or 'access' in e[1].lower()]

    print('\n-- Run summary --------------------------------------------------')
    print(f'SQL corrupt/malformed errors : {len(corrupt_sql)}')
    print(f'SQL locked/busy errors       : {len(lock_sql)}')
    print(f'SQL other errors             : {len(sql_errors) - len(corrupt_sql) - len(lock_sql)}')
    print(f'Hist file warnings           : {len(hist_warnings)}')
    print(f'Hist file PermissionErrors   : {len(perm_hist)}')
    print(f'Hist other errors            : {len(hist_errors) - len(perm_hist)}')
