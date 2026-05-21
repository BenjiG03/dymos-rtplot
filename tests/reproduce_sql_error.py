"""
Reproducer for the SQL malformed-disk-image / stops-updating bug.

Runs the brachistochrone optimisation in a background thread while the main
thread hammers the same read pattern that the dymos-rtplot dashboard uses.
All output goes to stdout so errors can be captured in one place.

Usage
-----
    python tests/reproduce_sql_error.py
"""

import os
import sys
import sqlite3
import threading
import time
import pathlib

# ── ensure local src is importable ────────────────────────────────────────────
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "tests"))

import slow_brachistochrone as sb
import dymos as dm
import openmdao.api as om
from openmdao.recorders.sqlite_reader import SqliteCaseReader
from dymos_rtplot.realtime_plot.realtime_data import CaseRecorderAccess

RECORDER = str(repo_root / "tests" / "reproduce_sql_error.sqlite")


# ── Optimisation thread ────────────────────────────────────────────────────────

def run_optimization():
    if os.path.exists(RECORDER):
        os.remove(RECORDER)
    p, phase = sb.build_problem(RECORDER)
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
    """Replicate what LiveDataBroker + CaseRecorderAccess does every 300 ms."""
    db = RECORDER
    uri = pathlib.Path(db).resolve().as_uri() + '?mode=ro'

    access = None          # will become CaseRecorderAccess once file exists
    next_counter = 1
    n_ok = 0
    errors = []
    idle_since = None

    while not stop_event.is_set():
        # Wait for the recorder file to appear (mirrors dashboard startup)
        if not os.path.exists(db):
            time.sleep(0.1)
            continue

        # Lazy-init the access helper (mirrors get_case_reader lazy init)
        if access is None:
            try:
                access = CaseRecorderAccess(db, SqliteCaseReader)
            except Exception as e:
                print(f'[DASH] CaseRecorderAccess init error: {e}', flush=True)
                time.sleep(0.3)
                continue

        # ── Phase 1: lightweight row-existence check (readonly URI) ───────────
        try:
            with sqlite3.connect(uri, uri=True, timeout=1.0) as con:
                con.row_factory = sqlite3.Row
                cur = con.cursor()
                cur.execute(
                    'SELECT * FROM driver_iterations WHERE counter=:c',
                    {'c': next_counter},
                )
                row = cur.fetchone()
        except Exception as e:
            errors.append(('row_check', next_counter, str(e)))
            print(f'[DASH] ROW-CHECK ERROR counter={next_counter}: {e}', flush=True)
            time.sleep(0.3)
            continue

        if row is None:
            # Stop when optimizer is done and there has been nothing new for 3 s
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
        except Exception as e:
            errors.append(('case_read', next_counter, str(e)))
            print(f'[DASH] CASE-READ ERROR counter={next_counter}: {e}', flush=True)
            access.reset_case_reader()   # mirror what LiveDataBroker does
            # Do NOT advance counter — retry same row next cycle
            time.sleep(0.3)

    print(f'[DASH] Finished. {n_ok} successful reads, {len(errors)} errors.', flush=True)
    for kind, c, msg in errors:
        print(f'  [{kind}] counter={c}: {msg}')
    return errors


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    stop_event = threading.Event()

    opt_thread = threading.Thread(target=run_optimization, daemon=True)
    opt_thread.start()

    try:
        errors = dashboard_read_loop(stop_event, opt_thread)
    except KeyboardInterrupt:
        print('\n[MAIN] Interrupted.', flush=True)
        errors = []
    finally:
        stop_event.set()

    opt_thread.join(timeout=10)

    if os.path.exists(RECORDER):
        os.remove(RECORDER)

    n_errors = sum(1 for e in errors if 'malformed' in e[2].lower() or 'corrupt' in e[2].lower())
    if n_errors:
        print(f'\nREPRODUCED: {n_errors} malformed/corrupt errors during {len(errors)} total errors')
    else:
        print(f'\nNo malformed/corrupt errors in this run ({len(errors)} total errors of other kinds)')
