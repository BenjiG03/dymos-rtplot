"""Integration test: brachistochrone problems with pyoptsparse/IPOPT.

The convergence test uses a standard single-phase GaussLobatto brachistochrone
(well-understood to converge quickly) with IPOPT.

The metadata tests set up a three-phase trajectory (GaussLobatto + Radau + Birkhoff)
without running the optimizer; they exercise build_rtplot_metadata() for mixed
transcriptions and verify the recorder DB can be read concurrently.

Run requirements:
  - pyoptsparse installed with IPOPT backend (for IPOPTBrachistochroneConvergenceTests)
  - dymos-rtplot conda environment
"""

import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
import openmdao.api as om
import dymos as dm

try:
    from pyoptsparse import OPT as _pyo_opt
    import pyoptsparse  # noqa: F401
    _PYOPTSPARSE_AVAILABLE = True
except ImportError:
    _PYOPTSPARSE_AVAILABLE = False

try:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
    _DRIVER_AVAILABLE = True
except ImportError:
    _DRIVER_AVAILABLE = False


def _ipopt_available():
    if not _PYOPTSPARSE_AVAILABLE or not _DRIVER_AVAILABLE:
        return False
    try:
        opt = _pyo_opt('IPOPT')
        return opt is not None
    except Exception:
        return False


_SKIP_IPOPT = "pyoptsparse with IPOPT not available"
_SKIP_PYOPT = "pyoptsparse not available"


class BrachistochroneODE(om.ExplicitComponent):
    """Brachistochrone ODE with analytical Jacobians."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('v', val=np.zeros(nn), units='m/s')
        self.add_input('g', val=9.80665, units='m/s**2')
        self.add_input('theta', val=np.zeros(nn), units='rad')
        self.add_output('xdot', val=np.zeros(nn), units='m/s')
        self.add_output('ydot', val=np.zeros(nn), units='m/s')
        self.add_output('vdot', val=np.zeros(nn), units='m/s**2')

    def setup_partials(self):
        nn = self.options['num_nodes']
        arange = np.arange(nn)
        self.declare_partials('xdot', 'v', rows=arange, cols=arange)
        self.declare_partials('xdot', 'theta', rows=arange, cols=arange)
        self.declare_partials('ydot', 'v', rows=arange, cols=arange)
        self.declare_partials('ydot', 'theta', rows=arange, cols=arange)
        self.declare_partials('vdot', 'theta', rows=arange, cols=arange)
        self.declare_partials('vdot', 'g')

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']
        outputs['vdot'] = g * cos_theta
        outputs['xdot'] = v * sin_theta
        outputs['ydot'] = -v * cos_theta

    def compute_partials(self, inputs, partials):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']
        partials['xdot', 'v'] = sin_theta
        partials['xdot', 'theta'] = v * cos_theta
        partials['ydot', 'v'] = -cos_theta
        partials['ydot', 'theta'] = v * sin_theta
        partials['vdot', 'theta'] = -g * sin_theta
        partials['vdot', 'g'] = cos_theta


def _build_single_phase_brachistochrone(recorder_file=None, use_ipopt=True):
    """Classic single-phase brachistochrone with GaussLobatto transcription.

    This is the canonical dymos test problem; it converges reliably with IPOPT
    in ~20-30 iterations from the standard initial guess.
    """
    p = om.Problem()

    if use_ipopt and _DRIVER_AVAILABLE:
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.opt_settings['max_iter'] = 300
        p.driver.opt_settings['tol'] = 1.0e-8
        p.driver.opt_settings['print_level'] = 0
    else:
        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['maxiter'] = 300

    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)

    phase = dm.Phase(
        ode_class=BrachistochroneODE,
        transcription=dm.GaussLobatto(num_segments=10, order=3),
    )
    traj.add_phase('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0), units='s')
    phase.add_state('x', rate_source='xdot', units='m', fix_initial=True, fix_final=False)
    phase.add_state('y', rate_source='ydot', units='m', fix_initial=True, fix_final=False)
    phase.add_state('v', rate_source='vdot', units='m/s', fix_initial=True, fix_final=False)
    phase.add_control('theta', continuity=True, rate_continuity=True,
                      units='deg', lower=0.01, upper=179.9, shape=(1,))
    phase.add_parameter('g', val=9.80665, units='m/s**2', opt=False)

    phase.add_objective('time', loc='final', scaler=10.0)
    phase.add_boundary_constraint('x', loc='final', equals=10.0, units='m')
    phase.add_boundary_constraint('y', loc='final', equals=5.0, units='m')

    if recorder_file:
        rec = om.SqliteRecorder(recorder_file)
        p.driver.add_recorder(rec)
        p.driver.recording_options['record_outputs'] = True
        p.driver.recording_options['record_derivatives'] = True
        p.driver.recording_options['includes'] = ['*']

    p.setup(force_alloc_complex=False)

    p.set_val('traj.phase0.t_initial', 0.0)
    p.set_val('traj.phase0.t_duration', 1.8)
    p.set_val('traj.phase0.states:x', phase.interp('x', ys=[0, 10]))
    p.set_val('traj.phase0.states:y', phase.interp('y', ys=[10, 5]))
    p.set_val('traj.phase0.states:v', phase.interp('v', ys=[0, 9.9]))
    p.set_val('traj.phase0.controls:theta', phase.interp('theta', ys=[5, 100.5]))
    p.set_val('traj.phase0.parameters:g', 9.80665)

    return p, phase


def _build_three_phase_metadata_problem():
    """Three-phase trajectory for metadata tests (not for optimization convergence).

    Uses GaussLobatto + Radau + Birkhoff to exercise mixed-transcription metadata.
    Problem is set up but not optimized — initial guesses are rough.
    """
    p = om.Problem()
    p.driver = om.ScipyOptimizeDriver()

    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)

    tx0 = dm.GaussLobatto(num_segments=5, order=3)
    tx1 = dm.Radau(num_segments=5, order=3)
    tx2 = dm.Birkhoff(num_segments=1, order=9, grid_type='lgl')

    phase0 = dm.Phase(ode_class=BrachistochroneODE, transcription=tx0)
    phase1 = dm.Phase(ode_class=BrachistochroneODE, transcription=tx1)
    phase2 = dm.Phase(ode_class=BrachistochroneODE, transcription=tx2)

    traj.add_phase('phase0', phase0)
    traj.add_phase('phase1', phase1)
    traj.add_phase('phase2', phase2)

    for phase in (phase0, phase1, phase2):
        phase.set_time_options(fix_initial=False, duration_bounds=(0.01, 5.0), units='s')
        phase.add_state('x', rate_source='xdot', units='m', fix_initial=False, fix_final=False)
        phase.add_state('y', rate_source='ydot', units='m', fix_initial=False, fix_final=False)
        phase.add_state('v', rate_source='vdot', units='m/s', fix_initial=False, fix_final=False)
        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9, shape=(1,))
        phase.add_parameter('g', val=9.80665, units='m/s**2', opt=False)

    phase0.set_time_options(fix_initial=True, duration_bounds=(0.01, 5.0), units='s')
    traj.link_phases(['phase0', 'phase1'], vars=['time', 'x', 'y', 'v'])
    traj.link_phases(['phase1', 'phase2'], vars=['time', 'x', 'y', 'v'])
    phase2.add_boundary_constraint('x', loc='final', equals=10.0, units='m')
    phase2.add_boundary_constraint('y', loc='final', equals=5.0, units='m')
    phase2.add_objective('time', loc='final', scaler=10.0)

    p.setup(force_alloc_complex=False)
    return p


@unittest.skipUnless(_ipopt_available(), _SKIP_IPOPT)
class IPOPTBrachistochroneConvergenceTests(unittest.TestCase):
    """Full IPOPT optimization tests — require pyoptsparse + IPOPT."""

    def test_ipopt_single_phase_gl_converges(self):
        """Standard GaussLobatto brachistochrone must converge < 1.8 s with IPOPT."""
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / 'brach.db')
            p, phase = _build_single_phase_brachistochrone(recorder_file=db, use_ipopt=True)
            dm.run_problem(p, run_driver=True, simulate=False)
            p.record('final')
            p.cleanup()

            t_final = float(np.asarray(p.get_val('traj.phase0.timeseries.time')).flat[-1])
            self.assertLess(t_final, 2.0, f"Did not converge: final time = {t_final:.4f} s")

    def test_metadata_built_correctly_after_ipopt_run(self):
        """Metadata built after a converged IPOPT run has correct interp modes."""
        from dymos_rtplot.realtime_plot.realtime_metadata import build_rtplot_metadata
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / 'brach_meta.db')
            p, phase = _build_single_phase_brachistochrone(recorder_file=db, use_ipopt=True)
            dm.run_problem(p, run_driver=True, simulate=False)
            p.cleanup()

            meta = build_rtplot_metadata(p, db)

        phases = meta['trajectories'][0]['phases']
        self.assertEqual(len(phases), 1)
        ph = phases[0]
        self.assertEqual(ph['transcription'], 'GaussLobatto')
        self.assertIn('state_interp', ph)
        self.assertEqual(ph['state_interp']['mode'], 'hermite')
        self.assertIn('control_interp', ph)
        # Defect paths present for collocation-based transcription
        self.assertTrue(any('collocation' in k for k in ph['defect_outputs']))

    def test_recorder_db_readable_during_ipopt_run(self):
        """Concurrent recorder DB reads during IPOPT optimization must not corrupt."""
        import sqlite3
        from dymos_rtplot.realtime_plot.realtime_data import CaseRecorderAccess
        from openmdao.recorders.sqlite_reader import SqliteCaseReader

        errors = []
        stop = threading.Event()

        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / 'concurrent.db')
            p, phase = _build_single_phase_brachistochrone(recorder_file=db, use_ipopt=True)

            def _reader_thread():
                access = CaseRecorderAccess(db, SqliteCaseReader)
                counter = 1
                while not stop.is_set():
                    try:
                        case = access.get_case_by_counter(counter)
                        if case is not None:
                            counter += 1
                    except sqlite3.DatabaseError as exc:
                        if 'malformed' in str(exc).lower():
                            errors.append(str(exc))
                    except Exception:
                        pass
                    time.sleep(0.05)

            reader = threading.Thread(target=_reader_thread, daemon=True)
            reader.start()
            time.sleep(0.2)

            dm.run_problem(p, run_driver=True, simulate=False)
            p.cleanup()
            stop.set()
            reader.join(timeout=5)

        self.assertEqual(errors, [],
            msg=f"Malformed DB errors during concurrent read: {errors}")


@unittest.skipUnless(
    _PYOPTSPARSE_AVAILABLE and _DRIVER_AVAILABLE,
    _SKIP_PYOPT,
)
class MetadataBuildWithoutRunTests(unittest.TestCase):
    """Metadata tests that require pyoptsparse driver class to be present."""

    def test_driver_class_reported_correctly(self):
        from dymos_rtplot.realtime_plot.realtime_metadata import build_rtplot_metadata
        p, _ = _build_single_phase_brachistochrone(use_ipopt=True)
        meta = build_rtplot_metadata(p, 'dummy.db')
        self.assertEqual(meta['driver_class'], 'pyOptSparseDriver')
        self.assertEqual(meta['optimizer'], 'IPOPT')


class ThreePhaseMetadataTests(unittest.TestCase):
    """Mixed-transcription metadata tests that don't require IPOPT or pyoptsparse."""

    def setUp(self):
        self.p = _build_three_phase_metadata_problem()

    def test_metadata_all_three_phases_present(self):
        from dymos_rtplot.realtime_plot.realtime_metadata import build_rtplot_metadata
        meta = build_rtplot_metadata(self.p, 'dummy.db')
        self.assertEqual(len(meta['trajectories']), 1)
        phases = meta['trajectories'][0]['phases']
        self.assertEqual(len(phases), 3)

    def test_phase0_gl_has_hermite_state_interp(self):
        from dymos_rtplot.realtime_plot.realtime_metadata import build_rtplot_metadata
        meta = build_rtplot_metadata(self.p, 'dummy.db')
        ph = meta['trajectories'][0]['phases'][0]
        self.assertEqual(ph['transcription'], 'GaussLobatto')
        self.assertIn('state_interp', ph)
        self.assertEqual(ph['state_interp']['mode'], 'hermite')
        self.assertIn('control_interp', ph)
        self.assertTrue(any('collocation' in k for k in ph['defect_outputs']))

    def test_phase1_radau_has_lagrange_state_interp(self):
        from dymos_rtplot.realtime_plot.realtime_metadata import build_rtplot_metadata
        meta = build_rtplot_metadata(self.p, 'dummy.db')
        ph = meta['trajectories'][0]['phases'][1]
        self.assertEqual(ph['transcription'], 'Radau')
        self.assertIn('state_interp', ph)
        self.assertEqual(ph['state_interp']['mode'], 'lagrange')
        self.assertIn('control_interp', ph)

    def test_phase2_birkhoff_has_lagrange_no_continuity(self):
        from dymos_rtplot.realtime_plot.realtime_metadata import build_rtplot_metadata
        meta = build_rtplot_metadata(self.p, 'dummy.db')
        ph = meta['trajectories'][0]['phases'][2]
        self.assertEqual(ph['transcription'], 'Birkhoff')
        self.assertIn('state_interp', ph)
        self.assertEqual(ph['state_interp']['mode'], 'lagrange')
        # Birkhoff with 1 segment has no interior continuity nodes
        for key, val in ph['defect_outputs'].items():
            if 'continuity' in key:
                self.assertEqual(val['node_ptau'], [],
                    "Birkhoff 1-segment phase should have no continuity ptau")

    def test_recorder_db_readable_during_setup_phase(self):
        """CaseRecorderAccess on a freshly created DB returns no cases but no errors."""
        import sqlite3
        from dymos_rtplot.realtime_plot.realtime_data import CaseRecorderAccess
        from openmdao.recorders.sqlite_reader import SqliteCaseReader

        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / 'setup_only.db')
            rec = om.SqliteRecorder(db)
            self.p.driver.add_recorder(rec)
            self.p.setup(force_alloc_complex=False)
            self.p.run_model()
            self.p.cleanup()

            access = CaseRecorderAccess(db, SqliteCaseReader)
            # No driver iterations recorded yet → should return None without error
            case = access.get_case_by_counter(1)
            self.assertIsNone(case)


if __name__ == '__main__':
    unittest.main()
