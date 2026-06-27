"""
Slow Brachistochrone problem for replicating the SQL malformed-disk-image /
stops-updating bug in dymos-rtplot.

The ODE sleeps briefly on each compute call so that every major iteration takes
a few seconds.  This gives the Bokeh server time to load while keeping the
OpenMDAO SQLite writer and the dashboard reader in frequent contention.

Usage
-----
Run via the rtplot wrapper so the dashboard launches automatically::

    python -m dymos_rtplot.rtplot rtplot tests/slow_brachistochrone.py

Or run the script directly (produces only the recorder file)::

    python tests/slow_brachistochrone.py
"""

import os
import time

import numpy as np
import openmdao.api as om
import dymos as dm


# Per-call sleep: compute() receives all nn nodes as one batch, so this fires
# once per nonlinear-solver iteration, not once per node.  Analytical Jacobians
# ensure this sleep does NOT multiply out over CS perturbations.
_SLEEP_PER_EVAL = 0.3


class SlowBrachistochroneODE(om.ExplicitComponent):
    """Brachistochrone ODE with analytical Jacobians and a per-call sleep.

    Using analytical partials keeps the sleep from multiplying over the ~120
    complex-step perturbations that method='cs' would require.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('v',     val=np.zeros(nn), units='m/s')
        self.add_input('g',     val=9.80665,       units='m/s**2')
        self.add_input('theta', val=np.zeros(nn), units='rad')

        self.add_output('xdot',  val=np.zeros(nn), units='m/s')
        self.add_output('ydot',  val=np.zeros(nn), units='m/s')
        self.add_output('vdot',  val=np.zeros(nn), units='m/s**2')
        self.add_output('check', val=np.zeros(nn), units='m/s**3')

    def setup_partials(self):
        nn = self.options['num_nodes']
        arange = np.arange(nn)

        # Diagonal sparse patterns for (nn,)->(nn,) pairs
        self.declare_partials('xdot',  'v',     rows=arange, cols=arange)
        self.declare_partials('xdot',  'theta', rows=arange, cols=arange)
        self.declare_partials('ydot',  'v',     rows=arange, cols=arange)
        self.declare_partials('ydot',  'theta', rows=arange, cols=arange)
        self.declare_partials('vdot',  'theta', rows=arange, cols=arange)
        self.declare_partials('check', 'v',     rows=arange, cols=arange)
        self.declare_partials('check', 'theta', rows=arange, cols=arange)

        # Dense patterns for scalar g -> (nn,) outputs
        self.declare_partials('vdot',  'g')
        self.declare_partials('check', 'g')

    def compute(self, inputs, outputs):
        time.sleep(_SLEEP_PER_EVAL)

        theta     = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g         = inputs['g']
        v         = inputs['v']

        outputs['vdot']  = g * cos_theta
        outputs['xdot']  = v * sin_theta
        outputs['ydot']  = -v * cos_theta
        outputs['check'] = -g * cos_theta / v

    def compute_partials(self, inputs, partials):
        theta     = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g         = inputs['g']
        v         = inputs['v']

        partials['xdot',  'v']     = sin_theta
        partials['xdot',  'theta'] = v * cos_theta
        partials['ydot',  'v']     = -cos_theta
        partials['ydot',  'theta'] = v * sin_theta
        partials['vdot',  'theta'] = -g * sin_theta
        partials['vdot',  'g']     = cos_theta           # shape (nn,) broadcast
        partials['check', 'v']     = g * cos_theta / v**2
        partials['check', 'theta'] = g * sin_theta / v
        partials['check', 'g']     = -cos_theta / v      # shape (nn,) broadcast


def build_problem(recorder_file='slow_brachistochrone.sqlite'):
    """Return (problem, phase) so callers can use phase.interp for ICs."""
    p = om.Problem(model=om.Group())

    p.driver = om.ScipyOptimizeDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    p.driver.options['maxiter'] = 300
    p.driver.options['tol']     = 1e-8

    # Recorder is added here so the rtplot hook finds it pre-existing and
    # augments (record_outputs, includes) rather than creating its own.
    recorder = om.SqliteRecorder(recorder_file)
    p.driver.add_recorder(recorder)
    p.driver.recording_options['record_outputs']     = True
    p.driver.recording_options['record_derivatives'] = True
    p.driver.recording_options['includes']           = ['*']

    traj  = dm.Trajectory()
    # 10 segments GaussLobatto.  With analytical Jacobians and a 0.3 s sleep
    # per compute() call, each nonlinear-solver iteration costs ~0.3 s.
    # A typical major iteration runs 3-6 solver steps → 1-2 s per iteration.
    # At 50 major iterations that is ~1-2 minutes total, enough for the
    # Bokeh server to make many reads and hit the write/read race window.
    phase = dm.Phase(
        ode_class=SlowBrachistochroneODE,
        transcription=dm.GaussLobatto(num_segments=10, order=3),
    )
    traj.add_phase('phase0', phase)
    p.model.add_subsystem('traj', traj)

    phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0),
                           units='s')
    phase.add_state('x', rate_source='xdot', units='m',
                    fix_initial=True, fix_final=False)
    phase.add_state('y', rate_source='ydot', units='m',
                    fix_initial=True, fix_final=False)
    phase.add_state('v', rate_source='vdot', units='m/s',
                    fix_initial=True, fix_final=False)

    phase.add_control('theta', continuity=True, rate_continuity=True,
                      units='deg', lower=0.01, upper=179.9)
    phase.add_parameter('g', val=9.80665, units='m/s**2')

    # Minimise final time
    phase.add_objective('time', loc='final', scaler=10.0)

    # Terminal constraints via boundary constraints
    phase.add_boundary_constraint('x', loc='final', equals=10.0, units='m')
    phase.add_boundary_constraint('y', loc='final', equals=5.0,  units='m')

    return p, phase


if __name__ == '__main__':
    # Use an absolute path so the recorder is findable regardless of CWD
    recorder_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'slow_brachistochrone.sqlite')
    if os.path.exists(recorder_file):
        os.remove(recorder_file)

    p, phase = build_problem(recorder_file)
    p.setup(force_alloc_complex=True)

    # Reasonable initial guesses so SLSQP starts near a feasible point
    p.set_val('traj.phase0.t_initial', 0.0)
    p.set_val('traj.phase0.t_duration', 1.8)
    p.set_val('traj.phase0.states:x',     phase.interp('x',     ys=[0, 10]))
    p.set_val('traj.phase0.states:y',     phase.interp('y',     ys=[0, 5]))
    p.set_val('traj.phase0.states:v',     phase.interp('v',     ys=[0, 9.9]))
    p.set_val('traj.phase0.controls:theta', phase.interp('theta', ys=[5, 100.5]))
    p.set_val('traj.phase0.parameters:g', 9.80665)

    dm.run_problem(p, run_driver=True, simulate=False)

    p.record('final')
    p.cleanup()
    print('Done. Recorder written to', recorder_file)
