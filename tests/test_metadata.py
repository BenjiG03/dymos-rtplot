"""Unit tests for build_rtplot_metadata() across all dymos transcription types."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import openmdao.api as om
import dymos as dm

from dymos_rtplot.realtime_plot.realtime_metadata import (
    build_rtplot_metadata,
    load_rtplot_metadata,
    write_rtplot_metadata,
    _is_shooting_transcription,
    _supports_exact_state_interp,
    _supports_exact_control_interp,
)


class SimpleODE(om.ExplicitComponent):
    """Minimal ODE: xdot = x, used across all transcription tests."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('x', val=np.zeros(nn), units='m')
        self.add_output('xdot', val=np.zeros(nn), units='m/s')

    def setup_partials(self):
        nn = self.options['num_nodes']
        arange = np.arange(nn)
        self.declare_partials('xdot', 'x', rows=arange, cols=arange, val=1.0)

    def compute(self, inputs, outputs):
        outputs['xdot'] = inputs['x']


def _build_single_phase_problem(transcription, recorder_file=None):
    """Set up a minimal single-phase trajectory problem without running it."""
    p = om.Problem()
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)
    phase = dm.Phase(ode_class=SimpleODE, transcription=transcription)
    traj.add_phase('phase0', phase)
    phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0), units='s')
    phase.add_state('x', rate_source='xdot', units='m', fix_initial=True)
    if recorder_file:
        rec = om.SqliteRecorder(recorder_file)
        p.driver.add_recorder(rec)
    p.setup()
    return p, phase


class ShootingTranscriptionHelperTests(unittest.TestCase):
    def test_is_shooting_explicit(self):
        self.assertTrue(_is_shooting_transcription(dm.ExplicitShooting(num_segments=2, order=3)))

    def test_is_shooting_picard(self):
        self.assertTrue(_is_shooting_transcription(dm.PicardShooting(num_segments=2, order=3)))

    def test_not_shooting_gausslobatto(self):
        self.assertFalse(_is_shooting_transcription(dm.GaussLobatto(num_segments=2, order=3)))

    def test_not_shooting_radau(self):
        self.assertFalse(_is_shooting_transcription(dm.Radau(num_segments=2, order=3)))

    def test_not_shooting_birkhoff(self):
        self.assertFalse(_is_shooting_transcription(dm.Birkhoff(num_segments=1, order=5, grid_type='lgl')))


class ExactInterpSupportTests(unittest.TestCase):
    def test_state_interp_gausslobatto(self):
        self.assertTrue(_supports_exact_state_interp(dm.GaussLobatto(num_segments=2, order=3)))

    def test_state_interp_radau(self):
        self.assertTrue(_supports_exact_state_interp(dm.Radau(num_segments=2, order=3)))

    def test_state_interp_birkhoff(self):
        self.assertTrue(_supports_exact_state_interp(dm.Birkhoff(num_segments=1, order=5, grid_type='lgl')))

    def test_no_state_interp_explicit_shooting(self):
        self.assertFalse(_supports_exact_state_interp(dm.ExplicitShooting(num_segments=2, order=3)))

    def test_no_state_interp_picard_shooting(self):
        self.assertFalse(_supports_exact_state_interp(dm.PicardShooting(num_segments=2, order=3)))

    def test_control_interp_gausslobatto(self):
        t = dm.GaussLobatto(num_segments=2, order=3)
        self.assertTrue(_supports_exact_control_interp(t, t.grid_data))

    def test_control_interp_radau(self):
        t = dm.Radau(num_segments=2, order=3)
        self.assertTrue(_supports_exact_control_interp(t, t.grid_data))

    def test_no_control_interp_explicit_shooting(self):
        t = dm.ExplicitShooting(num_segments=2, order=3)
        self.assertFalse(_supports_exact_control_interp(t, t.grid_data))

    def test_no_control_interp_picard_shooting(self):
        t = dm.PicardShooting(num_segments=2, order=3)
        self.assertFalse(_supports_exact_control_interp(t, t.grid_data))


class GaussLobattoMetadataTests(unittest.TestCase):
    def setUp(self):
        self.p, self.phase = _build_single_phase_problem(
            dm.GaussLobatto(num_segments=4, order=3)
        )
        self.meta = build_rtplot_metadata(self.p, 'dummy.db')

    def test_single_trajectory(self):
        self.assertEqual(len(self.meta['trajectories']), 1)

    def test_single_phase(self):
        self.assertEqual(len(self.meta['trajectories'][0]['phases']), 1)

    def test_transcription_name(self):
        ph = self.meta['trajectories'][0]['phases'][0]
        self.assertEqual(ph['transcription'], 'GaussLobatto')

    def test_state_interp_hermite(self):
        ph = self.meta['trajectories'][0]['phases'][0]
        self.assertIn('state_interp', ph)
        self.assertEqual(ph['state_interp']['mode'], 'hermite')
        for key in ('Ai', 'Bi', 'Ad', 'Bd'):
            self.assertIn(key, ph['state_interp'])

    def test_control_interp_present(self):
        ph = self.meta['trajectories'][0]['phases'][0]
        self.assertIn('control_interp', ph)
        for key in ('L', 'D', 'D2'):
            self.assertIn(key, ph['control_interp'])

    def test_defect_outputs_have_collocation_paths(self):
        ph = self.meta['trajectories'][0]['phases'][0]
        defects = ph['defect_outputs']
        self.assertTrue(any('collocation' in k for k in defects))
        col_x = defects.get('collocation:x')
        self.assertIsNotNone(col_x)
        self.assertIn('collocation_constraint', col_x['path'])


class RadauMetadataTests(unittest.TestCase):
    def setUp(self):
        self.p, self.phase = _build_single_phase_problem(
            dm.Radau(num_segments=4, order=3)
        )
        self.meta = build_rtplot_metadata(self.p, 'dummy.db')

    def test_state_interp_lagrange(self):
        ph = self.meta['trajectories'][0]['phases'][0]
        self.assertIn('state_interp', ph)
        self.assertEqual(ph['state_interp']['mode'], 'lagrange')
        for key in ('L', 'D'):
            self.assertIn(key, ph['state_interp'])

    def test_control_interp_present(self):
        ph = self.meta['trajectories'][0]['phases'][0]
        self.assertIn('control_interp', ph)

    def test_defect_outputs_non_empty(self):
        ph = self.meta['trajectories'][0]['phases'][0]
        self.assertTrue(len(ph['defect_outputs']) > 0)


class BirkhoffMetadataTests(unittest.TestCase):
    def setUp(self):
        self.p, self.phase = _build_single_phase_problem(
            dm.Birkhoff(num_segments=1, order=9, grid_type='lgl')
        )
        self.meta = build_rtplot_metadata(self.p, 'dummy.db')

    def test_state_interp_lagrange(self):
        ph = self.meta['trajectories'][0]['phases'][0]
        self.assertIn('state_interp', ph)
        self.assertEqual(ph['state_interp']['mode'], 'lagrange')

    def test_empty_continuity_ptau_single_segment(self):
        ph = self.meta['trajectories'][0]['phases'][0]
        for key, val in ph['defect_outputs'].items():
            if 'continuity-state' in key:
                self.assertEqual(val['node_ptau'], [],
                    "Birkhoff 1-segment should have no interior continuity nodes")

    def test_control_interp_present(self):
        ph = self.meta['trajectories'][0]['phases'][0]
        self.assertIn('control_interp', ph)


class ExplicitShootingMetadataTests(unittest.TestCase):
    def setUp(self):
        self.p, self.phase = _build_single_phase_problem(
            dm.ExplicitShooting(num_segments=3, order=3)
        )
        self.meta = build_rtplot_metadata(self.p, 'dummy.db')

    def test_no_state_interp(self):
        ph = self.meta['trajectories'][0]['phases'][0]
        self.assertNotIn('state_interp', ph)

    def test_no_control_interp(self):
        ph = self.meta['trajectories'][0]['phases'][0]
        self.assertNotIn('control_interp', ph)

    def test_empty_defect_outputs(self):
        ph = self.meta['trajectories'][0]['phases'][0]
        self.assertEqual(ph['defect_outputs'], {})

    def test_state_present_in_metadata(self):
        ph = self.meta['trajectories'][0]['phases'][0]
        self.assertIn('x', ph['states'])


class PicardShootingMetadataTests(unittest.TestCase):
    def setUp(self):
        self.p, self.phase = _build_single_phase_problem(
            dm.PicardShooting(num_segments=3, order=3)
        )
        self.meta = build_rtplot_metadata(self.p, 'dummy.db')

    def test_no_state_interp(self):
        ph = self.meta['trajectories'][0]['phases'][0]
        self.assertNotIn('state_interp', ph)

    def test_no_control_interp(self):
        ph = self.meta['trajectories'][0]['phases'][0]
        self.assertNotIn('control_interp', ph)

    def test_empty_defect_outputs(self):
        ph = self.meta['trajectories'][0]['phases'][0]
        self.assertEqual(ph['defect_outputs'], {})


class MixedTranscriptionMetadataTests(unittest.TestCase):
    def _build_two_phase_problem(self, tx0, tx1):
        p = om.Problem()
        traj = dm.Trajectory()
        p.model.add_subsystem('traj', traj)
        for name, tx in (('phase0', tx0), ('phase1', tx1)):
            phase = dm.Phase(ode_class=SimpleODE, transcription=tx)
            traj.add_phase(name, phase)
            phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 5.0), units='s')
            phase.add_state('x', rate_source='xdot', units='m', fix_initial=True)
        p.setup()
        return p

    def test_gl_plus_radau(self):
        p = self._build_two_phase_problem(
            dm.GaussLobatto(num_segments=3, order=3),
            dm.Radau(num_segments=3, order=3),
        )
        meta = build_rtplot_metadata(p, 'dummy.db')
        phases = meta['trajectories'][0]['phases']
        self.assertEqual(len(phases), 2)
        ph0, ph1 = phases
        self.assertEqual(ph0['state_interp']['mode'], 'hermite')
        self.assertEqual(ph1['state_interp']['mode'], 'lagrange')

    def test_gl_plus_birkhoff(self):
        p = self._build_two_phase_problem(
            dm.GaussLobatto(num_segments=3, order=3),
            dm.Birkhoff(num_segments=1, order=7, grid_type='lgl'),
        )
        meta = build_rtplot_metadata(p, 'dummy.db')
        phases = meta['trajectories'][0]['phases']
        ph0, ph1 = phases
        self.assertEqual(ph0['state_interp']['mode'], 'hermite')
        self.assertEqual(ph1['state_interp']['mode'], 'lagrange')

    def test_gl_plus_explicit_shooting(self):
        p = self._build_two_phase_problem(
            dm.GaussLobatto(num_segments=3, order=3),
            dm.ExplicitShooting(num_segments=3, order=3),
        )
        meta = build_rtplot_metadata(p, 'dummy.db')
        phases = meta['trajectories'][0]['phases']
        ph0, ph1 = phases
        self.assertIn('state_interp', ph0)
        self.assertNotIn('state_interp', ph1)
        self.assertNotIn('control_interp', ph1)
        self.assertEqual(ph1['defect_outputs'], {})

    def test_radau_plus_picard_shooting(self):
        p = self._build_two_phase_problem(
            dm.Radau(num_segments=3, order=3),
            dm.PicardShooting(num_segments=3, order=3),
        )
        meta = build_rtplot_metadata(p, 'dummy.db')
        phases = meta['trajectories'][0]['phases']
        ph0, ph1 = phases
        self.assertIn('state_interp', ph0)
        self.assertNotIn('state_interp', ph1)
        self.assertEqual(ph1['defect_outputs'], {})

    def test_five_phase_all_different_transcriptions(self):
        p = om.Problem()
        traj = dm.Trajectory()
        p.model.add_subsystem('traj', traj)
        transcriptions = [
            ('phase_gl', dm.GaussLobatto(num_segments=2, order=3)),
            ('phase_radau', dm.Radau(num_segments=2, order=3)),
            ('phase_birkhoff', dm.Birkhoff(num_segments=1, order=5, grid_type='lgl')),
            ('phase_es', dm.ExplicitShooting(num_segments=2, order=3)),
            ('phase_picard', dm.PicardShooting(num_segments=2, order=3)),
        ]
        for name, tx in transcriptions:
            phase = dm.Phase(ode_class=SimpleODE, transcription=tx)
            traj.add_phase(name, phase)
            phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 5.0), units='s')
            phase.add_state('x', rate_source='xdot', units='m', fix_initial=True)
        p.setup()
        meta = build_rtplot_metadata(p, 'dummy.db')
        phases = meta['trajectories'][0]['phases']
        self.assertEqual(len(phases), 5)
        for ph in phases:
            self.assertIn('states', ph)
            self.assertIn('x', ph['states'])


class MultipleTrajectoryMetadataTests(unittest.TestCase):
    def test_two_trajectories(self):
        p = om.Problem()
        for traj_name in ('traj0', 'traj1'):
            traj = dm.Trajectory()
            p.model.add_subsystem(traj_name, traj)
            phase = dm.Phase(
                ode_class=SimpleODE,
                transcription=dm.GaussLobatto(num_segments=3, order=3),
            )
            traj.add_phase('phase0', phase)
            phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 5.0), units='s')
            phase.add_state('x', rate_source='xdot', units='m', fix_initial=True)
        p.setup()
        meta = build_rtplot_metadata(p, 'dummy.db')
        self.assertEqual(len(meta['trajectories']), 2)
        names = {t['name'] for t in meta['trajectories']}
        self.assertEqual(names, {'traj0', 'traj1'})


class PolynomialControlMetadataTests(unittest.TestCase):
    def test_polynomial_control_type_in_metadata(self):
        p = om.Problem()
        traj = dm.Trajectory()
        p.model.add_subsystem('traj', traj)
        phase = dm.Phase(
            ode_class=SimpleODE,
            transcription=dm.GaussLobatto(num_segments=4, order=3),
        )
        traj.add_phase('phase0', phase)
        phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 5.0), units='s')
        phase.add_state('x', rate_source='xdot', units='m', fix_initial=True)
        phase.add_control('u', units='N', lower=-10.0, upper=10.0, shape=(1,))
        phase.add_control('u_poly', units='N', control_type='polynomial', order=2, shape=(1,))
        p.setup()
        meta = build_rtplot_metadata(p, 'dummy.db')
        ph = meta['trajectories'][0]['phases'][0]
        self.assertIn('u', ph['controls'])
        self.assertEqual(ph['controls']['u']['control_type'], 'full')
        self.assertIn('u_poly', ph['controls'])
        self.assertEqual(ph['controls']['u_poly']['control_type'], 'polynomial')
        # polynomial control path should point to polynomial_controls:
        self.assertIn('polynomial_controls', ph['controls']['u_poly']['path'])


class MetadataRoundtripTests(unittest.TestCase):
    def test_json_roundtrip(self):
        p, _ = _build_single_phase_problem(dm.GaussLobatto(num_segments=3, order=3))
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / 'test.db')
            meta_path = write_rtplot_metadata(p, db)
            loaded = load_rtplot_metadata(case_recorder_filename=db)
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded['trajectories']), 1)
        ph = loaded['trajectories'][0]['phases'][0]
        self.assertIn('state_interp', ph)
        # matrices survive JSON round-trip as lists
        self.assertIsInstance(ph['state_interp']['Ai'], list)

    def test_missing_meta_file_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            result = load_rtplot_metadata(case_recorder_filename=str(Path(td) / 'nonexistent.db'))
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
