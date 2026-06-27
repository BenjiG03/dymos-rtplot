"""Unit tests for _TrajectoryTab trace logic — no browser, no Bokeh server."""

import unittest
from unittest import mock

import numpy as np

from dymos_rtplot.realtime_plot import realtime_dashboard
from dymos_rtplot.realtime_plot.realtime_dashboard import _TrajectoryTab


def _make_broker(metadata=None):
    broker = mock.Mock()
    broker.metadata = metadata or {}
    return broker


def _dense_grid(n=21):
    ptau = np.linspace(-1, 1, n)
    stau = np.linspace(-1, 1, n)
    return {
        'num_nodes': n,
        'node_ptau': ptau.tolist(),
        'node_stau': stau.tolist(),
        'node_dptau_dstau': np.ones(n).tolist(),
        'segment_indices': list(range(n)),
    }


def _lagrange_interp(n_state=4, n_dense=21):
    """Return a minimal Lagrange interp dict with correct matrix shapes."""
    L = np.eye(n_dense, n_state).tolist()
    D = np.zeros((n_dense, n_state)).tolist()
    return {'mode': 'lagrange', 'L': L, 'D': D}


def _hermite_interp(n_state=4, n_dense=21):
    """Return a minimal Hermite interp dict."""
    Ai = np.eye(n_dense, n_state).tolist()
    Bi = np.zeros((n_dense, n_state)).tolist()
    Ad = np.zeros((n_dense, n_state)).tolist()
    Bd = np.eye(n_dense, n_state).tolist()
    return {'mode': 'hermite', 'Ai': Ai, 'Bi': Bi, 'Ad': Ad, 'Bd': Bd}


def _make_phase_meta(name='phase0', with_state_interp=None, rate_source=None):
    """Build a minimal phase metadata dict."""
    return {
        'name': name,
        'promoted_path': f'traj.{name}',
        'transcription': 'GaussLobatto',
        'transcription_name': 'gauss-lobatto',
        'segment_ends': [-1.0, 1.0],
        'compressed': False,
        'state_input_to_disc': [0, 1, 2, 3],
        'dense_grid': _dense_grid(21),
        'state_input_node_ptau': [-1.0, -0.5, 0.5, 1.0],
        'control_input_node_ptau': [-1.0, 0.0, 1.0],
        'defect_outputs': {},
        'states': {
            'x': {
                'units': 'm',
                'shape': [1],
                'lower': None,
                'upper': None,
                'rate_source': rate_source,
            }
        },
        'controls': {},
        'timeseries_outputs': {
            'states:x': {
                'path': 'timeseries.states:x',
                'name': 'x',
                'units': 'm',
                'category': 'state',
            }
        },
        'path_constraints': [],
        'boundary_constraints': {'initial': [], 'final': []},
        **(({'state_interp': with_state_interp} if with_state_interp is not None else {})),
    }


class StateLagrangeTraceTests(unittest.TestCase):
    def setUp(self):
        meta = {
            'trajectories': [{'name': 'traj', 'phases': []}]
        }
        self.tab = _TrajectoryTab(_make_broker(meta))

    def _make_case(self, x_vals, t_initial=0.0, t_duration=1.0):
        n = len(x_vals)
        timeseries_time = np.linspace(t_initial, t_initial + t_duration, n)
        case = mock.Mock()
        case.get_val = mock.Mock(side_effect=lambda path: {
            'traj.phase0.states:x': np.array(x_vals).reshape(-1, 1),
            'traj.phase0.timeseries.states:x': np.column_stack([timeseries_time]),
            'traj.phase0.timeseries.time': timeseries_time.reshape(-1, 1),
            'traj.phase0.t_duration': np.array([t_duration]),
            'traj.phase0.t_initial': np.array([t_initial]),
        }[path])
        return case

    def test_lagrange_state_trace_happy_path(self):
        n_state = 4
        phase_meta = _make_phase_meta(
            with_state_interp=_lagrange_interp(n_state=n_state, n_dense=21),
        )
        case = self._make_case(list(range(n_state)))
        xvals, yvals, violation, warning = self.tab._state_trace(case, phase_meta, 'x')
        self.assertIsNone(warning)
        self.assertIsNotNone(xvals)
        self.assertEqual(len(xvals), len(yvals))
        self.assertFalse(np.any(violation))

    def test_no_state_interp_falls_back_to_timeseries(self):
        phase_meta = _make_phase_meta(with_state_interp=None)
        n = 5
        timeseries = np.linspace(0, 1, n)
        case = mock.Mock()
        case.get_val = mock.Mock(side_effect=lambda path: {
            'traj.phase0.timeseries.states:x': timeseries.reshape(-1, 1),
            'traj.phase0.timeseries.time': timeseries.reshape(-1, 1),
        }[path])
        xvals, yvals, violation, warning = self.tab._state_trace(case, phase_meta, 'x')
        self.assertIsNotNone(xvals)
        self.assertEqual(len(xvals), n)

    def test_unknown_variable_returns_none(self):
        phase_meta = _make_phase_meta()
        case = mock.Mock()
        xvals, yvals, violation, warning = self.tab._state_trace(case, phase_meta, 'nonexistent')
        self.assertIsNone(xvals)
        self.assertIsNone(yvals)


class StateHermiteTraceTests(unittest.TestCase):
    def setUp(self):
        meta = {'trajectories': [{'name': 'traj', 'phases': []}]}
        self.tab = _TrajectoryTab(_make_broker(meta))

    def _make_case(self, x_vals, rate_path='traj.phase0.ode.xdot', t_duration=2.0):
        n = len(x_vals)
        case = mock.Mock()
        state_vals = np.array(x_vals, dtype=float).reshape(-1, 1)
        rate_vals = np.ones(n, dtype=float) * 0.5

        def _get(path):
            if path == 'traj.phase0.states:x':
                return state_vals
            if path == rate_path:
                return rate_vals
            if path == 'traj.phase0.t_duration':
                return np.array([t_duration])
            if path == 'traj.phase0.timeseries.time':
                return np.linspace(0, t_duration, n).reshape(-1, 1)
            if path == 'traj.phase0.timeseries.states:x':
                return state_vals
            raise KeyError(path)

        case.get_val = mock.Mock(side_effect=_get)
        return case

    def test_hermite_happy_path(self):
        n = 4
        rate_source = {'path': 'traj.phase0.ode.xdot', 'rows': [0, 1, 2, 3]}
        phase_meta = _make_phase_meta(
            with_state_interp=_hermite_interp(n_state=n, n_dense=21),
            rate_source=rate_source,
        )
        case = self._make_case(list(range(n)), rate_path='traj.phase0.ode.xdot')
        xvals, yvals, violation, warning = self.tab._state_trace(case, phase_meta, 'x')
        self.assertIsNone(warning)
        self.assertIsNotNone(xvals)

    def test_hermite_none_rate_source_falls_back_no_crash(self):
        """rate_source=None with Hermite interp must NOT crash with secondary TypeError."""
        n = 4
        phase_meta = _make_phase_meta(
            with_state_interp=_hermite_interp(n_state=n, n_dense=21),
            rate_source=None,  # the bug condition
        )
        timeseries = np.linspace(0, 1, n).reshape(-1, 1)
        state_vals = np.arange(n, dtype=float).reshape(-1, 1)

        def _get(path):
            if path == 'traj.phase0.states:x':
                return state_vals
            if path == 'traj.phase0.timeseries.states:x':
                return state_vals
            if path == 'traj.phase0.timeseries.time':
                return timeseries
            if path == 'traj.phase0.t_duration':
                return np.array([1.0])
            raise KeyError(path)

        case = mock.Mock()
        case.get_val = mock.Mock(side_effect=_get)

        # Must not raise — the secondary-exception bug would propagate TypeError here
        xvals, yvals, violation, warning = self.tab._state_trace(case, phase_meta, 'x')
        # Should fall back to timeseries
        self.assertIsNotNone(xvals)
        self.assertIsNotNone(warning)
        self.assertIn('None', warning)

    def test_hermite_missing_rate_path_falls_back(self):
        """rate_source['path'] raises on get_val → fall back to timeseries with warning."""
        n = 4
        rate_source = {'path': 'traj.phase0.nonexistent_rate', 'rows': [0, 1, 2, 3]}
        phase_meta = _make_phase_meta(
            with_state_interp=_hermite_interp(n_state=n, n_dense=21),
            rate_source=rate_source,
        )
        state_vals = np.arange(n, dtype=float).reshape(-1, 1)
        timeseries = np.linspace(0, 1, n).reshape(-1, 1)

        def _get(path):
            if path == 'traj.phase0.states:x':
                return state_vals
            if path == 'traj.phase0.timeseries.states:x':
                return state_vals
            if path == 'traj.phase0.timeseries.time':
                return timeseries
            if path == 'traj.phase0.t_duration':
                return np.array([1.0])
            raise KeyError(path)

        case = mock.Mock()
        case.get_val = mock.Mock(side_effect=_get)
        xvals, yvals, violation, warning = self.tab._state_trace(case, phase_meta, 'x')
        self.assertIsNotNone(xvals)
        self.assertIsNotNone(warning)


class ControlTraceTests(unittest.TestCase):
    def setUp(self):
        meta = {'trajectories': [{'name': 'traj', 'phases': []}]}
        self.tab = _TrajectoryTab(_make_broker(meta))

    def _make_phase_with_control(self, control_type='full', with_ctrl_interp=True):
        n_ctrl = 3
        n_dense = 21
        ctrl_meta = {
            'name': 'phase0',
            'promoted_path': 'traj.phase0',
            'transcription': 'GaussLobatto',
            'transcription_name': 'gauss-lobatto',
            'segment_ends': [-1.0, 1.0],
            'compressed': False,
            'state_input_to_disc': [0, 1, 2, 3],
            'dense_grid': _dense_grid(n_dense),
            'state_input_node_ptau': [-1.0, 0.0, 1.0],
            'control_input_node_ptau': [-1.0, 0.0, 1.0],
            'defect_outputs': {},
            'states': {},
            'controls': {
                'u': {
                    'units': 'N',
                    'shape': [1],
                    'lower': None,
                    'upper': None,
                    'control_type': control_type,
                    'path': 'traj.phase0.controls:u',
                }
            },
            'timeseries_outputs': {
                'controls:u': {
                    'path': 'timeseries.controls:u',
                    'name': 'u',
                    'units': 'N',
                    'category': 'control',
                }
            },
            'path_constraints': [],
            'boundary_constraints': {'initial': [], 'final': []},
        }
        if with_ctrl_interp:
            L = np.eye(n_dense, n_ctrl).tolist()
            D = np.zeros((n_dense, n_ctrl)).tolist()
            D2 = np.zeros((n_dense, n_ctrl)).tolist()
            ctrl_meta['control_interp'] = {'L': L, 'D': D, 'D2': D2}
        return ctrl_meta

    def test_polynomial_control_uses_timeseries(self):
        phase_meta = self._make_phase_with_control(control_type='polynomial', with_ctrl_interp=False)
        n = 5
        time_arr = np.linspace(0, 1, n).reshape(-1, 1)
        ctrl_arr = np.ones(n).reshape(-1, 1)
        case = mock.Mock()
        case.get_val = mock.Mock(side_effect=lambda p: {
            'traj.phase0.timeseries.controls:u': ctrl_arr,
            'traj.phase0.timeseries.time': time_arr,
        }[p])
        xvals, yvals, violation, warning = self.tab._control_trace(case, phase_meta, 'u')
        self.assertIsNotNone(xvals)
        self.assertEqual(len(xvals), n)

    def test_control_interp_matrix_applied(self):
        phase_meta = self._make_phase_with_control(control_type='full', with_ctrl_interp=True)
        n_ctrl = 3
        ctrl_input = np.array([1.0, 2.0, 3.0]).reshape(-1, 1)
        case = mock.Mock()
        case.get_val = mock.Mock(side_effect=lambda p: {
            'traj.phase0.controls:u': ctrl_input,
            'traj.phase0.t_duration': np.array([2.0]),
            'traj.phase0.timeseries.time': np.linspace(0, 2, 21).reshape(-1, 1),
        }[p])
        xvals, yvals, violation, warning = self.tab._control_trace(case, phase_meta, 'u')
        self.assertIsNotNone(xvals)
        self.assertEqual(len(yvals), 21)

    def test_control_interp_shape_mismatch_falls_back(self):
        """Control get_val raises → fall back to recorded timeseries."""
        phase_meta = self._make_phase_with_control(control_type='full', with_ctrl_interp=True)
        n = 5
        time_arr = np.linspace(0, 1, n).reshape(-1, 1)
        ctrl_arr = np.ones(n).reshape(-1, 1)
        case = mock.Mock()

        def _get(p):
            if p == 'traj.phase0.controls:u':
                raise KeyError('not found')
            return {
                'traj.phase0.timeseries.controls:u': ctrl_arr,
                'traj.phase0.timeseries.time': time_arr,
            }[p]

        case.get_val = mock.Mock(side_effect=_get)
        xvals, yvals, violation, warning = self.tab._control_trace(case, phase_meta, 'u')
        self.assertIsNotNone(xvals)

    def test_unknown_control_variable_returns_none(self):
        phase_meta = self._make_phase_with_control()
        case = mock.Mock()
        xvals, yvals, violation, warning = self.tab._control_trace(case, phase_meta, 'nonexistent')
        self.assertIsNone(xvals)


class DefectTraceTests(unittest.TestCase):
    def setUp(self):
        meta = {'trajectories': [{'name': 'traj', 'phases': []}]}
        self.tab = _TrajectoryTab(_make_broker(meta))

    def _make_phase_with_defect(self):
        n = 21
        return {
            'name': 'phase0',
            'promoted_path': 'traj.phase0',
            'transcription': 'GaussLobatto',
            'transcription_name': 'gauss-lobatto',
            'segment_ends': [-1.0, 0.0, 1.0],
            'compressed': False,
            'state_input_to_disc': [0, 1, 2, 3],
            'dense_grid': _dense_grid(n),
            'state_input_node_ptau': [-1.0, 0.0, 1.0],
            'control_input_node_ptau': [-1.0, 0.0, 1.0],
            'defect_outputs': {
                'collocation:x': {
                    'path': 'traj.phase0.collocation_constraint.defects:x',
                    'kind': 'collocation',
                    'node_ptau': [-0.5, 0.5],
                }
            },
            'states': {},
            'controls': {},
            'timeseries_outputs': {},
            'path_constraints': [],
            'boundary_constraints': {'initial': [], 'final': []},
        }

    def test_defect_trace_happy_path(self):
        phase_meta = self._make_phase_with_defect()
        defect_vals = np.array([0.1, 0.2]).reshape(-1, 1)
        case = mock.Mock()
        case.get_val = mock.Mock(side_effect=lambda p: {
            'traj.phase0.collocation_constraint.defects:x': defect_vals,
            'traj.phase0.t_duration': np.array([2.0]),
            'traj.phase0.t_initial': np.array([0.0]),
        }[p])
        xvals, yvals, violation, warning = self.tab._defect_trace(
            case, phase_meta, 'collocation:x'
        )
        self.assertIsNotNone(xvals)
        self.assertEqual(len(xvals), 2)

    def test_defect_trace_missing_path_returns_none(self):
        """get_val raises for defect path → graceful None return."""
        phase_meta = self._make_phase_with_defect()
        case = mock.Mock()
        case.get_val = mock.Mock(side_effect=KeyError('not found'))
        xvals, yvals, violation, warning = self.tab._defect_trace(
            case, phase_meta, 'collocation:x'
        )
        self.assertIsNone(xvals)

    def test_defect_trace_unknown_key_returns_none(self):
        phase_meta = self._make_phase_with_defect()
        case = mock.Mock()
        xvals, yvals, violation, warning = self.tab._defect_trace(
            case, phase_meta, 'no_such_defect'
        )
        self.assertIsNone(xvals)


class MixedPhasesRefreshTests(unittest.TestCase):
    """Integration-style tests for refresh() with multiple phases using different interp modes."""

    def _make_snapshot(self, cases_by_phase):
        """Return a BrokerSnapshot-like mock."""
        def _get_val(path):
            for phase_vals in cases_by_phase.values():
                if path in phase_vals:
                    return phase_vals[path]
            raise KeyError(path)

        case = mock.Mock()
        case.get_val = mock.Mock(side_effect=_get_val)
        snapshot = mock.Mock()
        snapshot.case = case
        snapshot.major_iteration = 1
        snapshot.counter = 1
        return snapshot

    def test_two_phase_states_both_produce_traces(self):
        n_state = 4
        n_dense = 21
        time_arr = np.linspace(0, 1, n_state).reshape(-1, 1)
        state_vals = np.ones(n_state).reshape(-1, 1)
        timeseries_time = np.linspace(0, 1, n_state).reshape(-1, 1)

        traj_meta = {
            'name': 'traj',
            'phases': [
                _make_phase_meta(
                    name='phase0',
                    with_state_interp=_lagrange_interp(n_state=n_state, n_dense=n_dense),
                ),
                _make_phase_meta(
                    name='phase1',
                    with_state_interp=None,  # ExplicitShooting-style: no state_interp
                ),
            ],
        }

        meta = {'trajectories': [traj_meta]}
        broker = _make_broker(meta)
        broker.snapshots = []

        cases = {
            'phase0': {
                'traj.phase0.states:x': state_vals,
                'traj.phase0.t_duration': np.array([1.0]),
                'traj.phase0.t_initial': np.array([0.0]),
                'traj.phase0.timeseries.time': timeseries_time,
                'traj.phase0.timeseries.states:x': state_vals,
            },
            'phase1': {
                'traj.phase1.states:x': state_vals,
                'traj.phase1.t_duration': np.array([1.0]),
                'traj.phase1.t_initial': np.array([1.0]),
                'traj.phase1.timeseries.time': (timeseries_time + 1.0),
                'traj.phase1.timeseries.states:x': state_vals,
            },
        }
        snapshot = self._make_snapshot(cases)
        broker.latest_snapshot = mock.Mock(return_value=snapshot)

        tab = _TrajectoryTab(broker)
        # Manually call _ensure_initialized so phase_paths are set up
        tab._ensure_initialized()
        tab._traj_select.value = 'traj'

        ph0 = traj_meta['phases'][0]
        ph1 = traj_meta['phases'][1]

        x0, y0, v0, w0 = tab._state_trace(snapshot.case, ph0, 'x')
        x1, y1, v1, w1 = tab._state_trace(snapshot.case, ph1, 'x')

        self.assertIsNotNone(x0, "phase0 (Lagrange) should produce x values")
        self.assertIsNotNone(x1, "phase1 (no interp) should fall back to timeseries")


class HistoryDivergenceTests(unittest.TestCase):
    """Test that optimizer-history vs driver-case count divergence is handled."""

    def test_series_tab_warns_on_diverged_counts(self):
        from dymos_rtplot.realtime_plot.realtime_dashboard import _SeriesTab
        from dymos_rtplot.realtime_plot.realtime_broker import BrokerSnapshot
        import numpy as np

        broker = mock.Mock()
        broker.metadata = {}

        # 3 driver snapshots but 5 history entries → diverged
        snapshots = []
        for i in range(3):
            s = mock.Mock()
            s.major_iteration = i + 1
            s.scaled_objs = {'obj': np.array([float(i)])}
            s.scaled_desvars = {}
            s.scaled_cons = {}
            snapshots.append(s)
        broker.snapshots = snapshots
        broker.latest_snapshot = mock.Mock(return_value=snapshots[-1])
        broker.get_history_keys = mock.Mock(return_value=['iter', 'obj'])
        broker.get_history_entries = mock.Mock(return_value=[
            {'iter': np.array([float(i)]), 'obj': np.array([float(i)])} for i in range(5)
        ])
        broker.get_history_warning = mock.Mock(return_value=None)
        broker.get_series = mock.Mock(return_value=[np.array([0.0])] * 3)

        tab = _SeriesTab(broker)
        tab._group_select.value = 'Optimizer History'
        tab._var_select.options = ['obj']
        tab._var_select.value = ['obj']
        tab.refresh(force=True)
        # Warning should be set because len(snapshots)=3 != len(history_entries)=5
        self.assertIn('diverged', tab._warning.text.lower())


class InterpCacheTests(unittest.TestCase):
    """Test that _get_interp caches and returns numpy arrays."""

    def setUp(self):
        meta = {'trajectories': [{'name': 'traj', 'phases': []}]}
        self.tab = _TrajectoryTab(_make_broker(meta))

    def test_cache_converts_lists_to_arrays(self):
        phase_meta = _make_phase_meta(
            with_state_interp=_lagrange_interp(n_state=4, n_dense=10),
        )
        interp = self.tab._get_interp(phase_meta, 'state_interp')
        self.assertIsInstance(interp['L'], np.ndarray)
        self.assertIsInstance(interp['D'], np.ndarray)
        self.assertEqual(interp['mode'], 'lagrange')

    def test_cache_returns_same_object_on_second_call(self):
        phase_meta = _make_phase_meta(
            with_state_interp=_lagrange_interp(n_state=4, n_dense=10),
        )
        interp1 = self.tab._get_interp(phase_meta, 'state_interp')
        interp2 = self.tab._get_interp(phase_meta, 'state_interp')
        self.assertIs(interp1, interp2)

    def test_cache_keyed_by_phase_path_and_key(self):
        meta0 = _make_phase_meta(name='phase0', with_state_interp=_lagrange_interp(n_state=4, n_dense=10))
        meta1 = _make_phase_meta(name='phase1', with_state_interp=_lagrange_interp(n_state=6, n_dense=10))
        interp0 = self.tab._get_interp(meta0, 'state_interp')
        interp1 = self.tab._get_interp(meta1, 'state_interp')
        self.assertIsNot(interp0, interp1)
        self.assertEqual(interp0['L'].shape, (10, 4))
        self.assertEqual(interp1['L'].shape, (10, 6))


if __name__ == '__main__':
    unittest.main()
