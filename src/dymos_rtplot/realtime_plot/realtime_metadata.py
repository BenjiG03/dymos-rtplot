"""Metadata helpers for the Dymos realtime plot dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import dymos as dm
import numpy as np
from scipy.linalg import block_diag

from dymos.transcriptions.grid_data import UniformGrid
from dymos.utils.hermite import hermite_matrices
from dymos.utils.lagrange import lagrange_matrices


def _as_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _as_jsonable(val) for key, val in value.items()}
    return value


def _phase_promoted_path(phase):
    return phase.pathname.replace('.phases.', '.')


def _resolve_phase_var_path(phase_promoted_path, relative_path):
    if relative_path.startswith(f'{phase_promoted_path}.'):
        return relative_path
    return f'{phase_promoted_path}.{relative_path}'


def _aligned_dense_lgl_state_matrices(grid_data, output_grid_data):
    ai_blocks = []
    bi_blocks = []
    ad_blocks = []
    bd_blocks = []

    for iseg in range(grid_data.num_segments):
        i1, i2 = grid_data.subset_segment_indices['state_disc'][iseg, :]
        given_indices = grid_data.subset_node_indices['state_disc'][i1:i2]
        nodes_given = grid_data.node_stau[given_indices]

        i1, i2 = output_grid_data.subset_segment_indices['all'][iseg, :]
        eval_indices = output_grid_data.subset_node_indices['all'][i1:i2]
        nodes_eval = output_grid_data.node_stau[eval_indices]

        ai_seg, bi_seg, ad_seg, bd_seg = hermite_matrices(nodes_given, nodes_eval)
        ai_blocks.append(ai_seg)
        bi_blocks.append(bi_seg)
        ad_blocks.append(ad_seg)
        bd_blocks.append(bd_seg)

    return {
        'mode': 'hermite',
        'Ai': block_diag(*ai_blocks).tolist(),
        'Bi': block_diag(*bi_blocks).tolist(),
        'Ad': block_diag(*ad_blocks).tolist(),
        'Bd': block_diag(*bd_blocks).tolist(),
    }


def _aligned_dense_radau_state_matrices(grid_data, output_grid_data):
    l_blocks = []
    d_blocks = []

    for iseg in range(grid_data.num_segments):
        i1, i2 = grid_data.subset_segment_indices['state_disc'][iseg, :]
        disc_indices = grid_data.subset_node_indices['state_disc'][i1:i2]
        nodes_given = grid_data.node_stau[disc_indices]

        i1, i2 = output_grid_data.subset_segment_indices['all'][iseg, :]
        eval_indices = output_grid_data.subset_node_indices['all'][i1:i2]
        nodes_eval = output_grid_data.node_stau[eval_indices]

        l_seg, d_seg = lagrange_matrices(nodes_given, nodes_eval)
        l_blocks.append(l_seg)
        d_blocks.append(d_seg)

    return {
        'mode': 'lagrange',
        'L': block_diag(*l_blocks).tolist(),
        'D': block_diag(*d_blocks).tolist(),
    }


def _aligned_dense_control_matrices(grid_data, output_grid_data):
    l_blocks = []
    d_blocks = []

    for iseg in range(grid_data.num_segments):
        i1, i2 = grid_data.subset_segment_indices['control_disc'][iseg, :]
        disc_indices = grid_data.subset_node_indices['control_disc'][i1:i2]
        nodes_given = grid_data.node_stau[disc_indices]

        i1, i2 = output_grid_data.subset_segment_indices['all'][iseg, :]
        eval_indices = output_grid_data.subset_node_indices['all'][i1:i2]
        nodes_eval = output_grid_data.node_stau[eval_indices]

        l_seg, d_seg = lagrange_matrices(nodes_given, nodes_eval)
        l_blocks.append(l_seg)
        d_blocks.append(d_seg)

    l_da = block_diag(*l_blocks)
    d_da = block_diag(*d_blocks)

    num_disc_nodes = grid_data.subset_num_nodes['control_disc']
    num_input_nodes = grid_data.subset_num_nodes['control_input']
    l_id = np.zeros((num_disc_nodes, num_input_nodes), dtype=float)
    l_id[np.arange(num_disc_nodes, dtype=int), grid_data.input_maps['dynamic_control_input_to_disc']] = 1.0

    _, d_dd = grid_data.phase_lagrange_matrices('control_disc', 'control_disc', sparse=False)
    return {
        'L': (l_da.dot(l_id)).tolist(),
        'D': (d_da.dot(l_id)).tolist(),
        'D2': (d_da.dot(d_dd.dot(l_id))).tolist(),
    }


def _aligned_dense_state_matrices(grid_data, output_grid_data):
    transcription = getattr(grid_data, 'transcription', None)
    if transcription == 'gauss-lobatto':
        return _aligned_dense_lgl_state_matrices(grid_data, output_grid_data)
    if transcription in ('radau-ps', 'birkhoff'):
        return _aligned_dense_radau_state_matrices(grid_data, output_grid_data)
    return None


def _supports_exact_state_interp(transcription):
    cls_name = type(transcription).__name__
    return cls_name in {'GaussLobatto', 'Radau', 'RadauNew', 'Birkhoff'}


def _supports_exact_control_interp(grid_data):
    try:
        return (
            'control_disc' in grid_data.subset_num_nodes
            and 'control_input' in grid_data.subset_num_nodes
            and 'dynamic_control_input_to_disc' in grid_data.input_maps
        )
    except Exception:
        return False


def _constraint_to_meta(constraint_name, constraint_data):
    if isinstance(constraint_data, dict) and 'constraint_name' in constraint_data and isinstance(constraint_data['constraint_name'], dict):
        unpacked = {}
        for key, value in constraint_data.items():
            if isinstance(value, dict) and 'val' in value:
                unpacked[key] = value['val']
            else:
                unpacked[key] = value
        constraint_data = unpacked
    def _get(key, default=None):
        if isinstance(constraint_data, dict):
            return constraint_data[key] if key in constraint_data else default
        try:
            return constraint_data[key]
        except Exception:
            return default
    return {
        'constraint_name': constraint_name,
        'name': _get('name', constraint_name),
        'constraint_path': _get('constraint_path'),
        'lower': _as_jsonable(_get('lower')),
        'upper': _as_jsonable(_get('upper')),
        'equals': _as_jsonable(_get('equals')),
        'shape': _as_jsonable(_get('shape')),
        'units': _get('units'),
    }


def _timeseries_output_meta(phase):
    outputs = {}
    for ts_name, ts_meta in phase._timeseries.items():
        for output_name, output_meta in ts_meta['outputs'].items():
            category = 'ode'
            if output_name.startswith('states:'):
                category = 'state'
            elif output_name.startswith('controls:'):
                category = 'control'
            elif output_name.startswith('state_rates:'):
                category = 'state_rate'
            elif output_name.startswith('control_rates:'):
                category = 'control_rate'
            elif output_name in ('time', 'time_phase'):
                category = 'time'
            outputs[output_name] = {
                'path': f'{ts_name}.{output_name}',
                'name': output_meta['name'] if 'name' in output_meta else output_name,
                'units': output_meta['units'] if 'units' in output_meta else None,
                'shape': _as_jsonable(output_meta['shape'] if 'shape' in output_meta else None),
                'category': category,
            }
    return outputs


def _defect_output_meta(phase):
    outputs = {}
    phase_path = _phase_promoted_path(phase)
    transcription = phase.options['transcription']
    grid_data = transcription.grid_data

    collocation_ptau = (
        grid_data.node_ptau[grid_data.subset_node_indices['col']].tolist()
        if 'col' in grid_data.subset_node_indices else []
    )
    continuity_ptau = grid_data.segment_ends[1:-1].tolist()

    for state_name in phase.state_options:
        outputs[f"collocation:{state_name}"] = {
            'path': f'{phase_path}.collocation_constraint.defects:{state_name}',
            'kind': 'collocation',
            'node_ptau': collocation_ptau,
        }
        outputs[f"continuity-state:{state_name}"] = {
            'path': f'{phase_path}.continuity_comp.defect_states:{state_name}',
            'kind': 'continuity-state',
            'node_ptau': continuity_ptau,
        }

    for control_name in phase.control_options:
        outputs[f"continuity-control:{control_name}"] = {
            'path': f'{phase_path}.continuity_comp.defect_controls:{control_name}',
            'kind': 'continuity-control',
            'node_ptau': continuity_ptau,
        }
        outputs[f"continuity-control-rate:{control_name}_rate"] = {
            'path': f'{phase_path}.continuity_comp.defect_control_rates:{control_name}_rate',
            'kind': 'continuity-control-rate',
            'node_ptau': continuity_ptau,
        }

    return outputs


def _state_rate_metadata(phase, state_name):
    transcription = phase.options['transcription']
    try:
        rate_path, src_idxs = transcription._get_rate_source_path(state_name, nodes='state_disc', phase=phase)
    except TypeError:
        try:
            rate_path, src_idxs = transcription._get_rate_source_path(state_name, phase=phase)
        except Exception:
            return None
    except Exception:
        return None
    phase_path = _phase_promoted_path(phase)
    rows = list(range(len(src_idxs[0]))) if isinstance(src_idxs, tuple) else None

    # The promoted source paths returned by Dymos are phase-relative.
    full_path = _resolve_phase_var_path(phase_path, rate_path)

    try:
        rows = src_idxs[0].tolist()
    except Exception:
        if rows is None:
            rows = list(phase.options['transcription'].grid_data.input_maps['state_input_to_disc'])

    return {
        'path': full_path,
        'rows': rows,
    }


def _iter_constraint_entries(constraints):
    if isinstance(constraints, dict):
        return constraints.items()
    entries = []
    for idx, constraint in enumerate(constraints or []):
        name = constraint.get('constraint_name', {}).get('val') if isinstance(constraint, dict) else None
        if name is None:
            name = constraint.get('name', {}).get('val') if isinstance(constraint, dict) else None
        if name is None:
            name = f'constraint_{idx}'
        entries.append((name, constraint))
    return entries


def build_rtplot_metadata(problem, case_recorder_filename):
    """Build metadata needed by the richer realtime dashboard tabs."""
    hist_file = None
    driver = problem.driver
    if driver is not None and 'hist_file' in driver.options and driver.options['hist_file']:
        hist_file = str(Path(driver.options['hist_file']).resolve())

    metadata = {
        'case_recorder_filename': str(Path(case_recorder_filename).resolve()),
        'hist_file': hist_file,
        'driver_class': type(driver).__name__ if driver is not None else None,
        'optimizer': driver.options['optimizer'] if driver is not None and 'optimizer' in driver.options else None,
        'trajectories': [],
    }

    for system in problem.model.system_iter(include_self=True, recurse=True):
        if not isinstance(system, dm.Trajectory):
            continue

        traj_meta = {
            'name': system.name,
            'pathname': system.pathname,
            'phases': [],
        }

        for phase_name, phase in system._phases.items():
            transcription = phase.options['transcription']
            grid_data = transcription.grid_data
            dense_nodes_per_segment = int(max(21, 4 * int(np.max(np.atleast_1d(grid_data.transcription_order)))))
            output_grid = UniformGrid(
                num_segments=grid_data.num_segments,
                nodes_per_seg=dense_nodes_per_segment,
                segment_ends=grid_data.segment_ends,
                compressed=False,
            )

            phase_meta = {
                'name': phase_name,
                'pathname': phase.pathname,
                'promoted_path': _phase_promoted_path(phase),
                'transcription': type(transcription).__name__,
                'transcription_name': getattr(grid_data, 'transcription', None),
                'grid_type': getattr(grid_data, 'grid_type', None),
                'segment_ends': grid_data.segment_ends.tolist(),
                'compressed': bool(grid_data.compressed),
                'state_input_to_disc': grid_data.input_maps['state_input_to_disc'].tolist(),
                'dense_grid': {
                    'num_nodes': int(output_grid.subset_num_nodes['all']),
                    'node_ptau': output_grid.node_ptau.tolist(),
                    'node_stau': output_grid.node_stau.tolist(),
                    'node_dptau_dstau': output_grid.node_dptau_dstau.tolist(),
                    'segment_indices': output_grid.segment_indices.tolist(),
                },
                'state_input_node_ptau': grid_data.node_ptau[
                    grid_data.subset_node_indices['state_input']
                ].tolist(),
                'control_input_node_ptau': grid_data.node_ptau[
                    grid_data.subset_node_indices['control_input']
                ].tolist(),
                'defect_outputs': _defect_output_meta(phase),
                'states': {},
                'controls': {},
                'timeseries_outputs': _timeseries_output_meta(phase),
                'path_constraints': [],
                'boundary_constraints': {
                    'initial': [],
                    'final': [],
                },
            }

            state_interp = None
            if _supports_exact_state_interp(transcription):
                state_interp = _aligned_dense_state_matrices(grid_data, output_grid)
            if state_interp is not None:
                phase_meta['state_interp'] = state_interp
            if _supports_exact_control_interp(grid_data):
                phase_meta['control_interp'] = _aligned_dense_control_matrices(grid_data, output_grid)

            for state_name, options in phase.state_options.items():
                phase_meta['states'][state_name] = {
                    'units': options['units'],
                    'shape': list(options['shape']),
                    'lower': _as_jsonable(options['lower'] if 'lower' in options else None),
                    'upper': _as_jsonable(options['upper'] if 'upper' in options else None),
                    'rate_source': _state_rate_metadata(phase, state_name),
                }

            for control_name, options in phase.control_options.items():
                phase_meta['controls'][control_name] = {
                    'units': options['units'],
                    'shape': list(options['shape']),
                    'lower': _as_jsonable(options['lower'] if 'lower' in options else None),
                    'upper': _as_jsonable(options['upper'] if 'upper' in options else None),
                    'control_type': options['control_type'] if 'control_type' in options else 'full',
                }

            for name, data in _iter_constraint_entries(getattr(phase, '_path_constraints', {})):
                phase_meta['path_constraints'].append(_constraint_to_meta(name, data))

            for name, data in _iter_constraint_entries(getattr(phase, '_initial_boundary_constraints', {})):
                phase_meta['boundary_constraints']['initial'].append(_constraint_to_meta(name, data))

            for name, data in _iter_constraint_entries(getattr(phase, '_final_boundary_constraints', {})):
                phase_meta['boundary_constraints']['final'].append(_constraint_to_meta(name, data))

            traj_meta['phases'].append(phase_meta)

        metadata['trajectories'].append(traj_meta)

    return metadata


def metadata_path_for_case(case_recorder_filename):
    case_path = Path(case_recorder_filename)
    return case_path.with_suffix(case_path.suffix + '.rtplot_meta.json')


def write_rtplot_metadata(problem, case_recorder_filename):
    metadata = build_rtplot_metadata(problem, case_recorder_filename)
    meta_path = metadata_path_for_case(case_recorder_filename)
    meta_path.write_text(json.dumps(metadata), encoding='utf-8')
    return meta_path


def load_rtplot_metadata(meta_file=None, case_recorder_filename=None):
    candidate = None
    if meta_file:
        candidate = Path(meta_file)
    elif case_recorder_filename:
        candidate = metadata_path_for_case(case_recorder_filename)

    if candidate is None or not candidate.exists():
        return None

    return json.loads(candidate.read_text(encoding='utf-8'))
