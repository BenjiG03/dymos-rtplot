"""Tabbed Dymos realtime plot dashboard."""

from __future__ import annotations

import math
import numbers

import numpy as np
from openmdao.core.constants import INF_BOUND
from openmdao.utils.shell_proc import _is_process_running

from dymos_rtplot.realtime_plot.realtime_broker import LiveDataBroker
from dymos_rtplot.realtime_plot.realtime_optimizer_plot import _RealTimeOptimizerPlot

from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    CheckboxGroup,
    ColumnDataSource,
    Div,
    HoverTool,
    LinearColorMapper,
    MultiChoice,
    Range1d,
    Select,
    ScrollBox,
    TabPanel,
    Tabs,
)
from bokeh.palettes import Category10, Category20, RdBu11
from bokeh.plotting import figure


_HEAVY_TAB_STRIDE = 5
_ZERO_JAC_THRESHOLD = 1.0e-16
_DEFAULT_LINE_COLORS = Category20[20]
_PHASE_COLORS = Category10[10]
CASE_PLOTTER_TAB = "case-plotter"
TRAJECTORY_TAB = "trajectory"
SERIES_TAB = "series"
JACOBIAN_ENTRIES_TAB = "jacobian-entries"
JACOBIAN_HEATMAP_TAB = "jacobian-heatmap"
DASHBOARD_TAB_ORDER = (
    CASE_PLOTTER_TAB,
    TRAJECTORY_TAB,
    SERIES_TAB,
    JACOBIAN_ENTRIES_TAB,
    JACOBIAN_HEATMAP_TAB,
)
DASHBOARD_TAB_TITLES = {
    CASE_PLOTTER_TAB: "Case Plotter",
    TRAJECTORY_TAB: "Trajectory",
    SERIES_TAB: "Scaled + Opt",
    JACOBIAN_ENTRIES_TAB: "Jacobian Entries",
    JACOBIAN_HEATMAP_TAB: "Jacobian Heatmap",
}
_HEAVY_TABS = {JACOBIAN_ENTRIES_TAB, JACOBIAN_HEATMAP_TAB}


def get_dashboard_tab_names():
    return list(DASHBOARD_TAB_ORDER)


def _ensure_figure_legend(fig):
    if not fig.legend:
        return
    legend = fig.legend[0]
    legend.visible = True
    legend.location = "top_left"
    legend.click_policy = "hide"


def _flatten(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.array([], dtype=float)
    return np.ravel(arr.astype(float))


def _scalar_for_plot(arr):
    flat = _flatten(arr)
    if flat.size == 0:
        return 0.0
    if flat.size == 1:
        return float(flat[0])
    return float(np.linalg.norm(flat, ord=np.inf))


def _scalar_item(value):
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0])


def _is_finite_bound(value, sign=1):
    if value is None:
        return False
    if isinstance(value, numbers.Number):
        if sign > 0:
            return value != INF_BOUND
        return value != -INF_BOUND
    return False


def _constraint_violation(values, lower=None, upper=None, equals=None):
    vals = np.asarray(values, dtype=float)
    if vals.ndim == 1:
        vals = vals[:, np.newaxis]
    violation = np.zeros(vals.shape[0], dtype=bool)
    reduce_axes = tuple(range(1, vals.ndim))
    if equals is not None:
        eq = np.asarray(equals, dtype=float)
        violation |= np.any(np.abs(vals - eq) > 1.0e-8, axis=reduce_axes)
        return violation
    if lower is not None and _is_finite_bound(lower, sign=-1):
        violation |= np.any(vals < float(lower) - 1.0e-8, axis=reduce_axes)
    if upper is not None and _is_finite_bound(upper, sign=1):
        violation |= np.any(vals > float(upper) + 1.0e-8, axis=reduce_axes)
    return violation


class _TrajectoryTab:
    _CATEGORY_LABELS = {
        "controls": "Controls",
        "states": "States",
        "ode": "ODE Values",
        "state_rates": "State Rates",
        "control_rates": "Control Rates",
    }

    def __init__(self, broker):
        self._broker = broker
        self._meta = broker.metadata or {}
        self._initialized = False
        self._phase_paths = {}
        self._phase_meta = {}
        self._last_order = None
        self._status = Div(text="Waiting for trajectory data...")
        self._warning = Div(text="")
        self._traj_select = Select(title="Trajectory", options=[], value=None)
        self._order_selects = []
        self._plot_sources = {}
        self._plot_node_sources = {}
        self._plot_violation_sources = {}
        self._plot_figures = {}
        self._plot_node_renderers = {}
        self._plot_violation_renderers = {}
        self._plots_column = column(Div(text="Waiting for trajectory plots..."), sizing_mode="stretch_width")
        self._plots_scroll = ScrollBox(child=self._plots_column, height_policy="max")
        self._display_check = CheckboxGroup(
            labels=["Show node markers", "Show violation markers"],
            active=[0, 1],
        )

        order_controls = []
        category_options = [(key, label) for key, label in self._CATEGORY_LABELS.items()]
        default_order = ["controls", "states", "ode", "state_rates", "control_rates"]
        for idx, category in enumerate(default_order, start=1):
            select = Select(title=f"Order {idx}", options=category_options, value=category, width=180)
            select.on_change("value", self._selection_changed)
            self._order_selects.append(select)
            order_controls.append(select)

        self._traj_select.on_change("value", self._selection_changed)
        self._display_check.on_change("active", self._selection_changed)
        controls = row(self._traj_select, *order_controls, self._display_check, sizing_mode="stretch_width")
        self.panel = TabPanel(
            child=column(self._status, self._warning, controls, self._plots_scroll, sizing_mode="stretch_both"),
            title=DASHBOARD_TAB_TITLES[TRAJECTORY_TAB],
        )

    def _ensure_initialized(self):
        if self._initialized:
            return
        options = []
        for traj_meta in self._meta.get("trajectories", []):
            options.append(traj_meta["name"])
            for phase_meta in traj_meta["phases"]:
                self._phase_meta[phase_meta["promoted_path"]] = phase_meta
                self._phase_paths[phase_meta["promoted_path"]] = traj_meta["name"]
        if options:
            self._traj_select.options = options
            self._traj_select.value = options[0]
        self._initialized = True

    def _selection_changed(self, attr, old, new):
        self._rebuild_plots()
        self.refresh(force=True)

    def _selected_order(self):
        ordered = []
        for select in self._order_selects:
            if select.value not in ordered:
                ordered.append(select.value)
        for category in self._CATEGORY_LABELS:
            if category not in ordered:
                ordered.append(category)
        return ordered

    def _traj_meta(self):
        traj_name = self._traj_select.value
        for traj_meta in self._meta.get("trajectories", []):
            if traj_meta["name"] == traj_name:
                return traj_meta
        return None

    def _category_variables(self, traj_meta, category):
        found = set()
        for phase_meta in traj_meta["phases"]:
            if category == "states":
                found.update(phase_meta["states"].keys())
            elif category == "controls":
                found.update(phase_meta["controls"].keys())
            elif category == "state_rates":
                found.update(phase_meta["states"].keys())
            elif category == "control_rates":
                found.update(phase_meta["controls"].keys())
            elif category == "ode":
                for name, meta in phase_meta.get("timeseries_outputs", {}).items():
                    if meta.get("category") == "ode":
                        found.add(name)
        return sorted(found)

    def _plot_key(self, category, variable):
        return f"{category}:{variable}"

    def _make_plot(self, category, variable):
        source = ColumnDataSource(data=dict(xs=[], ys=[], phase=[], segment_color=[]))
        node_source = ColumnDataSource(data=dict(x=[], y=[], phase=[], node_color=[]))
        viol_source = ColumnDataSource(data=dict(x=[], y=[], reason=[]))
        fig = figure(
            title=f"{self._CATEGORY_LABELS[category]}: {variable}",
            sizing_mode="stretch_width",
            height=220,
            min_width=420,
            x_axis_label="Time",
            y_axis_label=variable,
            output_backend="webgl",
        )
        line_renderer = fig.multi_line(xs="xs", ys="ys", source=source, line_width=3, line_color="segment_color")
        node_renderer = fig.scatter("x", "y", source=node_source, size=6, color="node_color", alpha=0.95, line_color="black")
        viol_renderer = fig.scatter("x", "y", source=viol_source, size=8, color="red")
        fig.add_tools(HoverTool(renderers=[line_renderer], tooltips=[("phase", "@phase")]))
        fig.add_tools(HoverTool(renderers=[node_renderer], tooltips=[("phase", "@phase"), ("time", "@x"), ("value", "@y")]))
        fig.add_tools(HoverTool(renderers=[viol_renderer], tooltips=[("time", "@x"), ("value", "@y"), ("reason", "@reason")]))
        key = self._plot_key(category, variable)
        self._plot_sources[key] = source
        self._plot_node_sources[key] = node_source
        self._plot_violation_sources[key] = viol_source
        self._plot_figures[key] = fig
        self._plot_node_renderers[key] = node_renderer
        self._plot_violation_renderers[key] = viol_renderer
        return fig

    def _rebuild_plots(self):
        self._ensure_initialized()
        traj_meta = self._traj_meta()
        if traj_meta is None:
            return

        ordered_categories = self._selected_order()
        if ordered_categories == self._last_order and self._plots_column.children:
            return

        children = []
        for category in ordered_categories:
            variables = self._category_variables(traj_meta, category)
            if not variables:
                continue
            children.append(Div(text=f"<b>{self._CATEGORY_LABELS[category]}</b>"))
            figures = []
            for variable in variables:
                key = self._plot_key(category, variable)
                if key not in self._plot_figures:
                    self._make_plot(category, variable)
                figures.append(self._plot_figures[key])
            children.append(
                gridplot(
                    figures,
                    ncols=3,
                    sizing_mode="stretch_width",
                    merge_tools=False,
                    toolbar_location=None,
                )
            )
        self._plots_column.children = children
        self._last_order = ordered_categories

    def refresh(self, force=False):
        self._ensure_initialized()
        snapshot = self._broker.latest_snapshot()
        if snapshot is None:
            return
        self._rebuild_plots()
        traj_meta = self._traj_meta()
        if traj_meta is None:
            return

        banner_parts = []
        show_nodes = 0 in self._display_check.active
        show_violations = 1 in self._display_check.active
        phase_color_map = {
            phase_meta["name"]: _PHASE_COLORS[idx % len(_PHASE_COLORS)]
            for idx, phase_meta in enumerate(traj_meta["phases"])
        }

        for category in self._selected_order():
            for variable in self._category_variables(traj_meta, category):
                key = self._plot_key(category, variable)
                source = self._plot_sources[key]
                node_source = self._plot_node_sources[key]
                viol_source = self._plot_violation_sources[key]
                self._plot_node_renderers[key].visible = show_nodes
                self._plot_violation_renderers[key].visible = show_violations
                traces_x = []
                traces_y = []
                trace_phase = []
                colors = []
                node_x = []
                node_y = []
                node_phase = []
                node_color = []
                viol_x = []
                viol_y = []
                viol_reason = []

                for phase_meta in traj_meta["phases"]:
                    xvals, yvals, violation_mask, warning = self._phase_trace(snapshot, phase_meta, category, variable)
                    if warning:
                        banner_parts.append(warning)
                    if xvals is None:
                        continue

                    phase_name = phase_meta["name"]
                    traces_x.append(xvals.tolist())
                    traces_y.append(yvals.tolist())
                    trace_phase.append(phase_name)
                    colors.append(phase_color_map[phase_name])

                    marker_x, marker_y = self._phase_marker_trace(snapshot.case, phase_meta, category, variable)
                    if show_nodes and marker_x is not None:
                        node_x.extend(marker_x.tolist())
                        node_y.extend(marker_y.tolist())
                        node_phase.extend([phase_name] * len(marker_x))
                        node_color.extend([phase_color_map[phase_name]] * len(marker_x))

                    if show_violations:
                        bad_idx = np.where(violation_mask)[0]
                        for idx in bad_idx:
                            viol_x.append(float(xvals[idx]))
                            viol_y.append(float(yvals[idx]))
                            viol_reason.append(f"{phase_name} violation")

                source.data = dict(xs=traces_x, ys=traces_y, phase=trace_phase, segment_color=colors)
                node_source.data = dict(x=node_x, y=node_y, phase=node_phase, node_color=node_color)
                viol_source.data = dict(x=viol_x, y=viol_y, reason=viol_reason)

        self._status.text = f"Major iteration {snapshot.major_iteration}, driver counter {snapshot.counter}"
        self._warning.text = " ".join(sorted(set(banner_parts)))

    def _phase_trace(self, snapshot, phase_meta, category, variable):
        case = snapshot.case
        if category == "states":
            return self._state_trace(case, phase_meta, variable)
        if category == "state_rates":
            return self._state_trace(case, phase_meta, variable, rates_only=True)
        if category == "controls":
            return self._control_trace(case, phase_meta, variable)
        if category == "control_rates":
            return self._control_trace(case, phase_meta, variable, derivative_order=1)
        return self._ode_trace(case, phase_meta, variable)

    def _physical_time(self, case, phase_meta):
        phase_path = phase_meta["promoted_path"]
        t_initial = self._phase_initial_time(case, phase_meta)
        t_duration = _scalar_item(case.get_val(f"{phase_path}.t_duration"))
        ptau = np.asarray(phase_meta["dense_grid"]["node_ptau"], dtype=float)
        return t_initial + 0.5 * (ptau + 1.0) * t_duration

    def _phase_initial_time(self, case, phase_meta):
        phase_path = phase_meta["promoted_path"]
        for candidate in (
            f"{phase_path}.timeseries.time",
            f"{phase_path}.t_initial",
        ):
            try:
                value = np.asarray(case.get_val(candidate))
                return float(value.flat[0])
            except Exception:
                pass

        trajectory_name = self._phase_paths.get(phase_path)
        if trajectory_name:
            for traj_meta in self._meta.get("trajectories", []):
                if traj_meta["name"] != trajectory_name:
                    continue
                start = 0.0
                for other_phase in traj_meta["phases"]:
                    if other_phase["promoted_path"] == phase_path:
                        return start
                    other_path = other_phase["promoted_path"]
                    try:
                        start += _scalar_item(case.get_val(f"{other_path}.t_duration"))
                    except Exception:
                        break
        return 0.0

    def _state_trace(self, case, phase_meta, variable, rates_only=False):
        if variable not in phase_meta["states"]:
            return None, None, np.array([], dtype=bool), None
        if "state_interp" not in phase_meta:
            if rates_only:
                rate_meta = phase_meta.get("timeseries_outputs", {}).get(f"state_rates:{variable}")
                if rate_meta:
                    return self._timeseries_trace(case, phase_meta, rate_meta["path"])
                return None, None, np.array([], dtype=bool), None
            return self._timeseries_trace(case, phase_meta, f"timeseries.states:{variable}", boundary_state=phase_meta["states"][variable])

        state_meta = phase_meta["states"][variable]
        phase_path = phase_meta["promoted_path"]
        state_input = np.asarray(case.get_val(f"{phase_path}.states:{variable}"))
        state_disc = state_input[np.asarray(phase_meta["state_input_to_disc"], dtype=int)]
        shape = tuple(state_meta["shape"])
        size = int(np.prod(shape))
        xd_flat = np.reshape(state_disc, (state_disc.shape[0], size))

        interp = phase_meta["state_interp"]
        dt_dstau = 0.5 * _scalar_item(case.get_val(f"{phase_path}.t_duration")) * np.asarray(
            phase_meta["dense_grid"]["node_dptau_dstau"], dtype=float
        )
        dense_x = self._physical_time(case, phase_meta)
        if interp.get("mode") == "lagrange":
            lmat = np.asarray(interp["L"], dtype=float)
            dmat = np.asarray(interp["D"], dtype=float)
            if rates_only:
                dense_val = dmat.dot(xd_flat) / dt_dstau[:, np.newaxis]
            else:
                dense_val = lmat.dot(xd_flat)
        else:
            rate_meta = state_meta["rate_source"]
            try:
                rate_values = np.asarray(case.get_val(rate_meta["path"]))
            except Exception:
                return self._timeseries_trace(
                    case,
                    phase_meta,
                    f"timeseries.states:{variable}",
                    boundary_state=state_meta,
                    warning=f"Missing exact state rate source {rate_meta['path']}; using recorded nodes.",
                )

            row_idx = np.asarray(rate_meta["rows"], dtype=int)
            rate_disc = rate_values[row_idx]
            fd_flat = np.reshape(rate_disc, (rate_disc.shape[0], size))
            ai = np.asarray(interp["Ai"], dtype=float)
            bi = np.asarray(interp["Bi"], dtype=float)
            ad = np.asarray(interp["Ad"], dtype=float)
            bd = np.asarray(interp["Bd"], dtype=float)
            if rates_only:
                dense_val = ad.dot(xd_flat) / dt_dstau[:, np.newaxis] + bd.dot(fd_flat)
            else:
                dense_val = ai.dot(xd_flat) + bi.dot(fd_flat) * dt_dstau[:, np.newaxis]
        dense_val = np.reshape(dense_val, (dense_val.shape[0],) + shape)
        yvals = dense_val[:, 0] if dense_val.ndim > 1 else dense_val
        violation = self._bounds_violation(yvals[:, np.newaxis], state_meta)
        return dense_x, np.asarray(yvals, dtype=float), violation, None

    def _control_trace(self, case, phase_meta, variable, derivative_order=0):
        if variable not in phase_meta["controls"]:
            return None, None, np.array([], dtype=bool), None
        if "control_interp" not in phase_meta:
            suffix = variable
            if derivative_order == 1:
                suffix = f"control_rates:{variable}_rate"
            elif derivative_order == 2:
                suffix = f"control_rates:{variable}_rate2"
            else:
                suffix = f"controls:{variable}"
            return self._timeseries_trace(case, phase_meta, f"timeseries.{suffix}", boundary_state=phase_meta["controls"][variable])

        phase_path = phase_meta["promoted_path"]
        control = np.asarray(case.get_val(f"{phase_path}.controls:{variable}"))
        shape = tuple(phase_meta["controls"][variable]["shape"])
        size = int(np.prod(shape))
        control_flat = np.reshape(control, (control.shape[0], size))
        interp = phase_meta["control_interp"]
        if derivative_order == 0:
            mat = np.asarray(interp["L"], dtype=float)
        elif derivative_order == 1:
            mat = np.asarray(interp["D"], dtype=float)
        else:
            mat = np.asarray(interp["D2"], dtype=float)
        dense_val = mat.dot(control_flat)
        dense_x = self._physical_time(case, phase_meta)
        duration = _scalar_item(case.get_val(f"{phase_path}.t_duration"))
        if derivative_order == 1:
            dense_val = dense_val * (2.0 / duration)
        elif derivative_order == 2:
            dense_val = dense_val * (2.0 / duration) ** 2
        dense_val = np.reshape(dense_val, (dense_val.shape[0],) + shape)
        yvals = dense_val[:, 0] if dense_val.ndim > 1 else dense_val
        violation = self._bounds_violation(yvals[:, np.newaxis], phase_meta["controls"][variable])
        return dense_x, np.asarray(yvals, dtype=float), violation, None

    def _ode_trace(self, case, phase_meta, variable):
        meta = phase_meta.get("timeseries_outputs", {}).get(variable)
        if not meta:
            return None, None, np.array([], dtype=bool), None
        return self._timeseries_trace(case, phase_meta, meta["path"], boundary_state=meta)

    def _timeseries_trace(self, case, phase_meta, relative_path, boundary_state=None, warning=None):
        phase_path = phase_meta["promoted_path"]
        path = relative_path if relative_path.startswith(phase_path) else f"{phase_path}.{relative_path}"
        try:
            yraw = np.asarray(case.get_val(path))
        except Exception:
            return None, None, np.array([], dtype=bool), warning
        xvals = self._timeseries_xvals(case, phase_meta, yraw.shape[0])
        yvals = yraw[:, 0] if yraw.ndim > 1 else yraw
        violation = np.zeros(len(xvals), dtype=bool)
        if boundary_state:
            violation |= self._bounds_violation(yraw, boundary_state)
        constraint_violation = self._constraint_violation_for_path(case, phase_meta, path)
        if constraint_violation is not None and len(constraint_violation) == len(violation):
            violation |= constraint_violation
        return xvals, np.asarray(yvals, dtype=float), violation, warning

    def _phase_marker_trace(self, case, phase_meta, category, variable):
        if category == "states":
            return self._marker_trace(case, phase_meta, f"timeseries.states:{variable}")
        if category == "controls":
            return self._marker_trace(case, phase_meta, f"timeseries.controls:{variable}")
        if category == "state_rates":
            meta = phase_meta.get("timeseries_outputs", {}).get(f"state_rates:{variable}")
            if meta:
                return self._marker_trace(case, phase_meta, meta["path"])
            return None, None
        if category == "control_rates":
            for candidate in (
                f"control_rates:{variable}_rate",
                f"control_rates:{variable}",
            ):
                meta = phase_meta.get("timeseries_outputs", {}).get(candidate)
                if meta:
                    return self._marker_trace(case, phase_meta, meta["path"])
            return None, None
        meta = phase_meta.get("timeseries_outputs", {}).get(variable)
        if meta:
            return self._marker_trace(case, phase_meta, meta["path"])
        return None, None

    def _marker_trace(self, case, phase_meta, relative_path):
        phase_path = phase_meta["promoted_path"]
        path = relative_path if relative_path.startswith(phase_path) else f"{phase_path}.{relative_path}"
        try:
            yraw = np.asarray(case.get_val(path))
        except Exception:
            return None, None
        xvals = self._timeseries_xvals(case, phase_meta, yraw.shape[0])
        yvals = yraw[:, 0] if yraw.ndim > 1 else yraw
        return np.asarray(xvals, dtype=float), np.asarray(yvals, dtype=float)

    def _timeseries_xvals(self, case, phase_meta, count):
        phase_path = phase_meta["promoted_path"]
        for candidate in (
            f"{phase_path}.timeseries.time",
            f"{phase_path}.timeseries.time_phase",
        ):
            try:
                values = np.asarray(case.get_val(candidate))
                return values[:, 0]
            except Exception:
                pass

        t_initial = self._phase_initial_time(case, phase_meta)
        t_duration = _scalar_item(case.get_val(f"{phase_path}.t_duration"))
        if count <= 1:
            return np.array([t_initial], dtype=float)
        return np.linspace(t_initial, t_initial + t_duration, count)

    def _bounds_violation(self, values, meta):
        vals = np.asarray(values, dtype=float)
        if vals.ndim == 1:
            vals = vals[:, np.newaxis]
        return _constraint_violation(vals, lower=meta.get("lower"), upper=meta.get("upper"), equals=meta.get("equals"))

    def _constraint_violation_for_path(self, case, phase_meta, path):
        rel_path = path.split(f"{phase_meta['promoted_path']}.", 1)[-1]
        for con_meta in phase_meta["path_constraints"]:
            con_path = con_meta.get("constraint_path")
            if con_path == rel_path:
                values = np.asarray(case.get_val(path))
                return _constraint_violation(values, con_meta["lower"], con_meta["upper"], con_meta["equals"])
        for location in ("initial", "final"):
            for con_meta in phase_meta["boundary_constraints"][location]:
                con_path = con_meta.get("constraint_path")
                if con_path != rel_path:
                    continue
                values = np.asarray(case.get_val(path))
                mask = np.zeros(values.shape[0], dtype=bool)
                idx = 0 if location == "initial" else -1
                mask[idx] = np.any(_constraint_violation(values[idx:idx + 1], con_meta["lower"], con_meta["upper"], con_meta["equals"]))
                return mask
        return None


class _SeriesTab:
    _GROUP_LABEL_TO_KEY = {
        "Scaled Objectives": "scaled_objs",
        "Scaled Design Vars": "scaled_desvars",
        "Scaled Constraints": "scaled_cons",
        "Optimizer History": "opt_history",
    }

    def __init__(self, broker):
        self._broker = broker
        self._updating_widgets = False
        self._source = ColumnDataSource(data=dict(iteration=[]))
        self._warning = Div(text="")
        self._group_select = Select(
            title="Group",
            options=list(self._GROUP_LABEL_TO_KEY.keys()),
            value="Scaled Objectives",
        )
        self._var_select = MultiChoice(title="Variables", options=[], value=[])
        self._figure = figure(
            title="Scaled Driver And Optimizer Diagnostics",
            sizing_mode="stretch_both",
            x_axis_label="Major iteration",
            y_axis_label="Value",
            output_backend="webgl",
        )
        placeholder = ColumnDataSource(data=dict(x=[], y=[]))
        self._figure.line("x", "y", source=placeholder, visible=False)
        self._renderers = {}
        self._group_select.on_change("value", self._selection_changed)
        self._var_select.on_change("value", self._selection_changed)
        controls = row(self._group_select, self._var_select, sizing_mode="stretch_width")
        self.panel = TabPanel(
            child=column(self._warning, controls, self._figure, sizing_mode="stretch_both"),
            title=DASHBOARD_TAB_TITLES[SERIES_TAB],
        )

    def _selection_changed(self, attr, old, new):
        if self._updating_widgets:
            return
        if attr == "value" and self._group_select.value != old and self._group_select.value == new:
            self._updating_widgets = True
            try:
                self._var_select.value = []
            finally:
                self._updating_widgets = False
        self.refresh(force=True)

    def _options_for_group(self):
        snapshot = self._broker.latest_snapshot()
        if snapshot is None:
            return []
        group_key = self._GROUP_LABEL_TO_KEY.get(self._group_select.value, "scaled_objs")
        if group_key == "scaled_objs":
            return sorted(snapshot.scaled_objs.keys())
        if group_key == "scaled_desvars":
            return sorted(snapshot.scaled_desvars.keys())
        if group_key == "scaled_cons":
            return sorted(snapshot.scaled_cons.keys())
        return sorted(self._broker.get_history_keys())

    def refresh(self, force=False):
        options = self._options_for_group()
        selected = [name for name in self._var_select.value if name in options]
        if not selected and options:
            selected = options[: min(4, len(options))]
        self._updating_widgets = True
        try:
            self._var_select.options = options
            if self._var_select.value != selected:
                self._var_select.value = selected
        finally:
            self._updating_widgets = False
        if not selected:
            for renderer in self._renderers.values():
                renderer.visible = False
            self._source.data = dict(iteration=[])
            return
        iterations = [snapshot.major_iteration for snapshot in self._broker.snapshots]
        data = {"iteration": iterations}
        group_key = self._GROUP_LABEL_TO_KEY.get(self._group_select.value, "scaled_objs")
        for name in selected:
            series = self._broker.get_series(group_key, name)
            data[name] = [_scalar_for_plot(item) for item in series]
        self._source.data = data
        existing = set(self._renderers)
        for idx, name in enumerate(selected):
            if name not in self._renderers:
                renderer = self._figure.line(
                    "iteration",
                    name,
                    source=self._source,
                    line_width=3,
                    color=_DEFAULT_LINE_COLORS[idx % len(_DEFAULT_LINE_COLORS)],
                    legend_label=name,
                )
                self._figure.add_tools(HoverTool(renderers=[renderer], tooltips=[("iter", "@iteration"), ("value", f"@{name}")]))
                self._renderers[name] = renderer
            self._renderers[name].visible = True
        for name in existing - set(selected):
            self._renderers[name].visible = False
        _ensure_figure_legend(self._figure)
        warning = self._broker.get_history_warning()
        if group_key == "opt_history" and not self._broker.get_history_keys():
            warning = "No pyOptSparse .hst file available, so optimizer diagnostics are unavailable."
        self._warning.text = warning or ""


class _JacobianEntriesTab:
    def __init__(self, broker):
        self._broker = broker
        self._updating_widgets = False
        self._selected_block = None
        self._block_select = Select(title="(of, wrt) block", options=[], value=None)
        self._entry_select = MultiChoice(title="Entries", options=[], value=[])
        self._log_check = CheckboxGroup(labels=["Plot log10(abs(value))"], active=[])
        self._warning = Div(text="")
        self._source = ColumnDataSource(data=dict(iteration=[]))
        self._renderers = {}
        self._figure = figure(
            title="Total Jacobian Entry Traces",
            sizing_mode="stretch_both",
            x_axis_label="Major iteration",
            y_axis_label="Derivative",
            output_backend="webgl",
        )
        placeholder = ColumnDataSource(data=dict(x=[], y=[]))
        self._figure.line("x", "y", source=placeholder, visible=False)
        self._block_select.on_change("value", self._block_changed)
        self._entry_select.on_change("value", self._entry_changed)
        self._log_check.on_change("active", self._log_changed)
        controls = row(self._block_select, self._entry_select, self._log_check, sizing_mode="stretch_width")
        self.panel = TabPanel(
            child=column(self._warning, controls, self._figure, sizing_mode="stretch_both"),
            title=DASHBOARD_TAB_TITLES[JACOBIAN_ENTRIES_TAB],
        )

    def _block_changed(self, attr, old, new):
        if self._updating_widgets:
            return
        self._selected_block = new
        self._updating_widgets = True
        try:
            self._entry_select.value = []
        finally:
            self._updating_widgets = False
        self.refresh(force=True)

    def _entry_changed(self, attr, old, new):
        if self._updating_widgets:
            return
        self.refresh(force=True)

    def _log_changed(self, attr, old, new):
        if self._updating_widgets:
            return
        self.refresh(force=True)

    def _latest_derivatives(self):
        snapshot = self._broker.latest_snapshot()
        if snapshot is None:
            return None
        return snapshot.derivatives

    def refresh(self, force=False):
        if not self._broker.snapshots:
            return
        latest = self._latest_derivatives()
        if latest is None:
            self._warning.text = "No driver total derivatives recorded yet."
            return
        blocks = sorted(latest.keys())
        block_labels = [f"{of} | {wrt}" for of, wrt in blocks]
        mapping = dict(zip(block_labels, blocks))

        entry_cache = {}

        def _entry_map_for_block(block_key):
            if block_key in entry_cache:
                return entry_cache[block_key]
            latest_block = np.asarray(latest[block_key])
            if latest_block.ndim == 1:
                latest_block = latest_block[:, np.newaxis]
            labels = []
            nonzero = {}
            for idx in np.ndindex(latest_block.shape):
                values = []
                for snapshot in self._broker.snapshots:
                    derivs = snapshot.derivatives
                    if derivs is None or block_key not in derivs:
                        continue
                    arr = np.asarray(derivs[block_key])
                    if arr.ndim == 1:
                        arr = arr[:, np.newaxis]
                    values.append(arr[idx])
                if values and np.max(np.abs(values)) > _ZERO_JAC_THRESHOLD:
                    label = f"{idx[0]},{idx[1]}"
                    labels.append(label)
                    nonzero[label] = idx
            entry_cache[block_key] = (latest_block, labels, nonzero)
            return entry_cache[block_key]

        nonzero_labels = [label for label in block_labels if _entry_map_for_block(mapping[label])[1]]
        if nonzero_labels:
            block_labels = nonzero_labels
            mapping = {label: mapping[label] for label in block_labels}

        current_label = self._selected_block if self._selected_block in block_labels else None
        if current_label is None and self._block_select.value in block_labels:
            current_label = self._block_select.value

        if current_label is None and block_labels:
            for label in block_labels:
                _, candidate_entries, _ = _entry_map_for_block(mapping[label])
                if candidate_entries:
                    current_label = label
                    break
            if current_label is None:
                current_label = block_labels[0]
        if not current_label:
            return

        block_key = mapping[current_label]
        latest_block, entry_labels, ever_nonzero = _entry_map_for_block(block_key)
        if not entry_labels:
            for label in block_labels:
                candidate_key = mapping[label]
                candidate_block, candidate_entries, candidate_nonzero = _entry_map_for_block(candidate_key)
                if candidate_entries:
                    current_label = label
                    block_key = candidate_key
                    latest_block = candidate_block
                    entry_labels = candidate_entries
                    ever_nonzero = candidate_nonzero
                    break
        valid_selection = [label for label in self._entry_select.value if label in ever_nonzero]
        if not valid_selection and entry_labels:
            if len(entry_labels) <= 25:
                valid_selection = entry_labels
            else:
                ranked = sorted(entry_labels, key=lambda label: abs(latest_block[ever_nonzero[label]]), reverse=True)
                valid_selection = ranked[:10]

        self._updating_widgets = True
        try:
            self._block_select.options = block_labels
            self._selected_block = current_label
            if self._block_select.value != current_label:
                self._block_select.value = current_label
            self._entry_select.options = entry_labels
            if self._entry_select.value != valid_selection:
                self._entry_select.value = valid_selection
        finally:
            self._updating_widgets = False

        selected = valid_selection
        if not selected:
            self._warning.text = "No nonzero recorded entries found for the selected Jacobian block yet."
            self._source.data = dict(iteration=[])
            for renderer in self._renderers.values():
                renderer.visible = False
            return
        data = {"iteration": [snapshot.major_iteration for snapshot in self._broker.snapshots]}
        use_log = 0 in self._log_check.active
        for label in selected:
            idx = ever_nonzero[label]
            values = []
            for snapshot in self._broker.snapshots:
                derivs = snapshot.derivatives
                if derivs is None or block_key not in derivs:
                    values.append(np.nan)
                    continue
                arr = np.asarray(derivs[block_key])
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]
                value = float(arr[idx])
                values.append(math.log10(abs(value)) if use_log and abs(value) > 0.0 else value)
            data[label] = values
        self._source.data = data

        existing = set(self._renderers)
        for idx, label in enumerate(selected):
            if label not in self._renderers:
                renderer = self._figure.line(
                    "iteration",
                    label,
                    source=self._source,
                    line_width=2,
                    color=_DEFAULT_LINE_COLORS[idx % len(_DEFAULT_LINE_COLORS)],
                    legend_label=label,
                )
                self._figure.add_tools(HoverTool(renderers=[renderer], tooltips=[("iter", "@iteration"), ("value", f"@{label}")]))
                self._renderers[label] = renderer
            self._renderers[label].visible = True
        for label in existing - set(selected):
            self._renderers[label].visible = False
        _ensure_figure_legend(self._figure)
        self._warning.text = ""


class _JacobianHeatmapTab:
    def __init__(self, broker):
        self._broker = broker
        self._warning = Div(text="")
        self._stats = Div(text="Waiting for derivatives...")
        self._source = ColumnDataSource(data=dict(x=[], y=[], value=[], label=[]))
        self._mapper = LinearColorMapper(palette=RdBu11[::-1], low=-1.0, high=1.0)
        self._figure = figure(
            title="Driver Total Jacobian Heatmap",
            sizing_mode="stretch_both",
            x_axis_label="Column",
            y_axis_label="Row",
            output_backend="webgl",
            tooltips=[("entry", "@label"), ("sym-log", "@value")],
        )
        self._figure.rect(x="x", y="y", width=1, height=1, source=self._source, fill_color={"field": "value", "transform": self._mapper}, line_color=None)
        self.panel = TabPanel(
            child=column(self._warning, self._stats, self._figure, sizing_mode="stretch_both"),
            title=DASHBOARD_TAB_TITLES[JACOBIAN_HEATMAP_TAB],
        )

    def refresh(self, force=False):
        if not self._broker.snapshots:
            return
        snapshot = self._broker.latest_snapshot()
        if snapshot.derivatives is None:
            self._warning.text = "No driver total derivatives recorded yet."
            return

        block_keys = sorted(snapshot.derivatives.keys())
        blocks = []
        nrows = 0
        ncols = 0
        for of, wrt in block_keys:
            block = np.asarray(snapshot.derivatives[(of, wrt)])
            if block.ndim == 1:
                block = block[:, np.newaxis]
            blocks.append((of, wrt, block))

        x = []
        y = []
        value = []
        label = []
        row_cursor = 0
        row_map = {}
        for of, wrt, block in blocks:
            row_start = row_map.setdefault(of, row_cursor)
            if row_start == row_cursor:
                row_cursor += block.shape[0]
        nrows = row_cursor

        col_cursor = 0
        col_map = {}
        for of, wrt, block in blocks:
            col_start = col_map.setdefault(wrt, col_cursor)
            if col_start == col_cursor:
                col_cursor += block.shape[1]
        ncols = col_cursor
        dense = np.zeros((nrows, ncols))

        for of, wrt, block in blocks:
            row_start = row_map[of]
            col_start = col_map[wrt]
            dense[row_start:row_start + block.shape[0], col_start:col_start + block.shape[1]] = block
            for i in range(block.shape[0]):
                for j in range(block.shape[1]):
                    raw = float(block[i, j])
                    x.append(col_start + j + 0.5)
                    y.append(row_start + i + 0.5)
                    value.append(np.sign(raw) * np.log10(1.0 + abs(raw)))
                    label.append(f"{of} wrt {wrt} [{i},{j}] = {raw:.6e}")

        self._source.data = dict(x=x, y=y, value=value, label=label)
        if value:
            bound = max(abs(min(value)), abs(max(value)))
            self._mapper.low = -bound
            self._mapper.high = bound
        self._figure.x_range = Range1d(0, ncols)
        self._figure.y_range = Range1d(nrows, 0)

        prev_dense = None
        for prev_snapshot in reversed(self._broker.snapshots[:-1]):
            if prev_snapshot.derivatives is None:
                continue
            prev_dense = np.zeros_like(dense)
            for of, wrt, block in blocks:
                prev_block = np.asarray(prev_snapshot.derivatives[(of, wrt)])
                if prev_block.ndim == 1:
                    prev_block = prev_block[:, np.newaxis]
                prev_dense[row_map[of]:row_map[of] + prev_block.shape[0], col_map[wrt]:col_map[wrt] + prev_block.shape[1]] = prev_block
            break

        rank = int(np.linalg.matrix_rank(dense)) if dense.size else 0
        cond = float(np.linalg.cond(dense)) if dense.size else float("nan")
        nnz = int(np.count_nonzero(np.abs(dense) > _ZERO_JAC_THRESHOLD))
        density = nnz / dense.size if dense.size else 0.0
        if prev_dense is None:
            delta_norm = float("nan")
            max_delta = float("nan")
        else:
            delta = dense - prev_dense
            delta_norm = float(np.linalg.norm(delta))
            max_delta = float(np.max(np.abs(delta)))
        self._stats.text = (
            f"shape={dense.shape}, nnz={nnz}, density={density:.4f}, rank={rank}, "
            f"cond={cond:.3e}, frob_delta={delta_norm:.3e}, max_abs_delta={max_delta:.3e}"
        )
        self._warning.text = ""


class _RealTimeDymosDashboard:
    def __init__(self, case_tracker, callback_period, doc, pid_of_calling_script, script, metadata=None, hist_file=None):
        self._broker = LiveDataBroker(case_tracker, metadata=metadata, hist_file=hist_file)
        self._doc = doc
        self._pid = pid_of_calling_script
        self._last_active = 0
        self._tab_rendered = {name: False for name in _HEAVY_TABS}
        self._tab_objects = {}
        self._tab_panels = []
        for tab_name in DASHBOARD_TAB_ORDER:
            tab_obj, panel = _build_dashboard_tab(
                tab_name,
                case_tracker,
                callback_period,
                doc,
                pid_of_calling_script,
                script,
                broker=self._broker,
            )
            self._tab_objects[tab_name] = tab_obj
            self._tab_panels.append(panel)
        self._tabs = Tabs(
            tabs=self._tab_panels,
            sizing_mode="stretch_both",
        )
        self._tabs.on_change("active", self._active_tab_changed)
        self._doc.add_root(self._tabs)
        self._doc.add_periodic_callback(self._update, callback_period)
        self._doc.title = "Dymos RTPlot Dashboard"

    def _active_tab_changed(self, attr, old, new):
        if not self._broker.snapshots:
            return
        active_name = DASHBOARD_TAB_ORDER[new]
        if active_name in _HEAVY_TABS:
            self._tab_objects[active_name].refresh(force=True)
            self._tab_rendered[active_name] = True

    def _update(self):
        self._tab_objects[CASE_PLOTTER_TAB]._update_wrapped_in_try()
        new_snapshots = self._broker.poll()
        if not new_snapshots and not self._broker.is_running():
            return
        if not self._broker.snapshots:
            return

        active = self._tabs.active
        self._tab_objects[TRAJECTORY_TAB].refresh()
        self._tab_objects[SERIES_TAB].refresh()
        latest_iter = self._broker.latest_snapshot().major_iteration
        active_changed = active != self._last_active
        active_name = DASHBOARD_TAB_ORDER[active]
        heavy_allowed = active_name in _HEAVY_TABS and (
            active_changed or len(new_snapshots) > 0 or latest_iter % _HEAVY_TAB_STRIDE == 0
        )
        if heavy_allowed:
            self._tab_objects[active_name].refresh(force=True)
            self._tab_rendered[active_name] = True
        self._last_active = active


def _build_dashboard_tab(tab_name, case_tracker, callback_period, doc, pid_of_calling_script, script, broker=None):
    if tab_name == CASE_PLOTTER_TAB:
        plot = _RealTimeOptimizerPlot(
            case_tracker,
            callback_period,
            doc,
            pid_of_calling_script,
            script,
            add_root=False,
            start_callback=False,
        )
        return plot, TabPanel(child=plot.layout, title=DASHBOARD_TAB_TITLES[CASE_PLOTTER_TAB])

    if broker is None:
        raise ValueError("A broker is required for non-case-plotter dashboard tabs.")
    if tab_name == TRAJECTORY_TAB:
        tab = _TrajectoryTab(broker)
    elif tab_name == SERIES_TAB:
        tab = _SeriesTab(broker)
    elif tab_name == JACOBIAN_ENTRIES_TAB:
        tab = _JacobianEntriesTab(broker)
    elif tab_name == JACOBIAN_HEATMAP_TAB:
        tab = _JacobianHeatmapTab(broker)
    else:
        raise KeyError(f"Unknown dashboard tab: {tab_name}")

    return tab, tab.panel


class _StandaloneDashboardTabApp:
    def __init__(self, tab_name, case_tracker, callback_period, doc, pid_of_calling_script, script, metadata=None, hist_file=None):
        self._tab_name = tab_name
        self._broker = LiveDataBroker(case_tracker, metadata=metadata, hist_file=hist_file)
        self._doc = doc
        self._tab, panel = _build_dashboard_tab(
            tab_name,
            case_tracker,
            callback_period,
            doc,
            pid_of_calling_script,
            script,
            broker=self._broker,
        )
        self._doc.add_root(panel.child)
        self._doc.add_periodic_callback(self._update, callback_period)
        self._doc.title = f"Dymos RTPlot - {DASHBOARD_TAB_TITLES[tab_name]}"

    def _update(self):
        new_snapshots = self._broker.poll()
        if not new_snapshots and not self._broker.is_running():
            return
        if not self._broker.snapshots:
            return
        self._tab.refresh(force=self._tab_name in _HEAVY_TABS)
