"""Offline matplotlib plotter for recorded Dymos timeseries outputs."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import uuid

import numpy as np
from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.utils.units import convert_units

from dymos_rtplot.realtime_plot.realtime_data import readonly_sqlite_connection
from dymos_rtplot.realtime_plot.realtime_metadata import load_rtplot_metadata


_IDENT_RE = re.compile(r"\W+")
_AXIS_NAMES = ("x1", "x2", "y1", "y2")
_X_AXIS_NAMES = ("x1", "x2")
_Y_AXIS_NAMES = ("y1", "y2")


@dataclass
class SeriesData:
    """One scalar timeseries with optional OpenMDAO units and search metadata."""

    name: str
    values: np.ndarray
    units: str | None = None
    source: str = ""
    trajectory: str = ""
    phase: str = ""
    category: str = ""
    variable: str = ""
    path: str = ""
    derived: bool = False

    @property
    def length(self):
        return int(np.asarray(self.values).reshape(-1).size)


@dataclass
class _LoadedRecorder:
    path: str
    label: str
    metadata: dict | None
    case: object


@dataclass
class AxisSettings:
    """Presentation settings for one matplotlib axis."""

    label: str = ""
    minimum: float | None = None
    maximum: float | None = None
    scale: str = "linear"

    def to_dict(self):
        return {
            "label": self.label,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(
            label=data.get("label", ""),
            minimum=data.get("minimum"),
            maximum=data.get("maximum"),
            scale=data.get("scale", "linear"),
        )


@dataclass
class PlotStyle:
    """Sticky cosmetic options for one plotted trace."""

    color: str | None = None
    line_style: str = "-"
    line_width: float = 2.0
    marker_style: str = ""
    marker_size: float = 5.0
    marker_face_color: str | None = None
    marker_edge_color: str | None = None

    def to_dict(self):
        return {
            "color": self.color,
            "line_style": self.line_style,
            "line_width": self.line_width,
            "marker_style": self.marker_style,
            "marker_size": self.marker_size,
            "marker_face_color": self.marker_face_color,
            "marker_edge_color": self.marker_edge_color,
        }

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(
            color=data.get("color"),
            line_style=data.get("line_style", "-"),
            line_width=float(data.get("line_width", 2.0)),
            marker_style=data.get("marker_style", ""),
            marker_size=float(data.get("marker_size", 5.0)),
            marker_face_color=data.get("marker_face_color"),
            marker_edge_color=data.get("marker_edge_color"),
        )


@dataclass
class PlotSettings:
    """Presentation-oriented figure settings."""

    title: str = ""
    axes: dict[str, AxisSettings] = field(
        default_factory=lambda: {name: AxisSettings() for name in _AXIS_NAMES}
    )
    show_grid: bool = True
    show_legend: bool = True
    legend_location: str = "best"
    figure_width: float = 10.0
    figure_height: float = 6.0
    dpi: int = 120

    def axis(self, name):
        return self.axes.setdefault(name, AxisSettings())

    def to_dict(self):
        return {
            "title": self.title,
            "axes": {name: self.axis(name).to_dict() for name in _AXIS_NAMES},
            "show_grid": self.show_grid,
            "show_legend": self.show_legend,
            "legend_location": self.legend_location,
            "figure_width": self.figure_width,
            "figure_height": self.figure_height,
            "dpi": self.dpi,
        }

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        axes = {
            name: AxisSettings.from_dict((data.get("axes") or {}).get(name))
            for name in _AXIS_NAMES
        }
        return cls(
            title=data.get("title", ""),
            axes=axes,
            show_grid=bool(data.get("show_grid", True)),
            show_legend=bool(data.get("show_legend", True)),
            legend_location=data.get("legend_location", "best"),
            figure_width=float(data.get("figure_width", 10.0)),
            figure_height=float(data.get("figure_height", 6.0)),
            dpi=int(data.get("dpi", 120)),
        )


@dataclass
class PlotTrace:
    """A request to plot one y series against one x series."""

    x: str
    y: str
    label: str | None = None
    x_axis: str = "x1"
    y_axis: str = "y1"
    style: PlotStyle = field(default_factory=PlotStyle)
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "x": self.x,
            "y": self.y,
            "label": self.label,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "style": self.style.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            trace_id=data.get("trace_id") or uuid.uuid4().hex,
            x=data["x"],
            y=data["y"],
            label=data.get("label"),
            x_axis=data.get("x_axis", "x1"),
            y_axis=data.get("y_axis", "y1"),
            style=PlotStyle.from_dict(data.get("style")),
        )


@dataclass
class DerivedSeriesSpec:
    """User-defined series created from a restricted expression."""

    name: str
    expression: str
    units: str | None = None

    def to_dict(self):
        return {"name": self.name, "expression": self.expression, "units": self.units}

    @classmethod
    def from_dict(cls, data):
        return cls(name=data["name"], expression=data["expression"], units=data.get("units"))


@dataclass
class PlotProject:
    """Serializable plotter project."""

    recorder_paths: list[str] = field(default_factory=list)
    metadata_path: str | None = None
    case: str = "last"
    auto_simulation: bool = True
    traces: list[PlotTrace] = field(default_factory=list)
    derived_series: list[DerivedSeriesSpec] = field(default_factory=list)
    settings: PlotSettings = field(default_factory=PlotSettings)

    def to_dict(self):
        return {
            "version": 1,
            "recorder_paths": self.recorder_paths,
            "metadata_path": self.metadata_path,
            "case": self.case,
            "auto_simulation": self.auto_simulation,
            "traces": [trace.to_dict() for trace in self.traces],
            "derived_series": [spec.to_dict() for spec in self.derived_series],
            "settings": self.settings.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            recorder_paths=list(data.get("recorder_paths", [])),
            metadata_path=data.get("metadata_path"),
            case=str(data.get("case", "last")),
            auto_simulation=bool(data.get("auto_simulation", True)),
            traces=[PlotTrace.from_dict(item) for item in data.get("traces", [])],
            derived_series=[
                DerivedSeriesSpec.from_dict(item) for item in data.get("derived_series", [])
            ],
            settings=PlotSettings.from_dict(data.get("settings")),
        )

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2)

    def missing_series(self, series):
        names = set(series)
        missing = []
        for trace in self.traces:
            if trace.x not in names:
                missing.append(trace.x)
            if trace.y not in names:
                missing.append(trace.y)
        return sorted(set(missing))


def _sanitize_identifier(name):
    clean = _IDENT_RE.sub("_", name).strip("_")
    if not clean or clean[0].isdigit():
        clean = f"v_{clean}"
    return clean


def _scalar_series(values):
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr.astype(float)
    return arr.reshape((arr.shape[0], -1))[:, 0].astype(float)


def _case_coordinate_for_counter(recorder_path, counter):
    with readonly_sqlite_connection(recorder_path) as con:
        cur = con.cursor()
        cur.execute(
            "SELECT iteration_coordinate FROM driver_iterations WHERE counter=:counter",
            {"counter": int(counter)},
        )
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"No driver case with counter {counter}.")
    return row[0]


def _default_recorder_label(path):
    stem = Path(path).stem
    if stem == "dymos_solution":
        return "solution"
    if stem == "dymos_simulation":
        return "simulation"
    return stem


def _expand_recorders(recorders):
    paths = [Path(path) for path in recorders]
    expanded = list(paths)
    for path in paths:
        if path.name != "dymos_solution.db":
            continue
        candidate = path.parent / "traj_simulation_0_out" / "dymos_simulation.db"
        if candidate.exists() and candidate not in expanded:
            expanded.append(candidate)
    return expanded


def _split_timeseries_path(path):
    source = ""
    parts = path.split(".")
    if parts and parts[0] in {"solution", "simulation"}:
        source = parts[0]
        parts = parts[1:]
    try:
        ts_idx = parts.index("timeseries")
    except ValueError:
        return source, "", "", "", "", path
    trajectory = parts[0] if parts else ""
    phase = parts[1] if len(parts) > 1 else ""
    output = ".".join(parts[ts_idx + 1 :])
    if ":" in output:
        category, variable = output.split(":", 1)
    else:
        category, variable = "ode", output
    return source, trajectory, phase, category, variable, ".".join(parts)


class TimeseriesDataset:
    """Load scalar Dymos timeseries from one or more recorder files."""

    def __init__(self, recorder_path, metadata_path=None, case="last", auto_simulation=True):
        raw_paths = recorder_path if isinstance(recorder_path, (list, tuple)) else [recorder_path]
        paths = _expand_recorders(raw_paths) if auto_simulation else [Path(path) for path in raw_paths]
        self.recorder_paths = [str(path) for path in paths]
        self.metadata_path = metadata_path
        self.case_name = str(case)
        self.auto_simulation = auto_simulation
        self.recorders = [self._load_recorder(path, metadata_path, case) for path in paths]
        self.recorder_path = self.recorders[0].path
        self.metadata = self.recorders[0].metadata
        self.case = self.recorders[0].case
        self.series = self._load_series()
        if not self.series:
            raise ValueError("No Dymos timeseries outputs were found in the recorder file(s).")
        self.aliases = self._build_aliases(self.series)

    def _load_recorder(self, recorder_path, metadata_path, case):
        path = str(recorder_path)
        metadata = load_rtplot_metadata(
            meta_file=metadata_path if len(self.recorder_paths) == 1 else None,
            case_recorder_filename=path,
        )
        return _LoadedRecorder(
            path=path,
            label=_default_recorder_label(path),
            metadata=metadata,
            case=self._load_case(path, case),
        )

    def _load_case(self, recorder_path, case):
        reader = SqliteCaseReader(recorder_path)
        if case == "last":
            cases = reader.list_cases("driver", out_stream=None)
            if not cases:
                cases = reader.list_cases("problem", out_stream=None)
            if not cases:
                cases = reader.list_cases(out_stream=None)
            if not cases:
                raise ValueError("Recorder does not contain cases.")
            return reader.get_case(cases[-1])
        if str(case).isdigit():
            coordinate = _case_coordinate_for_counter(recorder_path, case)
        else:
            coordinate = case
        return reader.get_case(coordinate)

    def _load_series(self):
        out = {}
        prefix_sources = len(self.recorders) > 1
        for recorder in self.recorders:
            loaded = self._load_metadata_series(recorder) if recorder.metadata else self._load_scanned_series(recorder)
            for name, series in loaded.items():
                out_name = f"{recorder.label}.{name}" if prefix_sources else name
                source, trajectory, phase, category, variable, raw_path = _split_timeseries_path(out_name)
                out[out_name] = SeriesData(
                    name=out_name,
                    values=series.values,
                    units=series.units,
                    source=source or recorder.label,
                    trajectory=trajectory,
                    phase=phase,
                    category=category,
                    variable=variable,
                    path=raw_path,
                )
        return out

    def _load_metadata_series(self, recorder):
        out = {}
        for traj_meta in recorder.metadata.get("trajectories", []):
            traj_name = traj_meta["name"]
            for phase_meta in traj_meta.get("phases", []):
                phase_name = phase_meta["name"]
                phase_path = phase_meta["promoted_path"]
                for output_name, output_meta in phase_meta.get("timeseries_outputs", {}).items():
                    rel_path = output_meta["path"]
                    path = rel_path if rel_path.startswith(phase_path) else f"{phase_path}.{rel_path}"
                    try:
                        values = _scalar_series(recorder.case.get_val(path))
                    except Exception:
                        continue
                    name = f"{traj_name}.{phase_name}.{output_name}"
                    out[name] = SeriesData(name=name, values=values, units=output_meta.get("units"))
        return out

    def _load_scanned_series(self, recorder):
        out = {}
        units_by_name = self._units_by_name(recorder.case)
        for path in recorder.case.outputs:
            if ".timeseries." not in path:
                continue
            try:
                values = _scalar_series(recorder.case.get_val(path))
            except Exception:
                continue
            if values.size == 0:
                continue
            out[path] = SeriesData(name=path, values=values, units=units_by_name.get(path))
        return out

    def _units_by_name(self, case):
        units = {}
        for mapping_name in ("_var_info", "_abs2meta"):
            mapping = getattr(case, mapping_name, None) or {}
            for name, meta in mapping.items():
                if isinstance(meta, dict) and meta.get("units") is not None:
                    units[name] = meta.get("units")
        return units

    def _build_aliases(self, series):
        aliases = {}
        used = set(series)
        for name, data in series.items():
            short = name.rsplit(".", 1)[-1]
            candidates = [
                short,
                data.variable,
                _sanitize_identifier(short),
                _sanitize_identifier(data.variable),
                _sanitize_identifier(name),
            ]
            for candidate in candidates:
                if candidate and candidate not in used and candidate not in aliases:
                    aliases[candidate] = name
                    used.add(candidate)
        return aliases

    def rebuild_aliases(self):
        self.aliases = self._build_aliases(self.series)

    def resolve_series(self, name):
        key = self.aliases.get(name, name)
        return self.series[key]

    def add_derived_series(self, spec):
        evaluator = FormulaEvaluator(self.series, aliases=self.aliases)
        result = evaluator.evaluate(spec.expression, name=spec.name)
        values = result.values
        units = result.units
        if spec.units:
            if units and units != spec.units:
                values = convert_units(values, units, spec.units)
            units = spec.units
        self.series[spec.name] = SeriesData(
            name=spec.name,
            values=np.asarray(values, dtype=float).reshape(-1),
            units=units,
            source="derived",
            category="derived",
            variable=spec.name,
            path=spec.expression,
            derived=True,
        )
        self.rebuild_aliases()
        return self.series[spec.name]

    def search(self, query="", source=None, category=None):
        terms = [term.lower() for term in query.split() if term]
        results = []
        for data in self.series.values():
            if source and data.source != source:
                continue
            if category and data.category != category:
                continue
            haystack = " ".join(
                str(part)
                for part in (
                    data.name,
                    data.source,
                    data.trajectory,
                    data.phase,
                    data.category,
                    data.variable,
                    data.units or "",
                )
            ).lower()
            if all(term in haystack for term in terms):
                results.append(data)
        return sorted(results, key=lambda item: item.name)


class FormulaEvaluator:
    """Evaluate restricted arithmetic expressions over named timeseries."""

    def __init__(self, series, aliases=None):
        self.series = series
        self.aliases = aliases or {}

    def evaluate(self, expression, name="derived"):
        tree = ast.parse(expression, mode="eval")
        result = self._eval(tree.body)
        return SeriesData(name=name, values=np.asarray(result.values, dtype=float).reshape(-1), units=result.units)

    def _eval(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return SeriesData("", np.asarray(float(node.value)), None)
        if isinstance(node, ast.Name):
            key = self.aliases.get(node.id, node.id)
            if key not in self.series:
                raise ValueError(f"Unknown timeseries '{node.id}'.")
            return self.series[key]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val = self._eval(node.operand)
            return SeriesData("", -np.asarray(val.values, dtype=float), val.units)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return self._eval(node.operand)
        if isinstance(node, ast.BinOp):
            return self._eval_binary(node)
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        raise ValueError(f"Unsupported expression element: {type(node).__name__}.")

    def _aligned_values(self, left, right):
        lval = np.asarray(left.values, dtype=float)
        rval = np.asarray(right.values, dtype=float)
        if lval.ndim == 0:
            lval = np.full(rval.shape, float(lval))
        if rval.ndim == 0:
            rval = np.full(lval.shape, float(rval))
        if lval.shape != rval.shape:
            raise ValueError("Derived series operands must have matching lengths.")
        return lval, rval

    def _eval_binary(self, node):
        left = self._eval(node.left)
        right = self._eval(node.right)
        lval, rval = self._aligned_values(left, right)
        if isinstance(node.op, (ast.Add, ast.Sub)) and left.units and right.units and left.units != right.units:
            rval = convert_units(rval, right.units, left.units)
            units = left.units
        elif isinstance(node.op, (ast.Add, ast.Sub)):
            units = left.units or right.units
        else:
            units = None
        if isinstance(node.op, ast.Add):
            return SeriesData("", lval + rval, units)
        if isinstance(node.op, ast.Sub):
            return SeriesData("", lval - rval, units)
        if isinstance(node.op, ast.Mult):
            return SeriesData("", lval * rval, None)
        if isinstance(node.op, ast.Div):
            return SeriesData("", lval / rval, None)
        if isinstance(node.op, ast.Pow):
            return SeriesData("", lval ** rval, None)
        raise ValueError(f"Unsupported expression element: {type(node.op).__name__}.")

    def _eval_call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in {"unit", "convert"}:
            raise ValueError("Only unit(series, 'target_units') is supported.")
        if len(node.args) != 2 or node.keywords:
            raise ValueError("unit() requires a series and target units.")
        value = self._eval(node.args[0])
        target_node = node.args[1]
        if not isinstance(target_node, ast.Constant) or not isinstance(target_node.value, str):
            raise ValueError("Target units must be a string literal.")
        if not value.units:
            raise ValueError("Cannot convert a unitless series.")
        target_units = target_node.value
        return SeriesData("", convert_units(value.values, value.units, target_units), target_units)


def _apply_axis_settings(ax, settings, x_axis, y_axis):
    x_settings = settings.axis(x_axis)
    y_settings = settings.axis(y_axis)
    ax.set_xlabel(x_settings.label)
    ax.set_ylabel(y_settings.label)
    ax.set_xscale(x_settings.scale)
    ax.set_yscale(y_settings.scale)
    if x_settings.minimum is not None or x_settings.maximum is not None:
        ax.set_xlim(left=x_settings.minimum, right=x_settings.maximum)
    if y_settings.minimum is not None or y_settings.maximum is not None:
        ax.set_ylim(bottom=y_settings.minimum, top=y_settings.maximum)
    if x_axis == "x2":
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
    if y_axis == "y2":
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()


def render_matplotlib_plot(figure, series, traces, settings):
    """Render traces on up to two x axes and two y axes."""
    figure.clear()
    figure.set_size_inches(settings.figure_width, settings.figure_height, forward=True)
    figure.set_dpi(settings.dpi)
    axes = {}

    def _axis_for(x_axis, y_axis):
        key = (x_axis, y_axis)
        if key in axes:
            return axes[key]
        if ("x1", "y1") not in axes:
            axes[("x1", "y1")] = figure.add_subplot(111)
        base = axes[("x1", "y1")]
        if key == ("x1", "y1"):
            return base
        ax = base
        if y_axis == "y2":
            ax = base.twinx()
        if x_axis == "x2":
            ax = ax.twiny()
        axes[key] = ax
        return ax

    for trace in traces:
        if trace.x not in series or trace.y not in series:
            continue
        ax = _axis_for(trace.x_axis, trace.y_axis)
        x = series[trace.x]
        y = series[trace.y]
        xvals = np.asarray(x.values, dtype=float).reshape(-1)
        yvals = np.asarray(y.values, dtype=float).reshape(-1)
        if xvals.shape != yvals.shape:
            raise ValueError(f"Trace '{trace.label or trace.y}' has mismatched x/y lengths.")
        style = trace.style
        marker_face_color = style.marker_face_color or style.color
        marker_edge_color = style.marker_edge_color or style.color
        ax.plot(
            xvals,
            yvals,
            color=style.color,
            linestyle=style.line_style,
            linewidth=style.line_width,
            marker=style.marker_style or None,
            markersize=style.marker_size,
            markerfacecolor=marker_face_color,
            markeredgecolor=marker_edge_color,
            label=trace.label or y.name,
        )

    if not axes:
        axes[("x1", "y1")] = figure.add_subplot(111)

    for (x_axis, y_axis), ax in axes.items():
        _apply_axis_settings(ax, settings, x_axis, y_axis)
        ax.grid(settings.show_grid)

    base = axes[("x1", "y1")]
    if settings.title:
        base.set_title(settings.title)
    if settings.show_legend:
        handles = []
        labels = []
        seen_axes = []
        for ax in axes.values():
            if ax in seen_axes:
                continue
            seen_axes.append(ax)
            axis_handles, axis_labels = ax.get_legend_handles_labels()
            handles.extend(axis_handles)
            labels.extend(axis_labels)
        if handles:
            base.legend(handles, labels, loc=settings.legend_location)
    figure.tight_layout()
    return axes


def export_matplotlib_plot(path, series, traces, settings):
    """Render and export a plot to a file."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(settings.figure_width, settings.figure_height), dpi=settings.dpi)
    render_matplotlib_plot(figure, series, traces, settings)
    figure.savefig(path, bbox_inches="tight", dpi=settings.dpi)
    return path


def _series_axis_label(series):
    label = series.variable or series.name
    if series.phase:
        label = f"{series.phase} {label}"
    if series.source and series.source not in label:
        label = f"{series.source} {label}"
    if series.units:
        label = f"{label} ({series.units})"
    return label


def _series_limits(series):
    values = np.asarray(series.values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    return float(np.min(finite)), float(np.max(finite))


def _find_default_trace(series):
    names = list(series)
    if len(names) < 2:
        return None
    time_candidates = [
        name for name, data in series.items()
        if data.variable in {"time", "time_phase"} or name.endswith(".timeseries.time")
    ]
    x_name = time_candidates[0] if time_candidates else names[0]
    y_candidates = [
        name for name, data in series.items()
        if name != x_name and data.variable not in {"time", "time_phase"}
    ]
    if not y_candidates:
        return None
    priority = {"states": 0, "controls": 1, "ode": 2, "state_rates": 3, "control_rates": 4}
    y_name = sorted(
        y_candidates,
        key=lambda name: (priority.get(series[name].category, 99), name),
    )[0]
    return PlotTrace(x=x_name, y=y_name, label=series[y_name].variable or y_name)


def apply_trace_axis_defaults(settings, traces, series, replace=False):
    """Populate blank axis labels and limits from the plotted series."""
    for trace in traces:
        if trace.x not in series or trace.y not in series:
            continue
        for axis_name, series_name in ((trace.x_axis, trace.x), (trace.y_axis, trace.y)):
            axis = settings.axis(axis_name)
            data = series[series_name]
            if replace or not axis.label:
                axis.label = _series_axis_label(data)
            lower, upper = _series_limits(data)
            if lower is None:
                continue
            if replace or axis.minimum is None:
                axis.minimum = lower
            if replace or axis.maximum is None:
                axis.maximum = upper
    return settings


class TimeseriesPlotterApp:
    """Tkinter shell around matplotlib for interactive chart styling."""

    def __init__(self, dataset, project_path=None):
        self.dataset = dataset
        self.settings = PlotSettings()
        self.traces = []
        self.derived_specs = []
        self.project_path = project_path
        self._series_rows = {}
        self._trace_rows = {}
        if project_path:
            self._apply_project(PlotProject.load(project_path))

    def _apply_project(self, project):
        self.settings = project.settings
        self.traces = list(project.traces)
        self.derived_specs = []
        for spec in project.derived_series:
            try:
                self.dataset.add_derived_series(spec)
                self.derived_specs.append(spec)
            except Exception:
                pass

    def _project(self):
        return PlotProject(
            recorder_paths=list(self.dataset.recorder_paths),
            metadata_path=self.dataset.metadata_path,
            case=self.dataset.case_name,
            auto_simulation=self.dataset.auto_simulation,
            traces=list(self.traces),
            derived_series=list(self.derived_specs),
            settings=self.settings,
        )

    def run(self):
        import tkinter as tk
        from tkinter import colorchooser, filedialog, messagebox, ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
        from matplotlib import colormaps
        from matplotlib.colors import to_hex

        root = tk.Tk()
        root.title("Dymos Timeseries Plotter")
        root.geometry("1480x860")
        figure = Figure(figsize=(self.settings.figure_width, self.settings.figure_height), dpi=self.settings.dpi)

        main = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(main, padding=6)
        center = ttk.Frame(main, padding=6)
        right = ttk.Frame(main, padding=6)
        main.add(left, weight=1)
        main.add(center, weight=3)
        main.add(right, weight=1)

        search_var = tk.StringVar()
        x_pick_var = tk.StringVar()
        y_pick_var = tk.StringVar()

        ttk.Label(left, text="Search").pack(anchor=tk.W)
        search_entry = ttk.Entry(left, textvariable=search_var)
        search_entry.pack(fill=tk.X)
        browser_cols = ("source", "phase", "category", "units", "length")
        browser = ttk.Treeview(left, columns=browser_cols, show="tree headings", height=25)
        browser.heading("#0", text="Series")
        browser.column("#0", width=280)
        for col in browser_cols:
            browser.heading(col, text=col.title())
            browser.column(col, width=80, anchor=tk.W)
        browser.pack(fill=tk.BOTH, expand=True, pady=(6, 6))

        pick_frame = ttk.Frame(left)
        pick_frame.pack(fill=tk.X)
        ttk.Button(pick_frame, text="Use as X", command=lambda: set_pick("x")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(pick_frame, text="Use as Y", command=lambda: set_pick("y")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(left, textvariable=x_pick_var, wraplength=340).pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(left, textvariable=y_pick_var, wraplength=340).pack(anchor=tk.W)
        ttk.Button(left, text="Add Trace", command=lambda: add_trace_from_picks()).pack(fill=tk.X, pady=(6, 0))

        ttk.Label(left, text="Derived Series").pack(anchor=tk.W, pady=(12, 0))
        derived_name_var = tk.StringVar()
        derived_units_var = tk.StringVar()
        derived_expr = tk.Text(left, height=4, width=42)
        ttk.Label(left, text="Name").pack(anchor=tk.W)
        ttk.Entry(left, textvariable=derived_name_var).pack(fill=tk.X)
        ttk.Label(left, text="Output units (optional)").pack(anchor=tk.W)
        ttk.Entry(left, textvariable=derived_units_var).pack(fill=tk.X)
        ttk.Label(left, text="Expression").pack(anchor=tk.W)
        derived_expr.pack(fill=tk.X)
        ttk.Label(
            left,
            text="Use series names or aliases. Example: unit(thrust, 'lbf') / unit(mass_rate, 'lbm/s')",
            wraplength=340,
        ).pack(anchor=tk.W)
        ttk.Button(left, text="Add Derived Series", command=lambda: add_derived()).pack(fill=tk.X)

        trace_cols = ("x", "y", "axes", "label")
        trace_table = ttk.Treeview(center, columns=trace_cols, show="tree headings", height=7)
        trace_table.heading("#0", text="ID")
        trace_table.column("#0", width=0, stretch=False)
        for col in trace_cols:
            trace_table.heading(col, text=col.upper())
            trace_table.column(col, width=220 if col in {"x", "y"} else 90)
        trace_table.pack(fill=tk.X)
        trace_buttons = ttk.Frame(center)
        trace_buttons.pack(fill=tk.X, pady=(4, 6))
        ttk.Button(trace_buttons, text="Remove Trace", command=lambda: remove_selected_trace()).pack(side=tk.LEFT)
        ttk.Button(trace_buttons, text="Apply Color Cycle", command=lambda: apply_color_cycle()).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(trace_buttons, text="Redraw", command=lambda: redraw()).pack(side=tk.LEFT, padx=(4, 0))

        canvas = FigureCanvasTkAgg(figure, master=center)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.configure(width=780, height=520)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, center, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)

        title_var = tk.StringVar(value=self.settings.title)
        legend_var = tk.BooleanVar(value=self.settings.show_legend)
        grid_var = tk.BooleanVar(value=self.settings.show_grid)
        legend_loc_var = tk.StringVar(value=self.settings.legend_location)
        fig_w_var = tk.StringVar(value=str(self.settings.figure_width))
        fig_h_var = tk.StringVar(value=str(self.settings.figure_height))
        dpi_var = tk.StringVar(value=str(self.settings.dpi))
        axis_vars = {}
        trace_style_vars = {
            "label": tk.StringVar(),
            "x_axis": tk.StringVar(value="x1"),
            "y_axis": tk.StringVar(value="y1"),
            "color": tk.StringVar(),
            "line_style": tk.StringVar(value="-"),
            "line_width": tk.StringVar(value="2.0"),
            "marker_style": tk.StringVar(),
            "marker_size": tk.StringVar(value="5.0"),
            "marker_face_color": tk.StringVar(),
            "marker_edge_color": tk.StringVar(),
            "x": tk.StringVar(),
            "y": tk.StringVar(),
        }

        def labeled_entry(parent, label, variable):
            ttk.Label(parent, text=label).pack(anchor=tk.W)
            entry = ttk.Entry(parent, textvariable=variable)
            entry.pack(fill=tk.X)
            return entry

        def labeled_combo(parent, label, variable, values):
            ttk.Label(parent, text=label).pack(anchor=tk.W)
            combo = ttk.Combobox(parent, textvariable=variable, values=values)
            combo.pack(fill=tk.X)
            return combo

        def color_row(parent, label, variable):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X)
            ttk.Label(frame, text=label).pack(side=tk.LEFT)
            ttk.Entry(frame, textvariable=variable, width=12).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Button(frame, text="Pick", command=lambda: pick_color(variable)).pack(side=tk.LEFT)

        def pick_color(variable):
            chosen = colorchooser.askcolor(color=variable.get() or None)
            if chosen and chosen[1]:
                variable.set(chosen[1])

        ttk.Label(right, text="Figure").pack(anchor=tk.W)
        labeled_entry(right, "Title", title_var)
        labeled_entry(right, "Width", fig_w_var)
        labeled_entry(right, "Height", fig_h_var)
        labeled_entry(right, "DPI", dpi_var)
        ttk.Checkbutton(right, text="Legend", variable=legend_var).pack(anchor=tk.W)
        ttk.Checkbutton(right, text="Grid", variable=grid_var).pack(anchor=tk.W)
        labeled_combo(
            right,
            "Legend location",
            legend_loc_var,
            ["best", "upper right", "upper left", "lower left", "lower right", "center right", "center"],
        )

        ttk.Label(right, text="Axes").pack(anchor=tk.W, pady=(12, 0))
        for axis_name in _AXIS_NAMES:
            frame = ttk.LabelFrame(right, text=axis_name)
            frame.pack(fill=tk.X, pady=(2, 2))
            settings = self.settings.axis(axis_name)
            label = tk.StringVar(value=settings.label)
            minimum = tk.StringVar(value="" if settings.minimum is None else str(settings.minimum))
            maximum = tk.StringVar(value="" if settings.maximum is None else str(settings.maximum))
            scale = tk.StringVar(value=settings.scale)
            axis_vars[axis_name] = (label, minimum, maximum, scale)
            labeled_entry(frame, "Label", label)
            labeled_entry(frame, "Min", minimum)
            labeled_entry(frame, "Max", maximum)
            labeled_combo(frame, "Scale", scale, ["linear", "log"])

        ttk.Label(right, text="Selected Trace").pack(anchor=tk.W, pady=(12, 0))
        labeled_combo(right, "X series", trace_style_vars["x"], list(self.dataset.series))
        labeled_combo(right, "Y series", trace_style_vars["y"], list(self.dataset.series))
        labeled_entry(right, "Label", trace_style_vars["label"])
        labeled_combo(right, "X axis", trace_style_vars["x_axis"], list(_X_AXIS_NAMES))
        labeled_combo(right, "Y axis", trace_style_vars["y_axis"], list(_Y_AXIS_NAMES))
        color_row(right, "Color", trace_style_vars["color"])
        labeled_combo(right, "Line style", trace_style_vars["line_style"], ["-", "--", "-.", ":", "None"])
        labeled_entry(right, "Line width", trace_style_vars["line_width"])
        labeled_combo(right, "Marker", trace_style_vars["marker_style"], ["", ".", "o", "s", "^", "v", "x", "+", "D", "*"])
        labeled_entry(right, "Marker size", trace_style_vars["marker_size"])
        color_row(right, "Marker face", trace_style_vars["marker_face_color"])
        color_row(right, "Marker edge", trace_style_vars["marker_edge_color"])
        ttk.Button(right, text="Apply Trace Style", command=lambda: apply_trace_style()).pack(fill=tk.X, pady=(6, 0))

        io_frame = ttk.Frame(right)
        io_frame.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(io_frame, text="Save Project", command=lambda: save_project()).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(io_frame, text="Load Project", command=lambda: load_project()).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(right, text="Export...", command=lambda: export()).pack(fill=tk.X, pady=(4, 0))

        def selected_series_name():
            selection = browser.selection()
            if not selection:
                return None
            return self._series_rows.get(selection[0])

        def set_pick(which):
            name = selected_series_name()
            if not name:
                return
            if which == "x":
                x_pick_var.set(name)
            else:
                y_pick_var.set(name)

        def refresh_browser(*_args):
            browser.delete(*browser.get_children())
            self._series_rows.clear()
            for data in self.dataset.search(search_var.get()):
                row_id = browser.insert(
                    "",
                    tk.END,
                    text=data.name,
                    values=(data.source, data.phase, data.category, data.units or "", data.length),
                )
                self._series_rows[row_id] = data.name

        def refresh_trace_table():
            trace_table.delete(*trace_table.get_children())
            self._trace_rows.clear()
            for trace in self.traces:
                row_id = trace_table.insert(
                    "",
                    tk.END,
                    text=trace.trace_id,
                    values=(trace.x, trace.y, f"{trace.x_axis}/{trace.y_axis}", trace.label or ""),
                )
                self._trace_rows[row_id] = trace.trace_id

        def sync_axis_settings_to_ui():
            for axis_name, variables in axis_vars.items():
                label, minimum, maximum, scale = variables
                axis = self.settings.axis(axis_name)
                label.set(axis.label)
                minimum.set("" if axis.minimum is None else f"{axis.minimum:g}")
                maximum.set("" if axis.maximum is None else f"{axis.maximum:g}")
                scale.set(axis.scale)

        def apply_auto_axis_defaults(replace=False):
            apply_trace_axis_defaults(self.settings, self.traces, self.dataset.series, replace=replace)
            sync_axis_settings_to_ui()

        def selected_trace():
            selection = trace_table.selection()
            if not selection:
                return None
            trace_id = self._trace_rows.get(selection[0])
            for trace in self.traces:
                if trace.trace_id == trace_id:
                    return trace
            return None

        def load_trace_into_editor(*_args):
            trace = selected_trace()
            if not trace:
                return
            trace_style_vars["x"].set(trace.x)
            trace_style_vars["y"].set(trace.y)
            trace_style_vars["label"].set(trace.label or "")
            trace_style_vars["x_axis"].set(trace.x_axis)
            trace_style_vars["y_axis"].set(trace.y_axis)
            trace_style_vars["color"].set(trace.style.color or "")
            trace_style_vars["line_style"].set(trace.style.line_style)
            trace_style_vars["line_width"].set(str(trace.style.line_width))
            trace_style_vars["marker_style"].set(trace.style.marker_style)
            trace_style_vars["marker_size"].set(str(trace.style.marker_size))
            trace_style_vars["marker_face_color"].set(trace.style.marker_face_color or "")
            trace_style_vars["marker_edge_color"].set(trace.style.marker_edge_color or "")

        def add_trace_from_picks():
            x_name = x_pick_var.get()
            y_name = y_pick_var.get()
            if not x_name or not y_name:
                messagebox.showerror("Missing series", "Select both an x series and y series.")
                return
            self.traces.append(PlotTrace(x=x_name, y=y_name, label=y_name))
            apply_auto_axis_defaults()
            refresh_trace_table()
            redraw()

        def remove_selected_trace():
            trace = selected_trace()
            if trace:
                self.traces = [item for item in self.traces if item.trace_id != trace.trace_id]
                refresh_trace_table()
                redraw()

        def apply_trace_style():
            trace = selected_trace()
            if not trace:
                return
            trace.x = trace_style_vars["x"].get()
            trace.y = trace_style_vars["y"].get()
            trace.label = trace_style_vars["label"].get() or None
            trace.x_axis = trace_style_vars["x_axis"].get()
            trace.y_axis = trace_style_vars["y_axis"].get()
            trace.style.color = trace_style_vars["color"].get() or None
            trace.style.line_style = trace_style_vars["line_style"].get()
            trace.style.line_width = parse_float(trace_style_vars["line_width"].get(), 2.0)
            trace.style.marker_style = trace_style_vars["marker_style"].get()
            trace.style.marker_size = parse_float(trace_style_vars["marker_size"].get(), 5.0)
            trace.style.marker_face_color = trace_style_vars["marker_face_color"].get() or None
            trace.style.marker_edge_color = trace_style_vars["marker_edge_color"].get() or None
            apply_auto_axis_defaults()
            refresh_trace_table()
            redraw()

        def add_derived():
            spec = DerivedSeriesSpec(
                name=derived_name_var.get().strip(),
                expression=derived_expr.get("1.0", tk.END).strip(),
                units=derived_units_var.get().strip() or None,
            )
            if not spec.name or not spec.expression:
                messagebox.showerror("Invalid derived series", "Name and expression are required.")
                return
            try:
                self.dataset.add_derived_series(spec)
            except Exception as err:
                messagebox.showerror("Invalid derived series", str(err))
                return
            self.derived_specs.append(spec)
            refresh_browser()
            messagebox.showinfo("Derived series", f"Added {spec.name}.")

        def apply_settings_from_ui():
            self.settings.title = title_var.get()
            self.settings.show_legend = bool(legend_var.get())
            self.settings.show_grid = bool(grid_var.get())
            self.settings.legend_location = legend_loc_var.get()
            self.settings.figure_width = parse_float(fig_w_var.get(), 10.0)
            self.settings.figure_height = parse_float(fig_h_var.get(), 6.0)
            self.settings.dpi = int(parse_float(dpi_var.get(), 120))
            for axis_name, variables in axis_vars.items():
                label, minimum, maximum, scale = variables
                axis = self.settings.axis(axis_name)
                axis.label = label.get()
                axis.minimum = parse_optional_float(minimum.get())
                axis.maximum = parse_optional_float(maximum.get())
                axis.scale = scale.get()

        def redraw():
            apply_settings_from_ui()
            apply_auto_axis_defaults()
            try:
                render_matplotlib_plot(figure, self.dataset.series, self.traces, self.settings)
            except Exception as err:
                messagebox.showerror("Plot error", str(err))
                return
            canvas.draw()
            canvas.draw_idle()

        def apply_color_cycle():
            cmap = colormaps.get_cmap("tab10")
            for idx, trace in enumerate(self.traces):
                trace.style.color = to_hex(cmap(idx % 10))
            load_trace_into_editor()
            redraw()

        def save_project():
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("RTPlot project", "*.json"), ("All files", "*.*")],
            )
            if not path:
                return
            apply_settings_from_ui()
            self._project().save(path)
            self.project_path = path

        def load_project():
            path = filedialog.askopenfilename(filetypes=[("RTPlot project", "*.json"), ("All files", "*.*")])
            if not path:
                return
            project = PlotProject.load(path)
            self._apply_project(project)
            missing = project.missing_series(self.dataset.series)
            if missing:
                messagebox.showwarning("Missing series", "\n".join(missing))
            sync_settings_to_ui()
            refresh_browser()
            refresh_trace_table()
            redraw()

        def export():
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG", "*.png"),
                    ("PDF", "*.pdf"),
                    ("SVG", "*.svg"),
                    ("JPEG", "*.jpg"),
                ],
            )
            if not path:
                return
            apply_settings_from_ui()
            try:
                export_matplotlib_plot(path, self.dataset.series, self.traces, self.settings)
            except Exception as err:
                messagebox.showerror("Export error", str(err))

        def sync_settings_to_ui():
            title_var.set(self.settings.title)
            legend_var.set(self.settings.show_legend)
            grid_var.set(self.settings.show_grid)
            legend_loc_var.set(self.settings.legend_location)
            fig_w_var.set(str(self.settings.figure_width))
            fig_h_var.set(str(self.settings.figure_height))
            dpi_var.set(str(self.settings.dpi))
            sync_axis_settings_to_ui()

        def parse_float(text, default):
            try:
                return float(text)
            except Exception:
                return default

        def parse_optional_float(text):
            text = text.strip()
            if not text:
                return None
            return float(text)

        search_var.trace_add("write", refresh_browser)
        trace_table.bind("<<TreeviewSelect>>", load_trace_into_editor)
        refresh_browser()
        refresh_trace_table()
        if not self.traces and len(self.dataset.series) >= 2:
            default_trace = _find_default_trace(self.dataset.series)
            if default_trace:
                self.traces.append(default_trace)
                x_pick_var.set(default_trace.x)
                y_pick_var.set(default_trace.y)
                apply_auto_axis_defaults(replace=True)
            refresh_trace_table()
        redraw()
        root.mainloop()


def build_parser():
    parser = argparse.ArgumentParser(prog="dymos-rtplot timeseries-plot")
    parser.add_argument("recorder", nargs="+", help="OpenMDAO/Dymos recorder SQLite or DB file(s) containing timeseries cases.")
    parser.add_argument("--metadata", default=None, help="Optional .rtplot_meta.json sidecar path.")
    parser.add_argument("--case", default="last", help="Driver case to plot: 'last' or an integer counter.")
    parser.add_argument(
        "--no-auto-simulation",
        action="store_true",
        help="Do not auto-load traj_simulation_0_out/dymos_simulation.db next to dymos_solution.db.",
    )
    parser.add_argument("--project", default=None, help="Optional plot project JSON file to load on startup.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    dataset = TimeseriesDataset(
        [Path(path) for path in args.recorder],
        metadata_path=args.metadata,
        case=args.case,
        auto_simulation=not args.no_auto_simulation,
    )
    TimeseriesPlotterApp(dataset, project_path=args.project).run()
