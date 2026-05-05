import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np

from dymos_rtplot.timeseries_plotter import (
    AxisSettings,
    DerivedSeriesSpec,
    FormulaEvaluator,
    PlotProject,
    PlotSettings,
    PlotStyle,
    PlotTrace,
    SeriesData,
    TimeseriesDataset,
    _find_default_trace,
    apply_trace_axis_defaults,
    export_matplotlib_plot,
    render_matplotlib_plot,
)


class _FakeCase:
    def __init__(self, values):
        self.outputs = values
        self._var_info = {name: {"units": "s" if name.endswith(".time") else "m"} for name in values}

    def get_val(self, path):
        return self.outputs[path]


class _FakeReader:
    cases_by_path = {}

    def __init__(self, path):
        self.path = str(path)

    def list_cases(self, source=None, out_stream=None):
        if source == "driver":
            return []
        if source == "problem":
            return ["final"]
        return ["final"]

    def get_case(self, coordinate):
        return self.cases_by_path[self.path]


class TimeseriesDatasetTests(unittest.TestCase):
    def test_loads_dymos_solution_db_without_rtplot_metadata(self):
        with TemporaryDirectory() as tmp:
            recorder = str(Path(tmp) / "dymos_solution.db")
            Path(recorder).touch()
            _FakeReader.cases_by_path = {
                recorder: _FakeCase({
                    "traj.phase0.timeseries.time": np.array([[0.0], [1.0]]),
                    "traj.phase0.timeseries.states:x": np.array([[2.0], [3.0]]),
                })
            }

            with mock.patch("dymos_rtplot.timeseries_plotter.SqliteCaseReader", _FakeReader), \
                 mock.patch("dymos_rtplot.timeseries_plotter.load_rtplot_metadata", return_value=None):
                dataset = TimeseriesDataset(recorder, auto_simulation=False)

        self.assertIn("traj.phase0.timeseries.states:x", dataset.series)
        self.assertEqual(dataset.series["traj.phase0.timeseries.states:x"].units, "m")
        matches = dataset.search("phase0 x")
        self.assertEqual([item.name for item in matches], ["traj.phase0.timeseries.states:x"])

    def test_auto_loads_simulation_db_next_to_solution_db(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            solution = root / "dymos_solution.db"
            simulation = root / "traj_simulation_0_out" / "dymos_simulation.db"
            simulation.parent.mkdir()
            solution.touch()
            simulation.touch()
            _FakeReader.cases_by_path = {
                str(solution): _FakeCase({
                    "traj.phase0.timeseries.states:x": np.array([[1.0], [2.0]]),
                }),
                str(simulation): _FakeCase({
                    "traj.phase0.timeseries.states:x": np.array([[1.5], [2.5]]),
                }),
            }

            with mock.patch("dymos_rtplot.timeseries_plotter.SqliteCaseReader", _FakeReader), \
                 mock.patch("dymos_rtplot.timeseries_plotter.load_rtplot_metadata", return_value=None):
                dataset = TimeseriesDataset(str(solution))

        self.assertIn("solution.traj.phase0.timeseries.states:x", dataset.series)
        self.assertIn("simulation.traj.phase0.timeseries.states:x", dataset.series)


class FormulaEvaluatorTests(unittest.TestCase):
    def test_restricted_formula_converts_units(self):
        series = {
            "thrust": SeriesData("thrust", np.array([1.0, 2.0]), "N"),
            "mass_rate": SeriesData("mass_rate", np.array([1.0, 2.0]), "kg/s"),
        }
        evaluator = FormulaEvaluator(series)

        isp = evaluator.evaluate("-1 * unit(thrust, 'lbf') / unit(mass_rate, 'lbm/s')", name="ISP")

        self.assertEqual(isp.name, "ISP")
        self.assertTrue(np.all(np.isfinite(isp.values)))
        self.assertLess(isp.values[0], 0.0)

    def test_restricted_formula_rejects_attribute_access(self):
        evaluator = FormulaEvaluator({"x": SeriesData("x", np.array([1.0]), None)})

        with self.assertRaisesRegex(ValueError, "Unsupported"):
            evaluator.evaluate("x.__class__")

    def test_restricted_formula_uses_aliases(self):
        series = {"traj.phase.states:x": SeriesData("traj.phase.states:x", np.array([2.0]), "m")}
        evaluator = FormulaEvaluator(series, aliases={"x": "traj.phase.states:x"})

        result = evaluator.evaluate("x * 3")

        self.assertEqual(float(result.values[0]), 6.0)

    def test_restricted_formula_rejects_indexing_and_bad_calls(self):
        evaluator = FormulaEvaluator({"x": SeriesData("x", np.array([1.0]), None)})

        with self.assertRaisesRegex(ValueError, "Unsupported"):
            evaluator.evaluate("x[0]")
        with self.assertRaisesRegex(ValueError, "Only unit"):
            evaluator.evaluate("sum(x)")

    def test_restricted_formula_rejects_mismatched_lengths(self):
        evaluator = FormulaEvaluator({
            "x": SeriesData("x", np.array([1.0, 2.0]), None),
            "y": SeriesData("y", np.array([1.0, 2.0, 3.0]), None),
        })

        with self.assertRaisesRegex(ValueError, "matching lengths"):
            evaluator.evaluate("x + y")

    def test_dataset_adds_derived_series_with_units(self):
        dataset = mock.Mock()
        dataset.series = {
            "x": SeriesData("x", np.array([1.0, 2.0]), "m"),
        }
        dataset.aliases = {}
        dataset.add_derived_series = TimeseriesDataset.add_derived_series.__get__(dataset)
        dataset.rebuild_aliases = mock.Mock()

        series = dataset.add_derived_series(DerivedSeriesSpec("x_ft", "unit(x, 'ft')", "ft"))

        self.assertEqual(series.units, "ft")
        self.assertIn("x_ft", dataset.series)


class RenderingTests(unittest.TestCase):
    def test_default_trace_uses_time_for_x_and_populates_axis_defaults(self):
        series = {
            "traj.phase.timeseries.controls:u": SeriesData(
                "traj.phase.timeseries.controls:u",
                np.array([3.0, 4.0]),
                "deg",
                source="solution",
                phase="phase",
                category="controls",
                variable="u",
            ),
            "traj.phase.timeseries.time": SeriesData(
                "traj.phase.timeseries.time",
                np.array([0.0, 10.0]),
                "s",
                source="solution",
                phase="phase",
                category="ode",
                variable="time",
            ),
            "traj.phase.timeseries.states:x": SeriesData(
                "traj.phase.timeseries.states:x",
                np.array([1.0, 2.0]),
                "m",
                source="solution",
                phase="phase",
                category="states",
                variable="x",
            ),
        }

        trace = _find_default_trace(series)
        settings = apply_trace_axis_defaults(PlotSettings(), [trace], series, replace=True)

        self.assertEqual(trace.x, "traj.phase.timeseries.time")
        self.assertEqual(trace.y, "traj.phase.timeseries.controls:u")
        self.assertEqual(settings.axis("x1").label, "solution phase time (s)")
        self.assertEqual(settings.axis("x1").minimum, 0.0)
        self.assertEqual(settings.axis("x1").maximum, 10.0)
        self.assertEqual(settings.axis("y1").label, "solution phase u (deg)")
        self.assertEqual(settings.axis("y1").minimum, 3.0)
        self.assertEqual(settings.axis("y1").maximum, 4.0)

    def test_render_supports_secondary_x_and_y_axes_and_styles(self):
        series = {
            "time": SeriesData("time", np.array([0.0, 1.0]), "s"),
            "mach": SeriesData("mach", np.array([0.2, 0.3]), None),
            "alt": SeriesData("alt", np.array([10.0, 20.0]), "m"),
            "speed": SeriesData("speed", np.array([100.0, 200.0]), "m/s"),
        }
        settings = PlotSettings(
            axes={
                "x1": AxisSettings(label="time"),
                "x2": AxisSettings(label="mach", minimum=0.1, maximum=0.4),
                "y1": AxisSettings(label="alt"),
                "y2": AxisSettings(label="alt2", minimum=0.0, maximum=30.0),
            }
        )
        trace = PlotTrace(
            x="mach",
            y="alt",
            x_axis="x2",
            y_axis="y2",
            style=PlotStyle(color="red", line_width=3.0, marker_style="o", marker_size=7.0),
        )

        axes = render_matplotlib_plot(
            Figure(),
            series,
            [
                PlotTrace(x="time", y="alt", x_axis="x1", y_axis="y1"),
                PlotTrace(x="time", y="speed", x_axis="x1", y_axis="y2"),
                PlotTrace(x="mach", y="alt", x_axis="x2", y_axis="y1"),
                trace,
            ],
            settings,
        )

        self.assertIn(("x1", "y1"), axes)
        self.assertIn(("x1", "y2"), axes)
        self.assertIn(("x2", "y1"), axes)
        self.assertIn(("x2", "y2"), axes)
        self.assertEqual(len(axes), 4)
        line = axes[("x2", "y2")].lines[0]
        self.assertEqual(line.get_color(), "red")
        self.assertEqual(line.get_linewidth(), 3.0)
        self.assertEqual(line.get_marker(), "o")
        self.assertEqual(axes[("x2", "y2")].get_xlim(), (0.1, 0.4))
        self.assertEqual(axes[("x2", "y2")].get_ylim(), (0.0, 30.0))

    def test_export_writes_nonempty_file(self):
        with TemporaryDirectory() as tmp:
            series = {
                "time": SeriesData("time", np.array([0.0, 1.0]), "s"),
                "alt": SeriesData("alt", np.array([10.0, 20.0]), "m"),
            }
            for suffix in ("png", "pdf", "svg"):
                path = Path(tmp) / f"plot.{suffix}"
                export_matplotlib_plot(
                    path,
                    series,
                    [PlotTrace(x="time", y="alt")],
                    PlotSettings(),
                )

                self.assertGreater(path.stat().st_size, 0)


class PlotProjectTests(unittest.TestCase):
    def test_project_round_trips_and_reports_missing_series(self):
        project = PlotProject(
            recorder_paths=["dymos_solution.db"],
            traces=[PlotTrace(x="time", y="alt", label="Altitude")],
            derived_series=[DerivedSeriesSpec("isp", "thrust / mdot", "s")],
            settings=PlotSettings(title="Mission"),
        )

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "plot.json"
            project.save(path)
            loaded = PlotProject.load(path)

        self.assertEqual(loaded.settings.title, "Mission")
        self.assertEqual(loaded.traces[0].label, "Altitude")
        self.assertEqual(loaded.derived_series[0].name, "isp")
        self.assertEqual(loaded.missing_series({"time": object()}), ["alt"])


if __name__ == "__main__":
    unittest.main()
