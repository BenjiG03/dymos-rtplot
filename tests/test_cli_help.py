import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from dymos_rtplot import rtplot


class CliHelpTests(unittest.TestCase):
    def test_entrypoint_help_lists_string_choices(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            with self.assertRaises(SystemExit):
                rtplot.main(["--help"])

        help_text = buf.getvalue()
        for expected in (
            "case-plotter",
            "trajectory",
            "series",
            "jacobian-entries",
            "jacobian-heatmap",
        ):
            self.assertIn(expected, help_text)

    def test_timeseries_plot_subcommand_is_registered(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rtplot.main([])

        self.assertIn("timeseries-plot", buf.getvalue())
        self.assertIn("timeseries-plots", buf.getvalue())

    def test_timeseries_plots_alias_help_uses_subcommand_parser(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            with self.assertRaises(SystemExit):
                rtplot.main(["timeseries-plots", "--help"])

        help_text = buf.getvalue()
        self.assertIn("recorder", help_text)
        self.assertIn("--metadata", help_text)
        self.assertIn("--case", help_text)
        self.assertNotIn("Python file containing the model", help_text)

    def test_entrypoint_dashboard_options_without_file_report_file_error(self):
        err = io.StringIO()
        with redirect_stderr(err):
            with self.assertRaises(SystemExit):
                rtplot.main(["--dashboard-mode", "multiwindow", "-b", "-s", "30", "-D", "-J"])

        message = err.getvalue()
        self.assertIn("the following arguments are required: file", message)
        self.assertIn("multiwindow", message)
        self.assertNotIn("invalid choice: 'multiwindow'", message)


if __name__ == "__main__":
    unittest.main()
