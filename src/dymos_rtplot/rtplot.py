"""Run the local rtplot fork without patching the installed OpenMDAO package."""

import argparse
import sys

from dymos_rtplot.realtime_plot.realtime_plot import (
    _add_dashboard_cli_arguments,
    _realtime_plot_cmd,
    _realtime_plot_setup_parser,
    _rtplot_cmd,
    _rtplot_setup_parser,
    clean_rtplot_artifacts,
)


_SUBCOMMANDS = {'realtime_plot', 'rtplot', 'clean', 'timeseries-plot', 'timeseries-plots'}
_TIMESERIES_PLOT_COMMANDS = {'timeseries-plot', 'timeseries-plots'}


def _looks_like_rtplot_invocation(raw_args):
    if not raw_args:
        return False
    if raw_args[0] in _SUBCOMMANDS:
        return False
    if raw_args[0] in {'-h', '--help'}:
        return True
    return True


def _build_entrypoint_parser():
    parser = argparse.ArgumentParser(prog='python -m dymos_rtplot.rtplot')
    parser.add_argument('file', nargs='?', help='Python file containing the model.')
    _add_dashboard_cli_arguments(parser)
    return parser


def _has_entrypoint_options(raw_args):
    return any(arg.startswith('-') for arg in raw_args)


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if _looks_like_rtplot_invocation(raw_args):
        entrypoint_parser = _build_entrypoint_parser()
        entry_args, user_args = entrypoint_parser.parse_known_args(raw_args)
        if entry_args.file:
            rtplot_args = ['rtplot', entry_args.file]
            for flag in (
                'open_browser',
                'host',
                'dashboard_mode',
                'tabs',
                'tab_core',
                'base_port',
                'idle_shutdown_seconds',
                'highlight_jacobian_structure',
                'dark_mode',
            ):
                value = getattr(entry_args, flag, None)
                if flag == 'open_browser':
                    if value:
                        rtplot_args.append('--open-browser')
                elif flag == 'highlight_jacobian_structure':
                    if value is False:
                        rtplot_args.append('--disable-jacobian-highlighting')
                elif flag == 'dark_mode':
                    if value is False:
                        rtplot_args.append('--light-mode')
                elif value is not None:
                    rtplot_args.extend([f"--{flag.replace('_', '-')}", str(value)])
            raw_args = rtplot_args + user_args
        elif raw_args and raw_args[0] in {'-h', '--help'}:
            entrypoint_parser.print_help()
            return
        elif _has_entrypoint_options(raw_args):
            entrypoint_parser.error("the following arguments are required: file")

    parser = argparse.ArgumentParser(prog='python -m dymos_rtplot.rtplot')
    subparsers = parser.add_subparsers(dest='command')

    realtime_parser = subparsers.add_parser('realtime_plot')
    _realtime_plot_setup_parser(realtime_parser)

    rtplot_parser = subparsers.add_parser('rtplot')
    _rtplot_setup_parser(rtplot_parser)

    clean_parser = subparsers.add_parser('clean')
    clean_parser.add_argument('path', nargs='?', default='.', help='Root directory to clean RTPlot artifacts from.')
    clean_parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without deleting it.')

    timeseries_parser = subparsers.add_parser(
        'timeseries-plot',
        aliases=['timeseries-plots'],
        help='Open the offline Dymos timeseries plotting tool.',
    )
    timeseries_parser.add_argument(
        'recorder',
        nargs='+',
        help='OpenMDAO/Dymos recorder SQLite or DB file(s) containing timeseries cases.',
    )
    timeseries_parser.add_argument('--metadata', default=None, help='Optional .rtplot_meta.json sidecar path for one recorder.')
    timeseries_parser.add_argument('--case', default='last', help="Driver case to plot: 'last' or an integer counter.")
    timeseries_parser.add_argument(
        '--no-auto-simulation',
        action='store_true',
        help='Do not auto-load traj_simulation_0_out/dymos_simulation.db next to dymos_solution.db.',
    )
    timeseries_parser.add_argument(
        '--project',
        default=None,
        help='Optional plot project JSON file to load on startup.',
    )

    args, user_args = parser.parse_known_args(raw_args)

    if args.command == 'realtime_plot':
        _realtime_plot_cmd(args, user_args)
    elif args.command == 'rtplot':
        _rtplot_cmd(args, user_args)
    elif args.command == 'clean':
        result = clean_rtplot_artifacts(args.path, dry_run=args.dry_run)
        dry = result.get('dry_run', args.dry_run)
        verb = "Would remove" if dry else "Removed"
        prefix = "[DRY RUN] " if dry else ""
        total = len(result['files']) + len(result['dirs'])
        print(f"{prefix}RTPlot cleanup in: {result['root']}")
        if total == 0:
            print("  Nothing to remove.")
        else:
            for path in result['files']:
                print(f"  {verb}: {path}")
            for path in result['dirs']:
                print(f"  {verb}: {path}  (directory tree)")
            print(f"  {verb} {total} item(s).")
        if dry:
            print("  Re-run without --dry-run to delete.")

    elif args.command in _TIMESERIES_PLOT_COMMANDS:
        from dymos_rtplot.timeseries_plotter import TimeseriesDataset, TimeseriesPlotterApp

        dataset = TimeseriesDataset(
            args.recorder,
            metadata_path=args.metadata,
            case=args.case,
            auto_simulation=not args.no_auto_simulation,
        )
        TimeseriesPlotterApp(dataset, project_path=args.project).run()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
