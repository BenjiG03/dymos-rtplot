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


def _looks_like_rtplot_invocation(raw_args):
    if not raw_args:
        return False
    if raw_args[0] in {'realtime_plot', 'rtplot', 'clean'}:
        return False
    if raw_args[0] in {'-h', '--help'}:
        return True
    return True


def _build_entrypoint_parser():
    parser = argparse.ArgumentParser(prog='python -m dymos_rtplot.rtplot')
    parser.add_argument('file', nargs='?', help='Python file containing the model.')
    _add_dashboard_cli_arguments(parser)
    return parser


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

    parser = argparse.ArgumentParser(prog='python -m dymos_rtplot.rtplot')
    subparsers = parser.add_subparsers(dest='command')

    realtime_parser = subparsers.add_parser('realtime_plot')
    _realtime_plot_setup_parser(realtime_parser)

    rtplot_parser = subparsers.add_parser('rtplot')
    _rtplot_setup_parser(rtplot_parser)

    clean_parser = subparsers.add_parser('clean')
    clean_parser.add_argument('path', nargs='?', default='.', help='Root directory to clean RTPlot artifacts from.')
    clean_parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without deleting it.')

    args, user_args = parser.parse_known_args(raw_args)

    if args.command == 'realtime_plot':
        _realtime_plot_cmd(args, user_args)
    elif args.command == 'rtplot':
        _rtplot_cmd(args, user_args)
    elif args.command == 'clean':
        result = clean_rtplot_artifacts(args.path, dry_run=args.dry_run)
        print(f"RTPlot cleanup root: {result['root']}")
        print(f"Directories: {len(result['dirs'])}")
        for path in result['dirs']:
            print(path)
        print(f"Files: {len(result['files'])}")
        for path in result['files']:
            print(path)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
