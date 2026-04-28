"""Run the local rtplot fork without patching the installed OpenMDAO package."""

import argparse
import sys

from dymos_rtplot.realtime_plot.realtime_plot import (
    _realtime_plot_cmd,
    _realtime_plot_setup_parser,
    _rtplot_cmd,
    _rtplot_setup_parser,
)


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] not in {'realtime_plot', 'rtplot', '-h', '--help'}:
        raw_args = ['rtplot', *raw_args]

    parser = argparse.ArgumentParser(prog='python -m dymos_rtplot.rtplot')
    subparsers = parser.add_subparsers(dest='command')

    realtime_parser = subparsers.add_parser('realtime_plot')
    _realtime_plot_setup_parser(realtime_parser)

    rtplot_parser = subparsers.add_parser('rtplot')
    _rtplot_setup_parser(rtplot_parser)

    args, user_args = parser.parse_known_args(raw_args)

    if args.command == 'realtime_plot':
        _realtime_plot_cmd(args, user_args)
    elif args.command == 'rtplot':
        _rtplot_cmd(args, user_args)
    else:
        # Match the OpenMDAO CLI convenience behavior: treat a lone filename
        # as the rtplot target even when the user omits the explicit subcommand.
        parser.print_help()


if __name__ == '__main__':
    main()
