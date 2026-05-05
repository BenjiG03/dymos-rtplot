"""
The abstract class for the two forms of real time plot which are for AnalysisDriver and optimizers.

Put in a separate class to avoid circular imports
"""

try:
    from tornado.ioloop import IOLoop
    from openmdao.utils.shell_proc import _is_process_running

    class _RealTimePlot(object):
        def __init__(self, case_tracker, callback_period, doc, pid_of_calling_script, script):
            """
            Construct and initialize _RealTimeOptPlot instance.
            """
            self._case_tracker = case_tracker
            self._pid_of_calling_script = pid_of_calling_script
            self._doc = doc
            self._callback_period = callback_period
            self._source = None
            self._source_stream_dict = {}
            self._script = script

        def _source_process_running(self):
            return self._pid_of_calling_script is not None and _is_process_running(
                self._pid_of_calling_script
            )

        def _setup_data_source(self):
            raise NotImplementedError("_setup_data_source not implemented")

        def _setup_figure(self):
            raise NotImplementedError("_setup_figure not implemented")

        def _update(self):
            raise NotImplementedError("_update not implemented")

        def _update_wrapped_in_try(self):
            try:
                self._update()
            except Exception as e:
                print(f"Realtime plot update will retry after refresh error: {e}")
                return
except ImportError:
    class _RealTimePlot(object):
        pass
