"""Live data broker for the Dymos realtime plot dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from pyoptsparse.pyOpt_history import History
except Exception:  # pragma: no cover
    History = None


_HISTORY_EXCLUDED_KEYS = {
    "funcs",
    "funcsSens",
    "xuser",
    "isMajor",
    "fail",
}


def _to_numpy(value):
    if value is None:
        return None
    return np.asarray(value)


def _flatten_value(value):
    arr = np.asarray(value)
    if arr.size == 0:
        return np.array([], dtype=float)
    return np.ravel(arr.astype(float))


@dataclass
class BrokerSnapshot:
    """One merged major-iteration snapshot."""

    counter: int
    timestamp: float
    major_iteration: int
    case: object
    objs: dict
    desvars: dict
    cons: dict
    scaled_objs: dict
    scaled_desvars: dict
    scaled_cons: dict
    derivatives: object | None
    opt_history: dict


class LiveDataBroker:
    """Poll driver cases and optional optimizer history into a unified stream."""

    def __init__(self, case_tracker, metadata=None, hist_file=None):
        self.case_tracker = case_tracker
        self.metadata = metadata or {}
        self.hist_file = hist_file or self.metadata.get("hist_file")
        self.snapshots = []
        self.warning_messages = []
        self._next_counter = 1
        self._history_cache = []
        self._history_keys = []
        self._history_mtime = None
        self._history_aligned_warning = None

    def poll(self):
        """Read all new major iterations available right now."""
        new_snapshots = []
        while True:
            case = self.case_tracker.get_case_by_counter(self._next_counter)
            if case is None:
                break
            snapshot = self._build_snapshot(case)
            self.snapshots.append(snapshot)
            new_snapshots.append(snapshot)
            self._next_counter += 1

        self._reload_history_if_needed()
        self._align_history()
        return new_snapshots

    def latest_snapshot(self):
        return self.snapshots[-1] if self.snapshots else None

    def is_running(self):
        return self.case_tracker.is_source_process_running()

    def _build_snapshot(self, case):
        major_iteration = len(self.snapshots) + 1
        return BrokerSnapshot(
            counter=int(case.counter),
            timestamp=float(case.timestamp),
            major_iteration=major_iteration,
            case=case,
            objs={name: _to_numpy(val) for name, val in case.get_objectives(scaled=False).items()},
            desvars={name: _to_numpy(val) for name, val in case.get_design_vars(scaled=False).items()},
            cons={name: _to_numpy(val) for name, val in case.get_constraints(scaled=False).items()},
            scaled_objs={name: _to_numpy(val) for name, val in case.get_objectives(scaled=True).items()},
            scaled_desvars={name: _to_numpy(val) for name, val in case.get_design_vars(scaled=True).items()},
            scaled_cons={name: _to_numpy(val) for name, val in case.get_constraints(scaled=True).items()},
            derivatives=getattr(case, "derivatives", None),
            opt_history={},
        )

    def _reload_history_if_needed(self):
        if not self.hist_file or History is None:
            return

        hist_path = Path(self.hist_file)
        if not hist_path.exists():
            return

        mtime = hist_path.stat().st_mtime
        if self._history_mtime is not None and mtime == self._history_mtime:
            return

        try:
            hist = History(str(hist_path), flag="r")
        except Exception as exc:
            self._history_aligned_warning = f"Unable to read optimizer history: {exc}"
            return

        iter_keys = [key for key in hist.getIterKeys() if key not in _HISTORY_EXCLUDED_KEYS]
        values = hist.getValues(names=iter_keys, major=True, scale=False) if iter_keys else {}
        entries = []
        num_rows = 0
        if values:
            num_rows = max(value.shape[0] for value in values.values())

        for row_idx in range(num_rows):
            entry = {}
            for key in iter_keys:
                entry[key] = values[key][row_idx]
            entries.append(entry)

        self._history_cache = entries
        self._history_keys = iter_keys
        self._history_mtime = mtime

    def _align_history(self):
        if not self.snapshots:
            return

        if not self._history_cache:
            for snapshot in self.snapshots:
                snapshot.opt_history = {}
            return

        shared_len = min(len(self.snapshots), len(self._history_cache))
        for idx, snapshot in enumerate(self.snapshots):
            if idx < shared_len:
                snapshot.opt_history = self._history_cache[idx]
            else:
                snapshot.opt_history = {}

        if len(self.snapshots) != len(self._history_cache):
            self._history_aligned_warning = (
                "Optimizer history and OpenMDAO case counts diverged; "
                f"aligned only the first {shared_len} major iterations."
            )
        else:
            self._history_aligned_warning = None

    def get_history_keys(self):
        return list(self._history_keys)

    def get_history_warning(self):
        return self._history_aligned_warning

    def get_series(self, group, name):
        out = []
        for snapshot in self.snapshots:
            if group == "objs":
                out.append(_flatten_value(snapshot.objs[name]))
            elif group == "desvars":
                out.append(_flatten_value(snapshot.desvars[name]))
            elif group == "cons":
                out.append(_flatten_value(snapshot.cons[name]))
            elif group == "scaled_objs":
                out.append(_flatten_value(snapshot.scaled_objs[name]))
            elif group == "scaled_desvars":
                out.append(_flatten_value(snapshot.scaled_desvars[name]))
            elif group == "scaled_cons":
                out.append(_flatten_value(snapshot.scaled_cons[name]))
            elif group == "opt_history":
                value = snapshot.opt_history.get(name)
                if value is None:
                    out.append(np.array([], dtype=float))
                else:
                    out.append(_flatten_value(value))
            else:
                raise KeyError(group)
        return out
