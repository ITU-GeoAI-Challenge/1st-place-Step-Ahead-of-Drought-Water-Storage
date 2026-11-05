"""One-line CodeCarbon instrumentation, so measuring is cheaper than not measuring.

The organisers were explicit (chat #34403): the sustainability criterion wants a measurement
from a tool like CodeCarbon, and "cannot be applied afterwards -- instrument your training runs
if you intend to compete for a top-10 placing". Three scripts were already instrumented by hand
and three were not, which is how the report ended up quoting 220 g measured plus ~450 g
estimated. A shared context manager removes the reason for that gap.

    from tws.carbon import track
    with track("tws-mlp-bag") as t:
        model.fit(X, y)
    print(t.grams)          # None if the tracker could not start -- never fatal

Every run appends to research/emissions.csv, which is the single ledger the report reads.
GPU energy is captured automatically when one is present, which matters here: the torch work
runs on an RTX 3070 and the LightGBM work does not use a GPU at all, so the two workloads have
very different power profiles and must not be extrapolated from each other.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from codecarbon import EmissionsTracker

LEDGER_DIR = "research"


class _Result:
    __slots__ = ("kg", "seconds")

    def __init__(self):
        self.kg = None
        self.seconds = None

    @property
    def grams(self):
        return None if self.kg is None else self.kg * 1000.0

    def __repr__(self):
        if self.kg is None:
            return "<carbon: not measured>"
        return f"<carbon: {self.grams:.2f} gCO2e over {self.seconds:.0f}s>"


@contextmanager
def track(project_name: str, output_dir: str = LEDGER_DIR):
    """Measure the enclosed block. Never raises: a tracker that cannot start yields None."""
    res = _Result()
    tracker = None
    try:
        tracker = EmissionsTracker(project_name=project_name, log_level="error",
                                   output_dir=output_dir, save_to_file=True)
        tracker.start()
    except Exception as exc:                      # for example, no permission to read RAPL
        print(f"  [carbon] not tracking ({type(exc).__name__}: {exc})")
        tracker = None
    t0 = time.time()
    try:
        yield res
    finally:
        res.seconds = time.time() - t0
        if tracker is not None:
            try:
                res.kg = tracker.stop()
            except Exception as exc:
                print(f"  [carbon] stop failed ({type(exc).__name__})")
        if res.kg is not None:
            print(f"  [carbon] {project_name}: {res.grams:.2f} gCO2e "
                  f"over {res.seconds:.0f}s")
