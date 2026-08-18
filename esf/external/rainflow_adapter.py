"""Rainflow cycle counting, on top of the MIT-licensed ``rainflow`` package.

The counting itself is ASTM E1049-85 three-point rainflow, provided by
`rainflow <https://pypi.org/project/rainflow/>`_ (MIT, Piotr Janiszewski).
This module is the thin adapter between that package and
:class:`esf.models.cycle_counting_algorithm.CycleCounter`, which works with
turning points already extracted by the peak detector and wants the result as
a single array.

Earlier versions of this project vendored a GPL-3.0 rainflow implementation
here. It was replaced by the MIT-licensed package so that the whole project can
be distributed under the MIT license; the two agree exactly (see
``tests/test_cycle_counting.py::test_rainflow_matches_astm_reference``).
"""

import numpy as np
import rainflow as _rainflow

# Row layout of the array returned by :func:`rainflow`.
ROW_RANGE = 0
ROW_MEAN = 1
ROW_COUNT = 2
N_ROWS = 3


def rainflow(array_ext):
    """Count rainflow cycles in a sequence of turning points.

    Args:
        array_ext (numpy.ndarray): 1-D array of turning points (alternating
            maxima and minima), as produced by the peak detector.

    Returns:
        numpy.ndarray: a ``(3, n_cycle)`` array, one column per counted cycle:

        0. cycle range (peak-to-peak; for SoC profiles this is the DoD)
        1. cycle mean
        2. cycle count -- ``1.0`` for a full cycle, ``0.5`` for a half cycle

        Cycles come out in the order the algorithm closes them, which is what
        ``CycleCounter`` relies on to map each cycle back onto the time axis.

    A sequence shorter than two points yields no cycles. Exactly two points are
    a single half cycle -- handled here because the upstream package's
    ``reversals()`` needs three points before it reports anything.
    """
    array_ext = np.asarray(array_ext, dtype=float).ravel()

    if array_ext.size < 2:
        return np.zeros((N_ROWS, 0))

    if array_ext.size == 2:
        first, last = array_ext
        lrange = abs(last - first)
        if lrange == 0:
            return np.zeros((N_ROWS, 0))
        return np.array([[lrange], [(first + last) / 2.0], [0.5]])

    cycles = [
        (rng, mean, count)
        for rng, mean, count, _i_start, _i_end in _rainflow.extract_cycles(array_ext)
        # zero-range cycles carry no degradation and were dropped by the
        # previous implementation too
        if rng > 0
    ]

    if not cycles:
        return np.zeros((N_ROWS, 0))

    return np.array(cycles, dtype=float).T
