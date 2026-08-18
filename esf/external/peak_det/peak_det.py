import sys

import numpy as np


def peakdet(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit("Input vectors v and x must have same length")

    if not np.isscalar(delta):
        sys.exit("Input argument delta must be a scalar")

    if delta <= 0:
        sys.exit("Input argument delta must be positive")

    mn, mx = np.inf, -np.inf
    mnpos, mxpos = np.nan, np.nan

    lookformax = True
    len_v = len(v) - 1

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if i == 0 and len(v) > 1:
            if this < v[i + 1]:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
            elif this > v[i + 1]:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False

        if lookformax:
            if (this < mx - delta) or (i == len_v):
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if (this > mn + delta) or (i == len_v):
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


if __name__ == "__main__":
    import numpy as np
    from matplotlib.pyplot import plot, scatter, show

    series = [0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 2, 0, 0, 0, -2, 0]

    maxtab, mintab = peakdet(series, 0.001)
    plot(series)
    scatter(np.array(maxtab)[:, 0], np.array(maxtab)[:, 1], color="blue")
    scatter(np.array(mintab)[:, 0], np.array(mintab)[:, 1], color="red")

    show()
