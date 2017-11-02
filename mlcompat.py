"""
MATLAB compatibility methods

dense           : Similar to full(sparse(I, J, S, ...))
interp2_linear  : Similar to interp2(..., 'linear')
interp3_linear  : Similar to interp3(..., 'linear')
unique_rows     : Similar to unique(..., 'rows')
setdiff_rows    : Similar to setdiff(..., 'rows')
"""

__all__ = [
    'dense',
    'interp2_linear',
    'interp3_linear',
    'setdiff_rows',
    'unique_rows'
]

import numpy as np
import scipy.sparse as sparse
# import scipy.interpolate as interpolate


def dense(i, j, s, shape=None, dtype=None):
    if np.isscalar(j):
        x = j
        j = np.empty(i.shape, dtype=int)
        j.fill(x)
    if np.isscalar(s):
        x = s
        s = np.empty(s.shape)
        s.fill(x)

    # Turn these into 1-d arrays for processing.
    s = s.flat
    i = i.flat
    j = j.flat
    return sparse.coo_matrix((s, (i, j)), shape, dtype).toarray()


def setdiff_row(a1, a2):
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
    a = np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])
    return a


def simpvol(p, t):
    """Signed volumes of the simplex elements in the mesh."""
    dim = p.shape[1]
    if dim == 1:
        d01 = p[t[:, 1]] - p[t[:, 0]]
        return d01
    elif dim == 2:
        d01 = p[t[:, 1]] - p[t[:, 0]]
        d02 = p[t[:, 2]] - p[t[:, 0]]
        return (d01[:, 0] * d02[:, 1] - d01[:, 1] * d02[:, 0]) / 2
    else:
        raise NotImplementedError
