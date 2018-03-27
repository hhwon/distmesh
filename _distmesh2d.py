# -*- coding: utf-8 -*-
"""DishMesh 2D"""

# -----------------------------------------------------------------------------
#  Copyright (C) 2004-2012 Per-Olof Persson
#  Copyright (C) 2012 Bradley Froehle

#  Distributed under the terms of the GNU General Public License. You should
#  have received a copy of the license along with this program. If not,
#  see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
import scipy.spatial as spatial

# Local imports
import mlcompat as ml
import matplotlib.pyplot as plt


def distmesh2d(fd, fh, h0, bbox, pfix=None):
    """
    distmesh2d: 2-D Mesh Generator using Distance Functions.

    Usage
    -----
    >>> p = distmesh2d(fd, fh, h0, bbox, pfix)

    Parameters
    ----------
    :param fd: Distance function d(x, y)
    :param fh:  Scaled edge length function h(x, y)
    :param h0:  Initial edge length
    :param bbox:  Bounding box, (xmin, ymin, xmax, ymax)
    :param pfix:  Fixed node positions, shape (nfix, 2)
    :return: p: Node positions (N×2)
    """

    # Constants Defined
    dptol = .001
    ttol = .1
    fscale = 1.2
    deltat = .2
    geps = .001 * h0

    deps = np.sqrt(np.finfo(np.double).eps) * h0
    densityctrlfreq = 30

    # Extract bounding box
    xmin, ymin, xmax, ymax = bbox
    if pfix is not None:
        pfix = np.array(pfix, dtype='d')

    # 1.Create initial distribution in bounding box (equilateral triangles)
    x, y = np.mgrid[xmin:(xmax + h0):h0, ymin:(ymax + h0 * np.sqrt(3) / 2):h0 * np.sqrt(3) / 2]
    x[:, 1::2] += h0 / 2  # Shift even rows
    p = np.vstack((x.flat, y.flat)).T  # List of node coordinates

    # 2.Remove points outside the region, apply the rejection method
    p = p[fd(p) < geps]  # Keep only d<0 points
    r0 = 1 / fh(p) ** 2  # Probability to keep point
    index = np.random.random((p.shape[0], 1)) < r0 / r0.max()
    p = p[index.ravel(), :]
    if pfix is not None:
        p = ml.setdiff_row(p, pfix)  # Remove duplicated nodes
        pfix = np.unique(pfix, axis=0)
        nfix = len(pfix)
        p = np.vstack((pfix, p))  # Prepend fix points
    else:
        nfix = 0
    n = len(p)  # Number of points n

    count = 0
    pold = float('inf')  # For first iteration

    while True:
        count += 1

        # 3. Retriangulation by the Delaunay algorithm
        distance = lambda p1, p2: np.sqrt(((p1 - p2) ** 2).sum(1))
        if (distance(p, pold) / h0).max() > ttol:  # Any large movement?
            pold = p.copy()  # Save current position
            t = spatial.Delaunay(p).simplices  # List of triangles
            pmid = p[t].sum(1) / 3  # Compute centroids
            t = t[fd(pmid) < -geps]  # Keep interior triangles
            # 4. Describe each bar by a unique pair of nodes
            bars = np.vstack((t[:, [0, 1]],
                              t[:, [1, 2]],
                              t[:, [2, 0]]))  # Interior bars duplicated
            bars.sort(axis=1)
            bars = np.unique(bars, axis=0)

            # 5.Graphical output of the current mesh

        # 6. Move mesh points based on bar lengths L and forces F
        barvec = p[bars[:, 0]] - p[bars[:, 1]]  # List of bar vectors
        L = np.sqrt((barvec * barvec).sum(1))  # L = Bar lengths
        L = np.transpose([L])
        hbars = fh(p[bars].sum(1) / 2)
        L0 = (hbars * fscale *
              np.sqrt((L ** 2).sum() / (hbars ** 2).sum()))  # L0 = Desired lengths

        # Density control - remove points that are too close
        if (count % densityctrlfreq) == 0 and (L0 > 2 * L).any():
            ixdel = np.setdiff1d(bars[(L0 > 2 * L).ravel()].reshape(-1), np.arange(nfix))
            p = p[np.setdiff1d(np.arange(n), ixdel)]
            n = len(p)
            pold = float('inf')
            continue

        F = L0 - L
        F[F < 0] = 0  # Bar forces (scalars)

        Fvec = (F / L).dot([[1, 1]]) * barvec  # Bar forces (x, y components)
        Ftot = ml.dense(bars[:, [0, 0, 1, 1]],
                        np.repeat([[0, 1, 0, 1]],
                                  len(F), axis=0),
                        np.hstack((Fvec, -Fvec)), shape=(n, 2))
        Ftot[:nfix] = 0  # force = 0 at fixed points
        p += deltat * Ftot  # Update node positions

        # 7. Bring outside points back to the boundary
        d = fd(p)  # Find points ouside (d>0)
        ix = d > 0
        if ix.any():
            dgradx = (fd(p[ix] + [deps, 0]) - d[ix]) / deps  # Numerical
            dgrady = (fd(p[ix] + [0, deps]) - d[ix]) / deps  # gradient
            dgrad2 = dgradx ** 2 + dgrady ** 2
            p[ix] -= (d[ix] * np.vstack((dgradx, dgrady)) / dgrad2).T  # Project

        # 8. Termination criterion: All interior nodes move less than dptol (scaled)
        if (np.sqrt((deltat * Ftot[d < -geps] ** 2).sum(1)) / h0).max() < dptol:
            break

        if abs(count-1000) <= 0.00001:
            break
    return p


def dcircle(p, xc, yc, r):
    """Signed distance to circle centered at xc, yc with radius r."""
    return np.sqrt(((p - np.array([xc, yc])) ** 2).sum(-1)) - r


def ddiff(d1, d2):
    """Signed distance to set difference between two regions described by
    signed distance functions d1 and d2.
    Not exact the true signed distance function for the difference,
    for example around corners.
    """
    return np.maximum(d1, -d2)


def dunion(d1, d2):
    """Signed stance function for the set union of two regions described by
    signed distance functions d1, d2.
    This not a true signed distance function for the union, for example around
    corners.
    """
    return np.minimum(d1, d2)


def dintersect(d1, d2):
    """Signed distance to set intersection of two regions described by signed
    distance functions d1 and d2.
    Not exact the true signed distance function for the difference,
    for example around corners.
    """
    return np.maximum(d1, d2)


def drectangle(p, x1, x2, y1, y2):
    """Signed distance function for rectangle with corners (x1,y1), (x2,y1),
    (x1,y2), (x2,y2).
    This has an incorrect distance to the four corners. See drectangle0 for a
    true distance function.
    """
    return -np.minimum(np.minimum(np.minimum(-y1 + p[:, 1], y2 - p[:, 1]), -x1 + p[:, 0]), x2 - p[:, 0])

# fd = lambda p: ddiff(drectangle(p,-1,1,-1,1), dcircle(p,0,0,0.5))
# fh = lambda p: np.ones((len(p), 1))
# p = distmesh2d(fd, fh, 0.05, (-1,-1,1,1),
#                           [(-1,-1), (-1,1), (1,-1), (1,1)])
# plt.plot(p[:, 0], p[:, 1], '.')
# plt.axis('equal')
# plt.show()


import dist
huniform = lambda p: np.ones((len(p), 1))
# pv = np.array([[0, 0], [0, 5], [5, 5], [5, 0]])
fd = lambda p: dunion(drectangle(p,0,5,0,5), dcircle(p, 5, 2.5, 1))
# fd = lambda p: np.minimum(np.sqrt((p**2).sum(1))-3, dist.poly(p, pv))
p = distmesh2d(fd, huniform, 0.3, (-8, -8, 8, 8), np.array([[0, 1]]))
# # print(p)
plt.plot(p[:, 0], p[:, 1], '.')
plt.axis('equal')
plt.show()
