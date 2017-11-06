# -*- coding: utf-8 -*-
"""DishMesh 2D"""

# -----------------------------------------------------------------------------
#  Copyright (C) 2004-2012 Per-Olof Persson
#  Copyright (C) 2012 Bradley Froehle

#  Distributed under the terms of the GNU General Public License. You should
#  have received a copy of the license along with this program. If not,
#  see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

import numpy as np
import scipy.spatial as spatial
import mlcompat as ml
import dist
import matplotlib.pyplot as plt


# Local import


def distmesh2d(fd, fh, h0, bbox, pfix=None):
    dptol = .001
    ttol = .1
    Fscale = 1.2
    deltat = .2
    geps = .0001 * h0
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
        p = np.vstack((pfix, p))
    else:
        nfix = 0
    n = len(p)  # Number of points n

    count = 0
    pold = float('inf')  # For first iteration

    while True:
        count += 1

        # 3.Retriangulation by the Delaunay algorithm
        dist = lambda p1, p2: np.sqrt(((p1 - p2) ** 2).sum(1))
        if (dist(p, pold) / h0).max() > ttol:  # Any large movement?
            pold = p.copy()  # Save current position
            t = spatial.Delaunay(p).simplices  # List of triangles
            pmid = p[t].sum(1) / 3
            t = t[fd(pmid) < -geps]
            # 4. Describe each bar by a unique pair of nodes
            bars = np.vstack((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]))  # Interior bars duplicated
            bars.sort(axis=1)
            bars = np.unique(bars, axis=0)
            # 5.Graphical output of the current mesh

        # 6. Move mesh points based on bar lengths L and forces F
        barvec = p[bars[:, 0]] - p[bars[:, 1]]  # List of bar vectors
        L = np.sqrt((barvec * barvec).sum(1))
        L = np.transpose([L])
        hbars = fh(p[bars].sum(1) / 2)
        L0 = (hbars * Fscale * np.sqrt((L ** 2).sum() / (hbars ** 2).sum()))  # L0 = Desired lengths

        # Density control - remove points that are too close
        if (count % densityctrlfreq) == 0 and (L0 > 2 * L).any():
            ixdel = np.setdiff1d(bars[(L0 > 2 * L).ravel()].reshape(-1), np.arange(nfix))
            p = p[np.setdiff1d(np.arange(n), ixdel)]
            n = len(p)
            pold = float('inf')
            continue

        F = L0 - L
        F[F < 0] = 0  # bar force

        Fvec = (F / L).dot([[1, 1]]) * barvec  # Bar forces (x, y components)
        Ftot = ml.dense(bars[:, [0, 0, 1, 1]], np.repeat([[0, 1, 0, 1]], len(F), axis=0),
                        np.hstack((Fvec, -Fvec)), shape=(n, 2))
        Ftot[:nfix] = 0  # force = 0 at fixed points
        p += deltat * Ftot  # Update node positions

        # 7.Bring outside points back to the boundary
        d = fd(p)
        ix = d > 0
        if ix.any():
            dgradx = (fd(p[ix] + [deps, 0]) - d[ix]) / deps  # Numerical
            dgrady = (fd(p[ix] + [0, deps]) - d[ix]) / deps  # gradient
            dgrad2 = dgradx ** 2 + dgrady ** 2
            p[ix] -= (d[ix] * np.vstack((dgradx, dgrady)) / dgrad2).T

        # 8.Termination criterion: All interior nodes move less than dptol (scaled)
        if (np.sqrt((deltat * Ftot[d < -geps] ** 2).sum(1)) / h0).max() < dptol:
            break
    return p


# huniform = lambda p: np.ones((len(p), 1))
# fd = lambda p: np.sqrt((p**2).sum(1))-1.0
# p = distmesh2d(fd, huniform, 0.1, (-1, -1, 1, 1), np.array([[0, 1]]))
# # print(p)
# plt.plot(p[:, 0], p[:, 1], '.')
# plt.axis('equal')
# plt.show()


pv = np.array([[0, 0], [0, 5], [10, 5], [10, 1], [0, 0]])
fd = lambda p: dist.poly(p, pv)
huniform = lambda p: np.ones((len(p), 1))
p = distmesh2d(fd, huniform, 1, (0, 0, 10, 5), pv)
plt.plot(p[:, 0], p[:, 1], '.')
plt.axis('equal')
plt.show()

# pv = np.array([[0, 0], [0, 5], [10, 5], [0, 0]])
# fd = lambda p: dist.poly(p, pv)
# huniform = lambda p: np.ones((len(p), 1))
# p = distmesh2d(fd, huniform, 1, (0, 0, 10, 5), pv)
# plt.plot(p[:, 0], p[:, 1], '.')
# plt.axis('equal')
# plt.show()
