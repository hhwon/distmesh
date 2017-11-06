# -*- coding: utf-8 -*-
# __author__ = 'huang_wang'

import numpy as np


def segment_dist(points, pv):
    n = len(pv) - 1
    d = np.zeros((len(points), n))
    j = 0
    for p in points:
        for i in range(n):
            a = pv[i]
            b = pv[i+1]
            ab = b - a
            ap = p - a
            t = sum(ab*ap)/sum(ab*ab)
            if t < 0:
                d[j][i] = np.sqrt(sum(ap*ap))
            elif t > 1:
                bp = p - b
                d[j][i] = np.sqrt(sum(bp*bp))
            else:
                ad = t*ab
                d[j][i] = np.sqrt(sum((ad-ap)*(ad-ap)))
        j += 1
    return d


# def polygon(point, pv):
#     count = 0
#     s = 2
#     n = len(pv) - 1
#     dd = np.zeros((n, 1))
#     d = np.zeros((len(point), 1))
#     j = 0
#     for p in point:
#         for i in range(n):
#             dd[i] = point_to_segment_dist(p, pv[i], pv[i+1])
#             if abs((pv[i, 0] - p[0]) * (pv[i + 1, 1] - p[1]) - (pv[i + 1, 0] - p[0]) * (
#                 pv[i, 1] - p[1])) <= sys.float_info.epsilon * 100 and min(pv[i:i + 2, 0]) <= p[0] <= max(
#                     pv[i:i + 2, 0]) and min(pv[i:i + 2, 1]) <= p[1] <= max(pv[i:i + 2, 1]):
#                 s = 0
#                 break
#             elif pv[i, 1] != pv[i+1, 1] and max(pv[i:i+2, 0]) <= p[0]:
#                 if any([max(pv[i:i+2, 1]) == p[1]]):
#                     count += 1
#                 elif min(pv[i:i+2, 1]) < p[1] < max(pv[i:i+2, 1]):
#                     count += 1
#         d[j] = min(dd.flat)
#         if count % 2 == 1:
#             d[j] = -d[j]
#         j += 1
#     return d.ravel()


def poly(p, pv):
    from matplotlib.path import Path
    return (-1)**Path(pv).contains_points(p)*segment_dist(p, pv).min(1)
