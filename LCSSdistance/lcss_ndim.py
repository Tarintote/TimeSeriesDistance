# -*- coding: UTF-8 -*-
"""
dtaidistance.lcss_ndim
~~~~~~~~~~~~~~~~~~~~~

Dynamic Time Warping (lcss) for N-dimensional series.

:author: Wannes Meert
:copyright: Copyright 2017-2018 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import os
import logging
import math
import numpy as np

logger = logging.getLogger("be.kuleuven.dtai.distance")
dtaidistance_dir = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), os.pardir)

try:
    from . import lcss_c
except ImportError:
    # logger.info('C library not available')
    lcss_c = None

try:
    from tqdm import tqdm
except ImportError:
    logger.info('tqdm library not available')
    tqdm = None


def distance(s1, s2, epsilon=None, window=None, max_dist=None,
             max_step=None, max_length_diff=None, penalty=None, psi=None,
             use_c=False):
    """Dynamic Time Warping using multidimensional sequences.

    cost = EuclideanDistance(s1[i], s2[j])

    See :py:meth:`dtaidistance.lcss.distance` for parameters.
    """
    if use_c:
        logger.error("No C version implemented (yet)")
        return
    r, c = len(s1), len(s2)
    if max_length_diff is not None and abs(r - c) > max_length_diff:
        return np.inf
    if epsilon is None:
        epsilon = 0.0
    if window is None:
        window = max(r, c)
    if not max_step:
        max_step = np.inf
    else:
        max_step *= max_step
    if not max_dist:
        max_dist = np.inf
    else:
        max_dist *= max_dist
    if not penalty:
        penalty = 0
    else:
        penalty *= penalty
    if psi is None:
        psi = 0
    length = min(c + 1, abs(r - c) + 2 * (window - 1) + 1 + 1 + 1)
    # print("length (py) = {}".format(length))
    lcss = np.full((2, length), np.inf)
    # lcss[0, 0] = 0
    for i in range(psi + 1):
        lcss[0, i] = 0
    last_under_max_dist = 0
    skip = 0
    i0 = 1
    i1 = 0
    psi_shortest = np.inf
    for i in range(r):
        # print("i={}".format(i))
        # print(lcss)
        if last_under_max_dist == -1:
            prev_last_under_max_dist = np.inf
        else:
            prev_last_under_max_dist = last_under_max_dist
        last_under_max_dist = -1
        skipp = skip
        skip = max(0, i - max(0, r - c) - window + 1)
        i0 = 1 - i0
        i1 = 1 - i1
        lcss[i1, :] = np.inf
        j_start = max(0, i - max(0, r - c) - window + 1)
        j_end = min(c, i + max(0, c - r) + window)
        if lcss.shape[1] == c + 1:
            skip = 0
        if psi != 0 and j_start == 0 and i < psi:
            lcss[i1, 0] = 0
        for j in range(j_start, j_end):
            d = np.sum((s1[i] - s2[j]) ** 2)
            if d > max_step:
                continue
            assert j + 1 - skip >= 0
            assert j - skipp >= 0
            assert j + 1 - skipp >= 0
            assert j - skip >= 0
            lcss[i1, j + 1 - skip] = d + min(lcss[i0, j - skipp],
                                             lcss[i0, j + 1 - skipp] + penalty,
                                             lcss[i1, j - skip] + penalty)
            # print('({},{}), ({},{}), ({},{})'.format(i0, j - skipp, i0, j + 1 - skipp, i1, j - skip))
            # print('{}, {}, {}'.format(lcss[i0, j - skipp], lcss[i0, j + 1 - skipp], lcss[i1, j - skip]))
            # print('i={}, j={}, d={}, skip={}, skipp={}'.format(i,j,d,skip,skipp))
            # print(lcss)
            if lcss[i1, j + 1 - skip] <= max_dist:
                last_under_max_dist = j
            else:
                # print('above max_dist', lcss[i1, j + 1 - skip], i1, j + 1 - skip)
                lcss[i1, j + 1 - skip] = np.inf
                if prev_last_under_max_dist + 1 - skipp < j + 1 - skip:
                    # print("break")
                    break
        if last_under_max_dist == -1:
            # print('early stop')
            # print(lcss)
            return np.inf
        if psi != 0 and j_end == len(s2) and len(s1) - 1 - i <= psi:
            psi_shortest = min(psi_shortest, lcss[i1, length - 1])
    if psi == 0:
        d = math.sqrt(lcss[i1, min(c, c + window - 1) - skip])
    else:
        ic = min(c, c + window - 1) - skip
        vc = lcss[i1, ic - psi:ic + 1]
        d = min(np.min(vc), psi_shortest)
        d = math.sqrt(d)
    return d


def _distance_with_params(t):
    return distance(t[0], t[1], **t[2])


def warping_paths(s1, s2, window=None, max_dist=None,
                  max_step=None, max_length_diff=None, penalty=None, psi=None,):
    """
    Dynamic Time Warping (keep full matrix) using multidimensional sequences.

    cost = EuclideanDistance(s1[i], s2[j])

    See :py:meth:`dtaidistance.lcss.warping_paths` for parameters.
    """
    r, c = len(s1), len(s2)
    if max_length_diff is not None and abs(r - c) > max_length_diff:
        return np.inf
    if window is None:
        window = max(r, c)
    if not max_step:
        max_step = np.inf
    else:
        max_step *= max_step
    if not max_dist:
        max_dist = np.inf
    else:
        max_dist *= max_dist
    if not penalty:
        penalty = 0
    else:
        penalty *= penalty
    if psi is None:
        psi = 0
    lcss = np.full((r + 1, c + 1), np.inf)
    # lcss[0, 0] = 0
    for i in range(psi + 1):
        lcss[0, i] = 0
        lcss[i, 0] = 0
    last_under_max_dist = 0
    i0 = 1
    i1 = 0
    for i in range(r):
        if last_under_max_dist == -1:
            prev_last_under_max_dist = np.inf
        else:
            prev_last_under_max_dist = last_under_max_dist
        last_under_max_dist = -1
        i0 = i
        i1 = i + 1
        # print('i =', i, 'skip =',skip, 'skipp =', skipp)
        # jmin = max(0, i - max(0, r - c) - window + 1)
        # jmax = min(c, i + max(0, c - r) + window)
        # print(i,jmin,jmax)
        # x = lcss[i, jmin-skipp:jmax-skipp]
        # y = lcss[i, jmin+1-skipp:jmax+1-skipp]
        # print(x,y,lcss[i+1, jmin+1-skip:jmax+1-skip])
        # lcss[i+1, jmin+1-skip:jmax+1-skip] = np.minimum(x,
        #                                                y)
        for j in range(max(0, i - max(0, r - c) - window + 1), min(c, i + max(0, c - r) + window)):
            # print('j =', j, 'max=',min(c, c - r + i + window))
            d = np.sum((s1[i] - s2[j]) ** 2)
            if max_step is not None and d > max_step:
                continue
            # print(i, j + 1 - skip, j - skipp, j + 1 - skipp, j - skip)
            lcss[i1, j + 1] = d + min(lcss[i0, j],
                                      lcss[i0, j + 1] + penalty,
                                      lcss[i1, j] + penalty)
            # lcss[i + 1, j + 1 - skip] = d + min(lcss[i + 1, j + 1 - skip], lcss[i + 1, j - skip])
            if max_dist is not None:
                if lcss[i1, j + 1] <= max_dist:
                    last_under_max_dist = j
                else:
                    lcss[i1, j + 1] = np.inf
                    if prev_last_under_max_dist < j + 1:
                        break
        if max_dist is not None and last_under_max_dist == -1:
            # print('early stop')
            # print(lcss)
            return np.inf, lcss
    lcss = np.sqrt(lcss)
    if psi == 0:
        d = lcss[i1, min(c, c + window - 1)]
    else:
        ir = i1
        ic = min(c, c + window - 1)
        vr = lcss[ir-psi:ir+1, ic]
        vc = lcss[ir, ic-psi:ic+1]
        mir = np.argmin(vr)
        mic = np.argmin(vc)
        if vr[mir] < vc[mic]:
            lcss[ir-psi+mir+1:ir+1, ic] = -1
            d = vr[mir]
        else:
            lcss[ir, ic - psi + mic + 1:ic+1] = -1
            d = vc[mic]
    return d, lcss


def distance_matrix(s, epsilon=None, max_dist=None, max_length_diff=None,
                    window=None, max_step=None, penalty=None, psi=None,
                    block=None, parallel=False,
                    use_c=False, show_progress=False):
    """Dynamic Time Warping distance matrix using multidimensional sequences.

    cost = EuclideanDistance(s1[i], s2[j])

    See :py:meth:`dtaidistance.lcss.distance_matrix` for parameters.
    """
    if parallel and not use_c:
        try:
            import multiprocessing as mp
            logger.info('Using multiprocessing')
        except ImportError:
            parallel = False
            mp = None
    else:
        mp = None
    dist_opts = {
        'epsilon': epsilon,
        'max_dist': max_dist,
        'max_step': max_step,
        'window': window,
        'max_length_diff': max_length_diff,
        'penalty': penalty,
        'psi': psi
    }
    dists = None
    if max_length_diff is None:
        max_length_diff = np.inf
    large_value = np.inf
    logger.info('Computing distances')
    if use_c:
        logger.error("No C version available (yet)")
    if not use_c:
        logger.info("Compute distances in Python")
        if isinstance(s, np.matrix):
            ss = [np.asarray(s[i]).reshape(-1) for i in range(s.shape[0])]
            s = ss
        if parallel:
            logger.info("Use parallel computation")
            dists = np.zeros((len(s), len(s))) + large_value
            if block is None:
                idxs = np.triu_indices(len(s), k=1)
            else:
                idxsl_r = []
                idxsl_c = []
                for r in range(block[0][0], block[0][1]):
                    for c in range(max(r + 1, block[1][0]), min(len(s), block[1][1])):
                        idxsl_r.append(r)
                        idxsl_c.append(c)
                idxs = (np.array(idxsl_r), np.array(idxsl_c))
            with mp.Pool() as p:
                dists[idxs] = p.map(_distance_with_params, [
                                    (s[r], s[c], dist_opts) for c, r in zip(*idxs)])
                # pbar = tqdm(total=int((len(s)*(len(s)-1)/2)))
                # for r in range(len(s)):
                #     dists[r,r+1:len(s)] = p.map(distance, [(s[r],s[c], dist_opts) for c in range(r+1,len(cur))])
                #     pbar.update(len(s) - r - 1)
                # pbar.close()
        else:
            logger.info("Use serial computation")
            dists = np.zeros((len(s), len(s))) + large_value
            if block is None:
                it_r = range(len(s))
            else:
                it_r = range(block[0][0], block[0][1])
            if show_progress:
                it_r = tqdm(it_r)
            for r in it_r:
                if block is None:
                    it_c = range(r + 1, len(s))
                else:
                    it_c = range(
                        max(r + 1, block[1][0]), min(len(s), block[1][1]))
                for c in it_c:
                    if abs(len(s[r]) - len(s[c])) <= max_length_diff:
                        dists[r, c] = distance(s[r], s[c], **dist_opts)
    return dists
