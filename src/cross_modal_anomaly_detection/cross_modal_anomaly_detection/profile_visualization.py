#!/usr/bin/env python3
"""Display-only diagnostic helpers for support-filtered RGB–LiDAR curves.

These functions must never influence NCC, PEAK DIFF, dropout or voting.
"""
import cv2
import numpy as np


def normalize_using_reference(full_values, accepted_values):
    """Normalize full signal with scale of metric-accepted observations.

    This keeps dotted excluded traces on the SAME vertical scale as the
    solid metric curves; never lets excluded returns distort normalization.
    """
    full = np.asarray(full_values, dtype=np.float64)
    accepted = np.asarray(accepted_values, dtype=np.float64)
    out = np.full(full.shape, np.nan, dtype=np.float64)
    fit = accepted[np.isfinite(accepted)]
    if fit.size < 3:
        return out
    median = float(np.median(fit))
    scale = float(np.percentile(fit, 75)-np.percentile(fit, 25))
    if scale < 1e-8:
        scale = float(np.std(fit))
    if scale < 1e-8:
        return out
    good = np.isfinite(full)
    out[good] = (full[good]-median)/scale
    return out


def draw_dotted_region(canvas, xs, values, excluded, map_x, map_y, color,
                       *, spacing_px=8, radius_px=2):
    """Draw dotted runs only where excluded; never bridge actual NaN gaps.

    A segment can include one adjacent accepted bin for continuity, but
    drawing never modifies data or the anomaly decision. Dot spacing is
    measured in rendered image pixels, not input grid bins.
    """
    xs = np.asarray(xs, dtype=np.float64)
    yy = np.asarray(values, dtype=np.float64)
    excluded = np.asarray(excluded, dtype=bool)
    if len(xs) != len(yy) or len(xs) != len(excluded):
        raise ValueError('plot arrays must have equal lengths')
    if xs.size < 2:
        return
    good = np.isfinite(xs) & np.isfinite(yy)
    dotted = good & excluded
    transitions = np.diff(np.r_[False, dotted, False].astype(np.int8))
    for lo, hi in zip(np.flatnonzero(transitions == 1),
                      np.flatnonzero(transitions == -1)):
        # Include adjacent endpoints when the full curve is valid. Do NOT
        # extend through a real missing/invalid observation.
        start = lo-1 if lo > 0 and good[lo-1] else lo
        stop = hi+1 if hi < len(xs) and good[hi] else hi
        if stop-start == 1:
            cv2.circle(canvas, (map_x(xs[lo]), map_y(yy[lo])),
                       radius_px, color, -1, cv2.LINE_AA)
            continue
        points = np.asarray(
            [(map_x(x), map_y(y)) for x,y in zip(xs[start:stop], yy[start:stop])],
            dtype=np.float64)
        lengths = np.linalg.norm(np.diff(points,axis=0),axis=1)
        distance = float(np.sum(lengths))
        if distance < 1e-8:
            cv2.circle(canvas, tuple(np.rint(points[0]).astype(int)),
                       radius_px, color, -1, cv2.LINE_AA)
            continue
        arc = np.r_[0.0,np.cumsum(lengths)]
        for t in np.arange(0.0,distance+0.01,max(2,int(spacing_px))):
            k = min(max(0,int(np.searchsorted(arc,t,side='right')-1)),len(lengths)-1)
            frac = float(np.clip((t-arc[k])/max(lengths[k],1e-8),0.0,1.0))
            p = (1-frac)*points[k]+frac*points[k+1]
            cv2.circle(canvas, tuple(np.rint(p).astype(int)),
                       radius_px, color, -1, cv2.LINE_AA)
