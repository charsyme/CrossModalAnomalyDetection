#!/usr/bin/env python3
"""Shared evidence fusion for RGB–LiDAR depth-profile anomaly monitoring.

Calibration: begin with healthy sensor streams, collect `coverage_baseline_frames`
valid frames, then freeze the regional coverage reference. During the calibrated
phase no dropout can silently be treated as a valid normal comparison.

This detector is for controlled fault-injection research, NOT navigation safety.
"""
from collections import deque
import numpy as np


def projected_lidar_mask(points, width, min_depth=0.0, max_depth=float('inf')):
    """One boolean per CAMERA pixel, not per surviving (shrunk) LiDAR support.

    The frame-wide fixed denominator is essential for detecting full-sector or
    complete projected-LiDAR dropouts.
    """
    width=int(width)
    mask=np.zeros(max(width,0),dtype=bool)
    if width == 0:
        return mask
    for point in (() if points is None else points):
        try:
            u=float(point['depth_u_float'])
            z=float(point['camera_depth'])
        except (TypeError, KeyError, ValueError):
            continue
        if np.isfinite(u) and 0 <= u < width and np.isfinite(z) and min_depth < z <= max_depth:
            mask[int(np.floor(u))]=True
    return mask


def fixed_band_depth_mask(depth, *, row_fraction=0.55, half_height_px=12,
                          min_valid_per_column=1, min_depth=0.4, max_depth=15.0):
    """Independent fixed-width availability check; never alters source depth."""
    a=np.asarray(depth)
    if a.ndim != 2:
        raise ValueError('expected a 2-D depth image')
    h,w=a.shape
    cy=int(round((h-1)*row_fraction))
    y0=max(0,cy-int(half_height_px))
    y1=min(h,cy+int(half_height_px)+1)
    block=a[y0:y1,:]
    ok=np.isfinite(block) & (block>=min_depth) & (block<=max_depth)
    return np.sum(ok,axis=0)>=max(1,int(min_valid_per_column))


def fixed_band_estimate_mask(estimated, *, row_fraction=0.55, half_height_px=12,
                             min_valid_per_column=1):
    a=np.asarray(estimated)
    if a.ndim != 2:
        raise ValueError('expected a 2-D estimated depth image')
    h,w=a.shape
    cy=int(round((h-1)*row_fraction))
    block=a[max(0,cy-int(half_height_px)):min(h,cy+int(half_height_px)+1),:]
    good=np.isfinite(block)
    if np.issubdtype(a.dtype,np.integer):
        good &= block>0
    return np.sum(good,axis=0)>=max(1,int(min_valid_per_column))


class CoverageBaseline:
    """Frozen, sensor-specific per-region baseline; never learns from later faults.

    Caution: a sparse LiDAR has variable geometric support; relative drop
    thresholds need calibration on healthy moving and stationary sequences.
    """
    def __init__(self, name, *, frames=20, regions=5, global_ratio=0.55,
                 region_ratio=0.45, min_reference_global=0.05,
                 min_reference_region=0.12, required_regions=1):
        self.name=str(name)
        self.frames=max(3,int(frames))
        self.regions=max(1,int(regions))
        self.global_ratio=float(global_ratio)
        self.region_ratio=float(region_ratio)
        self.min_reference_global=float(min_reference_global)
        self.min_reference_region=float(min_reference_region)
        self.required_regions=max(1,int(required_regions))
        self.samples=[]
        self.reference=None

    @property
    def ready(self):
        return self.reference is not None

    def observe(self, mask, *, learn=True):
        a=np.asarray(mask,dtype=bool).reshape(-1)
        if not a.size:
            return dict(name=self.name,valid=False,bad=False,reason='empty_grid',
                        observed_fraction=None,reference_fraction=None,
                        ratio=None,bad_regions=[],calibration_frames=len(self.samples),
                        required_calibration_frames=self.frames)
        windows=np.array_split(a,self.regions)
        fractions=np.array([np.mean(a)] +
                           [np.mean(w) if w.size else 0.0 for w in windows],dtype=float)
        if not self.ready and learn and fractions[0] >= self.min_reference_global:
            self.samples.append(fractions)
            if len(self.samples)>=self.frames:
                self.reference=np.median(np.asarray(self.samples),axis=0)
                self.samples=[]
        if not self.ready:
            return dict(name=self.name,valid=False,bad=False,reason='calibrating',
                        observed_fraction=float(fractions[0]),reference_fraction=None,
                        ratio=None,bad_regions=[],calibration_frames=len(self.samples),
                        required_calibration_frames=self.frames)
        reference=self.reference
        global_bad=(reference[0]>=self.min_reference_global and
                    fractions[0] < self.global_ratio*reference[0])
        bad_regions=[k+1 for k in range(self.regions)
                     if reference[k+1]>=self.min_reference_region and
                     fractions[k+1]<self.region_ratio*reference[k+1]]
        return dict(name=self.name,valid=True,
                    bad=bool(global_bad or len(bad_regions)>=self.required_regions),
                    global_bad=bool(global_bad),bad_regions=bad_regions,
                    reason='ok',observed_fraction=float(fractions[0]),
                    reference_fraction=float(reference[0]),
                    ratio=float(fractions[0]/max(reference[0],1e-12)),
                    global_threshold_ratio=float(self.global_ratio),
                    region_threshold_ratio=float(self.region_ratio),
                    region_fraction=[float(z) for z in fractions[1:]],
                    reference_regions=[float(z) for z in reference[1:]],
                    calibration_frames=self.frames,
                    required_calibration_frames=self.frames)


def merge_evidence(result, coverage, votes, *, required_bad_frames,
                   shape_key='check_valid', metric=None, extra_bad=False):
    """Combine independent evidence, even when shape Pearson is undefined."""
    shape_valid=bool(result.get(shape_key,False))
    shape_bad=shape_valid and bool(result.get('global_bad',False))
    available_coverage=[e for e in coverage.values() if e.get('valid',False)]
    coverage_bad=any(e.get('bad',False) for e in available_coverage)
    metric=metric or {}
    metric_valid=bool(metric.get('metric_valid',False))
    metric_bad=metric_valid and bool(metric.get('metric_bad',False))
    evidence_valid=bool(shape_valid or available_coverage or metric_valid or extra_bad)
    frame_bad=bool(shape_bad or coverage_bad or metric_bad or extra_bad)
    if evidence_valid:
        votes.append(frame_bad)
    else:
        votes.clear()
    num_bad=int(sum(votes))
    anomaly=bool(evidence_valid and len(votes)>=required_bad_frames and
                 num_bad>=required_bad_frames)
    result.update({
        shape_key:evidence_valid, 'shape_valid':shape_valid,
        'shape_bad':bool(shape_bad), 'coverage_bad':bool(coverage_bad),
        'coverage_evidence':coverage, 'metric_valid':metric_valid,
        'metric_bad':bool(metric_bad), 'frame_bad':frame_bad,
        'anomaly':anomaly, 'bad_frames_in_window':num_bad,
        'history_length':len(votes),
        'decision_metric':'ncc_or_coverage_or_metric',
        'reason':('evidence_detected' if frame_bad and not shape_valid
                  else result.get('reason','ok')),
    })
    return result
