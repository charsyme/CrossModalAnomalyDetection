#!/usr/bin/env python3
"""Localized relative-depth peak mismatch evidence for LiDAR/RGB curves.

Unlike Pearson correlation, this detects an anomalous local peak after a
robust global affine fit.  Both signals are RELATIVE/normalized profiles;
there is no absolute depth-scale inference or full calibration here.
"""
import numpy as np


def peak_mismatch_evidence(u, lidar_profile, rgb_profile_aligned, *,
                           residual_threshold=0.50, min_bad_fraction=0.05,
                           min_run_px=8, min_points=24,
                           min_paired_coverage=0.35):
    """Check for spatially coherent residuals after fitting RGB->LiDAR.

    Inputs are already on the same IMAGE x-grid and rgb is aligned using the
    bounded shift chosen by NCC.  Missing depth remains NaN and is excluded.
    Residuals are divided by the LiDAR interquartile range, so thresholds are
    dimensionless (not metres).  A robust affine fit absorbs unknown global
    scale/offset but leaves unusual local peak-height differences visible.
    """
    output = dict(peak_valid=False, peak_bad=False,
                  peak_reason='insufficient_paired_samples',
                  peak_residual_threshold=float(residual_threshold),
                  peak_min_fraction=float(min_bad_fraction),
                  peak_min_run_px=int(min_run_px),
                  peak_outlier_fraction=None, peak_residual_p95=None,
                  peak_residual_median=None, peak_longest_run_px=0,
                  peak_paired_count=0, peak_affine_scale=None,
                  peak_affine_offset=None)
    x=np.asarray(u, dtype=np.float64)
    a=np.asarray(lidar_profile, dtype=np.float64)
    b=np.asarray(rgb_profile_aligned, dtype=np.float64)
    if x.ndim!=1 or a.shape!=x.shape or b.shape!=x.shape or len(x)<3:
        output['peak_reason']='invalid_shapes'
        return output
    paired=np.isfinite(x)&np.isfinite(a)&np.isfinite(b)
    n=int(np.count_nonzero(paired))
    output['peak_paired_count']=n
    if n<max(6,int(min_points)) or n/max(1,len(x))<min_paired_coverage:
        return output
    a1=a[paired]
    b1=b[paired]
    iqr=float(np.percentile(a1,75)-np.percentile(a1,25))
    if iqr<0.06 or float(np.std(b1))<0.06:
        output['peak_reason']='flat_paired_profile'
        return output
    design=np.column_stack((b1,np.ones_like(b1)))
    try:
        params=np.linalg.lstsq(design,a1,rcond=None)[0]
        for _ in range(6):
            err=a1-design@params
            center=float(np.median(err))
            mad=float(np.median(np.abs(err-center)))
            sigma=max(1.4826*mad,0.02*iqr)
            weights=np.minimum(1.0,1.5*sigma/np.maximum(np.abs(err-center),1e-12))
            root=np.sqrt(weights)
            params=np.linalg.lstsq(design*root[:,None],a1*root,rcond=None)[0]
    except np.linalg.LinAlgError:
        output['peak_reason']='affine_fit_failed'
        return output
    if not np.all(np.isfinite(params)):
        output['peak_reason']='nonfinite_affine_fit'
        return output
    err_abs=np.abs(a1-design@params)/iqr
    bad=(err_abs>=float(residual_threshold))
    # Compute longest consecutive bad run without joining across invalid x.
    full_flags=np.zeros(len(x),dtype=bool)
    full_flags[paired]=bad
    step=np.median(np.diff(x))
    if not np.isfinite(step) or step<=0:
        output['peak_reason']='invalid_grid'
        return output
    # Count consecutive columns in pixels; gaps and nonuniform jumps break runs.
    longest=0
    current=0
    previous_idx=None
    for i,flag in enumerate(full_flags):
        if flag:
            if previous_idx is not None and abs((x[i]-x[previous_idx])-step)>0.25*step:
                current=0
            current+=1
            longest=max(longest,current)
            previous_idx=i
        else:
            current=0
            previous_idx=None
    # u-spacing is in image pixels. A 1-pixel grid yields same units as count.
    longest_px=int(round(longest*step))
    fraction=float(np.mean(bad))
    output.update(peak_valid=True,peak_reason='ok',
                  peak_outlier_fraction=fraction,
                  peak_residual_p95=float(np.percentile(err_abs,95)),
                  peak_residual_median=float(np.median(err_abs)),
                  peak_longest_run_px=longest_px,
                  peak_affine_scale=float(params[0]),
                  peak_affine_offset=float(params[1]),
                  peak_bad=bool(fraction>=min_bad_fraction and
                                longest_px>=min_run_px))
    return output
