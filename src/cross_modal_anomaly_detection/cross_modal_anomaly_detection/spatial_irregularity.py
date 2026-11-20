#!/usr/bin/env python3
"""Proposed bilateral, median-neighborhood RGB/LiDAR irregularity monitor.

PROTOTYPE, NOT SAFETY-QUALIFIED. No startup healthy calibration. Prefer input
at actual projected LiDAR x-bins, with RGB depth sampled at those bins, over
profiles densely interpolated between LiDAR returns. The symmetric two-sided
score has NOT been validated on actual monocular-depth predictions.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class IrregularityConfig:
    regions: int = 9  # nine overlapping windows, approximately five-window widths
    region_overlap_fraction: float = 0.50
    min_unique_bad_points: int = 8
    neighborhood: int = 7
    threshold: float = 0.20  # vibration-recovery default; requires field validation
    min_affected_fraction: float = 0.02
    min_bad_regions: int = 2
    min_fit_samples: int = 45
    max_cached_fit_scans: int = 2
    cached_fit_median_residual_ratio: float = 0.35
    cached_fit_min_pairs: int = 24
    min_region_support: int = 8
    min_supported_fraction: float = 0.35
    scale_floor: float = 0.04  # inverse-metre if L=1/z in metres
    max_neighbor_gap_px: float = 16.0
    alignment_smooth_sigma: float = 12.0  # bins; fit LOW-frequency geometry, not injected high-frequency noise
    moving_multiplier: float = 1.35
    skip_rotation: bool = True


def _smooth_for_affine(x, sigma):
    """Gaussian, NaN-aware smoothing for ALIGNMENT ONLY, not anomaly evidence."""
    x = np.asarray(x, dtype=float)
    if sigma <= 0:
        return x.copy()
    rad = min(int(np.ceil(3 * sigma)), (len(x) - 1) // 2)
    k = np.arange(-rad, rad+1, dtype=float)
    weights = np.exp(-0.5*(k/sigma)**2)
    weights /= np.sum(weights)
    valid = np.isfinite(x)
    numerator = np.convolve(np.where(valid, x, 0.), weights, mode='same')
    effective = np.convolve(valid.astype(float), weights, mode='same')
    possible = np.convolve(np.ones(len(x)), weights, mode='same')
    out = np.full_like(x, np.nan)
    usable = (effective > 0) & (effective >= 0.70*possible)
    out[usable] = numerator[usable]/effective[usable]
    return out


def _robust_affine(rgb, lidar, min_points):
    ok = np.isfinite(rgb) & np.isfinite(lidar)
    if np.count_nonzero(ok) < min_points:
        return None
    x, y = rgb[ok], lidar[ok]
    # Detect essentially flat inputs, where scale can be unstable.
    if np.std(x) < 0.004 or np.std(y) < 0.004:
        return None
    A = np.column_stack((x, np.ones(len(x))))
    try:
        coeff = np.linalg.lstsq(A, y, rcond=None)[0]
        for _ in range(5):
            error = y - A @ coeff
            cen = np.median(error)
            sigma = max(1.4826 * np.median(np.abs(error-cen)),
                        0.008 * np.median(np.abs(y)), 1e-5)
            weights = np.minimum(1., 1.5 * sigma / (np.abs(error-cen) + 1e-12))
            w = np.sqrt(weights)
            coeff = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    return coeff if np.all(np.isfinite(coeff)) else None


def _local_irregularity(u, x, width, max_gap):
    """Absolute deviation from local median of UNSMOOTHED measured bins.

    For each measured beam j, compare x[j] with the median over the centered
    odd-length neighborhood. Never median-filter x itself, interpolate across
    missing returns, or bridge neighboring beams separated by excessive x-gap.
    Both LiDAR and affine-aligned RGB depth use exactly this same operation.
    """
    n = len(u)
    radius = width // 2
    out = np.full(n, np.nan)
    for j in range(radius, n-radius):
        sl = slice(j-radius, j+radius+1)
        if (np.all(np.isfinite(x[sl])) and
                np.all(np.diff(u[sl]) > 0) and
                np.max(np.diff(u[sl])) <= max_gap):
            reference_median = float(np.median(x[sl]))
            out[j] = abs(x[j] - reference_median)
    return out


class CrossModalSpatialIrregularity:
    def __init__(self, **params):
        self.cfg = IrregularityConfig(**params)
        c = self.cfg
        if c.neighborhood < 3 or c.neighborhood % 2 == 0:
            raise ValueError('neighborhood must be an odd integer >= 3')
        if not (1 <= c.min_bad_regions <= c.regions <= 32):
            raise ValueError('invalid regional layout')
        if not (0 <= c.region_overlap_fraction < 1):
            raise ValueError('region_overlap_fraction must be in [0, 1)')
        if c.min_unique_bad_points < 1:
            raise ValueError('min_unique_bad_points must be positive')
        if c.max_cached_fit_scans < 0 or c.cached_fit_min_pairs < 3:
            raise ValueError('invalid cached fit limits')
        if not (0 <= c.min_supported_fraction <= 1):
            raise ValueError('min_supported_fraction must be within [0, 1]')
        self._last_stamp = None
        self._last_result = None
        self._last_good_affine = None
        self._cached_fit_age = 0

    def unavailable(self, reason):
        return dict(spatial_irregularity_valid=False,
                    spatial_irregularity_bad=False,
                    spatial_irregularity_reason=reason,
                    spatial_irregularity_bad_regions=[],
                    spatial_irregularity_bad_region_count=0,
                    spatial_irregularity_max_score=None,
                    spatial_irregularity_dominance='unavailable',
                    spatial_irregularity_fit_source='none',
                    spatial_irregularity_fit_age=None,
                    spatial_irregularity_unique_bad_points=0,
                    spatial_irregularity_median_window=self.cfg.neighborhood,
                    spatial_irregularity_required_regions=self.cfg.min_bad_regions,
                    spatial_irregularity_required_unique_points=self.cfg.min_unique_bad_points,
                    spatial_irregularity_decision_reason=reason)

    def observe(self, u, lidar, rgb, *, stamp_key=None,
                moving=False, turning=False):
        if stamp_key is not None and stamp_key == self._last_stamp and self._last_result:
            return dict(self._last_result)
        if stamp_key is not None:
            self._last_stamp = stamp_key
        if turning and self.cfg.skip_rotation:
            r = self.unavailable('rotation_guard')
            self._last_result = r
            return r
        u = np.asarray(u, dtype=float).reshape(-1)
        l = np.asarray(lidar, dtype=float).reshape(-1)
        e = np.asarray(rgb, dtype=float).reshape(-1)
        c = self.cfg
        if (u.shape != l.shape or l.shape != e.shape or len(u) < c.min_fit_samples
                or not np.all(np.isfinite(u)) or np.any(np.diff(u) <= 0)):
            r = self.unavailable('shape_or_support')
            self._last_result = r
            return r
        # Fit once on the CURRENT NCC-selected, paired geometry (never per
        # overlapping window). This keeps the model independent of window count.
        fit = _robust_affine(_smooth_for_affine(e, c.alignment_smooth_sigma),
                             _smooth_for_affine(l, c.alignment_smooth_sigma),
                             c.min_fit_samples)
        fit_source = 'fresh'
        if fit is not None:
            self._last_good_affine = np.asarray(fit, dtype=float).copy()
            self._cached_fit_age = 0
        else:
            # A short fallback may bridge one or two intermittently degenerate
            # fits. Never learn an affine model from the corrupted current data.
            # Never silently mark insufficient support as OK.
            previous = self._last_good_affine
            next_age = self._cached_fit_age + 1
            if previous is None or next_age > c.max_cached_fit_scans:
                r = self.unavailable('affine_fit_unavailable')
                self._last_result = r
                return r
            finite = np.isfinite(l) & np.isfinite(e)
            if np.count_nonzero(finite) < c.cached_fit_min_pairs:
                r = self.unavailable('cached_fit_insufficient_pairs')
                self._last_result = r
                return r
            cur_scale = max(float(np.median(np.abs(l[finite]))), c.scale_floor)
            residual = np.abs(l[finite] - (previous[0] * e[finite] + previous[1]))
            if float(np.median(residual)) / cur_scale > c.cached_fit_median_residual_ratio:
                r = self.unavailable('cached_fit_mismatch')
                self._last_result = r
                return r
            fit = previous
            self._cached_fit_age = next_age
            fit_source = 'cached'
        e_hat = fit[0] * e + fit[1]
        rl = _local_irregularity(u, l, c.neighborhood, c.max_neighbor_gap_px)
        re = _local_irregularity(u, e_hat, c.neighborhood, c.max_neighbor_gap_px)
        positive = l[np.isfinite(l) & (l > 0)]
        if not len(positive):
            r = self.unavailable('no_positive_lidar_depth')
            self._last_result = r
            return r
        scale = max(float(np.median(positive)), c.scale_floor)
        signed = (rl - re) / scale
        score = np.abs(signed)
        threshold = c.threshold * (c.moving_multiplier if moving else 1.)
        # Equal-width windows overlap in IMAGE x-coordinates, not beam indices.
        # A beam may occur in multiple windows; the unique affected-beam gate
        # limits double-counting from overlap and the launch config can reduce
        # overlap while requiring evidence in several regions.
        if c.regions == 1:
            windows = [(float(u[0]), float(u[-1]))]
        else:
            span = float(u[-1] - u[0])
            width = span / (1.0 + (c.regions - 1) * (1.0 - c.region_overlap_fraction))
            starts = np.linspace(float(u[0]), float(u[-1] - width), c.regions)
            windows = [(float(start), float(start + width)) for start in starts]
        bad = []
        fractions = []
        support = []
        distinct_bad = np.zeros(len(u), dtype=bool)
        eps = 1e-7 * max(1., float(u[-1]-u[0]))
        for k, (low, high) in enumerate(windows, 1):
            reg = np.flatnonzero((u >= low-eps) & (u <= high+eps))
            if len(reg) == 0:
                support.append(False)
                fractions.append(None)
                continue
            valid = np.isfinite(score[reg])
            n_valid = int(np.count_nonzero(valid))
            supported = (n_valid >= c.min_region_support and
                         n_valid / len(reg) >= c.min_supported_fraction)
            support.append(bool(supported))
            bad_mask = np.isfinite(score[reg]) & (score[reg] > threshold)
            fraction = (float(np.count_nonzero(bad_mask)) / len(reg)
                        if supported else None)
            fractions.append(fraction)
            if supported and fraction > c.min_affected_fraction:
                bad.append(k)
                distinct_bad[reg[bad_mask]] = True
        overall_valid = sum(support) >= c.min_bad_regions
        n_distinct_bad = int(np.count_nonzero(distinct_bad))
        score_valid = score[np.isfinite(score)]
        lid = int(np.count_nonzero(signed > threshold))
        cam = int(np.count_nonzero(signed < -threshold))
        if lid == 0 and cam == 0:
            dominance = 'none'
        elif lid > cam:
            dominance = 'lidar_profile_more_irregular'
        elif cam > lid:
            dominance = 'rgb_profile_more_irregular'
        else:
            dominance = 'mixed'
        # Diagnostic only: explain WHICH gate prevented an irregularity vote.
        # Core score, spatial windows and anomaly formulation are unchanged.
        if not overall_valid:
            decision_reason = 'insufficient_support'
        elif len(bad) < c.min_bad_regions:
            decision_reason = 'too_few_affected_regions'
        elif n_distinct_bad < c.min_unique_bad_points:
            decision_reason = 'too_few_distinct_bad_points'
        else:
            decision_reason = 'irregularity_confirmed'
        result = dict(
            spatial_irregularity_valid=bool(overall_valid),
            spatial_irregularity_bad=bool(decision_reason == 'irregularity_confirmed'),
            spatial_irregularity_required_regions=int(c.min_bad_regions),
            spatial_irregularity_required_unique_points=int(c.min_unique_bad_points),
            spatial_irregularity_decision_reason=decision_reason,
            spatial_irregularity_reason='ok' if overall_valid else 'insufficient_support',
            spatial_irregularity_bad_regions=bad,
            spatial_irregularity_bad_region_count=len(bad),
            spatial_irregularity_unique_bad_points=n_distinct_bad,
            spatial_irregularity_region_windows=windows,
            spatial_irregularity_region_overlap_fraction=c.region_overlap_fraction,
            spatial_irregularity_median_window=c.neighborhood,
            # Diagnostics only; do not modify the paper's bilateral score.
            spatial_irregularity_threshold_applied=float(threshold),
            spatial_irregularity_supported_region_count=int(sum(support)),
            spatial_irregularity_max_affected_fraction=(
                float(max(f for f in fractions if f is not None))
                if any(f is not None for f in fractions) else None),
            spatial_irregularity_exceeding_point_count=int(np.count_nonzero(
                np.isfinite(score) & (score > threshold))),
            spatial_irregularity_fit_source=fit_source,
            spatial_irregularity_fit_age=int(self._cached_fit_age),
            spatial_irregularity_max_score=(float(np.max(score_valid)) if len(score_valid) else None),
            spatial_irregularity_mean_score=(float(np.mean(score_valid)) if len(score_valid) else None),
            spatial_irregularity_region_fractions=fractions,
            spatial_irregularity_region_supported=support,
            spatial_irregularity_dominance=dominance,  # diagnostic, NOT source attribution
            spatial_irregularity_dominant_lidar_points=lid,
            spatial_irregularity_dominant_rgb_points=cam,
            spatial_irregularity_affine_scale=float(fit[0]),
            spatial_irregularity_affine_offset=float(fit[1]),
            spatial_irregularity_shared_scale=scale,
        )
        self._last_result = result
        return result
