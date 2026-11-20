import json
from collections import deque

import cv2

# Shared visualization style (OpenCV uses BGR, not RGB).
# The color consistently identifies the sensor, independent of which
# sensor is plotted first or which diagnostic is being visualized.
PLOT_COLOR_LIDAR = (255, 80, 40)   # blue
PLOT_COLOR_RGB = (40, 70, 230)    # red; depth estimated from RGB
PLOT_FONT = cv2.FONT_HERSHEY_SIMPLEX
PLOT_TITLE_SCALE = 0.65
PLOT_INFO_SCALE = 0.48
PLOT_LEGEND_SCALE = 0.45
PLOT_AXIS_SCALE = 0.42
PLOT_LINE_WIDTH = 2


def draw_sensor_legend(canvas, sensor_names):
    """Consistent, right-aligned legend for every diagnostic curve."""
    colors = {'LiDAR': PLOT_COLOR_LIDAR,
              'RGB depth': PLOT_COLOR_RGB}
    items = []
    for name in sensor_names:
        (tw, _), _ = cv2.getTextSize(name, PLOT_FONT, PLOT_LEGEND_SCALE, 1)
        items.append((name, tw))
    total = sum(30 + 7 + tw + 16 for _, tw in items) - 16
    x = max(8, canvas.shape[1] - 18 - total)
    y = 20
    for name, tw in items:
        cv2.line(canvas, (x, y), (x + 30, y), colors[name],
                 PLOT_LINE_WIDTH, cv2.LINE_AA)
        x += 37
        cv2.putText(canvas, name, (x, y + 5), PLOT_FONT,
                    PLOT_LEGEND_SCALE, (35, 35, 35), 1, cv2.LINE_AA)
        x += tw + 16


def draw_anomaly_check_strip(canvas, result, y0=122):
    """Display current scan status; irregularity is diagnostic unless voting is enabled."""
    data = result or {}
    width = canvas.shape[1]
    xmargin, gap = 14, 7

    def score(value, digits=2):
        try:
            value = float(value)
            return f'{value:.{digits}f}' if np.isfinite(value) else '--'
        except (TypeError, ValueError):
            return '--'

    if data.get('shape_valid', False):
        ncc = ('BAD' if data.get('shape_bad', False) else 'OK')
        ncc_extra = 'r=' + score(data.get('global_shape_similarity',
                                           data.get('aligned_profile_corr')))
        if data.get('local_ncc_affected_regions'):
            ncc_extra += ' g=' + score(data.get('global_shape_similarity_raw'))
        if data.get('local_ncc_pair_bad'):
            pairs = data.get('local_ncc_confirmed_pairs') or []
            if pairs:
                ncc_extra += ' pair=' + '-'.join(str(i) for i in pairs[0])
    else:
        ncc, ncc_extra = 'N/A', 'no comparison'

    lidar_coverage = (data.get('coverage_evidence') or {}).get('lidar') or {}
    if lidar_coverage.get('valid', False):
        coverage = 'BAD' if lidar_coverage.get('bad', False) else 'OK'
        coverage_extra = 'c=' + score(
            100.0 * float(lidar_coverage.get('ratio') or 0), 0) + '%'
        if lidar_coverage.get('return_coverage_valid', False):
            coverage_extra += ' r=' + score(
                100.0 * float(lidar_coverage.get('return_coverage_ratio') or 0), 0) + '%'
        elif lidar_coverage.get('return_coverage_reason') in ('calibrating', 'calibration_complete'):
            coverage_extra += ' r=CAL'
    elif lidar_coverage.get('reason') == 'calibrating':
        coverage, coverage_extra = 'CAL', (
            f"{lidar_coverage.get('calibration_frames', 0)}/"
            f"{lidar_coverage.get('required_calibration_frames', '?')}")
    else:
        coverage, coverage_extra = 'N/A', 'no reference'

    if not data.get('spatial_irregularity_enabled', False):
        irregularity, irregularity_extra = 'OFF', 'disabled'
    elif not data.get('spatial_irregularity_valid', False):
        irregularity, irregularity_extra = 'N/A', str(data.get('spatial_irregularity_reason', 'no support'))
    else:
        irregularity = 'BAD' if data.get('spatial_irregularity_bad', False) else 'OK'
        # Show both decision gates: overlapping BAD windows AND DISTINCT
        # anomalous measured LiDAR positions. A high max d by itself is not
        # enough to count as BAD and can hide a unique-points gate failure.
        n_regions = data.get('spatial_irregularity_bad_region_count', 0)
        req_regions = data.get('spatial_irregularity_required_regions', '?')
        n_unique = data.get('spatial_irregularity_unique_bad_points', 0)
        req_unique = data.get('spatial_irregularity_required_unique_points', '?')
        irregularity_extra = f'r={n_regions}/{req_regions} u={n_unique}/{req_unique}'
        # The irregularity metric uses each modality's UNSMOOTHED value minus
        # the median of a centered local neighborhood (not median-smoothing
        # the input profile). Display that neighborhood size for verification.
        median_size = data.get('spatial_irregularity_median_window')
        if median_size is not None:
            irregularity_extra += ' med' + str(int(median_size))
        if data.get('spatial_irregularity_max_score') is not None:
            irregularity_extra += ' d=' + score(data.get('spatial_irregularity_max_score'), 2)
        if not data.get('spatial_irregularity_vote_enabled', False):
            irregularity_extra += ' (diag)'

    if 'rgb_depth_invalid_bad' in data:
        rgb = 'BAD' if data['rgb_depth_invalid_bad'] else 'OK'
        rgb_extra = 'validity'
    else:
        rgb, rgb_extra = 'N/A', 'not evaluated'

    checks = [
        ('NCC', ncc, ncc_extra),
        ('LiDAR COVERAGE', coverage, coverage_extra),
        ('IRREGULARITY', irregularity, irregularity_extra),
        ('RGB DEPTH', rgb, rgb_extra),
    ]
    backgrounds = {
        'OK': (228, 248, 229),
        'BAD': (220, 224, 255),
        'ALARM': (220, 224, 255),
        'EVIDENCE': (207, 239, 255),
        'RECOVERY': (207, 239, 255),
        'HELD': (207, 239, 255),
        'CAL': (207, 239, 255),
        'OFF': (236, 236, 236),
        'N/A': (236, 236, 236),
    }
    foregrounds = {
        'OK': (38, 120, 39),
        'BAD': (25, 30, 190),
        'ALARM': (25, 30, 190),
        'EVIDENCE': (22, 110, 160),
        'RECOVERY': (22, 110, 160),
        'HELD': (22, 110, 160),
        'CAL': (22, 110, 160),
        'OFF': (110, 110, 110),
        'N/A': (110, 110, 110),
    }
    card_width = max(32, (width - 2*xmargin - (len(checks)-1)*gap)//len(checks))
    for i, (label, state, detail) in enumerate(checks):
        x = xmargin + i*(card_width+gap)
        x2 = min(width-xmargin, x+card_width)
        cv2.rectangle(canvas, (x, y0), (x2, y0+33),
                      backgrounds.get(state, backgrounds['N/A']), -1)
        cv2.rectangle(canvas, (x, y0), (x2, y0+33),
                      (198, 198, 198), 1)
        label_scale = 0.34
        cv2.putText(canvas, label, (x+6, y0+12), PLOT_FONT,
                    label_scale, (45, 45, 45), 1, cv2.LINE_AA)
        main = f'{state}  {detail}'
        main_scale = 0.42
        while (main_scale > 0.29 and
               cv2.getTextSize(main, PLOT_FONT, main_scale, 1)[0][0] > card_width-10):
            main_scale -= 0.02
        cv2.putText(canvas, main, (x+6, y0+27), PLOT_FONT,
                    main_scale, foregrounds.get(state, (35, 35, 35)),
                    1, cv2.LINE_AA)
    return checks


import numpy as np
import time
from .decision_visualization import draw_decision_banner, make_unavailable_plot
from .fault_evidence import (CoverageBaseline, projected_lidar_mask, merge_evidence)
from .peak_evidence import peak_mismatch_evidence
from .profile_support import comparison_support_mask
from .profile_visualization import normalize_using_reference, draw_dotted_region
from .spatial_irregularity import CrossModalSpatialIrregularity
from .return_coverage import CameraSectorReturnMonitor

import rclpy
from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from cv_bridge import CvBridge

from sensor_msgs.msg import (
    Image,
    PointCloud2,
    LaserScan,
)

from sensor_msgs_py import point_cloud2

from vision_msgs.msg import Detection2DArray
from nav_msgs.msg import Odometry

from std_msgs.msg import (
    Bool,
    Float32,
    String,
)





def minimal_detection_readout(score, valid, anomaly):
    """Identical, minimal status line for all 3 detector plot types.

    NCC is the alignment-adjusted profile correlation, not a fused anomaly
    probability. The state is the FINAL decision after coverage/metric
    evidence and temporal voting, never the intermediate NCC-only decision.
    """
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = float('nan')
    score_text = f'{score_value:.3f}' if np.isfinite(score_value) else 'N/A'
    state = ('YES' if anomaly else ('UNKNOWN' if not valid else 'NO'))
    color = ((35, 35, 220) if state == 'YES' else
             (35, 135, 35) if state == 'NO' else (100, 100, 100))
    return f'NCC: {score_text}   |   ANOMALY: {state}', color


def profile_problem_regions(x, first, second, num_regions=5,
                            overlap_fraction=0.0, min_points=12, min_coverage=0.50,
                            local_corr_threshold=0.15,
                            residual_threshold=0.45,
                            residual_excess_threshold=0.15):
    """Regional diagnostics used for red shading and optional NCC scoring.

    Marks localized disagreement (robust global affine-fit residual excess,
    optionally supported by low local Pearson) and large missing-data gaps.
    This function alone does not vote; local_aware_ncc_score() may separately
    use its supported geometric mismatch flags to adjust the NCC score.
    """
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if len(x) < 2 or len(x) != len(a) or len(x) != len(b):
        return []
    valid_x = np.isfinite(x)
    if not np.any(valid_x):
        return []
    x = x[valid_x]
    a = a[valid_x]
    b = b[valid_x]
    # Seven windows with 50% overlap cover the whole valid x span.  For
    # W=1152 px: window width ~=288 px, stride ~=144 px.  The previous
    # non-overlapping partition remains available by setting overlap=0.
    num_regions = int(num_regions)
    overlap_fraction = float(overlap_fraction)
    if not 0.0 <= overlap_fraction < 1.0 or num_regions < 1:
        raise ValueError('invalid region count or overlap_fraction')
    xmin, xmax = float(x.min()), float(x.max())
    span = max(xmax - xmin, 0.0)
    window_width = span / (1.0 + (num_regions - 1) * (1.0 - overlap_fraction))
    stride = window_width * (1.0 - overlap_fraction)
    paired = np.isfinite(a) & np.isfinite(b)
    residual = np.full(x.shape, np.nan, dtype=np.float64)
    if paired.sum() >= 6 and np.std(a[paired]) > 1e-8 and np.std(b[paired]) > 1e-8:
        target, predictor = a[paired], b[paired]
        design = np.column_stack((predictor, np.ones_like(predictor)))
        scale = max(float(np.percentile(target, 75) - np.percentile(target, 25)), 1e-6)
        try:
            coef = np.linalg.lstsq(design, target, rcond=None)[0]
            for _ in range(6):
                error = target - design @ coef
                center = float(np.median(error))
                sigma = max(1.4826 * float(np.median(np.abs(error - center))),
                            0.02 * scale)
                weights = np.minimum(1.0, 1.5 * sigma / np.maximum(np.abs(error), 1e-12))
                root = np.sqrt(weights)
                coef = np.linalg.lstsq(design * root[:, None],
                                       target * root, rcond=None)[0]
            residual[paired] = np.abs(target - design @ coef) / scale
        except np.linalg.LinAlgError:
            pass
    regions = []
    for k in range(num_regions):
        start = xmin + k * stride
        stop = xmax if k == num_regions - 1 else min(xmax, start + window_width)
        in_region = (x >= start) & (x <= stop if k == num_regions - 1 else x < stop)
        count = int(in_region.sum())
        if not count:
            continue
        a_finite = in_region & np.isfinite(a)
        b_finite = in_region & np.isfinite(b)
        matched = in_region & paired
        n_pair = int(matched.sum())
        a_fraction = float(a_finite.sum()) / count
        b_fraction = float(b_finite.sum()) / count
        paired_fraction = float(n_pair) / count
        # Only label a missing-data region when at least one sensor has data.
        given_available = n_pair / max(int(a_finite.sum()), int(b_finite.sum()), 1)
        coverage_bad = (max(a_fraction, b_fraction) >= 0.60 and
                        given_available < 0.25)
        local_bad = False
        corr = None
        med_inside = np.nan
        excess = np.nan
        if n_pair >= min_points and paired_fraction >= min_coverage:
            va = a[matched]
            vb = b[matched]
            if min(np.std(va), np.std(vb)) > 0.035:
                c = float(np.corrcoef(va, vb)[0, 1])
                if np.isfinite(c):
                    corr = c
            inner_res = residual[matched]
            inner_res = inner_res[np.isfinite(inner_res)]
            outer_res = residual[paired & ~in_region]
            outer_res = outer_res[np.isfinite(outer_res)]
            if inner_res.size and outer_res.size:
                med_inside = float(np.median(inner_res))
                excess = max(0.0, med_inside - float(np.median(outer_res)))
                high_residual = (med_inside > residual_threshold and
                                 excess > residual_excess_threshold)
                low_corr_and_residual = (corr is not None and
                    corr < local_corr_threshold and
                    med_inside > 0.5 * residual_threshold and
                    excess > 0.5 * residual_excess_threshold)
                local_bad = bool(high_residual or low_corr_and_residual)
        regions.append({'x_start': start, 'x_end': stop,
                        'bad': local_bad, 'coverage_bad': bool(coverage_bad),
                        'paired_coverage': paired_fraction,
                        'local_corr': corr,
                        'paired_points': int(n_pair),
                        'median_residual': float(med_inside) if np.isfinite(med_inside) else None,
                        'residual_excess': float(excess) if np.isfinite(excess) else None})
    return regions


def local_aware_ncc_score(u, first, second, global_corr, shift_px, *,
                          enabled=False, focus_gain=1.7,
                          min_corr_drop=0.18, regions=7, overlap_fraction=0.50,
                          require_adjacent_pair=True,
                          residual_enabled=True, min_region_coverage=0.70,
                          min_region_points=30, residual_min_median=0.50,
                          residual_min_excess=0.18, min_affected_regions=1):
    """Local-sensitive shape decision reusing the EXACT red-region diagnostic.

    Raw Pearson NCC is unchanged. A region may be RED due to a LOCAL affine-
    alignment residual even if its Pearson correlation stays high (or is
    undefined when RGB depth flattens). Earlier code incorrectly discarded
    those cases. Use the same robust residual and support mask as the plot.

    This creates a geometric consistency decision SCORE, not a new Pearson NCC.
    The 0.60 threshold and existing scan-based voting are left untouched.
    """
    raw = float(global_corr) if np.isfinite(global_corr) else None
    details = dict(
        local_ncc_fusion_enabled=bool(enabled),
        local_ncc_residual_enabled=bool(residual_enabled),
        local_ncc_focus_gain=float(focus_gain),
        local_ncc_min_region_points=int(min_region_points),
        local_ncc_min_region_coverage=float(min_region_coverage),
        local_ncc_min_affected_regions=int(min_affected_regions),
        local_ncc_regions=int(regions),
        local_ncc_overlap_fraction=float(overlap_fraction),
        local_ncc_require_adjacent_pair=bool(require_adjacent_pair),
        local_ncc_confirmed_pairs=[],
        local_ncc_pair_bad=False,
        local_ncc_window_bounds=[],
        local_ncc_candidate_regions=[],
        local_ncc_residual_min_median=float(residual_min_median),
        local_ncc_residual_min_excess=float(residual_min_excess),
        local_ncc_raw_global=raw,
        local_ncc_decision_score=raw,
        local_ncc_worst_corr=None,
        local_ncc_affected_regions=[],
        local_ncc_region_correlations=[],
        local_ncc_region_residual_medians=[],
        local_ncc_region_residual_excess=[],
        local_ncc_region_paired_coverages=[],
        local_ncc_visual_red_regions=[],
        local_ncc_residual_affected_regions=[],
        local_ncc_shape_affected_regions=[],
        local_ncc_reason='disabled' if not enabled else 'no_supported_local_disagreement',
    )
    if not enabled or raw is None:
        return details
    x = np.asarray(u, dtype=np.float64)
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if x.ndim != 1 or a.shape != x.shape or b.shape != x.shape or x.size < 80:
        details['local_ncc_reason'] = 'insufficient_profile'
        return details
    aligned = interp_with_holes(x, b, x + float(shift_px)) if shift_px else b
    # EXACT SAME default validity and red-region definitions as plot.
    region_info = profile_problem_regions(
        x, a, aligned, num_regions=int(regions),
        overlap_fraction=float(overlap_fraction))
    candidates = []
    for i, info in enumerate(region_info, start=1):
        details['local_ncc_window_bounds'].append([round(info['x_start'], 2), round(info['x_end'], 2)])
        corr = info.get('local_corr')
        med = info.get('median_residual')
        excess = info.get('residual_excess')
        coverage = float(info.get('paired_coverage') or 0.0)
        pairs = int(info.get('paired_points') or 0)
        red = bool(info.get('bad') or info.get('coverage_bad'))
        details['local_ncc_region_correlations'].append(round(float(corr), 4) if corr is not None and np.isfinite(corr) else None)
        details['local_ncc_region_residual_medians'].append(round(float(med), 4) if med is not None and np.isfinite(med) else None)
        details['local_ncc_region_residual_excess'].append(round(float(excess), 4) if excess is not None and np.isfinite(excess) else None)
        details['local_ncc_region_paired_coverages'].append(round(coverage, 4))
        if red:
            details['local_ncc_visual_red_regions'].append(i)
        # Coverage-only red shading is not RGB-LiDAR shape disagreement.
        # Limit low-signal, thin-object and missing-data false positives.
        if (not info.get('bad', False) or coverage < min_region_coverage or
                pairs < min_region_points):
            continue
        if corr is not None and np.isfinite(corr) and raw - float(corr) >= min_corr_drop:
            score = float(np.clip(raw - focus_gain * (raw - float(corr)), -1.0, 1.0))
            candidates.append((score, i, 'local_correlation'))
            details['local_ncc_shape_affected_regions'].append(i)
        # This captures *red regions that Pearson alone misses*, such as local
        # shifts or flattened RGB depth despite high local Pearson.
        if (residual_enabled and med is not None and np.isfinite(med) and
                excess is not None and np.isfinite(excess) and
                float(med) > residual_min_median and
                float(excess) > residual_min_excess):
            # Moderate margin beyond the visualization-only red criterion.
            # Reject borderline residual increases; preserve the existing
            # local-aware scoring and unchanged 0.60 NCC decision threshold.
            # This is a score derived from residuals, NOT Pearson correlation.
            score = float(np.clip(1.0 - float(med), -1.0, 1.0))
            candidates.append((score, i, 'local_residual'))
            details['local_ncc_residual_affected_regions'].append(i)
    details['local_ncc_candidate_regions'] = sorted({i for _, i, _ in candidates})
    if not candidates:
        if details['local_ncc_visual_red_regions']:
            details['local_ncc_reason'] = 'red_regions_but_no_supported_shape_evidence'
        return details
    # Consolidate candidate scores per window.  A window can qualify through
    # either local Pearson reduction or robust affine residual disagreement.
    # We deliberately do NOT count the same window twice when both are bad.
    by_window = {}
    for score_value, index, mechanism in candidates:
        if index not in by_window or score_value < by_window[index][0]:
            by_window[index] = (score_value, mechanism)
    if require_adjacent_pair:
        # Count only ADJACENT overlapping windows, not two arbitrary regions.
        # A single red window never changes the global NCC decision score.
        # For a pair, use its LESS severe score, so BOTH windows must carry
        # strong evidence; taking the minimum would overemphasize one window.
        pair_candidates = []
        for index in sorted(by_window):
            if index + 1 in by_window:
                pair_score = max(by_window[index][0], by_window[index + 1][0])
                pair_candidates.append((pair_score, index, index + 1))
        details['local_ncc_confirmed_pairs'] = [
            [left, right] for _, left, right in pair_candidates]
        if not pair_candidates:
            details['local_ncc_reason'] = 'no_adjacent_confirmed_overlapping_windows'
            return details
        if len(by_window) < max(2, int(min_affected_regions)):
            details['local_ncc_reason'] = 'insufficient_supported_local_regions'
            return details
        pair_score, left, right = min(pair_candidates)
        details['local_ncc_decision_score'] = min(raw, pair_score)
        details['local_ncc_affected_regions'] = [left, right]
        details['local_ncc_pair_bad'] = True
        details['local_ncc_reason'] = 'confirmed_overlapping_window_pair'
    else:
        # Explicit legacy A/B mode: independent windows, no adjacency gate.
        if len(by_window) < int(min_affected_regions):
            details['local_ncc_reason'] = 'insufficient_supported_local_regions'
            return details
        best = min(candidates, key=lambda item:item[0])
        details['local_ncc_decision_score'] = min(raw, best[0])
        details['local_ncc_affected_regions'] = sorted(by_window)
        details['local_ncc_reason'] = 'supported_' + best[2] + '_disagreement'
    corr = [c for c in details['local_ncc_region_correlations'] if c is not None]
    details['local_ncc_worst_corr'] = min(corr) if corr else None
    return details


def shade_problem_regions(canvas, regions, map_x, plot_x0, plot_x1, top, bottom):
    """Highlight the same suspect image-x regions on profile/slope/curvature."""
    # Union of bad windows: overlapping areas should not accumulate opacity.
    if not regions:
        return
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    for region in regions:
        x0 = int(np.clip(map_x(region['x_start']), plot_x0, plot_x1))
        x1 = int(np.clip(map_x(region['x_end']), plot_x0, plot_x1))
        cv2.line(canvas, (x0, top), (x0, bottom),
                 (230, 230, 230), 1, cv2.LINE_AA)
        if x1 > x0 and (region['bad'] or region['coverage_bad']):
            cv2.rectangle(mask, (x0, top), (x1, bottom), 255, -1)
    if np.any(mask):
        overlay = canvas.copy()
        overlay[mask > 0] = (210, 220, 255)
        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

def normalized_with_holes(values):
    """Robust normalization over the OWN sensor's finite pixels (not paired mask)."""
    a = np.asarray(values, dtype=np.float64)
    good = np.isfinite(a)
    out = np.full(a.shape, np.nan, dtype=np.float64)
    if good.sum() < 3:
        return out
    v = a[good]
    med = float(np.median(v))
    scale = float(np.percentile(v, 75) - np.percentile(v, 25))
    if scale < 1e-8:
        scale = float(np.std(v))
    if scale < 1e-8:
        # A genuinely flat but valid depth profile must remain visible.
        # Pearson still correctly reports undefined correlation for constants.
        scale = max(abs(med), 1.0)
    out[good] = (v - med) / scale
    return out


def interpolate_measured_support(x_known, y_known, x_query, max_gap_px=6.0):
    """Bounded spatial interpolation for a separate FULL-SUPPORT diagnostic.

    Uses only observed bins. No extrapolation and no interpolation over a
    missing sector wider than max_gap_px. This does not affect any alarm.
    """
    xk = np.asarray(x_known, dtype=float).reshape(-1)
    yk = np.asarray(y_known, dtype=float).reshape(-1)
    xq = np.asarray(x_query, dtype=float)
    out = np.full(xq.shape, np.nan, dtype=float)
    if xk.shape != yk.shape:
        raise ValueError('Known x and y must share shape')
    ok = np.isfinite(xk) & np.isfinite(yk)
    xk, yk = xk[ok], yk[ok]
    if xk.size == 0:
        return out
    order = np.argsort(xk)
    xk, yk = xk[order], yk[order]
    if np.any(np.diff(xk) <= 0):
        raise ValueError('Known positions must be unique')
    in_bounds = np.isfinite(xq) & (xq >= xk[0]) & (xq <= xk[-1])
    right = np.searchsorted(xk, xq[in_bounds], side='left')
    lo = np.maximum(0, right - 1)
    hi = np.minimum(xk.size - 1, right)
    x = xq[in_bounds]
    dxlo, dxhi = np.abs(x - xk[lo]), np.abs(x - xk[hi])
    exact_lo, exact_hi = dxlo < 1e-7, dxhi < 1e-7
    bracket_ok = ((xk[hi] - xk[lo]) <= float(max_gap_px))
    acceptable = exact_lo | exact_hi | bracket_ok
    y = np.full(x.shape, np.nan, dtype=float)
    y[exact_lo] = yk[lo[exact_lo]]
    y[exact_hi] = yk[hi[exact_hi]]
    between = acceptable & ~(exact_lo | exact_hi) & (hi != lo)
    weights = ((x[between] - xk[lo[between]]) /
               (xk[hi[between]] - xk[lo[between]]))
    y[between] = (yk[lo[between]] * (1 - weights) +
                  yk[hi[between]] * weights)
    out[in_bounds] = y
    return out


def normalize_full_profile_for_display(full_values, metric_reference):
    """Put original, valid RGB depths on the selected metric's plot scale.

    This function never affects comparison support, Pearson, faults or votes.
    When selected support is too flat, use the original complete RGB samples
    solely to choose display limits; do not fabricate absent measurements.
    """
    full = np.asarray(full_values, dtype=np.float64)
    ref = np.asarray(metric_reference, dtype=np.float64)
    valid = np.isfinite(full)
    result = np.full(full.shape, np.nan, dtype=np.float64)
    if not np.any(valid):
        return result
    source = ref[np.isfinite(ref)]
    if source.size < 3:
        source = full[valid]
    med = float(np.median(source))
    scale = float(np.percentile(source, 75) - np.percentile(source, 25))
    if scale < 1e-8:
        scale = float(np.std(source))
    if scale < 1e-8:
        scale = max(abs(med), 1.0)
    result[valid] = (full[valid] - med) / scale
    return result


def smooth_with_holes(values, width):
    """Smooth only inside consecutive valid runs. No bridges across missing depth."""
    a = np.asarray(values, dtype=np.float64)
    good = np.isfinite(a)
    out = np.full(a.shape, np.nan, dtype=np.float64)
    width = max(1, int(width))
    if width <= 1:
        return a.copy()
    cuts = np.diff(np.r_[False, good, False].astype(np.int8))
    starts = np.where(cuts == 1)[0]
    stops = np.where(cuts == -1)[0]
    for start, stop in zip(starts, stops):
        seg = a[start:stop]
        w = min(width | 1, len(seg) if len(seg) % 2 else len(seg) - 1)
        if w < 3:
            out[start:stop] = seg
        else:
            pad = w // 2
            out[start:stop] = np.convolve(
                np.pad(seg, (pad, pad), mode='edge'),
                np.ones(w, dtype=np.float64) / w,
                mode='valid')
    return out


def interp_with_holes(x_old, y_old, x_new):
    """Interpolate within uninterrupted valid runs; do not bridge missing data."""
    x_old=np.asarray(x_old,dtype=np.float64)
    y_old=np.asarray(y_old,dtype=np.float64)
    x_new=np.asarray(x_new,dtype=np.float64)
    result=np.full(x_new.shape,np.nan,dtype=np.float64)
    valid=np.isfinite(x_old)&np.isfinite(y_old)
    boundaries=np.diff(np.r_[False,valid,False].astype(np.int8))
    for lo,hi in zip(np.flatnonzero(boundaries==1),np.flatnonzero(boundaries==-1)):
        xs,ys=x_old[lo:hi],y_old[lo:hi]
        if xs.size>=2:
            region=(x_new>=xs[0])&(x_new<=xs[-1])
            result[region]=np.interp(x_new[region],xs,ys)
        elif xs.size==1:
            region=np.isclose(x_new,xs[0],atol=1e-5)
            result[region]=ys[0]
    return result


def select_irregularity_pairs(u_grid, lidar_grid_raw, rgb_grid_raw,
                              lidar_bin_u, lidar_bin_depth, shift_px):
    """Pair true LiDAR bins with RGB depth on the same selected NCC support.

    u_grid and both grid profiles must already carry identical geometric,
    rectification, narrow-object, and edge-trim selection.  No normalization,
    spatial smoothing or temporal EMA is performed here.  The NCC-selected
    horizontal shift is applied to RGB samples, exactly as for peak evidence.
    Invalid/missing pairs are NaN in BOTH modalities, not replaced by
    interpolated LiDAR values; the original LiDAR bins remain distinct.
    """
    u = np.asarray(u_grid, dtype=np.float64).reshape(-1)
    lg = np.asarray(lidar_grid_raw, dtype=np.float64).reshape(-1)
    eg = np.asarray(rgb_grid_raw, dtype=np.float64).reshape(-1)
    bu = np.asarray(lidar_bin_u, dtype=np.float64).reshape(-1)
    bl = np.asarray(lidar_bin_depth, dtype=np.float64).reshape(-1)
    if not (u.shape == lg.shape == eg.shape and bu.shape == bl.shape):
        raise ValueError('Irregularity pairing requires matched per-grid and per-beam arrays')
    if (u.size < 2 or not np.all(np.isfinite(u)) or
            np.any(np.diff(u) <= 0)):
        return (np.empty(0, dtype=np.float64),) * 3
    # Preserve actual beam-to-beam gaps for neighborhood checks, but omit all
    # beams outside the EXACT edge-trimmed camera-x interval used by NCC.
    in_domain = (np.isfinite(bu) & (bu >= u[0]) & (bu <= u[-1]))
    x, l = bu[in_domain], bl[in_domain]
    if x.size == 0:
        return (np.empty(0, dtype=np.float64),) * 3
    # A measured beam must lie at a supported regular-grid cell. The nearest
    # grid-cell gate is only for *selection*; its interpolated LiDAR depth is
    # never used in the actual irregularity calculation.
    idx_right = np.searchsorted(u, x)
    hi = np.clip(idx_right, 0, len(u)-1)
    lo = np.clip(idx_right-1, 0, len(u)-1)
    nearest = np.where(np.abs(x-u[lo]) <= np.abs(x-u[hi]), lo, hi)
    step = float(np.median(np.diff(u)))
    source_supported = ((np.abs(x-u[nearest]) <= 0.51*step) &
                        np.isfinite(lg[nearest]))
    # The RGB series was masked before any smoothing, so this interpolation
    # cannot bridge invalid, excluded or missing RGB support.
    e = interp_with_holes(u, eg, x + float(shift_px))
    paired = source_supported & np.isfinite(l) & np.isfinite(e)
    return x, np.where(paired, l, np.nan), np.where(paired, e, np.nan)


def fixed_grid_masked_ema(u, first, second, state, enabled, alpha):
    """Independent sensor EMA. Keep invalid pixels as gaps and handle moving u support."""
    u=np.asarray(u,dtype=np.float64)
    a=np.asarray(first,dtype=np.float64).copy()
    b=np.asarray(second,dtype=np.float64).copy()
    old_u=state.get('u')
    if enabled and old_u is not None and len(old_u)>1:
        for cur, old in ((a,state.get('first')), (b,state.get('second'))):
            if old is None: continue
            previous=interp_with_holes(old_u,old,u)
            good=np.isfinite(cur)&np.isfinite(previous)
            cur[good]=float(alpha)*cur[good]+(1-float(alpha))*previous[good]
    state['u']=u.copy()
    state['first']=a.copy()
    state['second']=b.copy()
    return a,b




def shift_invariant_pearson(u, first, second, *, max_shift_px=5,
                            min_points=7, min_coverage=0.35,
                            min_overlap_ratio=0.90, min_shift_gain=0.01):
    """Masked, bounded, shift-invariant Pearson of two image-x profiles.

    At shift d compare first(u) with second(u+d).  The original x grid is
    NEVER compressed: shifts are measured in IMAGE PIXELS, and interpolation
    is restricted to consecutive finite runs (no bridging invalid-depth gaps).

    Every candidate must have >=min_points valid pairs, >=min_coverage of
    the complete current grid, and >=min_overlap_ratio of the zero-shift
    paired count.  This prevents a short but well-matching overlap from
    concealing a corrupted or missing region.  Retain shift=0 unless an
    offset improves Pearson by at least min_shift_gain.

    Returns (best_r, best_shift_px, best_valid_pairs, original_r).
    Invalid original coverage produces NaN rather than a healthy score.
    """
    u = np.asarray(u, dtype=np.float64)
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if u.ndim != 1 or len(u) != len(first) or len(u) != len(second) or len(u) < 4:
        return np.nan, 0, 0, np.nan
    if not np.all(np.isfinite(u)) or np.any(np.diff(u) <= 0):
        return np.nan, 0, 0, np.nan

    zero_mask = np.isfinite(first) & np.isfinite(second)
    zero_count = int(zero_mask.sum())
    required = max(int(min_points), int(np.ceil(float(min_coverage) * len(u))))
    if zero_count < required:
        return np.nan, 0, zero_count, np.nan

    def safe_corr(a, b, mask):
        if mask.sum() < 4:
            return np.nan
        aa, bb = a[mask], b[mask]
        if np.std(aa) < 1e-9 or np.std(bb) < 1e-9:
            return np.nan
        return float(np.corrcoef(aa, bb)[0, 1])

    raw_r = safe_corr(first, second, zero_mask)
    if not np.isfinite(raw_r):
        return np.nan, 0, zero_count, np.nan
    best_r, best_shift, best_count = raw_r, 0, zero_count
    required_shift = max(required, int(np.ceil(float(min_overlap_ratio) * zero_count)))
    max_shift = max(0, int(max_shift_px))
    for shift in range(-max_shift, max_shift + 1):
        if shift == 0:
            continue
        # Re-sample only within actual finite segments; preserve all gaps.
        other_shifted = interp_with_holes(u, second, u + float(shift))
        mask = np.isfinite(first) & np.isfinite(other_shifted)
        n_valid = int(mask.sum())
        if n_valid < required_shift:
            continue
        value = safe_corr(first, other_shifted, mask)
        if not np.isfinite(value):
            continue
        # Favor the smaller shift when candidate quality is indistinguishable.
        gain = float(min_shift_gain) if best_shift == 0 else 1e-9
        if value > best_r + gain:
            best_r, best_shift, best_count = value, shift, n_valid
        elif best_shift != 0 and abs(value - best_r) <= 1e-9 and abs(shift) < abs(best_shift):
            best_shift, best_count = shift, n_valid
    return float(best_r), int(best_shift), int(best_count), float(raw_r)

def finite_gradient(values, u):
    """Finite derivative only inside continuous valid profile segments."""
    a = np.asarray(values, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    good = np.isfinite(a)
    output = np.full(a.shape, np.nan)
    cuts = np.diff(np.r_[False, good, False].astype(np.int8))
    for start, stop in zip(np.where(cuts==1)[0], np.where(cuts==-1)[0]):
        if stop-start >= 2:
            output[start:stop] = np.gradient(a[start:stop], u[start:stop])
    return output



class ScanUniqueVoter:
    """One vote per unique LiDAR scan, with optional weak-evidence confirmation.

    NCC, peak differences and local coverage losses are susceptible to
    intermittent thin-object returns. Their votes can require M bad scans in
    the previous N distinct scans, including the current scan. Severe *global*
    LiDAR loss and invalid RGB depth bypass the weak-evidence debouncer.

    The raw classification is retained in diagnostics; only the confirmed
    classification contributes to the alarm history and consecutive count.
    """
    def __init__(self, window=10, required=5, consecutive=4,
                 minimum_seconds=0.9, dropout_interval_s=0.25,
                 weak_confirm_window=1, weak_confirm_required=1):
        self.window = max(1, int(window))
        self.required = max(1, int(required))
        self.consecutive_required = max(1, int(consecutive))
        self.minimum_seconds = max(0.0, float(minimum_seconds))
        self.dropout_interval_s = max(0.05, float(dropout_interval_s))
        self.weak_confirm_window = max(1, int(weak_confirm_window))
        self.weak_confirm_required = max(1, min(
            self.weak_confirm_window, int(weak_confirm_required)))
        self.history = []
        self.raw_history = []
        self.weak_history = deque(maxlen=self.weak_confirm_window)
        self.seen_scans = deque(maxlen=128)
        self.reset()

    def reset(self):
        self.history.clear()
        self.raw_history.clear()
        self.weak_history.clear()
        self.seen_scans.clear()
        self.last_dropout_at = None
        self.bad_since = None
        self.consecutive_bad = 0
        self.anomaly = False
        self.last_raw_bad = False
        self.last_confirmed_bad = False
        self.last_strong_bad = False
        self.last_vote_reason = 'reset'

    def observe(self, *, valid, frame_bad, strong_bad=False,
                stamp_key=None, dropout=False, now=None):
        now = time.monotonic() if now is None else float(now)
        if not valid and not (dropout and frame_bad and strong_bad):
            # Invalid/unsynchronised frames do not provide a valid healthy vote.
            # Preserve the sliding window and an already-active alarm.  Break
            # consecutive evidence (onset must not span an unavailable gap).
            self.consecutive_bad = 0
            self.bad_since = None
            self.last_vote_reason = 'unavailable_hold'
            return self.details(False, 0.0)
        # A diagnosed strong LiDAR dropout may still contribute one bad vote
        # per dropout_interval_s, even though geometric comparison is invalid.

        new_evidence = False
        if dropout:
            # A missing LiDAR stream has no scan stamps: permit at most one
            # evidence sample per configured real-time interval.
            if (self.last_dropout_at is None or
                    now - self.last_dropout_at >= self.dropout_interval_s):
                new_evidence = True
                self.last_dropout_at = now
        elif stamp_key is not None:
            self.last_dropout_at = None
            if stamp_key not in self.seen_scans:
                new_evidence = True
                self.seen_scans.append(stamp_key)

        if new_evidence:
            raw_bad = bool(frame_bad)
            strong = bool(raw_bad and strong_bad)
            # A strong event (global coverage or depth invalidity) must not
            # accidentally confirm later weak evidence from an earlier scene.
            if strong:
                self.weak_history.clear()
            else:
                self.weak_history.append(raw_bad)
            weak_confirmed = bool(
                raw_bad and not strong and
                len(self.weak_history) >= self.weak_confirm_required and
                sum(self.weak_history) >= self.weak_confirm_required)
            confirmed_bad = bool(strong or weak_confirmed)

            self.last_raw_bad = raw_bad
            self.last_confirmed_bad = confirmed_bad
            self.last_strong_bad = strong
            self.last_vote_reason = (
                'strong_global_coverage_or_invalid_depth' if strong else
                'confirmed_weak_evidence' if weak_confirmed else
                'unconfirmed_weak_evidence' if raw_bad else 'healthy')
            self.raw_history.append(raw_bad)
            del self.raw_history[:-self.window]
            self.history.append(confirmed_bad)
            del self.history[:-self.window]

            if confirmed_bad:
                if self.consecutive_bad == 0:
                    self.bad_since = now
                self.consecutive_bad += 1
            else:
                self.consecutive_bad = 0
                self.bad_since = None

            duration = (max(0.0, now - self.bad_since)
                        if self.bad_since is not None else 0.0)
            count = int(sum(self.history))
            if self.anomaly:
                # Recovery hysteresis: once active, a single healthy scan
                # must NOT clear the alarm while >= required confirmed bad
                # scans remain in the rolling window.
                if count < self.required:
                    self.anomaly = False
            else:
                # Onset remains conservative: threshold + consecutive bad
                # scans + minimum elapsed bad-evidence duration.
                self.anomaly = bool(
                    len(self.history) >= self.required
                    and count >= self.required
                    and self.consecutive_bad >= self.consecutive_required
                    and duration >= self.minimum_seconds)

        duration = (max(0.0, now - self.bad_since)
                    if self.bad_since is not None else 0.0)
        return self.details(new_evidence, duration)

    def details(self, new_evidence, duration):
        return {
            'scan_vote_added': bool(new_evidence and self.last_confirmed_bad),
            'scan_evidence_added': bool(new_evidence),
            'raw_frame_bad': bool(self.last_raw_bad),
            'vote_frame_bad': bool(self.last_confirmed_bad),
            'strong_bad_bypass': bool(self.last_strong_bad),
            'vote_reason': self.last_vote_reason,
            'raw_bad_frames_in_window': int(sum(self.raw_history)),
            'bad_frames_in_window': int(sum(self.history)),
            'history_length': len(self.history),
            'consecutive_bad_scans': self.consecutive_bad,
            'bad_evidence_duration_s': round(float(duration), 3),
            'voting_mode': 'unique_lidar_scans_weak_evidence_debounced',
            'vote_weak_confirm_window': self.weak_confirm_window,
            'vote_weak_confirm_required': self.weak_confirm_required,
            'vote_weak_bad_in_window': int(sum(self.weak_history)),
            'required_consecutive_bad_scans': self.consecutive_required,
            'minimum_bad_evidence_duration_s': self.minimum_seconds,
            'anomaly': bool(self.anomaly),
            'anomaly_latched': bool(self.anomaly),
            'anomaly_release_below_bad_scans': self.required,
        }


class DepthLidarAnomalyNode(Node):

    def __init__(self):

        super().__init__('cross_modal_anomaly_detection')

        # ====================================================
        # Input topic parameters
        # ====================================================

        self.projected_points_topic = self.declare_parameter(
            'projected_points_topic',
            '/lidar_projected_points',
        ).value

        self.estimated_points_topic = self.declare_parameter(
            'estimated_points_topic',
            '/lidar_expected_projection',
        ).value
        self.scan_topic = str(self.declare_parameter(
            'return_coverage_scan_topic', '/scan').value)

        self.depth_topic = self.declare_parameter(
            'depth_topic',
            '/depth_anything/depth',
        ).value
        # Shared geometric mask: 255=rectification-valid; 0=discard.
        self.valid_mask_topic = str(self.declare_parameter(
            'valid_mask_topic', '/rgb_rectification/valid_mask').value)
        self.require_valid_mask = bool(self.declare_parameter(
            'require_valid_mask', True).value)
        self.rectification_valid_mask = None

        self.detection_topic = self.declare_parameter(
            'detection_topic',
            '/human_dets',
        ).value

        # ====================================================
        # Output topic parameters
        # ====================================================

        self.anomaly_topic = self.declare_parameter(
            'anomaly_topic',
            '/cross_modal_anomaly/anomaly',
        ).value

        self.frame_bad_topic = self.declare_parameter(
            'frame_bad_topic',
            '/cross_modal_anomaly/frame_bad',
        ).value

        self.similarity_topic = self.declare_parameter(
            'similarity_topic',
            '/cross_modal_anomaly/shape_similarity',
        ).value

        self.status_topic = self.declare_parameter(
            'status_topic',
            '/cross_modal_anomaly/status',
        ).value

        # ====================================================
        # Optional curve-visualization topics
        # ====================================================

        self.visualize_curves = bool(
            self.declare_parameter(
                'visualize_curves',
                True,
            ).value
        )

        self.profile_plot_topic = self.declare_parameter(
            'profile_plot_topic',
            '/cross_modal_anomaly/profile_plot',
        ).value

        self.slope_plot_topic = self.declare_parameter(
            'slope_plot_topic',
            '/cross_modal_anomaly/slope_plot',
        ).value

        self.curvature_plot_topic = self.declare_parameter(
            'curvature_plot_topic',
            '/cross_modal_anomaly/curvature_plot',
        ).value

        self.plot_width = max(
            400,
            int(
                self.declare_parameter(
                    'plot_width',
                    980,
                ).value
            ),
        )

        self.plot_height = max(
            240,
            int(
                self.declare_parameter(
                    'plot_height',
                    460,
                ).value
            ),
        )

        self.plot_every = max(
            1,
            int(
                self.declare_parameter(
                    'plot_every',
                    1,
                ).value
            ),
        )

        # ====================================================
        # Algorithm parameters
        # ====================================================

        self.mode = self.declare_parameter(
            'mode',
            'relative',
        ).value

        self.depth_patch_radius = max(
            0,
            int(
                self.declare_parameter(
                    'depth_patch_radius',
                    1,
                ).value
            ),
        )

        # Horizontal sampling grid used to create both 1-D signals.
        # 1 = sample every image pixel; 4 = one sample every 4 pixels.
        self.x_sampling_step_px = max(
            1,
            int(
                self.declare_parameter(
                    'x_sampling_step_px',
                    1,
                ).value
            ),
        )

        # Maximum time mismatch between the relative-depth frame and
        # the estimated projection cloud.
        self.max_estimated_time_difference = max(
            0.0,
            float(
                self.declare_parameter(
                    'max_estimated_time_difference',
                    0.25,
                ).value
            ),
        )

        # Kept for backward-compatible ROS parameters / diagnostics.
        # The anomaly decision below uses Pearson correlation of original depth profiles.
        slope_weight = float(
            self.declare_parameter(
                'slope_weight',
                1.0,
            ).value
        )

        curvature_weight = float(
            self.declare_parameter(
                'curvature_weight',
                0.0,
            ).value
        )

        if (
            slope_weight < 0.0
            or curvature_weight < 0.0
        ):
            raise ValueError(
                'Shape weights must be non-negative.'
            )

        weight_sum = (
            slope_weight
            + curvature_weight
        )

        if weight_sum <= 0.0:
            raise ValueError(
                'At least one shape weight must be positive.'
            )

        self.slope_weight = (
            slope_weight
            / weight_sum
        )

        self.curvature_weight = (
            curvature_weight
            / weight_sum
        )

        # Profile Pearson-correlation threshold. A valid frame is bad when
        # profile correlation is below this value.
        self.similarity_threshold = float(
            self.declare_parameter(
                'similarity_threshold',
                0.30,
            ).value
        )

        # Optional region-aware contribution to the EXISTING NCC score.
        # Does not change NCC thresholds, coverage/peak/temporal checks, or voting.
        self.local_ncc_fusion_enabled = bool(self.declare_parameter(
            'local_ncc_fusion_enabled', False).value)
        self.local_ncc_regions = int(self.declare_parameter(
            'local_ncc_regions', 7).value)
        self.local_ncc_overlap_fraction = float(self.declare_parameter(
            'local_ncc_overlap_fraction', 0.50).value)
        self.local_ncc_require_adjacent_pair = bool(self.declare_parameter(
            'local_ncc_require_adjacent_pair', True).value)
        if not 2 <= self.local_ncc_regions <= 32:
            raise ValueError('local_ncc_regions must be in [2, 32]')
        if not 0.0 <= self.local_ncc_overlap_fraction < 1.0:
            raise ValueError('local_ncc_overlap_fraction must be in [0, 1)')
        self.local_ncc_focus_gain = float(self.declare_parameter(
            'local_ncc_focus_gain', 1.7).value)
        self.local_ncc_min_corr_drop = float(self.declare_parameter(
            'local_ncc_min_corr_drop', 0.18).value)
        self.local_ncc_residual_enabled = bool(self.declare_parameter(
            'local_ncc_residual_enabled', True).value)
        self.local_ncc_min_region_coverage = float(self.declare_parameter(
            'local_ncc_min_region_coverage', 0.70).value)
        self.local_ncc_min_region_points = int(self.declare_parameter(
            'local_ncc_min_region_points', 30).value)
        self.local_ncc_residual_min_median = float(self.declare_parameter(
            'local_ncc_residual_min_median', 0.50).value)
        self.local_ncc_residual_min_excess = float(self.declare_parameter(
            'local_ncc_residual_min_excess', 0.18).value)
        if not 0 < self.local_ncc_min_region_points <= 4096:
            raise ValueError('local_ncc_min_region_points must be in [1,4096]')
        if not 0 <= self.local_ncc_min_region_coverage <= 1:
            raise ValueError('local_ncc_min_region_coverage must be in [0,1]')
        if not 0 <= self.local_ncc_residual_min_median <= 10:
            raise ValueError('local_ncc_residual_min_median must be in [0,10]')
        if not 0 <= self.local_ncc_residual_min_excess <= 10:
            raise ValueError('local_ncc_residual_min_excess must be in [0,10]')
        if not 0.0 <= self.local_ncc_focus_gain <= 3.0:
            raise ValueError('local_ncc_focus_gain must be in [0, 3]')
        if not 0.0 <= self.local_ncc_min_corr_drop <= 2.0:
            raise ValueError('local_ncc_min_corr_drop must be in [0, 2]')

        # Stricter LOCAL evidence requirements apply only while turning and
        # for a short post-turn settling interval. Global NCC / peak / coverage
        # thresholds and anomaly vote logic are not modified.
        self.local_ncc_rotation_guard_enabled = bool(self.declare_parameter(
            'local_ncc_rotation_guard_enabled', True).value)
        self.local_ncc_rotation_hold_s = float(self.declare_parameter(
            'local_ncc_rotation_hold_s', 0.75).value)
        self.local_ncc_rotation_focus_gain = float(self.declare_parameter(
            'local_ncc_rotation_focus_gain', 1.0).value)
        self.local_ncc_rotation_min_corr_drop = float(self.declare_parameter(
            'local_ncc_rotation_min_corr_drop', 0.30).value)
        self.local_ncc_rotation_min_region_coverage = float(self.declare_parameter(
            'local_ncc_rotation_min_region_coverage', 0.85).value)
        self.local_ncc_rotation_min_region_points = int(self.declare_parameter(
            'local_ncc_rotation_min_region_points', 45).value)
        self.local_ncc_rotation_residual_min_median = float(self.declare_parameter(
            'local_ncc_rotation_residual_min_median', 0.70).value)
        self.local_ncc_rotation_residual_min_excess = float(self.declare_parameter(
            'local_ncc_rotation_residual_min_excess', 0.30).value)
        self.local_ncc_rotation_min_affected_regions = int(self.declare_parameter(
            'local_ncc_rotation_min_affected_regions', 2).value)
        if not 0.0 <= self.local_ncc_rotation_hold_s <= 10.0:
            raise ValueError('local_ncc_rotation_hold_s must be in [0, 10]')
        if not 0.0 <= self.local_ncc_rotation_focus_gain <= 3.0:
            raise ValueError('local_ncc_rotation_focus_gain must be in [0, 3]')
        if not 0.0 <= self.local_ncc_rotation_min_corr_drop <= 2.0:
            raise ValueError('local_ncc_rotation_min_corr_drop must be in [0, 2]')
        if not 0.0 <= self.local_ncc_rotation_min_region_coverage <= 1.0:
            raise ValueError('local_ncc_rotation_min_region_coverage must be in [0, 1]')
        if not 1 <= self.local_ncc_rotation_min_region_points <= 4096:
            raise ValueError('local_ncc_rotation_min_region_points must be in [1,4096]')
        if not 0 <= self.local_ncc_rotation_residual_min_median <= 10:
            raise ValueError('local_ncc_rotation_residual_min_median must be in [0, 10]')
        if not 0 <= self.local_ncc_rotation_residual_min_excess <= 10:
            raise ValueError('local_ncc_rotation_residual_min_excess must be in [0, 10]')
        if not 1 <= self.local_ncc_rotation_min_affected_regions <= self.local_ncc_regions:
            raise ValueError('local_ncc_rotation_min_affected_regions must be within region count')

        # Shift-invariant Pearson (bounded image-horizontal offset, pixels).
        self.shift_max_px = max(0, int(self.declare_parameter('shift_max_px', 5).value))
        self.shift_min_overlap_ratio = float(self.declare_parameter('shift_min_overlap_ratio', 0.90).value)
        self.shift_min_gain = float(self.declare_parameter('shift_min_gain', 0.01).value)
        if not 0 < self.shift_min_overlap_ratio <= 1 or self.shift_min_gain < 0:
            raise ValueError('shift_min_overlap_ratio must be (0,1] and shift_min_gain >= 0')

        # Optional support-aware comparison. Thin chair/table legs are often
        # hit intermittently by the 2D LiDAR while monocular depth is smooth.
        # This mask is applied ONLY to NCC and peak-profile evaluation.
        # Original projected LiDAR points and raw dropout monitoring remain
        # completely unaffected.
        self.profile_support_filter_enabled = bool(self.declare_parameter(
            'profile_support_filter_enabled', False).value)
        self.profile_max_interp_gap_px = float(self.declare_parameter(
            'profile_max_interp_gap_px', 6.0).value)
        self.profile_narrow_object_width_px = float(self.declare_parameter(
            'profile_narrow_object_width_px', 10.0).value)
        self.profile_narrow_object_jump_iqr = float(self.declare_parameter(
            'profile_narrow_object_jump_iqr', 0.70).value)
        self.profile_narrow_object_margin_px = float(self.declare_parameter(
            'profile_narrow_object_margin_px', 3.0).value)
        if min(self.profile_max_interp_gap_px, self.profile_narrow_object_width_px,
               self.profile_narrow_object_jump_iqr, self.profile_narrow_object_margin_px) < 0:
            raise ValueError('profile-support parameters must be nonnegative')

        # Two-sided spatial irregularity: UNSMOOTHED measured LiDAR/RGB bins are
        # each compared with a configurable odd-sized LOCAL MEDIAN in the helper.
        # Both local deviations are compared after robust affine alignment.
        # A median is a neighborhood reference, NOT a filter of the input data:
        # isolated irregularities must remain observable. Diagnostic by default,
        # because the symmetric method has not been field-calibrated.
        # This irregularity check does NOT need a healthy startup baseline.
        self.spatial_irregularity_enabled = bool(self.declare_parameter(
            'spatial_irregularity_enabled', True).value)
        self.spatial_irregularity_vote_enabled = bool(self.declare_parameter(
            'spatial_irregularity_vote_enabled', False).value)
        self.spatial_irregularity_monitor = CrossModalSpatialIrregularity(
            regions=int(self.declare_parameter('spatial_irregularity_regions', 9).value),
            region_overlap_fraction=float(self.declare_parameter(
                'spatial_irregularity_overlap_fraction', 0.50).value),
            min_unique_bad_points=int(self.declare_parameter(
                'spatial_irregularity_min_unique_points', 8).value),
            # Odd length, e.g. 7 measured image-x samples: [i-3, ..., i+3].
            # The helper rejects neighborhoods with invalid data or wide gaps.
            neighborhood=int(self.declare_parameter('spatial_irregularity_neighborhood', 7).value),
            threshold=float(self.declare_parameter('spatial_irregularity_threshold', 0.20).value),
            min_affected_fraction=float(self.declare_parameter(
                'spatial_irregularity_min_fraction', 0.02).value),
            min_bad_regions=int(self.declare_parameter(
                'spatial_irregularity_min_bad_regions', 2).value),
            min_fit_samples=int(self.declare_parameter(
                'spatial_irregularity_min_fit_samples', 45).value),
            max_cached_fit_scans=int(self.declare_parameter(
                'spatial_irregularity_max_cached_fit_scans', 2).value),
            cached_fit_median_residual_ratio=float(self.declare_parameter(
                'spatial_irregularity_cached_fit_max_median_residual', 0.35).value),
            cached_fit_min_pairs=int(self.declare_parameter(
                'spatial_irregularity_cached_fit_min_pairs', 24).value),
            max_neighbor_gap_px=float(self.declare_parameter(
                'spatial_irregularity_max_neighbor_gap_px', 16.0).value),
            alignment_smooth_sigma=float(self.declare_parameter(
                'spatial_irregularity_alignment_smooth_sigma', 12.0).value),
            moving_multiplier=float(self.declare_parameter(
                'spatial_irregularity_moving_multiplier', 1.35).value),
            skip_rotation=bool(self.declare_parameter(
                'spatial_irregularity_skip_rotation', True).value))
        self._pending_spatial_irregularity_input = None

        # ====================================================
        # Optional human-ROI weighting
        # ====================================================

        self.use_person_roi_weighting = bool(
            self.declare_parameter(
                'use_person_roi_weighting',
                True,
            ).value
        )

        self.roi_point_weight = float(
            self.declare_parameter(
                'roi_point_weight',
                3.0,
            ).value
        )

        if self.roi_point_weight < 1.0:
            raise ValueError(
                'roi_point_weight must be >= 1.0.'
            )

        # ====================================================
        # Temporal / profile parameters
        # ====================================================

        self.anomaly_window = max(
            1,
            int(
                self.declare_parameter(
                    'anomaly_window',
                    10,
                ).value
            ),
        )

        self.required_bad_frames = max(
            1,
            int(
                self.declare_parameter(
                    'required_bad_frames',
                    10,
                ).value
            ),
        )

        if (
            self.required_bad_frames
            > self.anomaly_window
        ):
            raise ValueError(
                'required_bad_frames must be '
                '<= anomaly_window.'
            )

        # ====================================================
        # Spatial + temporal smoothing
        #
        # Spatial smoothing:
        #   moving average along the horizontal image axis u
        #   within the CURRENT frame.
        #
        # Temporal smoothing:
        #   exponential moving average (EMA) across frames,
        #   after spatial smoothing and before derivatives.
        #
        # Backward compatibility:
        #   smoothing_window is kept as an alias/default for
        #   spatial_smoothing_window.
        # ====================================================

        legacy_smoothing_window = max(
            1,
            int(
                self.declare_parameter(
                    'smoothing_window',
                    11,
                ).value
            ),
        )

        self.spatial_smoothing_window = max(
            1,
            int(
                self.declare_parameter(
                    'spatial_smoothing_window',
                    legacy_smoothing_window,
                ).value
            ),
        )

        self.temporal_smoothing_enabled = bool(
            self.declare_parameter(
                'temporal_smoothing_enabled',
                True,
            ).value
        )

        self.temporal_smoothing_alpha = float(
            self.declare_parameter(
                'temporal_smoothing_alpha',
                0.30,
            ).value
        )

        if not (
            0.0
            < self.temporal_smoothing_alpha
            <= 1.0
        ):
            raise ValueError(
                'temporal_smoothing_alpha must be '
                'in the interval (0, 1].'
            )

        self.edge_fraction = float(self.declare_parameter('edge_fraction', 0.05).value)
        if not 0.0 <= self.edge_fraction < 0.5:
            raise ValueError('edge_fraction must be [0,0.5)')
        # Motion-aware monitoring: odometry is used ONLY to mark fast scene
        # changes, not to control the TurtleBot or change sensor drivers.
        # During motion NCC is diagnostic (unreliable without scan deskew /
        # frame-to-frame geometric compensation), and coverage is judged by a
        # conservative global severe-loss threshold. Ordinary regional coverage
        # changes during turning are not interpreted as sensor faults.
        self.motion_aware_enabled = bool(self.declare_parameter(
            'motion_aware_enabled', True).value)
        self.odom_topic = str(self.declare_parameter('odom_topic', '/odom').value)
        self.motion_linear_threshold_mps = max(0.0, float(self.declare_parameter(
            'motion_linear_threshold_mps', 0.035).value))
        self.motion_angular_threshold_radps = max(0.0, float(self.declare_parameter(
            'motion_angular_threshold_radps', 0.10).value))
        self.motion_hold_s = max(0.0, float(self.declare_parameter(
            'motion_hold_s', 0.75).value))
        self.odom_timeout_s = max(0.1, float(self.declare_parameter(
            'odom_timeout_s', 1.0).value))
        self.motion_coverage_global_ratio = float(self.declare_parameter(
            'motion_coverage_global_ratio', 0.20).value)
        self.motion_similarity_threshold = float(self.declare_parameter(
            'motion_similarity_threshold', 0.30).value)
        self.rotation_similarity_threshold = float(self.declare_parameter(
            'rotation_similarity_threshold', 0.25).value)
        self.motion_coverage_region_ratio = float(self.declare_parameter(
            'motion_coverage_region_ratio', 0.18).value)
        self.motion_coverage_required_regions = max(1, int(self.declare_parameter(
            'motion_coverage_required_regions', 3).value))
        self.moving_required_consecutive = max(1, int(self.declare_parameter(
            'moving_required_consecutive_bad_scans', 5).value))
        self.moving_minimum_bad_seconds = max(0.0, float(self.declare_parameter(
            'moving_minimum_bad_evidence_duration_s', 1.2).value))
        # Peak-height differences can be large even when Pearson remains high.
        # Thresholds are relative to LiDAR profile IQR, not absolute metres.
        self.peak_mismatch_enabled = bool(self.declare_parameter(
            'peak_mismatch_enabled', False).value)
        self.peak_residual_threshold = float(self.declare_parameter(
            'peak_residual_threshold', 0.50).value)
        self.motion_peak_residual_threshold = float(self.declare_parameter(
            'motion_peak_residual_threshold', 0.65).value)
        self.rotation_peak_residual_threshold = float(self.declare_parameter(
            'rotation_peak_residual_threshold', 0.75).value)
        self.peak_min_fraction = float(self.declare_parameter(
            'peak_min_fraction', 0.05).value)
        self.motion_peak_min_fraction = float(self.declare_parameter(
            'motion_peak_min_fraction', 0.07).value)
        self.rotation_peak_min_fraction = float(self.declare_parameter(
            'rotation_peak_min_fraction', 0.09).value)
        self.peak_min_run_px = max(2, int(self.declare_parameter(
            'peak_min_run_px', 8).value))
        self.peak_max_motion_projection_dt_s = max(0.0, float(self.declare_parameter(
            'peak_max_motion_projection_dt_s', 0.15).value))
        if min(self.peak_residual_threshold, self.motion_peak_residual_threshold,
               self.rotation_peak_residual_threshold) <= 0.0:
            raise ValueError('Peak residual thresholds must be positive')
        if not all(0 < v <= 1 for v in (
                self.peak_min_fraction, self.motion_peak_min_fraction,
                self.rotation_peak_min_fraction)):
            raise ValueError('Peak mismatch fractions must be in (0, 1]')
        self.rgb_depth_invalid_alarm_enabled = bool(self.declare_parameter(
            'rgb_depth_invalid_alarm_enabled', True).value)
        self.rgb_depth_min_valid_fraction = float(self.declare_parameter(
            'rgb_depth_min_valid_fraction', 0.10).value)
        self.rgb_depth_min_std = float(self.declare_parameter(
            'rgb_depth_min_std', 1e-5).value)
        self.rgb_flat_lidar_min_cv = float(self.declare_parameter(
            'rgb_flat_lidar_min_cv', 0.08).value)
        if not 0 < self.motion_coverage_global_ratio <= 1:
            raise ValueError('motion_coverage_global_ratio must be (0,1]')
        if not -1 <= self.rotation_similarity_threshold <= 1 or not -1 <= self.motion_similarity_threshold <= 1:
            raise ValueError('NCC thresholds must lie in [-1,1]')
        if not 0 < self.motion_coverage_region_ratio <= 1:
            raise ValueError('motion_coverage_region_ratio must be (0,1]')
        if not 0 <= self.rgb_depth_min_valid_fraction <= 1:
            raise ValueError('rgb_depth_min_valid_fraction must be [0,1]')
        self.latest_odom_rx_mono = None
        self.last_motion_mono = None
        self.last_odom_linear_speed_mps = 0.0
        self.last_odom_angular_speed_radps = 0.0
        self._last_rotation_mono = None
        self._motion_was_active = False
        self._motion_info = self.motion_snapshot()

        self.min_paired_coverage = float(self.declare_parameter(
            'min_paired_coverage', 0.35).value)
        if not 0.0 <= self.min_paired_coverage <= 1.0:
            raise ValueError('min_paired_coverage must be between 0 and 1')

        self.min_points = max(
            3,
            int(
                self.declare_parameter(
                    'min_points',
                    7,
                ).value
            ),
        )

        self.max_projection_time_difference = max(
            0.0,
            float(
                self.declare_parameter(
                    'max_projection_time_difference',
                    0.25,
                ).value
            ),
        )

        self.projected_cloud_buffer_size = max(
            1,
            int(
                self.declare_parameter(
                    'projected_cloud_buffer_size',
                    20,
                ).value
            ),
        )

        self.estimated_cloud_buffer_size = max(
            1,
            int(
                self.declare_parameter(
                    'estimated_cloud_buffer_size',
                    20,
                ).value
            ),
        )

        self.print_every = max(
            1,
            int(
                self.declare_parameter(
                    'print_every',
                    10,
                ).value
            ),
        )

        # If these are <= 0, projected u/v are assumed to
        # already use the same pixel dimensions as the depth map.
        #
        # Set them only when /lidar_projected_points was produced
        # in a different image resolution than the depth image.
        self.projection_width = int(
            self.declare_parameter(
                'projection_width',
                0,
            ).value
        )

        self.projection_height = int(
            self.declare_parameter(
                'projection_height',
                0,
            ).value
        )

        if self.mode != 'relative':

            self.get_logger().warning(
                'This node implements the relative-depth '
                'shape anomaly detector. '
                'Use mode:=relative.'
            )

        # ====================================================
        # QoS
        # ====================================================

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        detection_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ====================================================
        # State
        # ====================================================

        self.bridge = CvBridge()

        self.projected_cloud_buffer = deque(
            maxlen=self.projected_cloud_buffer_size
        )

        self.estimated_cloud_buffer = deque(
            maxlen=self.estimated_cloud_buffer_size
        )

        self.person_boxes = []

        # One vote per distinct projected LiDAR scan. Repeated RGB depth
        # frames must not multiply the same LiDAR fault evidence.
        self.required_consecutive_bad_scans = max(1, int(self.declare_parameter(
            'required_consecutive_bad_scans', 4).value))
        self.minimum_bad_evidence_duration_s = max(0.0, float(self.declare_parameter(
            'minimum_bad_evidence_duration_s', 0.90).value))
        self.dropout_vote_interval_s = max(0.05, float(self.declare_parameter(
            'dropout_vote_interval_s', 0.25).value))
        self.coverage_region_votes_enabled = bool(self.declare_parameter(
            'coverage_region_votes_enabled', False).value)
        self.coverage_required_regions = max(1, int(self.declare_parameter(
            'coverage_required_regions', 2).value))
        # Confirm intermittent NCC/peak/local-coverage evidence before adding
        # a vote. Global LiDAR dropout and invalid RGB depth bypass the gate.
        self.vote_weak_confirm_window = max(1, int(self.declare_parameter(
            'vote_weak_confirm_window', 1).value))
        self.vote_weak_confirm_required = max(1, min(
            self.vote_weak_confirm_window, int(self.declare_parameter(
                'vote_weak_confirm_required', 1).value)))
        self.vote_guard = ScanUniqueVoter(
            window=self.anomaly_window, required=self.required_bad_frames,
            consecutive=self.required_consecutive_bad_scans,
            minimum_seconds=self.minimum_bad_evidence_duration_s,
            dropout_interval_s=self.dropout_vote_interval_s,
            weak_confirm_window=self.vote_weak_confirm_window,
            weak_confirm_required=self.vote_weak_confirm_required)
        self.bad_frame_history = self.vote_guard.history
        # Frozen, per-image-x reference learned during healthy startup.
        self.lidar_stream_timeout_s = float(self.declare_parameter('lidar_stream_timeout_s',0.75).value)
        self.last_projected_rx_mono = None
        # Complement projected-column occupancy with the valid original LaserScan
        # returns from camera-facing beams. The expected beam indices are learned
        # once from the projector's scan_index field, under fault-free operation.
        self.return_coverage_enabled = bool(self.declare_parameter(
            'return_coverage_enabled', True).value)
        self.return_coverage_monitor = CameraSectorReturnMonitor(
            baseline_scans=int(self.declare_parameter('return_coverage_baseline_scans', 24).value),
            min_expected_beams=int(self.declare_parameter('return_coverage_min_beams', 25).value),
            min_presence_fraction=float(self.declare_parameter('return_coverage_beam_presence', 0.25).value),
            static_ratio_threshold=float(self.declare_parameter('return_coverage_static_ratio', 0.70).value),
            moving_ratio_threshold=float(self.declare_parameter('return_coverage_moving_ratio', 0.60).value))
        self.raw_scan_buffer = deque(maxlen=100)

        self.coverage_monitor = CoverageBaseline(
            'lidar',frames=max(3,int(self.declare_parameter('coverage_baseline_frames',20).value)),
            regions=max(1,int(self.declare_parameter('coverage_regions',5).value)),
            global_ratio=float(self.declare_parameter('coverage_global_ratio',0.35).value),
            region_ratio=float(self.declare_parameter('coverage_region_ratio',0.30).value),
            min_reference_region=float(self.declare_parameter('coverage_min_reference_region',0.12).value),
            min_reference_global=float(self.declare_parameter('coverage_min_reference_global',0.01).value),
            required_regions=self.coverage_required_regions)


        self.frame_counter = 0

        self.plot_counter = 0
        self.latest_depth_header = None

        # Fixed plotting state.
        #
        # Each visualization (slope / curvature) gets its own fixed
        # x support and fixed y limits.  The limits are initialized
        # from the first valid plot and then kept unchanged.
        #
        # This is visualization-only state and does NOT change the
        # anomaly score or Pearson correlation calculation.
        self.latest_depth_width = None
        self.fixed_plot_state = {}

        # Previous temporally-smoothed profiles.
        #
        # The u grid can vary slightly from frame to frame, so
        # the previous profiles are stored together with their
        # image-x coordinates and interpolated onto the current
        # grid before EMA blending.
        self.temporal_u = None
        self.temporal_lidar_profile = None
        self.temporal_depth_profile = None
        # Separate visualization-only EMA: excluded LiDAR/RGB samples remain
        # visible as dotted traces, but never influence detection smoothing.
        self.support_plot_ema = {}

        self.warned_missing_cloud_fields = False
        self.warned_missing_estimated_fields = False

        # ====================================================
        # Subscribers
        # ====================================================

        self.projected_points_sub = self.create_subscription(
            PointCloud2,
            self.projected_points_topic,
            self.projected_points_callback,
            reliable_qos,
        )

        self.return_scan_sub = None
        if self.return_coverage_enabled:
            self.return_scan_sub = self.create_subscription(
                LaserScan, self.scan_topic, self.return_scan_callback, sensor_qos)
            self.get_logger().info(f'Camera-sector raw return availability: {self.scan_topic}')

        self.estimated_points_sub = self.create_subscription(
            PointCloud2,
            self.estimated_points_topic,
            self.estimated_points_callback,
            reliable_qos,
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            sensor_qos,
        )
        self.valid_mask_sub = self.create_subscription(
            Image, self.valid_mask_topic, self.valid_mask_callback, sensor_qos)

        self.odom_sub = None
        if self.motion_aware_enabled:
            self.odom_sub = self.create_subscription(
                Odometry, self.odom_topic, self.odom_callback, sensor_qos)
            self.get_logger().info(
                f'Motion-aware monitoring enabled: {self.odom_topic}; '
                f'linear > {self.motion_linear_threshold_mps:.3f} m/s, '
                f'angular > {self.motion_angular_threshold_radps:.3f} rad/s. '
                'NCC remains ACTIVE with a reduced threshold during motion; coverage remains active.')

        self.detection_sub = None

        if self.use_person_roi_weighting:

            self.detection_sub = self.create_subscription(
                Detection2DArray,
                self.detection_topic,
                self.detection_callback,
                detection_qos,
            )

        # ====================================================
        # Publishers
        # ====================================================

        self.anomaly_pub = self.create_publisher(
            Bool,
            self.anomaly_topic,
            10,
        )

        self.frame_bad_pub = self.create_publisher(
            Bool,
            self.frame_bad_topic,
            10,
        )

        self.similarity_pub = self.create_publisher(
            Float32, self.similarity_topic, 10)
        self.profile_similarity_pub = self.create_publisher(
            Float32, '/cross_modal_anomaly/profile_similarity', 10)
        self.valid_pub = self.create_publisher(
            Bool, '/cross_modal_anomaly/valid', 10)

        self.status_pub = self.create_publisher(
            String,
            self.status_topic,
            10,
        )

        self.profile_plot_pub = None
        self.slope_plot_pub = None
        self.curvature_plot_pub = None

        if self.visualize_curves:

            self.profile_plot_pub = self.create_publisher(
                Image,
                self.profile_plot_topic,
                10,
            )

            self.slope_plot_pub = self.create_publisher(
                Image,
                self.slope_plot_topic,
                10,
            )

            self.curvature_plot_pub = self.create_publisher(
                Image,
                self.curvature_plot_topic,
                10,
            )

        # ====================================================
        # Startup information
        # ====================================================

        self.get_logger().info(
            'Motion-aware Depth-LiDAR anomaly detector started (v5)'
        )

        self.get_logger().info(
            f'Projected LiDAR input: '
            f'{self.projected_points_topic}'
        )

        self.get_logger().info(
            f'Estimated projection input: '
            f'{self.estimated_points_topic}'
        )

        self.get_logger().info(
            'Depth sampling grid: '
            f'every {self.x_sampling_step_px} px along x'
        )

        self.get_logger().info(
            f'Depth input: '
            f'{self.depth_topic}'
        )

        if self.use_person_roi_weighting:

            self.get_logger().info(
                f'Human ROI input: '
                f'{self.detection_topic}'
            )

            self.get_logger().info(
                'Human ROI weighting: ENABLED '
                f'(weight={self.roi_point_weight:.2f})'
            )

        else:

            self.get_logger().info(
                'Human ROI weighting: DISABLED'
            )

        self.get_logger().info(
            'Required projected-cloud fields: '
            'u, v, camera_z'
        )

        self.get_logger().info(
            f'mode={self.mode} | '
            f'threshold={self.similarity_threshold:.3f} | '
            f'persistence='
            f'{self.required_bad_frames}/'
            f'{self.anomaly_window}'
        )

        self.get_logger().info(
            'Spatial smoothing: '
            f'window={self.spatial_smoothing_window}'
        )

        self.get_logger().info(
            'Temporal smoothing: '
            f'{"ENABLED" if self.temporal_smoothing_enabled else "DISABLED"}'
            f' | alpha={self.temporal_smoothing_alpha:.3f}'
        )

        if self.visualize_curves:

            self.get_logger().info(
                f'Profile plot: {self.profile_plot_topic}'
            )

            self.get_logger().info(
                f'Slope plot: {self.slope_plot_topic}'
            )

            self.get_logger().info(
                f'Curvature plot: {self.curvature_plot_topic}'
            )

    # ========================================================
    # Input callbacks
    # ========================================================

    def odom_callback(self, msg):
        """Use measured body velocity, including Create 3 action-driven motion."""
        self.latest_odom_rx_mono = time.monotonic()
        self.last_odom_linear_speed_mps = float(np.hypot(
            msg.twist.twist.linear.x, msg.twist.twist.linear.y))
        self.last_odom_angular_speed_radps = abs(float(msg.twist.twist.angular.z))
        if (self.last_odom_linear_speed_mps >= self.motion_linear_threshold_mps or
                self.last_odom_angular_speed_radps >= self.motion_angular_threshold_radps):
            self.last_motion_mono = self.latest_odom_rx_mono

    def motion_snapshot(self):
        """Conservative state with a short post-motion settling interval.

        If odometry never arrives, motion cannot be determined, so the original
        NCC decision is retained. status.motion_odom_fresh reveals this limitation.
        """
        now = time.monotonic()
        fresh = (self.latest_odom_rx_mono is not None and
                 now - self.latest_odom_rx_mono <= self.odom_timeout_s)
        active = bool(self.motion_aware_enabled and (
            (fresh and (self.last_odom_linear_speed_mps >= self.motion_linear_threshold_mps or
                        self.last_odom_angular_speed_radps >= self.motion_angular_threshold_radps)) or
            (self.last_motion_mono is not None and
             now - self.last_motion_mono <= self.motion_hold_s)))
        turning = bool(active and fresh and
                       self.last_odom_angular_speed_radps >= self.motion_angular_threshold_radps)
        if turning:
            self._last_rotation_mono = now
        turning_guard = bool(self.local_ncc_rotation_guard_enabled and active and
                             (turning or (self._last_rotation_mono is not None and
                              now - self._last_rotation_mono <= self.local_ncc_rotation_hold_s)))
        ncc_threshold = (self.rotation_similarity_threshold if turning else
                         self.motion_similarity_threshold if active else
                         self.similarity_threshold)
        return dict(
            motion_aware_enabled=bool(self.motion_aware_enabled),
            motion_active=active,
            motion_turning=turning,
            local_ncc_rotation_guard_active=turning_guard,
            motion_odom_fresh=bool(fresh),
            motion_linear_speed_mps=(float(self.last_odom_linear_speed_mps) if fresh else None),
            motion_angular_speed_radps=(float(self.last_odom_angular_speed_radps) if fresh else None),
            motion_ncc_suppressed=False,
            decision_ncc_threshold=float(ncc_threshold),
            motion_coverage_policy=('reduced_global_and_regional_thresholds' if active else 'full'),
        )

    def update_motion_state(self):
       self._motion_info = self.motion_snapshot()
       active = self._motion_info['motion_active']

       if active != self._motion_was_active:
           # Reset temporal profiles to prevent motion-induced lag.
           self.reset_temporal_smoothing()

           # Preserve historical anomaly evidence across
           # stationary/moving transitions.
           #
           # Do not reset the vote guard here:
           # self.vote_guard.reset()

           self._motion_was_active = active

    def decision_adjusted_coverage(self, evidence):
        """Keep raw coverage diagnostics but require strong, reproducible evidence.

        One projected LiDAR region can vanish as the visible scene changes.
        By default, regional losses are reported only, not used as votes; the
        global-loss vote remains active with a more conservative threshold.
        """
        if not evidence.get('valid', False):
            return evidence
        out = dict(evidence)
        out['raw_bad_regions'] = list(evidence.get('bad_regions', []))
        out['raw_coverage_bad'] = bool(evidence.get('bad', False))
        out['coverage_regions_vote_enabled'] = self.coverage_region_votes_enabled
        if not self.coverage_region_votes_enabled:
            out['bad_regions'] = []
        global_bad = bool(evidence.get('global_bad', False))
        region_bad = bool(self.coverage_region_votes_enabled and
                          len(out['raw_bad_regions']) >= self.coverage_required_regions)
        out['bad'] = bool(global_bad or region_bad)
        return out

    def fuse_and_vote(self, result, evidence, *, scan_msg=None, dropout=False):
        """Fuse the current evidence without adding a vote per RGB frame."""
        # merge_evidence appends to its supplied history: use a scratch list.
        # The guarded vote history only advances for genuinely new LiDAR scans.
        result = merge_evidence(
            result, {'lidar': evidence}, [],
            required_bad_frames=self.required_bad_frames,
            extra_bad=bool(result.get('rgb_depth_invalid_bad', False) or
                           result.get('peak_bad', False) or
                           (self.spatial_irregularity_vote_enabled and
                            result.get('spatial_irregularity_bad', False))))
        stamp_key = None
        if scan_msg is not None:
            stamp = scan_msg.header.stamp
            stamp_key = (int(stamp.sec), int(stamp.nanosec))
        # Severe global coverage loss is the strongest direct fog/dropout
        # signal and should not wait for weak NCC/peak confirmation. Regional
        # coverage losses are debounced like other locally unstable evidence.
        lidar_coverage = (result.get('coverage_evidence') or {}).get('lidar') or {}
        strong_bad = bool(lidar_coverage.get('global_bad', False) or
                          result.get('rgb_depth_invalid_bad', False))
        result.update(self.vote_guard.observe(
            valid=bool(result.get('check_valid', False)),
            frame_bad=bool(result.get('frame_bad', False)),
            strong_bad=strong_bad,
            stamp_key=stamp_key, dropout=bool(dropout)))
        # The new irregularity metric is diagnostic-only unless explicitly enabled
        # in voting after validation; no duplicate scan votes.
        result['decision_metric'] = 'ncc_coverage_peak_rgb_invalid_and_optional_irregularity'
        return result

    def motion_adjusted_coverage(self, evidence):
        """Keep true severe LiDAR loss detectable during turns.

        Motion changes projected coverage naturally, so only an extreme global
        loss relative to the stationary calibration is actionable in motion.
        Do not update the reference or censor raw coverage diagnostics.
        """
        if not self._motion_info.get('motion_active'):
            return evidence
        out = dict(evidence)
        out['motion_raw_coverage_bad'] = bool(evidence.get('bad', False))
        out['motion_raw_bad_regions'] = list(evidence.get('bad_regions', []))
        out['motion_raw_global_bad'] = bool(evidence.get('global_bad', False))
        out['global_threshold_ratio'] = self.motion_coverage_global_ratio
        out['region_threshold_ratio'] = self.motion_coverage_region_ratio
        out['motion_required_bad_regions'] = self.motion_coverage_required_regions
        ratio = evidence.get('ratio')
        ref = evidence.get('reference_fraction')
        severe = bool(evidence.get('valid', False) and ref is not None and
                      ref >= self.coverage_monitor.min_reference_global and
                      ratio is not None and
                      ratio < self.motion_coverage_global_ratio)
        current_regions = evidence.get('region_fraction') or []
        reference_regions = evidence.get('reference_regions') or []
        bad_regions = [i + 1 for i, (cur, baseline) in enumerate(
            zip(current_regions, reference_regions))
            if baseline >= self.coverage_monitor.min_reference_region and
            cur < self.motion_coverage_region_ratio * baseline]
        out['bad_regions'] = bad_regions
        out['global_bad'] = severe
        out['bad'] = bool(severe or
                          (self.coverage_region_votes_enabled and
                           len(bad_regions) >= self.motion_coverage_required_regions))
        return out

    def add_motion_diagnostics(self, result):
        # NEVER discard valid NCC just because the robot is moving.
        # A stricter stationary threshold and softer motion thresholds balance
        # fault detection with genuine scene change / rolling-scan mismatch.
        result.update(self._motion_info)
        threshold = float(self._motion_info['decision_ncc_threshold'])
        score = result.get('global_shape_similarity')
        shape_valid = bool(result.get('check_valid', False))
        if shape_valid and score is not None and np.isfinite(score):
            result['global_bad'] = bool(float(score) < threshold or
                                        result.get('local_ncc_pair_bad', False))
        weighted = result.get('weighted_shape_similarity')
        if weighted is not None and np.isfinite(weighted):
            result['weighted_bad'] = bool(float(weighted) < threshold)
        result['raw_shape_valid_while_moving'] = shape_valid
        result['raw_shape_bad_while_moving'] = bool(result.get('global_bad', False))
        return result

    def valid_mask_callback(self, msg):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            mask = np.asarray(mask, dtype=np.uint8)
            if mask.ndim != 2:
                raise ValueError('Expected HxW mask')
            self.rectification_valid_mask = mask != 0
        except Exception as exc:
            self.get_logger().error(f'Invalid rectification mask: {exc}')
            self.rectification_valid_mask = None

    def safe_pixel(self, u, v):
        mask = self.rectification_valid_mask
        if mask is None:
            return not self.require_valid_mask
        if not np.isfinite(u) or not np.isfinite(v):
            return False
        x, y = int(round(u)), int(round(v))
        return bool(0 <= y < mask.shape[0] and
                    0 <= x < mask.shape[1] and mask[y, x])

    def return_scan_callback(self, msg):
        self.raw_scan_buffer.append(msg)

    def matching_return_scan(self, projected_msg):
        stamp = projected_msg.header.stamp
        target = float(stamp.sec) + 1e-9 * float(stamp.nanosec)
        # Projector preserves the original /faults/scan timestamp exactly.
        # A narrow tolerance allows integer timestamp precision differences,
        # but prevents mixing current projected geometry with an old scan.
        for scan in reversed(self.raw_scan_buffer):
            t = float(scan.header.stamp.sec) + 1e-9 * float(scan.header.stamp.nanosec)
            if abs(t - target) <= 0.003:
                return scan
        return None

    @staticmethod
    def projector_scan_indices(projected_msg):
        if 'scan_index' not in {field.name for field in projected_msg.fields}:
            return None
        try:
            return [int(item[0]) for item in point_cloud2.read_points(
                projected_msg, field_names=('scan_index',), skip_nans=True)]
        except (ValueError, TypeError, RuntimeError):
            return None

    def projected_points_callback(
        self,
        msg,
    ):

        self.last_projected_rx_mono = time.monotonic()
        self.projected_cloud_buffer.append(
            msg
        )

    def estimated_points_callback(
        self,
        msg,
    ):

        self.estimated_cloud_buffer.append(
            msg
        )

    def detection_callback(
        self,
        msg,
    ):

        boxes = []

        for detection in msg.detections:

            bbox = detection.bbox
            center = bbox.center

            if hasattr(
                center,
                'position',
            ):

                cx = float(
                    center.position.x
                )

                cy = float(
                    center.position.y
                )

            else:

                cx = float(
                    center.x
                )

                cy = float(
                    center.y
                )

            width = float(
                bbox.size_x
            )

            height = float(
                bbox.size_y
            )

            boxes.append(
                (
                    cx - width / 2.0,
                    cy - height / 2.0,
                    cx + width / 2.0,
                    cy + height / 2.0,
                )
            )

        self.person_boxes = boxes

    def publish_lidar_dropout(self, reason, time_difference=None):
        self.update_motion_state()
        if reason == 'projected_cloud_time_mismatch' and self.last_projected_rx_mono is not None and                 time.monotonic() - self.last_projected_rx_mono < self.lidar_stream_timeout_s:
            self.reset_invalid()
            self.publish_invalid(reason, time_difference=time_difference)
            return
        if not self.coverage_monitor.ready or self.latest_depth_width is None:
            self.reset_invalid()
            self.publish_invalid(reason, time_difference=time_difference)
            return
        evidence = self.coverage_monitor.observe(
            np.zeros(self.latest_depth_width, dtype=bool),learn=False)
        evidence = self.decision_adjusted_coverage(evidence)
        evidence = self.motion_adjusted_coverage(evidence)
        result = {'check_valid':False,'reason':reason,'count':0,'roi_count':0,
                  'global_shape_similarity':None,'global_bad':False,
                  'weighted_bad':False,'weighted_shape_similarity':None,
                  'frame_bad':False,'anomaly':False}
        result.update(self._motion_info)
        result = self.fuse_and_vote(result, evidence, dropout=True)
        self.publish_result(result,time_difference)
        self.publish_missing_curves(result)

    # ========================================================
    # Depth callback
    #
    # Each depth frame is compared with the temporally closest
    # already-projected LiDAR cloud.
    # ========================================================

    def depth_callback(
        self,
        msg,
    ):

        if self.mode != 'relative':
            return

        self.update_motion_state()
        if not self.projected_cloud_buffer:
            self.publish_lidar_dropout('waiting_for_projected_lidar')
            return
        if not self.estimated_cloud_buffer:
            return

        try:

            depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='32FC1',
            )

        except Exception as exc:

            self.get_logger().error(
                f'Could not decode depth image: {exc}'
            )

            return

        self.latest_depth_header = msg.header

        depth = np.asarray(
            depth,
            dtype=np.float32,
        )

        if depth.ndim != 2:

            self.get_logger().warning(
                f'Expected HxW depth image, '
                f'got {depth.shape}'
            )

            return

        self.latest_depth_width = int(depth.shape[1])
        mask = self.rectification_valid_mask
        if self.require_valid_mask and mask is None:
            self.reset_invalid()
            self.publish_invalid(reason='waiting_for_rectification_valid_mask')
            return
        if mask is not None and mask.shape != depth.shape:
            self.reset_invalid()
            self.publish_invalid(reason='rectification_mask_depth_size_mismatch')
            return

        cloud_msg, dt = self.closest_cloud(
            self.projected_cloud_buffer,
            msg,
        )

        estimated_msg, estimated_dt = self.closest_cloud(
            self.estimated_cloud_buffer,
            msg,
        )

        if cloud_msg is None:
            self.publish_lidar_dropout('waiting_for_projected_lidar')
            return
        if estimated_msg is None:
            return

        if (
            self.max_projection_time_difference > 0.0
            and dt > self.max_projection_time_difference
        ):

            self.publish_lidar_dropout('projected_cloud_time_mismatch', dt)
            return

        if (
            self.max_estimated_time_difference > 0.0
            and estimated_dt > self.max_estimated_time_difference
        ):

            self.reset_invalid()

            self.publish_invalid(
                reason='estimated_cloud_time_mismatch',
                time_difference=estimated_dt,
            )

            return

        projected_points = self.read_projected_points(
            cloud_msg,
            depth.shape,
        )

        estimated_points = self.read_estimated_points(
            estimated_msg,
            depth.shape,
        )

        if projected_points is None:

            self.publish_lidar_dropout('invalid_projected_cloud', dt)
            return

        if estimated_points is None:

            self.reset_invalid()

            self.publish_invalid(
                reason='invalid_estimated_cloud',
                time_difference=estimated_dt,
            )

            return

        self._pending_curve_plot = None
        self._pending_spatial_irregularity_input = None
        result = self.compare_relative_depth(
            depth,
            projected_points,
            estimated_points,
        )

        result['estimated_time_difference'] = float(estimated_dt)
        result['projection_time_difference'] = float(dt)
        # NCC votes in motion too, with a softer threshold and scan-based persistence.
        allow_baseline_learning = bool(result.get('check_valid', False) and
                                       not self._motion_info['motion_active'])
        result = self.add_motion_diagnostics(result)
        good_depth = np.isfinite(depth) & (depth > 0)
        if mask is not None:
            good_depth &= mask
            # Denominator uses valid rectification area only; excluded borders
            # must never create a false invalid-depth anomaly.
            valid_fraction = float(np.count_nonzero(good_depth) /
                                   max(1, np.count_nonzero(mask)))
        else:
            valid_fraction = float(np.mean(good_depth))
        depth_std = float(np.std(depth[good_depth])) if np.any(good_depth) else 0.0
        lidar_z = np.asarray([point['camera_depth'] for point in projected_points
                              if point.get('camera_depth', 0.0) > 0], dtype=float)
        lidar_cv = (float(np.std(lidar_z) / max(np.mean(lidar_z), 1e-6))
                    if lidar_z.size >= 20 else 0.0)
        flat_depth_bad = bool(valid_fraction >= self.rgb_depth_min_valid_fraction
                              and depth_std < self.rgb_depth_min_std
                              and lidar_cv >= self.rgb_flat_lidar_min_cv)
        rgb_invalid = bool(self.rgb_depth_invalid_alarm_enabled and
                           (valid_fraction < self.rgb_depth_min_valid_fraction or
                            flat_depth_bad))
        result['rgb_depth_valid_fraction'] = valid_fraction
        result['rgb_depth_std'] = depth_std
        result['lidar_depth_coefficient_of_variation'] = lidar_cv
        result['rgb_depth_flat_bad'] = flat_depth_bad
        result['rgb_depth_invalid_bad'] = rgb_invalid
        # A completely invalid RGB-derived depth map must not be considered
        # healthy merely because the projected LiDAR is present.
        if rgb_invalid:
            result['rgb_depth_invalid_reason'] = ('depth_profile_collapsed' if flat_depth_bad else
                                                   'too_few_positive_finite_depth_pixels')
        # During motion, do not interpret non-synchronous peak geometry as a
        # sensor failure. NCC and availability checks remain enabled.
        if (self._motion_info.get('motion_active') and
                self.peak_max_motion_projection_dt_s > 0 and
                max(float(dt), float(estimated_dt)) >
                self.peak_max_motion_projection_dt_s):
            result['peak_bad'] = False
            result['peak_valid'] = False
            result['peak_reason'] = 'motion_projection_time_mismatch'
        lidar_mask = projected_lidar_mask(projected_points, depth.shape[1])
        availability = self.coverage_monitor.observe(
            lidar_mask, learn=allow_baseline_learning)
        availability = self.decision_adjusted_coverage(availability)
        availability = self.motion_adjusted_coverage(availability)

        # Use original scan validity at fixed camera-facing beam indices, not
        # surviving projected pixel columns. Keep the original column evidence
        # intact so an unavailable return check never suppresses coverage.
        if self.return_coverage_enabled:
            raw_scan = self.matching_return_scan(cloud_msg)
            if raw_scan is not None:
                source_msg = estimated_msg
                expected_indices = self.projector_scan_indices(source_msg)
                if not expected_indices:
                    expected_indices = self.projector_scan_indices(cloud_msg)
                stamp_key_ret = (int(cloud_msg.header.stamp.sec),
                                 int(cloud_msg.header.stamp.nanosec))
                return_evidence = self.return_coverage_monitor.observe(
                    raw_scan, expected_indices=expected_indices,
                    can_learn=allow_baseline_learning and not
                        bool(availability.get('bad', False)),
                    moving=bool(self._motion_info.get('motion_active', False)),
                    stamp_key=stamp_key_ret)
            else:
                return_evidence = self.return_coverage_monitor._unavailable(
                    'no_matching_raw_scan')
        else:
            return_evidence = self.return_coverage_monitor._unavailable('disabled')
        result.update(return_evidence)
        availability = dict(availability)
        availability.update(return_evidence)
        # Merging is OR, never an average: severe return dropout alone should
        # count even if enough points still occupy most image columns.
        if return_evidence.get('return_coverage_valid', False) and return_evidence.get('return_coverage_bad', False):
            availability['bad'] = True
            availability['global_bad'] = True
            availability['return_availability_bad'] = True
        else:
            availability['return_availability_bad'] = False

        # Evaluate current projected beam bins. No baseline learning.
        stamp = cloud_msg.header.stamp
        stamp_key = (int(stamp.sec), int(stamp.nanosec))
        moving_now = bool(self._motion_info.get('motion_active', False))
        turning_now = bool(self._motion_info.get('motion_turning', False))
        if self.spatial_irregularity_enabled and self._pending_spatial_irregularity_input is not None:
            ax, al, ad = self._pending_spatial_irregularity_input
            result.update(self.spatial_irregularity_monitor.observe(
                ax, al, ad, stamp_key=stamp_key,
                moving=moving_now, turning=turning_now))
        else:
            result.update(self.spatial_irregularity_monitor.unavailable(
                'disabled' if not self.spatial_irregularity_enabled else 'no_profile'))
        result['spatial_irregularity_enabled'] = self.spatial_irregularity_enabled
        result['spatial_irregularity_vote_enabled'] = self.spatial_irregularity_vote_enabled
        self._pending_spatial_irregularity_input = None
        result = self.fuse_and_vote(result, availability, scan_msg=cloud_msg)

        if self._pending_curve_plot is not None:
            self._plot_evidence_result = result
            curve_args, curve_kwargs = self._pending_curve_plot
            self.publish_curve_visualizations(
                *curve_args, **curve_kwargs,
                decision_valid=bool(result.get('check_valid', False)),
                decision_anomaly=bool(result.get('anomaly', False)))
            self._pending_curve_plot = None

        self.publish_result(
            result,
            dt,
        )

    # ========================================================
    # Timestamp helpers
    # ========================================================

    @staticmethod
    def stamp_seconds(
        stamp,
    ):

        return (
            float(stamp.sec)
            + float(stamp.nanosec)
            * 1e-9
        )

    def closest_cloud(
        self,
        cloud_buffer,
        depth_msg,
    ):

        if not cloud_buffer:
            return None, np.inf

        depth_time = self.stamp_seconds(
            depth_msg.header.stamp
        )

        best_cloud = None
        best_dt = np.inf

        for cloud_msg in cloud_buffer:

            cloud_time = self.stamp_seconds(
                cloud_msg.header.stamp
            )

            dt = abs(
                cloud_time
                - depth_time
            )

            if dt < best_dt:

                best_dt = dt
                best_cloud = cloud_msg

        return (
            best_cloud,
            float(best_dt),
        )

    # ========================================================
    # Read already-projected LiDAR cloud
    #
    # The lidar_rgb_native node is responsible for:
    #
    #   LaserScan -> TF -> camera optical frame -> image pixel
    #
    # This node therefore does NOT repeat camera projection,
    # TF lookup, distortion handling, or extrinsic correction.
    #
    # Required fields:
    #
    #   u
    #   v
    #   camera_z
    #
    # camera_z is used rather than radial "range", because the
    # image depth map corresponds to camera-axis depth.
    # ========================================================

    def read_projected_points(
        self,
        cloud_msg,
        depth_shape,
    ):

        available_fields = {
            field.name
            for field in cloud_msg.fields
        }

        required_fields = {
            'u',
            'v',
            'camera_z',
        }

        missing = (
            required_fields
            - available_fields
        )

        if missing:

            if not self.warned_missing_cloud_fields:

                self.get_logger().error(
                    'Projected LiDAR cloud is missing '
                    f'{sorted(missing)}. '
                    f'Available fields: '
                    f'{sorted(available_fields)}'
                )

                self.warned_missing_cloud_fields = True

            return None

        try:

            cloud_points = point_cloud2.read_points(
                cloud_msg,
                field_names=(
                    'u',
                    'v',
                    'camera_z',
                ),
                skip_nans=True,
            )

        except Exception as exc:

            self.get_logger().error(
                'Could not read projected LiDAR cloud: '
                f'{exc}'
            )

            return None

        depth_h, depth_w = depth_shape

        scale_x = 1.0
        scale_y = 1.0

        if self.projection_width > 0:

            scale_x = (
                float(depth_w)
                / float(self.projection_width)
            )

        if self.projection_height > 0:

            scale_y = (
                float(depth_h)
                / float(self.projection_height)
            )

        result = []

        for point in cloud_points:

            u_projection = float(
                point[0]
            )

            v_projection = float(
                point[1]
            )

            camera_depth = float(
                point[2]
            )

            if not (
                np.isfinite(u_projection)
                and np.isfinite(v_projection)
                and np.isfinite(camera_depth)
                and camera_depth > 1e-6
            ):
                continue

            # Coordinates in the depth-image grid.
            u_depth_float = (
                u_projection
                * scale_x
            )

            v_depth_float = (
                v_projection
                * scale_y
            )

            u_depth = int(
                round(u_depth_float)
            )

            v_depth = int(
                round(v_depth_float)
            )

            if not (
                0 <= u_depth < depth_w
                and 0 <= v_depth < depth_h
            ) or not self.safe_pixel(u_depth_float, v_depth_float):
                continue

            result.append(
                {
                    'u': u_projection,
                    'v': v_projection,
                    'depth_u': u_depth,
                    'depth_v': v_depth,
                    'depth_u_float': float(u_depth_float),
                    'depth_v_float': float(v_depth_float),
                    'camera_depth': camera_depth,
                }
            )

        return result

    # ========================================================
    # Read estimated image-space LiDAR path
    #
    # Only u and v are required.  The estimated cloud is NOT
    # matched beam-to-beam with the real LiDAR cloud.  Instead,
    # it defines a scene-dependent v(u) curve used to sample the
    # relative-depth image on a regular horizontal x grid.
    # ========================================================

    def read_estimated_points(
        self,
        cloud_msg,
        depth_shape,
    ):

        available_fields = {
            field.name
            for field in cloud_msg.fields
        }

        required_fields = {
            'u',
            'v',
        }

        missing = (
            required_fields
            - available_fields
        )

        if missing:

            if not self.warned_missing_estimated_fields:

                self.get_logger().error(
                    'Estimated projection cloud is missing '
                    f'{sorted(missing)}. '
                    f'Available fields: '
                    f'{sorted(available_fields)}'
                )

                self.warned_missing_estimated_fields = True

            return None

        try:

            cloud_points = point_cloud2.read_points(
                cloud_msg,
                field_names=(
                    'u',
                    'v',
                ),
                skip_nans=True,
            )

        except Exception as exc:

            self.get_logger().error(
                'Could not read estimated projection cloud: '
                f'{exc}'
            )

            return None

        depth_h, depth_w = depth_shape

        scale_x = 1.0
        scale_y = 1.0

        if self.projection_width > 0:
            scale_x = (
                float(depth_w)
                / float(self.projection_width)
            )

        if self.projection_height > 0:
            scale_y = (
                float(depth_h)
                / float(self.projection_height)
            )

        result = []

        for point in cloud_points:

            u_projection = float(point[0])
            v_projection = float(point[1])

            if not (
                np.isfinite(u_projection)
                and np.isfinite(v_projection)
            ):
                continue

            u_depth = float(
                u_projection * scale_x
            )

            v_depth = float(
                v_projection * scale_y
            )

            if not (
                0.0 <= u_depth < float(depth_w)
                and 0.0 <= v_depth < float(depth_h)
            ) or not self.safe_pixel(u_depth, v_depth):
                continue

            result.append(
                {
                    'u': u_projection,
                    'v': v_projection,
                    'depth_u': u_depth,
                    'depth_v': v_depth,
                }
            )

        return result

    # ========================================================
    # Median depth around projected pixel
    # ========================================================

    def depth_at_pixel(
        self,
        depth,
        u,
        v,
    ):

        h, w = depth.shape

        r = self.depth_patch_radius

        x0 = max(
            0,
            u - r,
        )

        x1 = min(
            w,
            u + r + 1,
        )

        y0 = max(
            0,
            v - r,
        )

        y1 = min(
            h,
            v + r + 1,
        )

        if not self.safe_pixel(u, v):
            return None
        patch = depth[y0:y1, x0:x1]
        valid_patch = np.isfinite(patch) & (patch > 0.0)
        if self.rectification_valid_mask is not None:
            valid_patch &= self.rectification_valid_mask[y0:y1, x0:x1]
        valid = patch[valid_patch]

        if valid.size == 0:
            return None

        return float(
            np.median(valid)
        )

    # ========================================================
    # Profile normalization
    # ========================================================

    @staticmethod
    def robust_standardize_profile(
        values,
    ):

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        if (
            values.size == 0
            or not np.all(
                np.isfinite(values)
            )
        ):
            return None

        center = float(
            np.median(values)
        )

        q25, q75 = np.percentile(
            values,
            [25.0, 75.0],
        )

        scale = float(
            q75 - q25
        )

        if scale <= 1e-8:

            scale = float(
                np.std(values)
            )

        if scale <= 1e-8:
            return None

        return (
            values - center
        ) / scale

    # ========================================================
    # Smooth profile
    # ========================================================

    @staticmethod
    def smooth_profile(
        values,
        window=5,
    ):

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        window = int(window)

        if (
            values.size < 3
            or window <= 1
        ):
            return values.copy()

        if window % 2 == 0:
            window += 1

        max_window = (
            values.size
            if values.size % 2 == 1
            else values.size - 1
        )

        window = min(
            window,
            max_window,
        )

        if window < 3:
            return values.copy()

        pad = (
            window // 2
        )

        padded = np.pad(
            values,
            (pad, pad),
            mode='edge',
        )

        kernel = (
            np.ones(
                window,
                dtype=np.float64,
            )
            / float(window)
        )

        return np.convolve(
            padded,
            kernel,
            mode='valid',
        )

    # ========================================================
    # Temporal smoothing of the two modality profiles
    #
    # Input profiles have already been spatially smoothed.
    #
    # EMA:
    #
    #   temporal =
    #       alpha * current
    #       + (1-alpha) * previous_temporal
    #
    # Because the available horizontal LiDAR support may change
    # slightly from frame to frame, the previous temporal profile
    # is interpolated onto the current u grid. No extrapolated old
    # values are used outside the previous frame's u range.
    # ========================================================

    def temporal_smooth_profiles(
        self,
        u_values,
        lidar_profile,
        depth_profile,
    ):

        u_values = np.asarray(
            u_values,
            dtype=np.float64,
        )

        lidar_profile = np.asarray(
            lidar_profile,
            dtype=np.float64,
        )

        depth_profile = np.asarray(
            depth_profile,
            dtype=np.float64,
        )

        if (
            not self.temporal_smoothing_enabled
            or self.temporal_smoothing_alpha >= 1.0
        ):

            lidar_output = lidar_profile.copy()
            depth_output = depth_profile.copy()

            # Keep the most recent profiles so temporal smoothing
            # can be enabled dynamically without stale state.
            self.temporal_u = u_values.copy()
            self.temporal_lidar_profile = (
                lidar_output.copy()
            )
            self.temporal_depth_profile = (
                depth_output.copy()
            )

            return (
                lidar_output,
                depth_output,
            )

        # First valid frame: initialize the EMA state.
        if (
            self.temporal_u is None
            or self.temporal_lidar_profile is None
            or self.temporal_depth_profile is None
            or len(self.temporal_u) < 2
        ):

            self.temporal_u = u_values.copy()
            self.temporal_lidar_profile = (
                lidar_profile.copy()
            )
            self.temporal_depth_profile = (
                depth_profile.copy()
            )

            return (
                lidar_profile.copy(),
                depth_profile.copy(),
            )

        alpha = float(
            self.temporal_smoothing_alpha
        )

        lidar_output = lidar_profile.copy()
        depth_output = depth_profile.copy()

        previous_u = np.asarray(
            self.temporal_u,
            dtype=np.float64,
        )

        previous_lidar = np.asarray(
            self.temporal_lidar_profile,
            dtype=np.float64,
        )

        previous_depth = np.asarray(
            self.temporal_depth_profile,
            dtype=np.float64,
        )

        previous_min = float(
            np.min(
                previous_u
            )
        )

        previous_max = float(
            np.max(
                previous_u
            )
        )

        overlap = (
            (u_values >= previous_min)
            & (u_values <= previous_max)
        )

        if np.any(overlap):

            current_overlap_u = (
                u_values[
                    overlap
                ]
            )

            previous_lidar_interp = np.interp(
                current_overlap_u,
                previous_u,
                previous_lidar,
            )

            previous_depth_interp = np.interp(
                current_overlap_u,
                previous_u,
                previous_depth,
            )

            lidar_output[
                overlap
            ] = (
                alpha
                * lidar_profile[
                    overlap
                ]
                + (
                    1.0 - alpha
                )
                * previous_lidar_interp
            )

            depth_output[
                overlap
            ] = (
                alpha
                * depth_profile[
                    overlap
                ]
                + (
                    1.0 - alpha
                )
                * previous_depth_interp
            )

        # Save the NEW temporally smoothed profile as the state
        # used by the next frame.
        self.temporal_u = u_values.copy()

        self.temporal_lidar_profile = (
            lidar_output.copy()
        )

        self.temporal_depth_profile = (
            depth_output.copy()
        )

        return (
            lidar_output,
            depth_output,
        )

    def reset_temporal_smoothing(
        self,
    ):

        self.temporal_u = None
        self.temporal_lidar_profile = None
        self.temporal_depth_profile = None
        self.support_plot_ema = {}

    # ========================================================
    # Pearson correlation
    # ========================================================

    @staticmethod
    def pearson_safe(a, b):
        a=np.asarray(a,dtype=np.float64)
        b=np.asarray(b,dtype=np.float64)
        if a.shape != b.shape: return np.nan
        good=np.isfinite(a)&np.isfinite(b)
        if int(good.sum()) < 3: return np.nan
        aa,bb=a[good],b[good]
        if np.std(aa)<1e-10 or np.std(bb)<1e-10: return np.nan
        return float(np.corrcoef(aa,bb)[0,1])

    # ========================================================
    # Weighted Pearson correlation
    # ========================================================

    @staticmethod
    def weighted_pearson_safe(
        a,
        b,
        weights,
    ):

        a = np.asarray(
            a,
            dtype=np.float64,
        )

        b = np.asarray(
            b,
            dtype=np.float64,
        )

        weights = np.asarray(
            weights,
            dtype=np.float64,
        )

        if (
            a.size < 2
            or b.size != a.size
            or weights.size != a.size
        ):
            return np.nan

        valid = (
            np.isfinite(a)
            & np.isfinite(b)
            & np.isfinite(weights)
            & (weights > 0.0)
        )

        a = a[valid]
        b = b[valid]
        weights = weights[valid]

        if a.size < 2:
            return np.nan

        weight_sum = float(
            np.sum(weights)
        )

        if weight_sum <= 0.0:
            return np.nan

        weights = (
            weights
            / weight_sum
        )

        mean_a = float(
            np.sum(
                weights * a
            )
        )

        mean_b = float(
            np.sum(
                weights * b
            )
        )

        da = a - mean_a
        db = b - mean_b

        var_a = float(
            np.sum(
                weights * da * da
            )
        )

        var_b = float(
            np.sum(
                weights * db * db
            )
        )

        if (
            var_a <= 1e-20
            or var_b <= 1e-20
        ):
            return np.nan

        covariance = float(
            np.sum(
                weights * da * db
            )
        )

        return (
            covariance
            / np.sqrt(
                var_a * var_b
            )
        )

    # ========================================================
    # ROI test
    # ========================================================

    def point_in_person_roi(
        self,
        u,
        v,
    ):

        if not self.use_person_roi_weighting:
            return False

        for (
            x1,
            y1,
            x2,
            y2,
        ) in self.person_boxes:

            if (
                x1 <= u <= x2
                and y1 <= v <= y2
            ):
                return True

        return False

    # ========================================================
    # Relative depth / LiDAR anomaly detector
    # ========================================================

    def compare_relative_depth(
        self,
        depth,
        projected_points,
        estimated_points,
    ):

        # ====================================================
        # Build TWO independent image-x descriptions:
        #
        # 1) Real LiDAR signal:
        #      measured camera_z values as a function of image x
        #
        # 2) Estimated sampling path:
        #      estimated projection v as a function of image x
        #
        # There is intentionally NO beam / scan_index matching.
        # Both are resampled onto the same regular x grid.
        # ====================================================

        step = float(
            self.x_sampling_step_px
        )

        # ----------------------------------------------------
        # Real LiDAR inverse-depth profile.
        # Aggregate points falling in the same horizontal bin.
        # ----------------------------------------------------

        lidar_bins = {}

        for point in projected_points:

            u = float(
                point['depth_u_float']
            )

            camera_depth = float(
                point['camera_depth']
            )

            if (
                not np.isfinite(u)
                or not np.isfinite(camera_depth)
                or camera_depth <= 1e-6
            ):
                continue

            bin_index = int(
                np.floor(u / step)
            )

            lidar_bins.setdefault(
                bin_index,
                [],
            ).append(
                1.0 / camera_depth
            )

        # ----------------------------------------------------
        # Estimated image path v(u).
        # Aggregate estimated pixels falling in the same x bin.
        # ----------------------------------------------------

        path_bins = {}

        for point in estimated_points:

            u = float(point['depth_u'])
            v = float(point['depth_v'])

            if not (
                np.isfinite(u)
                and np.isfinite(v)
            ):
                continue

            bin_index = int(
                np.floor(u / step)
            )

            path_bins.setdefault(
                bin_index,
                [],
            ).append(v)

        if (
            len(lidar_bins) < 2
            or len(path_bins) < 2
        ):

            # Voting is finalized using NCC + coverage in depth_callback.

            return {
                'check_valid': False,
                'reason': 'too_few_x_bins',
                'count': 0,
                'roi_count': 0,
                'global_shape_similarity': None,
                'weighted_shape_similarity': None,
                'global_bad': False,
                'weighted_bad': False,
                'frame_bad': False,
                'anomaly': False,
                'bad_frames_in_window': 0,
                'history_length': 0,
            }

        lidar_bin_ids = np.asarray(
            sorted(lidar_bins.keys()),
            dtype=np.int64,
        )

        path_bin_ids = np.asarray(
            sorted(path_bins.keys()),
            dtype=np.int64,
        )

        lidar_u = (
            lidar_bin_ids.astype(np.float64)
            + 0.5
        ) * step

        lidar_inverse = np.asarray(
            [
                np.median(
                    lidar_bins[int(bin_id)]
                )
                for bin_id in lidar_bin_ids
            ],
            dtype=np.float64,
        )

        path_u = (
            path_bin_ids.astype(np.float64)
            + 0.5
        ) * step

        path_v = np.asarray(
            [
                np.median(
                    path_bins[int(bin_id)]
                )
                for bin_id in path_bin_ids
            ],
            dtype=np.float64,
        )

        # Retain EVERY received, geometrically valid magenta point for the
        # depth profile visualization and coverage diagnostics. Sampling is
        # at the ORIGINAL estimated-projection bins, not interpolated over
        # gaps and not restricted by LiDAR support, NCC or edge trimming.
        full_estimated_depth_raw = np.asarray([
            (np.nan if sample is None else float(sample))
            for sample in (
                self.depth_at_pixel(depth, int(round(px)), int(round(py)))
                for px, py in zip(path_u, path_v)
            )
        ], dtype=np.float64)
        full_estimated_valid_count = int(np.count_nonzero(
            np.isfinite(full_estimated_depth_raw)))

        # ----------------------------------------------------
        # Common horizontal support.
        # ----------------------------------------------------

        support_min = max(
            float(np.min(lidar_u)),
            float(np.min(path_u)),
        )

        support_max = min(
            float(np.max(lidar_u)),
            float(np.max(path_u)),
        )

        if support_max <= support_min:

            # Voting is finalized using NCC + coverage in depth_callback.

            return {
                'check_valid': False,
                'reason': 'no_x_overlap',
                'count': 0,
                'roi_count': 0,
                'global_shape_similarity': None,
                'weighted_shape_similarity': None,
                'global_bad': False,
                'weighted_bad': False,
                'frame_bad': False,
                'anomaly': False,
                'bad_frames_in_window': 0,
                'history_length': 0,
            }

        first_sample = (
            np.ceil(
                support_min / step
            ) * step
        )

        last_sample = (
            np.floor(
                support_max / step
            ) * step
        )

        u_regular = np.arange(
            first_sample,
            last_sample + 0.5 * step,
            step,
            dtype=np.float64,
        )

        if len(u_regular) < self.min_points:

            # Voting is finalized using NCC + coverage in depth_callback.

            return {
                'check_valid': False,
                'reason': 'too_few_regular_points',
                'count': int(len(u_regular)),
                'roi_count': 0,
                'global_shape_similarity': None,
                'weighted_shape_similarity': None,
                'global_bad': False,
                'weighted_bad': False,
                'frame_bad': False,
                'anomaly': False,
                'bad_frames_in_window': 0,
                'history_length': 0,
            }

        # ----------------------------------------------------
        # Interpolate the real LiDAR signal and the estimated
        # vertical sampling path onto exactly the same x grid.
        # ----------------------------------------------------

        lidar_regular_raw = np.interp(
            u_regular,
            lidar_u,
            lidar_inverse,
        )

        v_regular = np.interp(
            u_regular,
            path_u,
            path_v,
        )

        # ----------------------------------------------------
        # Sample the relative-depth image at each regular x bin
        # using the estimated vertical coordinate v(u).
        # ----------------------------------------------------

        relative_samples = []
        valid_mask = []

        for u, v in zip(
            u_regular,
            v_regular,
        ):

            relative_depth = self.depth_at_pixel(
                depth,
                int(round(u)),
                int(round(v)),
            )

            is_valid = bool(
                relative_depth is not None
                and np.isfinite(relative_depth)
            )

            valid_mask.append(is_valid)

            relative_samples.append(
                float(relative_depth)
                if is_valid
                else np.nan
            )

        valid_mask = np.asarray(
            valid_mask,
            dtype=bool,
        )

        relative_samples = np.asarray(
            relative_samples,
            dtype=np.float64,
        )

        # Keep geometric holes as holes, not interpolated plausible values.
        # Both signals must be invalid at any regular-grid sample that lies
        # outside the rectification support; no NCC / peak evidence there.
        if self.rectification_valid_mask is not None:
            keep = np.asarray([self.safe_pixel(u, v)
                               for u, v in zip(u_regular, v_regular)], dtype=bool)
            lidar_regular_raw[~keep] = np.nan
            relative_samples[~keep] = np.nan

        # Keep geometry-valid, unfiltered observations for DIAGNOSTICS.
        # These contain LiDAR interpolations that may be unreliable, which
        # will be dotted rather than silently erased from plot images.
        full_lidar_for_plot = lidar_regular_raw.copy()
        full_rgb_for_plot = relative_samples.copy()
        metric_support = np.ones(u_regular.shape, dtype=bool)

        # Sensor-support-aware mask for the METRIC evaluation only.
        # Ordinary interpolation across missing/irregular LiDAR returns can
        # turn one intermittent thin-object hit into a long artificial peak.
        support_diag = {}
        if self.profile_support_filter_enabled:
            metric_support, support_diag = comparison_support_mask(
                u_regular, lidar_u, lidar_inverse,
                max_interp_gap_px=self.profile_max_interp_gap_px,
                narrow_object_width_px=self.profile_narrow_object_width_px,
                narrow_object_jump_iqr=self.profile_narrow_object_jump_iqr,
                narrow_object_margin_px=self.profile_narrow_object_margin_px)
            lidar_regular_raw[~metric_support] = np.nan
            relative_samples[~metric_support] = np.nan

        # Use the SAME complete horizontal x grid for both sensors.
        # Normalize each source independently: a faulty/deleted depth region
        # must not change healthy LiDAR samples or LiDAR plot scaling.
        relative_regular_raw = relative_samples.copy()
        lidar_regular = normalized_with_holes(lidar_regular_raw)
        relative_regular = normalized_with_holes(relative_regular_raw)

        # For excluded pixels, show the actual geometry-valid pre-mask
        # profiles, normalized using the same accepted-sample scale.
        # Their curves do NOT feed NCC, peak detection, voting or coverage.
        if self.profile_support_filter_enabled and self.visualize_curves:
            full_lidar_for_plot = normalize_using_reference(
                full_lidar_for_plot, lidar_regular_raw)
            full_rgb_for_plot = normalize_using_reference(
                full_rgb_for_plot, relative_regular_raw)
            full_lidar_for_plot = smooth_with_holes(
                full_lidar_for_plot, self.spatial_smoothing_window)
            full_rgb_for_plot = smooth_with_holes(
                full_rgb_for_plot, self.spatial_smoothing_window)
            full_lidar_for_plot, full_rgb_for_plot = fixed_grid_masked_ema(
                u_regular, full_lidar_for_plot, full_rgb_for_plot,
                self.support_plot_ema,
                self.temporal_smoothing_enabled and
                not self._motion_info.get('motion_active', False),
                self.temporal_smoothing_alpha)

        # ====================================================
        # 1) Spatial smoothing
        # ====================================================

        lidar_spatial = smooth_with_holes(
            lidar_regular, self.spatial_smoothing_window)
        relative_spatial = smooth_with_holes(
            relative_regular, self.spatial_smoothing_window)

        # ====================================================
        # 2) Temporal smoothing
        # ====================================================

        temporal_state = {'u': self.temporal_u,
                          'first': self.temporal_lidar_profile,
                          'second': self.temporal_depth_profile}
        lidar_smooth, relative_smooth = fixed_grid_masked_ema(
            u_regular, lidar_spatial, relative_spatial, temporal_state,
            self.temporal_smoothing_enabled and
            not self._motion_info.get('motion_active', False),
            self.temporal_smoothing_alpha)
        self.temporal_u = temporal_state['u']
        self.temporal_lidar_profile = temporal_state['first']
        self.temporal_depth_profile = temporal_state['second']

        # Cut the same left/right percentage from the COMPLETE regular grid
        # for BOTH curves before computing any Pearson or displaying plots.
        # This exclusion is independent of depth-estimation validity.
        trim = min(int(len(u_regular) * self.edge_fraction),
                   max(0, (len(u_regular) - self.min_points) // 2))
        if trim:
            keep = slice(trim, -trim)
            u_regular, v_regular = u_regular[keep], v_regular[keep]
            lidar_smooth, relative_smooth = lidar_smooth[keep], relative_smooth[keep]
            # The irregularity detector uses UNSMOOTHED samples, but MUST use
            # the same selected horizontal domain and support as NCC/peak.
            # Trim BOTH raw modalities and the support mask on the SAME grid.
            lidar_regular_raw = lidar_regular_raw[keep]
            relative_regular_raw = relative_regular_raw[keep]
            metric_support = metric_support[keep]
            if self.profile_support_filter_enabled and self.visualize_curves:
                full_lidar_for_plot = full_lidar_for_plot[keep]
                full_rgb_for_plot = full_rgb_for_plot[keep]

        paired = np.isfinite(lidar_smooth) & np.isfinite(relative_smooth)
        paired_count = int(np.sum(paired))
        paired_coverage = paired_count / max(len(u_regular), 1)
        profile_corr = self.pearson_safe(
            lidar_smooth,
            relative_smooth,
        )
        aligned_corr, best_shift_px, shift_pairs, _ = shift_invariant_pearson(
            u_regular, lidar_smooth, relative_smooth,
            max_shift_px=self.shift_max_px, min_points=self.min_points,
            min_coverage=self.min_paired_coverage,
            min_overlap_ratio=self.shift_min_overlap_ratio,
            min_shift_gain=self.shift_min_gain)

        # Localize shape disagreement WITHOUT changing the four checks or vote rules.
        # Use exactly the supported/smoothed signal and best shift used by NCC.
        rotation_guard = bool(self._motion_info.get('local_ncc_rotation_guard_active', False))
        local_ncc_diag = local_aware_ncc_score(
            u_regular, lidar_smooth, relative_smooth,
            aligned_corr, best_shift_px,
            enabled=self.local_ncc_fusion_enabled,
            regions=self.local_ncc_regions,
            overlap_fraction=self.local_ncc_overlap_fraction,
            require_adjacent_pair=self.local_ncc_require_adjacent_pair,
            focus_gain=(self.local_ncc_rotation_focus_gain if rotation_guard else
                        self.local_ncc_focus_gain),
            min_corr_drop=(self.local_ncc_rotation_min_corr_drop if rotation_guard else
                           self.local_ncc_min_corr_drop),
            residual_enabled=self.local_ncc_residual_enabled,
            min_region_coverage=(self.local_ncc_rotation_min_region_coverage if rotation_guard else
                                 self.local_ncc_min_region_coverage),
            min_region_points=(self.local_ncc_rotation_min_region_points if rotation_guard else
                               self.local_ncc_min_region_points),
            residual_min_median=(self.local_ncc_rotation_residual_min_median if rotation_guard else
                                 self.local_ncc_residual_min_median),
            residual_min_excess=(self.local_ncc_rotation_residual_min_excess if rotation_guard else
                                 self.local_ncc_residual_min_excess),
            min_affected_regions=(self.local_ncc_rotation_min_affected_regions if rotation_guard else 1))
        local_ncc_diag['local_ncc_mode'] = 'rotation_conservative' if rotation_guard else 'standard'
        ncc_decision_score = local_ncc_diag['local_ncc_decision_score']

        # LOCAL MEDIAN irregularity must run on UNSMOOTHED signals (the median
        # is computed inside CrossModalSpatialIrregularity). Keep NCC's SAME
        # selected x-interval, geometric support and horizontal shift. Only
        # actually measured LiDAR bins count; no synthetic LiDAR interpolation
        # is promoted to a measurement. Invalid paired positions remain NaN.
        if self.spatial_irregularity_enabled:
            self._pending_spatial_irregularity_input = select_irregularity_pairs(
                u_regular, lidar_regular_raw, relative_regular_raw,
                lidar_u, lidar_inverse, best_shift_px)

        # A high Pearson coefficient does NOT guarantee equal local peak
        # prominence. Align RGB to the exact shift used by the NCC decision,
        # fit only a global affine calibration, and evaluate local residuals.
        if self._motion_info.get('motion_turning'):
            peak_thr = self.rotation_peak_residual_threshold
            peak_fraction = self.rotation_peak_min_fraction
        elif self._motion_info.get('motion_active'):
            peak_thr = self.motion_peak_residual_threshold
            peak_fraction = self.motion_peak_min_fraction
        else:
            peak_thr = self.peak_residual_threshold
            peak_fraction = self.peak_min_fraction
        if self.peak_mismatch_enabled:
            relative_aligned = interp_with_holes(
                u_regular, relative_smooth,
                u_regular + float(best_shift_px))
            peak_evidence = peak_mismatch_evidence(
                u_regular, lidar_smooth, relative_aligned,
                residual_threshold=peak_thr,
                min_bad_fraction=peak_fraction,
                min_run_px=self.peak_min_run_px,
                min_points=max(self.min_points, 24),
                min_paired_coverage=self.min_paired_coverage)
        else:
            peak_evidence = dict(peak_valid=False, peak_bad=False,
                                 peak_reason='disabled')

        # ====================================================
        # 3) Derivatives on the actual x coordinates.
        # ====================================================

        lidar_slope = finite_gradient(lidar_smooth, u_regular)
        relative_slope = finite_gradient(relative_smooth, u_regular)
        lidar_curvature = finite_gradient(lidar_slope, u_regular)
        relative_curvature = finite_gradient(relative_slope, u_regular)

        slope_corr = self.pearson_safe(
            lidar_slope,
            relative_slope,
        )

        curvature_corr = self.pearson_safe(
            lidar_curvature,
            relative_curvature,
        )

        # The complete RGB-derived sampling path is DISPLAY-ONLY. Keep
        # geometrically valid magenta samples that lie outside LiDAR's x
        # range, the 5% edge trim, or the conservative comparison mask.
        # They are not promoted to paired NCC measurements.
        complete_estimated_for_plot = normalize_full_profile_for_display(
            full_estimated_depth_raw, relative_regular_raw)

        # Additional unfiltered FULL-SUPPORT Pearson is DIAGNOSTIC ONLY.
        # It uses valid, original LiDAR bins and the full magenta sampling
        # path after NCC's chosen shift. Missing sectors cannot be bridged
        # by more than the configured interpolation tolerance. This is not
        # the safety-critical NCC and has no influence on alarm voting.
        all_rgb_at_lidar = interpolate_measured_support(
            path_u, full_estimated_depth_raw,
            lidar_u + float(best_shift_px),
            max_gap_px=float(self.profile_max_interp_gap_px))
        full_diag_lidar = normalized_with_holes(lidar_inverse)
        full_diag_rgb = normalized_with_holes(all_rgb_at_lidar)
        full_pair_count = int(np.count_nonzero(
            np.isfinite(full_diag_lidar) & np.isfinite(full_diag_rgb)))
        full_support_corr = (self.pearson_safe(full_diag_lidar, full_diag_rgb)
                             if full_pair_count >= self.min_points else np.nan)

        # These counters distinguish depth availability from actual pairwise
        # comparison support. A valid magenta point with no LiDAR counterpart
        # is visible, but cannot contribute to Pearson's r.
        matched_estimated_bins = int(np.count_nonzero(
            np.isfinite(interp_with_holes(
                u_regular, lidar_smooth, path_u)) &
            np.isfinite(interp_with_holes(
                u_regular, relative_smooth, path_u)) &
            np.isfinite(full_estimated_depth_raw)))

        self.plot_counter += 1
        # Delay visualization until depth_callback has fused NCC, coverage
        # and optional irregularity evidence through the standard guarded vote.
        if self.visualize_curves and self.plot_counter % self.plot_every == 0:
            self._pending_curve_plot = (
                (u_regular, lidar_smooth, relative_smooth, lidar_slope,
                 relative_slope, lidar_curvature, relative_curvature,
                 profile_corr, slope_corr, curvature_corr),
                dict(aligned_corr=aligned_corr, best_shift_px=best_shift_px,
                     display_support_mask=(metric_support.copy() if
                         self.profile_support_filter_enabled else None),
                     display_lidar_profile=(full_lidar_for_plot.copy() if
                         self.profile_support_filter_enabled else None),
                     display_depth_profile=(full_rgb_for_plot.copy() if
                         self.profile_support_filter_enabled else None),
                     full_estimated_x=path_u.copy(),
                     full_estimated_depth=complete_estimated_for_plot.copy()))

        # ====================================================
        # PRIMARY DECISION: Pearson correlation of the two depth profiles
        # ====================================================

        global_valid = bool(
            paired_count >= self.min_points
            and paired_coverage >= self.min_paired_coverage
            and np.isfinite(aligned_corr)
        )

        if global_valid:
            global_similarity = float(ncc_decision_score)
            global_bad = bool(
                global_similarity < self.similarity_threshold or
                local_ncc_diag.get('local_ncc_pair_bad', False)
            )
        else:
            global_similarity = np.nan
            global_bad = False

        # ----------------------------------------------------
        # Convert sampling coordinates back to projection-image
        # coordinates for optional human ROI weighting.
        # ----------------------------------------------------

        depth_h, depth_w = depth.shape

        roi_scale_x = 1.0
        roi_scale_y = 1.0

        if self.projection_width > 0:
            roi_scale_x = (
                float(self.projection_width)
                / float(depth_w)
            )

        if self.projection_height > 0:
            roi_scale_y = (
                float(self.projection_height)
                / float(depth_h)
            )

        roi_u_regular = (
            u_regular * roi_scale_x
        )

        roi_v_regular = (
            v_regular * roi_scale_y
        )

        if self.use_person_roi_weighting:

            roi_mask = np.asarray(
                [
                    self.point_in_person_roi(
                        int(round(u)),
                        int(round(v)),
                    )
                    for u, v in zip(
                        roi_u_regular,
                        roi_v_regular,
                    )
                ],
                dtype=bool,
            )

            roi_count = int(
                np.sum(roi_mask)
            )

        else:

            roi_mask = np.zeros(
                len(u_regular),
                dtype=bool,
            )

            roi_count = 0

        if (
            self.use_person_roi_weighting
            and roi_count > 0
        ):

            weights = np.ones(
                len(u_regular),
                dtype=np.float64,
            )

            weights[roi_mask] = (
                self.roi_point_weight
            )

            weighted_profile_corr = self.weighted_pearson_safe(
                lidar_smooth, relative_smooth, weights
            )
            # Retain slope/curvature correlations only as diagnostics.
            weighted_slope_corr = self.weighted_pearson_safe(
                lidar_slope, relative_slope, weights
            )

            weighted_curvature_corr = (
                self.weighted_pearson_safe(
                    lidar_curvature,
                    relative_curvature,
                    weights,
                )
            )

            weighted_valid = bool(
                np.isfinite(weighted_profile_corr)
            )

            if weighted_valid:
                weighted_similarity = float(
                    weighted_profile_corr
                )
                weighted_bad = bool(
                    weighted_similarity
                    < self.similarity_threshold
                )
            else:
                weighted_similarity = np.nan
                weighted_bad = False

        else:

            weighted_profile_corr = np.nan
            weighted_slope_corr = np.nan
            weighted_curvature_corr = np.nan
            weighted_similarity = np.nan
            weighted_valid = False
            weighted_bad = False

        # NCC remains primary shape evidence; independent dropout evidence is
        # fused in depth_callback after fixed-width LiDAR coverage evaluation.
        frame_bad = bool(global_bad)
        check_valid = bool(global_valid)
        bad_frames = 0
        history_length = 0
        anomaly = False

        return {
            'check_valid': check_valid,
            'reason': (
                'ok'
                if check_valid
                else 'invalid_correlation'
            ),
            'count': paired_count,
            'paired_coverage': paired_coverage,
            'estimated_projection_total_points': int(len(estimated_points)),
            'estimated_projection_unique_x_bins': int(len(path_u)),
            'estimated_projection_valid_depth_bins': full_estimated_valid_count,
            'estimated_projection_paired_bins': matched_estimated_bins,
            'estimated_projection_unpaired_bins': max(
                0, full_estimated_valid_count - matched_estimated_bins),
            'full_support_paired_bin_count': int(full_pair_count),
            'full_support_profile_pearson_raw_diagnostic': self.finite_or_none(
                full_support_corr),
            'full_support_profile_votes_enabled': False,
            'profile_support_filter_enabled': self.profile_support_filter_enabled,
            'profile_support_diagnostics': support_diag,
            'edge_fraction': self.edge_fraction,
            'roi_count': roi_count,
            'roi_weighting_enabled':
                self.use_person_roi_weighting,
            'x_sampling_step_px':
                self.x_sampling_step_px,
            'spatial_smoothing_window':
                self.spatial_smoothing_window,
            'temporal_smoothing_enabled':
                self.temporal_smoothing_enabled,
            'temporal_smoothing_alpha':
                self.temporal_smoothing_alpha,
            'profile_corr':
                self.finite_or_none(profile_corr),
            'slope_corr':
                self.finite_or_none(slope_corr),
            'curvature_corr':
                self.finite_or_none(curvature_corr),
            'global_shape_similarity':
                self.finite_or_none(global_similarity),
            # True Pearson remains available separately from the fused score.
            'global_shape_similarity_raw': self.finite_or_none(aligned_corr),
            'peak_mismatch_enabled': self.peak_mismatch_enabled,
            **local_ncc_diag,
            'weighted_slope_corr':
                self.finite_or_none(weighted_slope_corr),
            'weighted_curvature_corr':
                self.finite_or_none(weighted_curvature_corr),
            'weighted_shape_similarity':
                self.finite_or_none(weighted_similarity),
            'weighted_profile_corr':
                self.finite_or_none(weighted_profile_corr),
            'aligned_profile_corr': self.finite_or_none(aligned_corr),
            'best_shift_px': int(best_shift_px),
            'shift_paired_count': int(shift_pairs),
            'shift_max_px': self.shift_max_px,
            'shift_min_overlap_ratio': self.shift_min_overlap_ratio,
            'decision_metric': 'aligned_pearson_with_supported_local_shape_score',
            **peak_evidence,
            'global_bad': global_bad,
            'weighted_bad': weighted_bad,
            'frame_bad': frame_bad,
            'bad_frames_in_window': bad_frames,
            'history_length': history_length,
            'anomaly': anomaly,
        }

    # ========================================================
    # Curve visualization
    #
    # Three separate ROS Image topics are published:
    #
    #   /cross_modal_anomaly/profile_plot
    #   /cross_modal_anomaly/slope_plot
    #   /cross_modal_anomaly/curvature_plot
    #
    # profile_plot:
    #   normalized + spatially/temporally smoothed LiDAR/depth
    #   profiles BEFORE differentiation.  profile_corr is shown
    #   only on this visualization.
    #
    # slope_plot:
    #   first derivatives and slope Pearson correlation.
    #
    # curvature_plot:
    #   second derivatives and curvature Pearson correlation.
    # ========================================================

    def publish_missing_curves(self, result):
        """Emit a fresh status image when a dropped source prevents plotting."""
        if (not self.visualize_curves or self.profile_plot_pub is None or
                self.slope_plot_pub is None or self.curvature_plot_pub is None):
            return
        for title, pub in (
            ('PROFILE (LiDAR vs %s)', self.profile_plot_pub),
            ('SLOPE (LiDAR vs %s)', self.slope_plot_pub),
            ('CURVATURE (LiDAR vs %s)', self.curvature_plot_pub),
        ):
            frame = make_unavailable_plot(
                self.plot_width, self.plot_height, title % 'RGB depth', result,
                threshold=self.similarity_threshold,
                votes_required=self.required_bad_frames,
                vote_window=self.anomaly_window,
                show_metric=False)
            if title.startswith('PROFILE ('):
                # The original unavailable plot reserves only a 135px header.
                # Move its empty-state panel down so the four-check strip fits.
                frame[120:, :, :] = 255
                draw_anomaly_check_strip(frame, result)
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (70, 179), (w-25, h-45),
                              (205, 205, 205), 1)
                cv2.putText(frame, 'No aligned profile available for this frame',
                            (92, 226), PLOT_FONT, .6, (100, 100, 100),
                            1, cv2.LINE_AA)
            message = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            if self.latest_depth_header is not None:
                message.header = self.latest_depth_header
            pub.publish(message)

    def publish_curve_visualizations(
        self,
        u_values,
        lidar_profile,
        depth_profile,
        lidar_slope,
        depth_slope,
        lidar_curvature,
        depth_curvature,
        profile_corr,
        slope_corr,
        curvature_corr,
        aligned_corr=None,
        best_shift_px=0,
        decision_valid=False,
        decision_anomaly=False,
        display_support_mask=None,
        display_lidar_profile=None,
        display_depth_profile=None,
        full_estimated_x=None,
        full_estimated_depth=None,
    ):

        if (
            self.profile_plot_pub is None
            or self.slope_plot_pub is None
            or self.curvature_plot_pub is None
        ):
            return


        # Display-only regional diagnostics, from the ORIGINAL profiles.
        # Use the same shaded x regions in all three curve visualizations.
        # Diagnostic highlighting must use the same horizontal registration
        # as the main shift-invariant score, while leaving the displayed raw
        # sensor traces unchanged.  Missing measurements remain NaN.
        aligned_depth_for_regions = (
            interp_with_holes(u_values, depth_profile,
                              np.asarray(u_values, dtype=np.float64) + best_shift_px)
            if best_shift_px else depth_profile)
        problem_regions = profile_problem_regions(
            u_values, lidar_profile, aligned_depth_for_regions,
            num_regions=self.local_ncc_regions,
            overlap_fraction=self.local_ncc_overlap_fraction)

        # Derivative plots use separate full (display-only) traces so excluded
        # regions are shown as dotted slopes/curvatures, not blank holes.
        full_lidar_slope = full_depth_slope = None
        full_lidar_curvature = full_depth_curvature = None
        if display_support_mask is not None:
            full_lidar_slope = finite_gradient(display_lidar_profile, u_values)
            full_depth_slope = finite_gradient(display_depth_profile, u_values)
            full_lidar_curvature = finite_gradient(full_lidar_slope, u_values)
            full_depth_curvature = finite_gradient(full_depth_slope, u_values)

        profile_image = self.render_curve_plot(
            u_values,
            lidar_profile,
            depth_profile,
            title='PROFILE (LiDAR vs RGB depth)',
            correlation=aligned_corr,
            decision_valid=decision_valid, decision_anomaly=decision_anomaly,
            problem_regions=problem_regions,
            support_mask=display_support_mask,
            display_lidar_values=display_lidar_profile,
            display_depth_values=display_depth_profile,
            extra_depth_x=full_estimated_x,
            extra_depth_values=full_estimated_depth,
        )

        slope_image = self.render_curve_plot(
            u_values,
            lidar_slope,
            depth_slope,
            title='SLOPE (LiDAR vs RGB depth)',
            correlation=aligned_corr,
            decision_valid=decision_valid, decision_anomaly=decision_anomaly,
            problem_regions=problem_regions,
            support_mask=display_support_mask,
            display_lidar_values=full_lidar_slope,
            display_depth_values=full_depth_slope,
        )

        curvature_image = self.render_curve_plot(
            u_values,
            lidar_curvature,
            depth_curvature,
            title='CURVATURE (LiDAR vs RGB depth)',
            correlation=aligned_corr,
            decision_valid=decision_valid, decision_anomaly=decision_anomaly,
            problem_regions=problem_regions,
            support_mask=display_support_mask,
            display_lidar_values=full_lidar_curvature,
            display_depth_values=full_depth_curvature,
        )

        profile_msg = self.bridge.cv2_to_imgmsg(
            profile_image,
            encoding='bgr8',
        )

        slope_msg = self.bridge.cv2_to_imgmsg(
            slope_image,
            encoding='bgr8',
        )

        curvature_msg = self.bridge.cv2_to_imgmsg(
            curvature_image,
            encoding='bgr8',
        )

        if self.latest_depth_header is not None:

            profile_msg.header = self.latest_depth_header
            slope_msg.header = self.latest_depth_header
            curvature_msg.header = self.latest_depth_header

        self.profile_plot_pub.publish(
            profile_msg
        )

        self.slope_plot_pub.publish(
            slope_msg
        )

        self.curvature_plot_pub.publish(
            curvature_msg
        )

    def render_curve_plot(
        self,
        x_values,
        lidar_values,
        depth_values,
        title,
        correlation,
        correlation_label='Pearson r',
        problem_regions=None,
        decision_valid=False,
        decision_anomaly=False,
        support_mask=None,
        display_lidar_values=None,
        display_depth_values=None,
        extra_depth_x=None,
        extra_depth_values=None,
    ):

        width = int(
            self.plot_width
        )

        height = int(
            self.plot_height
        )
        if title.startswith('PROFILE ('):
            # Reserve room for the five-check strip without covering the data.
            height = max(height, 360)

        canvas = np.full(
            (
                height,
                width,
                3,
            ),
            255,
            dtype=np.uint8,
        )

        x_values = np.asarray(
            x_values,
            dtype=np.float64,
        )

        lidar_values = np.asarray(
            lidar_values,
            dtype=np.float64,
        )

        depth_values = np.asarray(
            depth_values,
            dtype=np.float64,
        )

        # The two curves have INDEPENDENT masks.  Do not discard LiDAR
        # points merely because the second sensor has missing samples.
        valid_x = np.isfinite(x_values)
        x_values = x_values[valid_x]
        lidar_values = lidar_values[valid_x]
        depth_values = depth_values[valid_x]
        if support_mask is not None:
            support_mask = np.asarray(support_mask, dtype=bool)[valid_x]
            display_lidar_values = np.asarray(
                display_lidar_values, dtype=np.float64)[valid_x]
            display_depth_values = np.asarray(
                display_depth_values, dtype=np.float64)[valid_x]

        left = 70
        right = 25
        top = 179 if title.startswith('PROFILE (') else 135
        bottom = 45

        plot_x0 = left
        plot_x1 = max(
            left + 1,
            width - right,
        )

        plot_y0 = top
        plot_y1 = max(
            top + 1,
            height - bottom,
        )

        # Axes.
        cv2.rectangle(
            canvas,
            (
                plot_x0,
                plot_y0,
            ),
            (
                plot_x1,
                plot_y1,
            ),
            (
                210,
                210,
                210,
            ),
            1,
        )

        cv2.putText(
            canvas,
            title,
            (
                15,
                25,
            ),
            PLOT_FONT,
            PLOT_TITLE_SCALE,
            (
                20,
                20,
                20,
            ),
            2,
            cv2.LINE_AA,
        )

        # Current final decision is computed after NCC, coverage, metric
        # evidence and voting. Do not render an NCC-only label.
        draw_decision_banner(
            canvas, getattr(self, '_plot_evidence_result', {}),
            threshold=float(getattr(self, '_plot_evidence_result', {}).get(
                'decision_ncc_threshold', self.similarity_threshold)),
            votes_required=self.required_bad_frames,
            vote_window=self.anomaly_window,
            show_metric=False)

        draw_sensor_legend(canvas, ('LiDAR', 'RGB depth'))
        if title.startswith('PROFILE ('):
            draw_anomaly_check_strip(
                canvas, getattr(self, '_plot_evidence_result', {}))
        if support_mask is not None and np.any(~support_mask):
            cv2.putText(canvas, 'Dotted: excluded from NCC',
                        (70, 169 if title.startswith('PROFILE (') else 124),
                        PLOT_FONT, 0.38, (85, 85, 85), 1, cv2.LINE_AA)

        if x_values.size < 2:

            cv2.putText(
                canvas,
                'Not enough valid samples',
                (
                    left + 20,
                    top + 40,
                ),
                PLOT_FONT,
                0.55,
                (
                    30,
                    30,
                    30,
                ),
                1,
                cv2.LINE_AA,
            )

            return canvas

        # ====================================================
        # Fixed plot support and fixed Y scale
        #
        # The old visualization recomputed its axis limits from
        # every frame.  That made the curves appear to move/zoom
        # even for very small numerical changes.
        #
        # Here:
        #   1. x support is locked from the first valid plot;
        #   2. y limits are locked from the first valid plot;
        #   3. the current curves are interpolated onto the same
        #      fixed x grid for visualization.
        #
        # np.interp holds the closest edge value when the current
        # support is a few pixels shorter than the locked support.
        # This affects ONLY the displayed diagnostic image.
        # ====================================================

        current_x_min = float(np.min(x_values))
        current_x_max = float(np.max(x_values))
        # Keep the full image-width X axis for the main profile view.
        # NCC still uses the original trimmed pairwise interval.
        if title.startswith('PROFILE (') and self.latest_depth_width > 1:
            current_x_min = 0.0
            current_x_max = float(self.latest_depth_width - 1)

        all_y = np.concatenate(
            (
                lidar_values,
                depth_values,
            )
        )

        if support_mask is not None:
            # Stable display limits must include excluded-but-visible traces.
            all_y = np.concatenate((all_y, display_lidar_values,
                                    display_depth_values))
        # No hidden magenta-point values in Y-axis scaling. Use only the
        # visible curves so the plot scale matches the displayed data.
        finite_all_y = all_y[np.isfinite(all_y)]
        if finite_all_y.size == 0:
            return canvas
        current_y_min = float(np.min(finite_all_y))
        current_y_max = float(np.max(finite_all_y))

        state = self.fixed_plot_state.get(
            title
        )

        if state is None:

            # Lock horizontal curve support to the first valid frame.
            locked_x_min = float(
                np.ceil(
                    current_x_min
                )
            )

            locked_x_max = float(
                np.floor(
                    current_x_max
                )
            )

            if locked_x_max <= locked_x_min:
                locked_x_max = (
                    locked_x_min + 1.0
                )

            # Lock Y limits with a small margin.
            if abs(
                current_y_max
                - current_y_min
            ) <= 1e-12:

                y_pad = max(
                    1e-3,
                    abs(current_y_max) * 0.10,
                )

            else:

                y_pad = (
                    0.15
                    * (
                        current_y_max
                        - current_y_min
                    )
                )

            locked_y_min = (
                current_y_min
                - y_pad
            )

            locked_y_max = (
                current_y_max
                + y_pad
            )

            if (
                locked_y_min <= 0.0
                <= locked_y_max
            ):
                # Make a zero-centred symmetric range when zero is
                # already visible.  This makes slope plots easier
                # to compare visually from frame to frame.
                max_abs = max(
                    abs(locked_y_min),
                    abs(locked_y_max),
                    1e-3,
                )

                locked_y_min = -max_abs
                locked_y_max = max_abs

            state = {
                'x_min': locked_x_min,
                'x_max': locked_x_max,
                'y_min': locked_y_min,
                'y_max': locked_y_max,
            }

            self.fixed_plot_state[
                title
            ] = state

            self.get_logger().info(
                f'Locked {title} plot axes: '
                f'x=[{locked_x_min:.1f}, '
                f'{locked_x_max:.1f}] | '
                f'y=[{locked_y_min:.4f}, '
                f'{locked_y_max:.4f}]'
            )

        x_min = float(
            state['x_min']
        )

        x_max = float(
            state['x_max']
        )

        y_min = float(
            state['y_min']
        )

        y_max = float(
            state['y_max']
        )

        # ----------------------------------------------------
        # Draw both curves on the exact same fixed x grid.
        #
        # This prevents the visible curve from repeatedly
        # entering/leaving the plot from the left/right merely
        # because the available projected u support changed by
        # a few pixels.
        # ----------------------------------------------------

        fixed_x_values = np.arange(
            int(
                np.ceil(
                    x_min
                )
            ),
            int(
                np.floor(
                    x_max
                )
            ) + 1,
            dtype=np.float64,
        )

        if fixed_x_values.size >= 2:

            lidar_values = interp_with_holes(
                x_values, lidar_values, fixed_x_values)
            depth_values = interp_with_holes(
                x_values, depth_values, fixed_x_values)

            if support_mask is not None:
                original_x = x_values
                display_lidar_values = interp_with_holes(
                    original_x, display_lidar_values, fixed_x_values)
                display_depth_values = interp_with_holes(
                    original_x, display_depth_values, fixed_x_values)
                # Interpolation is display-only: if even one endpoint is
                # excluded, its interpolated bin remains excluded.
                support_mask = (np.interp(
                    fixed_x_values, original_x,
                    support_mask.astype(float), left=0.0, right=0.0) > 0.999)
            x_values = fixed_x_values

        if abs(
            x_max - x_min
        ) <= 1e-12:

            x_max = (
                x_min + 1.0
            )

        if abs(
            y_max - y_min
        ) <= 1e-12:

            y_max = (
                y_min + 1.0
            )

        def map_x(value):

            alpha = (
                (
                    float(value)
                    - x_min
                )
                / (
                    x_max
                    - x_min
                )
            )

            alpha = float(
                np.clip(
                    alpha,
                    0.0,
                    1.0,
                )
            )

            return int(
                round(
                    plot_x0
                    + alpha
                    * (
                        plot_x1
                        - plot_x0
                    )
                )
            )

        def map_y(value):

            alpha = (
                (
                    float(value)
                    - y_min
                )
                / (
                    y_max
                    - y_min
                )
            )

            alpha = float(
                np.clip(
                    alpha,
                    0.0,
                    1.0,
                )
            )

            return int(
                round(
                    plot_y1
                    - alpha
                    * (
                        plot_y1
                        - plot_y0
                    )
                )
            )

        # Visual explanation only: locally suspicious/missing x regions.
        # Shading does NOT affect any detection score or Pearson value.
        shade_problem_regions(canvas, problem_regions, map_x,
                              plot_x0, plot_x1, plot_y0, plot_y1)

        # Zero reference line when visible.
        if (
            y_min <= 0.0
            <= y_max
        ):

            zero_y = map_y(
                0.0
            )

            cv2.line(
                canvas,
                (
                    plot_x0,
                    zero_y,
                ),
                (
                    plot_x1,
                    zero_y,
                ),
                (
                    200,
                    200,
                    200,
                ),
                1,
                cv2.LINE_AA,
            )

        # Solid: accepted points used for NCC.
        # Dotted: the original geometry-valid trace over excluded support.
        # Never draw across actual holes (NaN, missing depth, invalid pixels).
        for values, full, color in (
            (lidar_values, display_lidar_values, PLOT_COLOR_LIDAR),
            (depth_values, display_depth_values, PLOT_COLOR_RGB)):
            good = np.isfinite(values)
            if support_mask is not None:
                good &= support_mask
            cuts = np.diff(np.r_[False, good, False].astype(np.int8))
            for low, high in zip(np.flatnonzero(cuts == 1),
                                 np.flatnonzero(cuts == -1)):
                if high - low < 2:
                    continue
                points = np.asarray(
                    [[map_x(x), map_y(y)] for x, y in
                     zip(x_values[low:high], values[low:high])],
                    dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [points], False, color,
                              PLOT_LINE_WIDTH, cv2.LINE_AA)
            if support_mask is not None:
                draw_dotted_region(canvas, x_values, full, ~support_mask,
                                   map_x, map_y, color, spacing_px=8,
                                   radius_px=2)

        # No per-point magenta markers: the plot contains only the
        # RGB-depth (red) and LiDAR (blue) profiles. This visualization
        # choice does not affect the underlying NCC or diagnostic metrics.

        # Axis values.
        cv2.putText(
            canvas,
            f'{x_min:.0f}',
            (
                plot_x0,
                height - 15,
            ),
            PLOT_FONT,
            PLOT_AXIS_SCALE,
            (
                60,
                60,
                60,
            ),
            1,
            cv2.LINE_AA,
        )

        x_max_text = (
            f'{x_max:.0f}'
        )

        cv2.putText(
            canvas,
            x_max_text,
            (
                max(
                    plot_x0,
                    plot_x1 - 45,
                ),
                height - 15,
            ),
            PLOT_FONT,
            PLOT_AXIS_SCALE,
            (
                60,
                60,
                60,
            ),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            'image u [px]',
            (
                max(
                    5,
                    width // 2 - 45,
                ),
                height - 12,
            ),
            PLOT_FONT,
            PLOT_AXIS_SCALE,
            (
                60,
                60,
                60,
            ),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f'{y_max:.3f}',
            (
                5,
                plot_y0 + 5,
            ),
            PLOT_FONT,
            PLOT_AXIS_SCALE,
            (
                60,
                60,
                60,
            ),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f'{y_min:.3f}',
            (
                5,
                plot_y1,
            ),
            PLOT_FONT,
            PLOT_AXIS_SCALE,
            (
                60,
                60,
                60,
            ),
            1,
            cv2.LINE_AA,
        )

        return canvas

    # ========================================================
    # Publishing helpers
    # ========================================================

    @staticmethod
    def finite_or_none(
        value,
    ):

        if value is None:
            return None

        value = float(
            value
        )

        if np.isfinite(value):
            return value

        return None

    def reset_invalid(self):

        self._pending_spatial_irregularity_input = None
        # Do not clear a confirmed alarm because one depth frame/mask/TF
        # comparison is unavailable; no valid healthy scan was observed.
        # Mark an unavailable interval without adding a healthy vote.
        self.vote_guard.observe(valid=False, frame_bad=False)

        # Do not blend a new valid sequence with stale profiles
        # from before an invalid/timestamp-mismatched interval.
        self.reset_temporal_smoothing()

    def publish_invalid(
        self,
        reason,
        time_difference=None,
    ):

        result = {
            'check_valid': False,
            'reason': reason,
            'count': 0,
            'roi_count': 0,
            'roi_weighting_enabled':
                self.use_person_roi_weighting,
            'spatial_smoothing_window':
                self.spatial_smoothing_window,
            'temporal_smoothing_enabled':
                self.temporal_smoothing_enabled,
            'temporal_smoothing_alpha':
                self.temporal_smoothing_alpha,
            'global_shape_similarity': None,
            'weighted_shape_similarity': None,
            'global_bad': False,
            'weighted_bad': False,
            'frame_bad': False,
            'anomaly': False,
            'bad_frames_in_window': 0,
            'history_length': 0,
        }

        result.update(self._motion_info)
        result.update(self.vote_guard.details(False, 0.0))
        self.publish_result(
            result,
            time_difference,
        )
        self.publish_missing_curves(result)

    def publish_result(
        self,
        result,
        time_difference,
    ):

        anomaly_msg = Bool()

        anomaly_msg.data = bool(
            result['anomaly']
        )

        self.anomaly_pub.publish(
            anomaly_msg
        )

        frame_bad_msg = Bool()

        frame_bad_msg.data = bool(
            result['frame_bad']
        )

        self.frame_bad_pub.publish(
            frame_bad_msg
        )

        similarity_msg = Float32()

        similarity = result.get(
            'global_shape_similarity'
        )

        similarity_msg.data = (
            float(similarity)
            if similarity is not None
            else float('nan')
        )

        self.similarity_pub.publish(similarity_msg)
        raw_corr = result.get('profile_corr')
        self.profile_similarity_pub.publish(Float32(
            data=float(raw_corr) if raw_corr is not None else float('nan')))
        self.valid_pub.publish(Bool(data=bool(result.get('check_valid', False))))

        status = dict(
            result
        )
        status['decision_metric'] = result.get('decision_metric','shift_invariant_pearson')
        status['similarity_threshold'] = float(result.get(
            'decision_ncc_threshold', self.similarity_threshold))

        status[
            'projection_depth_dt'
        ] = (
            float(
                time_difference
            )
            if time_difference is not None
            else None
        )

        status_msg = String()

        status_msg.data = json.dumps(
            status,
            allow_nan=False,
        )

        self.status_pub.publish(
            status_msg
        )

        self.frame_counter += 1

        if (
            self.frame_counter
            % self.print_every
            == 0
        ):

            self.get_logger().info(
                'Depth/LiDAR shift-NCC anomaly | '
                f"points={result['count']} | "
                f"roi={result['roi_count']} | "
                f"roi_weighting="
                f"{self.use_person_roi_weighting} | "
                f"spatial_w={self.spatial_smoothing_window} | "
                f"temporal_a="
                f"{self.temporal_smoothing_alpha:.2f} | "
                f"global="
                f"{result.get('global_shape_similarity')} | "
                f"dx={result.get('best_shift_px')}px | "
                f"raw={result.get('profile_corr')} | "
                f"weighted="
                f"{result.get('weighted_shape_similarity')} | "
                f"peak_bad={result.get('peak_bad', False)} | "
                f"peak_fraction={result.get('peak_outlier_fraction')} | "
                f"frame_bad={result['frame_bad']} | "
                f"history="
                f"{result['bad_frames_in_window']}/"
                f"{self.anomaly_window} | "
                f"ANOMALY={result['anomaly']}"
            )


def main(args=None):

    rclpy.init(args=args)

    node = DepthLidarAnomalyNode()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
