#!/usr/bin/env python3
"""Shared, explicit decision-evidence banner for ROS 2 depth anomaly plots.

Only include evidence that ACTUALLY contributes to the fused decision.
Slope/curvature and local affine residuals are diagnostic only; they are not
presented as trigger scores. The NCC is a correlation, not anomaly probability.
"""
import math
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _number(value, fmt='.3f', suffix=''):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 'N/A'
    return (format(f, fmt) + suffix) if math.isfinite(f) else 'N/A'


def _pct(value):
    try:
        return _number(float(value) * 100, '.1f', '%')
    except (TypeError, ValueError):
        return 'N/A'


def decision_lines(result, *, threshold=.60, votes_required=5, vote_window=10,
                   metric_threshold=None, metric_bad_fraction_threshold=None,
                   metric_abs_tolerance_m=None, show_metric=False, show_blur=False, blur_threshold=.30):
    """Return four descriptive, non-truncated rows for one processed frame."""
    r = result or {}
    valid = bool(r.get('check_valid', r.get('valid', False)))
    anomaly = bool(r.get('anomaly', False))
    bad = bool(r.get('frame_bad', False))
    state = 'ANOMALY' if anomaly else ('SUSPECT' if valid and bad else
            ('NORMAL' if valid else 'UNKNOWN'))
    score = r.get('aligned_profile_corr', r.get('global_shape_similarity'))
    ncc = _number(score)
    shape_ok = bool(r.get('shape_valid', r.get('valid', r.get('check_valid', False))))
    shape_flag = 'BAD' if r.get('shape_bad', False) else ('OK' if shape_ok else 'N/A')
    history = int(r.get('bad_frames_in_window') or 0)
    need = max(1, int(votes_required))
    window = max(1, int(vote_window))
    shift = r.get('best_shift_px', 0)
    motion = bool(r.get('motion_active', False))
    ncc_label = (f'[min {_number(r.get("decision_ncc_threshold", threshold), ".2f")}] {shape_flag}' +
                 (' (MOVING)' if motion else ''))
    row1 = (f'NCC {ncc} {ncc_label}'
            f'  |  FRAME {"BAD" if bad else ("OK" if valid else "N/A")}'
            f'  |  VOTES {history}/{window} (need {need})'
            f'  |  {state}')

    evidence = r.get('coverage_evidence') or {}
    cover = []
    for key, e in evidence.items():
        if not isinstance(e, dict):
            continue
        name = {'lidar':'LIDAR', 'rgb_depth':'RGB DEPTH'}.get(str(key), str(key).upper())
        if e.get('valid'):
            ratio = e.get('ratio')
            parts = (f'{name} {_pct(ratio)} ref'
                     f' [min {_pct(e.get("global_threshold_ratio", 0.55))}]')
            bad_regions = e.get('bad_regions') or []
            if bad_regions:
                parts += ' bad regions:' + ','.join(str(int(i)) for i in bad_regions[:8])
            parts += ' BAD' if e.get('bad') else ' OK'
        elif e.get('reason') == 'calibrating':
            parts = f'{name} CAL {e.get("calibration_frames", 0)}/{e.get("required_calibration_frames", "?")}'
        else:
            parts = f'{name} N/A'
        cover.append(parts)
    if not cover:
        cover.append('N/A')
    row2 = 'COVERAGE  |  ' + '  |  '.join(cover)

    if show_metric:
        if r.get('metric_valid', False):
            me = _pct(r.get('metric_median_relative_error'))
            frac = _pct(r.get('metric_outlier_fraction'))
            me_thr = _pct(metric_threshold)
            frac_thr = _pct(metric_bad_fraction_threshold)
            row3 = (f'METRIC  |  median relative {me} [max {me_thr}]'
                    f'  |  outliers {frac} [max {frac_thr}]'
                    f'  |  abs tolerance {_number(metric_abs_tolerance_m, ".2f", "m")}'
                    f'  |  {"BAD" if r.get("metric_bad") else "OK"}')
        else:
            row3 = 'METRIC  |  N/A (insufficient aligned metric pairs)'
    elif show_blur:
        if r.get('rgb_blur_valid'):
            row3 = (f'RGB BLUR  |  sharpness ratio {_number(r.get("rgb_sharpness_ratio"), ".2f")}'
                    f' [min {_number(blur_threshold, ".2f")}]'
                    f'  |  {"BAD" if r.get("rgb_blur_bad") else "OK"}')
        else:
            row3 = 'RGB BLUR  |  N/A (disabled/unavailable/calibrating)'
    elif r.get('peak_valid'):
        row3 = (f'PEAK DIFF | residual P95 {_number(r.get("peak_residual_p95"), ".2f")} IQR'
                f'  |  exceed {_pct(r.get("peak_outlier_fraction"))}'
                f' [min {_pct(r.get("peak_min_fraction"))}]'
                f'  |  run {r.get("peak_longest_run_px", 0)}/{r.get("peak_min_run_px", 0)} px'
                f'  |  {"BAD" if r.get("peak_bad") else "OK"}'
                + ('  (MOVING)' if motion else ''))
    elif motion:
        linear = _number(r.get('motion_linear_speed_mps'), '.3f')
        angular = _number(r.get('motion_angular_speed_radps'), '.3f')
        row3 = (f'MOTION GUARD  |  speed {linear} m/s | yaw {angular} rad/s'
                '  |  NCC + global/regional coverage ACTIVE (relaxed thresholds)')
    else:
        row3 = 'SHAPE  |  profile NCC is the only shape criterion; slope/curvature are diagnostic'

    causes = []
    if r.get('shape_bad'):
        causes.append('NCC')
    if r.get('coverage_bad'):
        causes.append('COVERAGE')
    if r.get('peak_bad'):
        causes.append('PEAK DIFF')
    if r.get('temporal_asymmetry_alarm') or r.get('temporal_asymmetry_confirmed'):
        phase = r.get('temporal_asymmetry_alarm_phase', 'ACTIVE')
        causes.append('TEMPORAL (RECOVERY)' if phase == 'RECOVERING' else
                      'TEMPORAL (HELD)' if phase == 'HELD_UNVERIFIED' else
                      'TEMPORAL')
    if r.get('rgb_depth_invalid_bad'):
        causes.append('RGB DEPTH')
    if show_metric and r.get('metric_bad'):
        causes.append('METRIC')
    if show_blur and r.get('rgb_blur_bad') and r.get('rgb_blur_valid'):
        causes.append('BLUR')
    reason = '+'.join(causes) if causes else ('HISTORICAL VOTES' if anomaly else
                                              ('NONE (MOVING)' if motion else 'NONE'))
    row4 = (f'TRIGGER  {reason}  |  shift {_number(shift, ".0f")} px'
            f'  |  reason {r.get("reason", "ok")}')
    return [row1, row2, row3, row4], state


def draw_decision_banner(canvas, result, *, threshold=.60, votes_required=5,
                         vote_window=10, metric_threshold=None,
                         metric_bad_fraction_threshold=None, metric_abs_tolerance_m=None,
                         show_metric=False,
                         show_blur=False, blur_threshold=.30, start_y=47,
                         line_height=21, font_scale=.45):
    """Render all active decision evidence on every profile/slope/curvature plot.

    Caller should reserve top >= start_y + 4*line_height + 5 for the plot.
    Returns the lines for offline testing.
    """
    lines, state = decision_lines(result, threshold=threshold,
                                 votes_required=votes_required, vote_window=vote_window,
                                 metric_threshold=metric_threshold,
                                 metric_bad_fraction_threshold=metric_bad_fraction_threshold,
                                 metric_abs_tolerance_m=metric_abs_tolerance_m,
                                 show_metric=show_metric, show_blur=show_blur,
                                 blur_threshold=blur_threshold)
    bg = {'ANOMALY': (233, 239, 255), 'SUSPECT': (222, 245, 255),
          'NORMAL': (234, 251, 234), 'UNKNOWN': (240, 240, 240)}[state]
    h, w = canvas.shape[:2]
    y0 = max(29, start_y - 17)
    y1 = min(h - 3, start_y + (len(lines)-1)*line_height + 8)
    if y1 > y0:
        cv2.rectangle(canvas, (9, y0), (w-10, y1), bg, -1)
    for i, line in enumerate(lines):
        # Keep the entire decision visible even when plot_width is reduced.
        scale = float(font_scale)
        while scale > .28 and cv2.getTextSize(line, FONT, scale, 1)[0][0] > w-33:
            scale -= .025
        cv2.putText(canvas, line, (17, int(start_y+i*line_height)), FONT,
                    scale, (32, 32, 32), 1, cv2.LINE_AA)
    return lines


def make_unavailable_plot(width, height, title, result, *, threshold=.60,
                          votes_required=5, vote_window=10,
                          show_metric=False, metric_threshold=None,
                          metric_bad_fraction_threshold=None, metric_abs_tolerance_m=None,
                          show_blur=False, blur_threshold=.30):
    """Fresh per-frame status image when profile sampling cannot form curves.

    Avoid showing a stale previously-published curve with an outdated NCC score
    during missing-data/invalid-input periods.
    """
    import numpy as np
    w, h = max(820, int(width)), max(360, int(height))
    canvas = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(canvas, title, (15, 25), FONT, .65, (20, 20, 20), 2, cv2.LINE_AA)
    draw_decision_banner(canvas, result, threshold=threshold,
                         votes_required=votes_required, vote_window=vote_window,
                         show_metric=show_metric, metric_threshold=metric_threshold,
                         metric_bad_fraction_threshold=metric_bad_fraction_threshold,
                         metric_abs_tolerance_m=metric_abs_tolerance_m,
                         show_blur=show_blur, blur_threshold=blur_threshold)
    cv2.rectangle(canvas, (70, 135), (w - 25, h - 45), (205, 205, 205), 1)
    cv2.putText(canvas, 'No aligned profile available for this frame',
                (92, 205), FONT, .65, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Inspect sensor coverage and timestamp status above.',
                (92, 240), FONT, .51, (110, 110, 110), 1, cv2.LINE_AA)
    return canvas
