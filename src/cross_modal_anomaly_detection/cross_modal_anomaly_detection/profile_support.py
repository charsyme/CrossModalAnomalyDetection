#!/usr/bin/env python3
"""Confidence/support mask for an RGB–2D LiDAR inverse-depth profile.

Conservative preprocessing for metric comparison only. Does NOT change raw
LiDAR points, projector outputs, or the independent dropout coverage monitor.
In particular, never fill a gap from absent LiDAR observations with fictional
measurements: long interpolation gaps are masked from NCC and peak evidence.
"""
import numpy as np


def comparison_support_mask(grid_u, observed_u, observed_inverse_depth, *,
                            max_interp_gap_px=6.0,
                            narrow_object_width_px=10.0,
                            narrow_object_jump_iqr=0.70,
                            narrow_object_margin_px=3.0):
    """Return (reliable-grid-mask, diagnostics).

    A LiDAR measurement at an observed position is eligible even beside a
    long gap. Interpolated positions are eligible only if bracketed by valid
    observations no farther apart than max_interp_gap_px.

    Isolated narrow LiDAR foreground/background excursions are recognized
    when two opposite steps return to approximately the former level within
    narrow_object_width_px. Small neighborhoods about these features are
    excluded from *NCC and peak metrics*, not from the original sensor clouds
    or the LiDAR coverage calculation.

    No temporal information or cross-modal agreement is used. This prevents
    a poor RGB estimate from deciding which LiDAR pixels are trustworthy.
    """
    x = np.asarray(grid_u, dtype=np.float64)
    u = np.asarray(observed_u, dtype=np.float64)
    z = np.asarray(observed_inverse_depth, dtype=np.float64)
    if x.ndim != 1 or u.ndim != 1 or z.shape != u.shape:
        raise ValueError('expected 1D arrays; observed coordinates and values must match')
    good = np.isfinite(u) & np.isfinite(z)
    u, z = u[good], z[good]
    if len(u) < 3:
        return np.zeros(len(x), dtype=bool), {'excluded_interp': len(x),
                                                'excluded_narrow': 0,
                                                'narrow_regions': 0}
    order = np.argsort(u, kind='stable')
    u, z = u[order], z[order]
    # Sample bins in the detector have unique ordered x coordinates.
    indices = np.searchsorted(u, x, side='left')
    right = np.clip(indices, 0, len(u)-1)
    left = np.clip(indices-1, 0, len(u)-1)
    at_measurement = np.isclose(u[right], x, rtol=0.0, atol=1e-5)
    between = (indices > 0) & (indices < len(u))
    gap = u[right] - u[left]
    reliable = (at_measurement | (between & (gap <= float(max_interp_gap_px))))
    reliable &= (x >= u[0]) & (x <= u[-1])
    excluded_gap = int((~reliable).sum())

    # Robust scale of actual measured inverse depth. Use the full spread of
    # valid measurements if the scene is otherwise nearly flat.
    iqr = float(np.percentile(z, 75)-np.percentile(z, 25))
    robust_scale = max(iqr, 1.4826 * np.median(np.abs(z-np.median(z))),
                       0.10 * abs(float(np.median(z))), 1e-3)
    jump_min = max(float(narrow_object_jump_iqr)*robust_scale, 1e-8)
    margin = max(0., float(narrow_object_margin_px))
    width = max(0., float(narrow_object_width_px))
    narrow_excluded = np.zeros(len(x), dtype=bool)
    regions = []
    if width > 0. and narrow_object_jump_iqr > 0.:
        # A narrow excursion must be bounded on *both* sides by observed
        # background samples. Keep larger real structural changes intact.
        for start in range(1, len(u)-1):
            left_value = z[start-1]
            first_jump = z[start] - left_value
            if abs(first_jump) < jump_min:
                continue
            for stop in range(start, len(u)-1):
                if (u[stop] - u[start]) > width:
                    break
                right_value = z[stop+1]
                last_jump = right_value - z[stop]
                # Require steps with opposite direction and comparable
                # background level on both sides of the suspected feature.
                if first_jump * last_jump >= 0 or abs(last_jump) < jump_min:
                    continue
                if abs(right_value-left_value) > 0.40 * min(abs(first_jump), abs(last_jump)):
                    continue
                middle = z[start:stop+1]
                background = 0.5*(left_value + right_value)
                if len(middle) == 0 or float(np.median(abs(middle-background))) < jump_min*0.70:
                    continue
                # Include interpolation slopes immediately outside the narrow
                # hit, not just the columns directly touching the return.
                lo = u[start-1] - margin
                hi = u[stop+1] + margin
                regions.append((float(lo), float(hi)))
                narrow_excluded |= (x >= lo) & (x <= hi)
                break
    reliable &= ~narrow_excluded
    return reliable, {
        'excluded_interp': excluded_gap,
        'excluded_narrow': int(narrow_excluded.sum()),
        'narrow_regions': len(regions),
        'excluded_total': int((~reliable).sum()),
        'total': int(len(x)),
    }
