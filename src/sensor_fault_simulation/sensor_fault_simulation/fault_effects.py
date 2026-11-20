#!/usr/bin/env python3
"""Subtle/persistent realistic sensor-fault approximations for ROS 2 evaluation.

Imports, without modifying, the existing 18 stronger fault effects and appends
additional keys. Every transformation is software-only and deterministic with
seeded RNG plus an explicit frame index / elapsed time.
"""
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from .camera_overlays import CameraFaultOverlay
import math
import numpy as np
import cv2

from .fault_effects_base import (
    FAULTS as ORIGINAL_FAULTS, KEYS as ORIGINAL_KEYS,
    corrupt_rgb as legacy_rgb, corrupt_scan as legacy_scan,
)

EXTRA_FAULTS = OrderedDict([
    ('6', ('RGB', 'Local lens smear / soft dirt (persistent)')),
    ('7', ('RGB', 'Exposure hunting + shot/read noise')),
    ('8', ('RGB', 'Rolling-shutter shear / jitter')),
    ('9', ('RGB', 'Localized optical glare / bloom')),
    ('0', ('RGB', 'Video codec compression artifacts')),
    ('y', ('RGB', 'RGB frame latency / stale frame replay')),
    ('u', ('LiDAR', 'Narrow low-reflectivity / glass return loss')),
    ('i', ('LiDAR', 'Sparse specular / multipath ghost echoes')),
    ('o', ('LiDAR', 'Slow small range calibration drift')),
    ('p', ('LiDAR', 'Small angular encoder index jitter')),
    ('g', ('LiDAR', 'Sparse optical / dust false returns')),
    ('h', ('LiDAR', 'Intermittent stale LiDAR scan replay')),
    ('j', ('LiDAR', 'Short contiguous packet / sector dropouts')),
    ('k', ('LiDAR', 'Reduced effective detection range')),
    ('m', ('RGB', 'Close-to-lens hand/finger partial occlusion')),
    ('l', ('RGB', 'Soft fabric / flap partially covering lens')),
    ('b', ('RGB', 'Opaque industrial tape covering part of camera lens')),
    ('s', ('RGB', 'Water splashes / droplets on camera lens')),
])
# Keep the historical Joy.buttons order unchanged for compatibility with
# any existing /faults/keys publisher. Removed faults retain their old index
# but are always transmitted as 0 by the supplied keyboard controller.
ALL_FAULTS = OrderedDict(ORIGINAL_FAULTS)
ALL_FAULTS.update(EXTRA_FAULTS)
KEYS = tuple(ALL_FAULTS)

# Only these 16 fault effects are enabled in the GUI and injector.
# RGB: partial obstruction, defocus, overexposure, cracked lens, smear,
#      glare, compression, industrial tape, water splashes.
# LiDAR: blackout, nonlinear distortion, severe dropout, 90-deg
#        misindexing, sparse dust interference.
# Shared: dense fog, platform vibration.
ENABLED_KEYS = frozenset(('1', '2', '3', '5', '6', '9', '0', 'b', 's',
                          'q', 'w', 'r', 't', 'g', 'x', 'v'))
FAULTS = OrderedDict((key, info) for key, info in ALL_FAULTS.items()
                     if key in ENABLED_KEYS)
GROUPS = OrderedDict(
    (group, [key for key, (g, _) in FAULTS.items() if g == group])
    for group in ('RGB', 'LiDAR', 'Shared')
)
assert len(FAULTS) == 16, 'Fault allowlist must contain exactly 16 keys'


@lru_cache(maxsize=4)
def _camera_tape(width, height):
    """Load tape only (avoid precomputing water refraction when B is selected)."""
    path = Path(__file__).resolve().parent / 'camera_fault_assets' / 'opaque_industrial_tape_camera_overlay.png'
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise FileNotFoundError('Expected transparent BGRA camera obstruction: '+str(path))
    # Resize tape to camera resolution
    rgba = cv2.resize(
       rgba, (width, height),
       interpolation=cv2.INTER_AREA)

    # Move tape upward by 7% of image height
    shift_up = int(height * 0.07)

    shifted = np.zeros_like(rgba)
    shifted[:-shift_up, :, :] = rgba[shift_up:, :, :]

    return shifted


@lru_cache(maxsize=4)
def _camera_overlay(width, height):
    """Load opaque tape + water RGBA assets and precompute water refraction once."""
    return CameraFaultOverlay(width, height)


def _unit_severity(severity):
    return float(np.clip(severity, 0.0, 1.0))


def _gaussian_mask(height, width, x, y, sx, sy):
    xx = (np.arange(width, dtype=np.float32) - x * width) / max(1.0, sx * width)
    yy = (np.arange(height, dtype=np.float32) - y * height) / max(1.0, sy * height)
    return np.exp(-.5 * (xx[None, :] ** 2 + yy[:, None] ** 2))




def _smoothstep01(x):
    """Anti-aliased opacity from 0 to 1 with no hard threshold edges."""
    z = np.clip(x, 0., 1.)
    return z*z*(3. - 2.*z)


@lru_cache(maxsize=8)
def _cover_texture(height, width, seed):
    """Repeatable low-frequency texture (fixed to the lens, not the scene)."""
    local = np.random.default_rng(seed)
    # Correlated, out-of-focus surface texture; intentionally not pixel noise.
    noise = local.normal(0., 1., (20, 30)).astype(np.float32)
    noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), max(1.0, min(height,width)*0.013))
    return noise / max(1e-5, float(np.std(noise)))


def _realistic_rgb_cover(img, kind, *, severity, elapsed_s):
    """Simulate an object a few centimeters from the lens, not an image overlay.

    Uses organic curved boundaries, broad soft shadows, out-of-focus material,
    partial transparency at the edge and small time-continuous motion.  The
    stationary fabric fault is fixed relative to the camera image.  This is
    physically motivated but not a measured OAK-D optical model.
    """
    h, w = img.shape[:2]
    if h < 2 or w < 2:
        return img
    s = _unit_severity(severity)
    if s < 1e-6:
        return img
    t = max(0., float(elapsed_s))
    yf, xf = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xf / max(1.0, float(w - 1))
    yn = yf / max(1.0, float(h - 1))

    if kind == 'hand':
        # Fingers/palm entering from the left: curved, imperfect silhouette.
        # Start with part of the hand in view and approach in ~0.8 seconds.
        onset = 0.22 + 0.78*float(_smoothstep01(t / 0.80))
        tremor = .012*math.sin(t*1.6) + .006*math.sin(t*3.7 + 1.3)
        contour = (.10 + (.13 + .32*s)*onset + tremor
                   + .11*np.exp(-((yn-.67)/.28)**2)
                   - .055*np.exp(-((yn-.15)/.15)**2)
                   + .015*np.sin(15.*yn+1.1))
        edge_width = .018 + .018*s
        mask = _smoothstep01((contour - xn) / edge_width + .50)
        # Diffuse penumbra beyond the actual cover boundary.
        shadow = np.clip((contour + .085 - xn)/.13, 0, 1)
        shadow *= (1-mask)
        texture = _cover_texture(h,w,124)
        # BGR warm neutral close object; low-frequency texture and shading.
        foreground = np.empty_like(img, dtype=np.float32)
        illumination = np.clip(.82 + .18*yn + .07*xn + .025*texture, .62, 1.10)
        for channel, val in enumerate((68., 121., 174.)):
            foreground[...,channel] = val * illumination
        opacity = (.84 + .13*s)*mask
        shadow_strength = .13 + .13*s
        # Scene behind translucent edge is naturally unfocused.
        sigma = max(2., .017*min(h,w))
    else:
        # Close black fabric/flap drifting down from the top. Thin uneven edge,
        # stable wrinkled folds and shadowing rather than rectangular blackout.
        contour = (.17 + .29*s + .13*(xn-.5)
                   + .025*np.sin(12*xn+1.3)
                   + .016*np.sin(31*xn))
        edge_width = .016 + .024*s
        mask = _smoothstep01((contour - yn) / edge_width + .50)
        shadow = np.clip((contour + .10 - yn)/.16, 0., 1.)*(1-mask)
        texture = _cover_texture(h,w,231)
        folds = .10*np.sin(13*xn + 5*yn) + .04*np.sin(31*xn-4*yn)
        foreground = np.empty_like(img, dtype=np.float32)
        illumination = np.clip(1.0 + .16*texture + folds, .45, 1.65)
        for channel, val in enumerate((17., 21., 25.)):
            foreground[...,channel] = val*illumination
        opacity = (.89 + .10*s)*mask
        shadow_strength = .17 + .10*s
        sigma = max(2., .018*min(h,w))

    # Cover close to the lens is visually soft; underlying scene loses contrast.
    scene = img.astype(np.float32)
    blurred = cv2.GaussianBlur(scene, (0, 0), sigmaX=sigma)
    # The physical obstruction also casts a soft shadow on the visible scene.
    scene = scene*(1. - shadow_strength*shadow[...,None])
    # Optical defocus in the transition band, not a sharp raster polygon.
    soft_band = np.clip(4.*mask*(1-mask), 0., 1.)
    scene = scene*(1. - .60*soft_band[...,None]) + blurred*(.60*soft_band[...,None])
    out = scene*(1. - opacity[...,None]) + foreground*opacity[...,None]
    return np.clip(out, 0, 255).astype(np.uint8)

def corrupt_rgb(bgr, active, rng, region_index=2, *,
                overexposure_gain=3.2, overexposure_offset=105.,
                crack_seed=137, crack_intensity=1.,
                severity=.65, frame_index=0, elapsed_s=0.):
    """Apply only the 16 enabled effects; ignore all other fault keys."""
    active = set(active) & ENABLED_KEYS
    out = legacy_rgb(bgr, active, rng, region_index,
                     overexposure_gain=overexposure_gain,
                     overexposure_offset=overexposure_offset,
                     crack_seed=crack_seed, crack_intensity=crack_intensity)
    if not any(key in active for key in ('6', '7', '8', '9', '0', 'm', 'l', 'b', 's')):
        return out
    s = _unit_severity(severity)
    if s < 1e-5:
        return out
    h, w = out.shape[:2]
    img = out.astype(np.float32)

    if '6' in active:  # stable patch on lens, not a moving black rectangle
        mask = _gaussian_mask(h, w, .63, .48, .15, .19)
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.5 + 6*s)
        strength = (0.55 + 0.35*s) * mask[..., None]
        img = img * (1 - strength) + blurred * strength
        img = img * (1 - .25*s*mask[..., None]) + (142.*.25*s*mask[..., None])

    if '7' in active:  # AE time response; read/shot noise strongest in shadows
        gain = 1 + (0.12 + .42*s) * math.sin(2*math.pi*max(0.,elapsed_s)/2.8)
        # Fixed per-frame noise, spatially independent but contrast dependent.
        img = np.maximum(0., img * gain)
        std = 1.2 + 7.*s*(1.0 - np.clip(img/255., 0., 1.))
        img += rng.normal(0., 1., (h, w, 1)).astype(np.float32) * std

    if '8' in active:  # row-wise readout shear / small rolling shutter wobble
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        wobble = (2. + 7.*s) * (yy / max(1, h-1) - .5)
        wobble += 1.5*s*np.sin(2*math.pi*yy/max(1,h) + .15*frame_index)
        img = cv2.remap(img, xx + wobble.astype(np.float32), yy,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT101)

    if '9' in active:  # localized lens bloom; can coexist with clear image elsewhere
        mask = _gaussian_mask(h, w, .27, .22, .12, .16)[..., None]
        halo = _gaussian_mask(h, w, .27, .22, .38, .42)[..., None]
        img = img + (90. + 145.*s)*mask + (10. + 24.*s)*halo

    out = np.clip(img, 0, 255).astype(np.uint8)
    if 'm' in active:  # close-to-lens organic hand/finger cover
        out = _realistic_rgb_cover(out, 'hand', severity=s, elapsed_s=elapsed_s)
    if 'l' in active:  # soft fabric/fold partially covering top of lens
        out = _realistic_rgb_cover(out, 'fabric', severity=s, elapsed_s=elapsed_s)

    if '0' in active:  # video codec errors only, not physical lens change
        q = int(np.clip(84 - 60*s, 18, 90))
        success, compressed = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if success:
            decoded = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
            if decoded is not None and decoded.shape == out.shape:
                out = decoded
    # Camera-only defects attached to the imaging surface. Do not modify LiDAR.
    # Geometry is fixed in camera coordinates and intersects the nominal scan band.
    # Both files have genuine RGBA transparency; the industrial tape core is opaque.
    if 'b' in active:
        out = CameraFaultOverlay.alpha_blend(out, _camera_tape(w, h))
    if 's' in active:
        out = _camera_overlay(w, h).apply(out, 'water')
    return out


def corrupt_scan(scan, active, rng, *, sector_deg=360.,
                 bias_factor=1.55, range_warp_amplitude=1.20,
                 fake_distance_m=.65, random_dropout_fraction=.70,
                 range_noise_fraction=.30, severity=.65, frame_index=0,
                 elapsed_s=0., lidar_sector_center_deg=0.):
    """Apply only enabled LiDAR/shared faults without hardware writes."""
    active = set(active) & ENABLED_KEYS
    scan = legacy_scan(scan, active, rng, sector_deg=sector_deg,
                       bias_factor=bias_factor,
                       range_warp_amplitude=range_warp_amplitude,
                       fake_distance_m=fake_distance_m,
                       random_dropout_fraction=random_dropout_fraction,
                       range_noise_fraction=range_noise_fraction)
    s = _unit_severity(severity)
    if s < 1e-5 or not any(key in active for key in 'uiopgjk'):
        return scan
    a = np.asarray(scan.ranges, dtype=np.float64).copy()
    if not len(a):
        return scan
    lower = float(scan.range_min)
    upper = float(scan.range_max)
    angles = float(scan.angle_min) + np.arange(len(a))*float(scan.angle_increment)
    wrap = np.arctan2(np.sin(angles), np.cos(angles))
    center_offset = math.radians(float(lidar_sector_center_deg))
    good = lambda x: np.isfinite(x) & (x >= lower) & (x <= upper)
    intensities = np.asarray(getattr(scan, 'intensities', []), dtype=np.float64).copy()
    has_intensity = intensities.shape == a.shape
    if not has_intensity:
        intensities = None

    if 'u' in active:  # fixed thin angular sector with weak/variable optical returns
        sel = abs(np.arctan2(np.sin(wrap - center_offset - .33), np.cos(wrap - center_offset - .33))) < math.radians(8. + 9*s)
        lost = sel & good(a) & (rng.random(len(a)) < .20 + .55*s)
        a[lost] = np.inf
        if has_intensity:
            intensities[sel] *= .25 + .35*(1-s)
        noisy = sel & good(a)
        if noisy.any():
            a[noisy] *= np.maximum(.60, 1 + rng.normal(0, .04+.07*s, noisy.sum()))

    if 'i' in active:  # sparse coherent reflectance-related ghost clusters
        sel = abs(np.arctan2(np.sin(wrap - center_offset + .48), np.cos(wrap - center_offset + .48))) < math.radians(7. + 9*s)
        ghost = sel & good(a) & (rng.random(len(a)) < .10 + .22*s)
        if ghost.any():
            offset = (.15 + .75*s) * (1 if frame_index % 4 < 2 else -1)
            a[ghost] += offset + rng.normal(0., .04, ghost.sum())

    if 'o' in active:  # slowly developing uniform range bias (NCC often misses)
        ramp = min(1., max(0., elapsed_s)/20.)
        gain = 1. + (.012 + .090*s) * ramp
        a[good(a)] *= gain

    if 'p' in active:  # angular phase jitter, subtly changing surfaces/edges
        steps = int(np.clip(round(1 + 4*s), 1, 5))
        shift = int(round(steps*math.sin(frame_index*.43)))
        if shift:
            a = np.roll(a, shift)
            if has_intensity:
                intensities = np.roll(intensities, shift)

    if 'g' in active:  # false short returns & missing echoes from optical interference
        originally_good = good(a)
        noise = rng.random(len(a))
        nearby = originally_good & (noise < .006 + .025*s)
        a[nearby] = np.maximum(lower + .03, np.minimum(a[nearby], .40 + 1.1*rng.random(nearby.sum())))
        missing = originally_good & (noise >= .96 - .018*s)
        a[missing] = np.inf

    if 'j' in active:  # short bursts of consecutive missing beams
        if frame_index % 5 in (0, 1, 2):
            center = center_offset + .32 + .11*math.sin(frame_index*.12)
            width = math.radians(2.5 + 10.*s)
            a[abs(np.arctan2(np.sin(wrap-center),np.cos(wrap-center))) < width/2] = np.inf

    if 'k' in active:  # reduced sensitivity: invalid far measurements
        effective_max = max(lower + .25, upper * (1 - .35*s))
        a[good(a) & (a > effective_max)] = np.inf

    a[(np.isfinite(a)) & ((a < lower) | (a > upper))] = np.inf
    scan.ranges = a.tolist()
    if has_intensity:
        intensities[~good(a)] = 0.
        scan.intensities = intensities.tolist()
    return scan


