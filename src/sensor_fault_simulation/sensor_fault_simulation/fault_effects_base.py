#!/usr/bin/env python3
"""Pure NumPy/OpenCV fault models for controlled ROS 2 sensor experiments.

Faults are software approximations. They do not establish that a particular
physical failure will produce the same readings on a real sensor.
"""
from collections import OrderedDict
from functools import lru_cache
import math
import numpy as np
import cv2

# Order is the Joy.buttons index mapping. Do not reorder without changing both ends.
FAULTS = OrderedDict([
    ('1', ('RGB', 'RGB partial lens obstruction')),
    ('2', ('RGB', 'RGB defocus blur')),
    ('3', ('RGB', 'RGB overexposure / saturation')),
    ('4', ('RGB', 'RGB output freeze (stale timestamp)')),
    ('q', ('LiDAR', 'LiDAR major blackout / missing scan')),
    ('w', ('LiDAR', 'LiDAR nonlinear range distortion')),
    ('e', ('LiDAR', 'LiDAR broad false near obstacles')),
    ('r', ('LiDAR', 'LiDAR severe dropout + range noise')),
    ('z', ('RGB', 'RGB shared lens obstruction')),
    ('x', ('Shared', 'Dense fog approximation: RGB + LiDAR')),
    ('c', ('RGB', 'RGB stream freeze')),
    ('v', ('Shared', 'Platform vibration: RGB + LiDAR')),
    # APPEND: existing Joy.buttons index order must not change.
    ('5', ('RGB', 'RGB cracked-lens / broken-camera image')),
    ('t', ('LiDAR', 'LiDAR angular misindexing (90-deg scan rotation)')),
])
KEYS = tuple(FAULTS)
GROUPS = OrderedDict((group, [k for k,(g,_) in FAULTS.items() if g==group])
                     for group in ('RGB', 'LiDAR', 'Shared'))


def _stripe(width, region_index=2, parts=5):
    region_index = int(np.clip(region_index, 0, parts - 1))
    return (region_index * width // parts, (region_index + 1) * width // parts)


def _motion_blur(bgr, length=17):
    length = max(3, int(length) | 1)
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1. / length
    return cv2.filter2D(bgr, -1, kernel)


@lru_cache(maxsize=12)
def _cracked_camera_masks(height, width, seed):
    """Static, reproducible per-resolution image fault masks.

    Models a damaged lens/camera output, NOT merely cosmetic screen cracks:
    opaque fractured sectors + saturated shards + jagged radial fractures.
    Masks remain in the same pixels from frame to frame.
    """
    h, w = int(height), int(width)
    local = np.random.default_rng(int(seed))
    black = np.zeros((h, w), np.uint8)
    bloom = np.zeros((h, w), np.uint8)
    cracks = np.zeros((h, w), np.uint8)
    xc = int(.51*w)
    yc = int(.53*h)
    radius = float(np.hypot(w, h))
    # Three damaged opaque wedges across the image; some neighboring wedges
    # are bright/saturated, reproducing frozen/broken camera-image artifacts.
    sectors = [(0.20,0.52), (1.66,2.10), (3.50,3.88), (4.82,5.22)]
    for i, (a0,a1) in enumerate(sectors):
        r = radius * float(local.uniform(.84, 1.08))
        pts = [(xc, yc)]
        for a in np.linspace(a0, a1, 9):
            j = float(local.uniform(.93,1.04))
            pts.append((int(xc + np.cos(a)*r*j), int(yc + np.sin(a)*r*j)))
        target = black if i % 3 != 1 else bloom
        cv2.fillPoly(target, [np.asarray(pts, dtype=np.int32)], 255)
    # A dark irregular failure zone near the impact point.
    cv2.ellipse(black,(xc,yc),(max(3,w//15),max(3,h//12)),0,0,360,255,-1)
    # Spider-web cracks: connected, jagged radial lines with short branches.
    for i in range(19):
        theta=2*np.pi*i/19+float(local.uniform(-.13,.13))
        dist=radius*float(local.uniform(.66,1.08))
        pts=[(xc,yc)]
        for j in range(1,10):
            radius_j=dist*j/9
            theta_j=theta+float(local.normal(0,.065))
            pts.append((int(xc+np.cos(theta_j)*radius_j),
                        int(yc+np.sin(theta_j)*radius_j)))
        poly=np.array(pts,dtype=np.int32)
        cv2.polylines(cracks,[poly],False,255,2 if i%4==0 else 1,cv2.LINE_AA)
        if i%2==0:
            p0=np.array(pts[5],dtype=float)
            p1=p0+np.array([np.cos(theta+.6),np.sin(theta+.6)])*radius*.14
            cv2.line(cracks,tuple(p0.astype(int)),tuple(p1.astype(int)),210,1,cv2.LINE_AA)
    # Concentric fractured rings and a sensor-like tear line.
    for scale in (.19,.35):
        angles=np.linspace(0,2*np.pi,34)
        r=radius*scale*(1+local.normal(0,.075,size=angles.size))
        poly=np.round(np.column_stack([xc+np.cos(angles)*r,yc+np.sin(angles)*r])).astype(np.int32)
        cv2.polylines(cracks,[poly],True,180,1,cv2.LINE_AA)
    cv2.line(cracks,(0,int(.79*h)),(w-1,int(.79*h)),200,1,cv2.LINE_AA)
    return black,bloom,cracks


def cracked_camera_image(img, rng, *, seed=137, intensity=1.0):
    """Severely distorted RGB camera output with persistent fracture geometry."""
    image=np.asarray(img,dtype=np.uint8)
    h,w=image.shape[:2]
    black,bloom,cracks=_cracked_camera_masks(h,w,int(seed))
    amount=float(np.clip(intensity,0,1))
    if amount<=0:
        return image.copy()
    out=image.astype(np.float32).copy()
    # Regions of permanently lost / saturated camera information.
    dark=(black.astype(np.float32)/255.*amount)[...,None]
    white=(bloom.astype(np.float32)/255.*amount)[...,None]
    out*=1.0-.97*dark
    out=out*(1.0-white)+255.0*white
    # Bright cracks with dark shadows for visibility at 320 x 180.
    dilated=cv2.dilate(cracks,np.ones((3,3),np.uint8),iterations=1)
    shadow=np.maximum(dilated.astype(np.float32)-cracks.astype(np.float32),0)/255.
    out*=1.0-.60*amount*shadow[...,None]
    light=(cracks.astype(np.float32)/255.*amount)[...,None]
    out=out*(1-.94*light)+245.0*(.94*light)
    # Random flicker inside damaged areas only, leaving crack geometry stable.
    affected=(black>0)|(bloom>0)|(cracks>0)
    grain=rng.normal(0,32*amount,size=(h,w,1)).astype(np.float32)
    out+=grain*affected[...,None]
    return np.clip(out,0,255).astype(np.uint8)


def corrupt_rgb(bgr, active, rng, region_index=2, *,
                overexposure_gain=3.2, overexposure_offset=105.0,
                crack_seed=137, crack_intensity=1.0):
    """Input/output BGR uint8; compose independent + shared faults."""
    img = np.asarray(bgr, dtype=np.uint8).copy()
    h, w = img.shape[:2]
    x0, x1 = _stripe(w, region_index)
    if '1' in active:
        img[:, x0:x1] = 0
    if '2' in active:
        img = cv2.GaussianBlur(img, (17, 17), 5.0)
    if '3' in active:
        # Much stronger than the original gain=1.7, offset=55. Most image
        # detail is lost to clipping, rather than a harmless brightness shift.
        img = np.clip(img.astype(np.float32) * float(overexposure_gain)
                      + float(overexposure_offset), 0, 255).astype(np.uint8)
    if 'z' in active:
        img[:, x0:x1] = 0
    if 'x' in active:
        img = np.clip(img.astype(np.float32) * .42 + 136, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (9, 9), 2.0)
    if 'v' in active:
        img = _motion_blur(img, 17)
    if '5' in active:
        img = cracked_camera_image(img,rng,seed=crack_seed,intensity=crack_intensity)
    return img


def corrupt_scan(scan, active, rng, *, sector_deg=360.0,
                 bias_factor=1.55, range_warp_amplitude=1.20,
                 fake_distance_m=0.65, random_dropout_fraction=0.70,
                 range_noise_fraction=0.30):
    """Software-only, severe LiDAR corruption; never modify a real sensor.

    q: wide angular outage; at 360 deg ALL ranges invalid (diagnostic INVALID).
    w: nonuniform angular-dependent distance warp, defeats scale invariance.
    e: false obstacles across multiple angular sectors, camera-facing or not.
    r: heavy random dropout plus strong noise on surviving ranges.
    t: 90-degree scan index misregistration without changing angle metadata.
    x/v: existing shared fog/vibration faults (intensified modestly).

    Important: the scanner's angle=0 may not point in camera-forward direction.
    Wide/multi-sector faults deliberately affect a substantial fraction of 360 deg.
    """
    arr=np.asarray(scan.ranges,dtype=np.float64).copy()
    if arr.size==0:
        return scan
    angles=float(scan.angle_min)+np.arange(arr.size)*float(scan.angle_increment)
    wrapped=np.arctan2(np.sin(angles),np.cos(angles))
    lower=float(scan.range_min)
    upper=float(scan.range_max)
    good=lambda v: np.isfinite(v)&(v>=lower)&(v<=upper)

    if 'q' in active:
        sector_width=float(np.clip(sector_deg,0.,360.))
        if sector_width>=359.99:
            # Complete missing scan, not a subtle 40-degree cutout.
            arr[:]=np.inf
        else:
            sector=np.abs(wrapped)<=math.radians(sector_width/2)
            arr[sector]=np.inf

    if 'w' in active:
        valid=good(arr)
        # Not a uniform scaling: local modulation creates distinct errors in
        # the spatial shape detected by Pearson/NCC.
        gain=(float(bias_factor)+float(range_warp_amplitude)*np.sin(7*angles+0.45)
              +0.24*np.cos(11*angles-0.2))
        gain=np.clip(gain,.25,3.4)
        arr[valid]*=gain[valid]
        arr[(arr<lower)|(arr>upper)]=np.inf

    if 'e' in active:
        valid=good(arr)
        # Several broad false surfaces (instead of a narrow 20-degree wedge).
        sector=(np.cos(2*angles-0.35)>.05) | (np.cos(3*angles+0.5)>.75)
        wall=np.maximum(lower+.02,
                        float(fake_distance_m)+.16*np.sin(5*angles)+.07*np.cos(11*angles))
        wall=np.minimum(wall,upper)
        use=sector & valid
        arr[use]=np.minimum(arr[use],wall[use])

    if 'r' in active:
        valid=good(arr)
        missing=valid & (rng.random(arr.size)<float(np.clip(random_dropout_fraction,0.,1.)))
        arr[missing]=np.inf
        valid=good(arr)
        if np.any(valid):
            jitter=np.clip(1+rng.normal(0,range_noise_fraction,valid.sum()),.15,2.5)
            arr[valid]*=jitter
        arr[(arr<lower)|(arr>upper)]=np.inf

    if 't' in active:
        arr=np.roll(arr,max(1,int(round(arr.size*0.25))))

    if 'x' in active:
        valid=good(arr)
        arr[valid & (rng.random(arr.size)<.45)]=np.inf
        valid=good(arr)
        if np.any(valid):
            arr[valid]*=np.maximum(.12,1+rng.normal(0,.14,valid.sum()))
        arr[(arr<lower)|(arr>upper)]=np.inf

    if 'v' in active:
        delta=math.radians(float(rng.normal(0.,4.0)))
        scan.angle_min+=delta
        scan.angle_max+=delta
        valid=good(arr)
        if np.any(valid):
            arr[valid]*=np.maximum(.25,1+rng.normal(0,.14,valid.sum()))
            arr[(arr<lower)|(arr>upper)]=np.inf

    scan.ranges=arr.tolist()
    return scan
