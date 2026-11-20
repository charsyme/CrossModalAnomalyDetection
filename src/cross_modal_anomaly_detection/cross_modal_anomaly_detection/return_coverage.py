#!/usr/bin/env python3
"""Camera-sector LiDAR return availability based on original LaserScan beams.

A one-time, healthy calibration collects the scan indices of beams expected to
project within the RGB camera's valid FoV (from the projector's `scan_index`
field). The reference beam set and healthy valid-return fraction are frozen.
During monitoring, *raw* LaserScan ranges are tested at those fixed indices,
not the number of surviving projected image columns.

This intentionally does not use 360-degree scan availability, so losses behind
or beside the camera cannot dominate the result. No fault is inferred when
input data or matching timestamps are missing; the original projected-column
coverage remains available as a separate check.
"""
from collections import Counter
import numpy as np


class CameraSectorReturnMonitor:
    def __init__(self, *, baseline_scans=24, min_expected_beams=25,
                 min_presence_fraction=0.25, min_healthy_fraction=0.40,
                 static_ratio_threshold=0.70, moving_ratio_threshold=0.60):
        self.baseline_scans = max(3, int(baseline_scans))
        self.min_expected_beams = max(1, int(min_expected_beams))
        self.min_presence_fraction = float(min_presence_fraction)
        self.min_healthy_fraction = float(min_healthy_fraction)
        self.static_ratio_threshold = float(static_ratio_threshold)
        self.moving_ratio_threshold = float(moving_ratio_threshold)
        if not 0 < self.min_presence_fraction <= 1:
            raise ValueError('min_presence_fraction must be in (0,1]')
        if not 0 < self.min_healthy_fraction <= 1:
            raise ValueError('min_healthy_fraction must be in (0,1]')
        for threshold in (self.static_ratio_threshold, self.moving_ratio_threshold):
            if not 0 < threshold <= 1:
                raise ValueError('return coverage thresholds must be in (0,1]')
        self.samples = []
        self.indices = None
        self.reference_fraction = None
        self.scan_geometry = None
        self.last_key = None
        self.last_result = None

    @property
    def ready(self):
        return self.indices is not None

    @staticmethod
    def _scan_data(scan):
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        if ranges.size == 0:
            return None, None
        valid = (np.isfinite(ranges) & (ranges >= float(scan.range_min)) &
                 (ranges <= float(scan.range_max)))
        geometry = (ranges.size, float(scan.angle_min),
                    float(scan.angle_increment), str(scan.header.frame_id))
        return valid, geometry

    @staticmethod
    def _geometry_matches(a, b):
        return bool(a is not None and b is not None and a[0] == b[0]
                    and abs(a[1]-b[1]) <= 0.003
                    and abs(a[2]-b[2]) <= 1e-5 and a[3] == b[3])

    def _unavailable(self, reason, **extra):
        result = dict(return_coverage_valid=False, return_coverage_bad=False,
                      return_coverage_reason=reason,
                      return_coverage_baseline_ready=self.ready,
                      return_coverage_baseline_scans=len(self.samples),
                      return_coverage_baseline_required=self.baseline_scans,
                      return_coverage_expected_beams=(len(self.indices) if self.ready else None),
                      return_coverage_observed_fraction=None,
                      return_coverage_reference_fraction=self.reference_fraction,
                      return_coverage_ratio=None)
        result.update(extra)
        return result

    def observe(self, scan, *, expected_indices=None, can_learn=False,
                moving=False, stamp_key=None):
        if stamp_key is not None and stamp_key == self.last_key and self.last_result is not None:
            return dict(self.last_result)  # same scan, not a new baseline observation
        valid, geometry = self._scan_data(scan)
        if valid is None:
            return self._unavailable('empty_scan')
        if self.scan_geometry is not None and not self._geometry_matches(geometry, self.scan_geometry):
            return self._unavailable('scan_geometry_changed')
        n = len(valid)
        if not self.ready:
            if not can_learn:
                return self._unavailable('baseline_wait_healthy')
            beams = set()
            if expected_indices is not None:
                beams = {int(i) for i in expected_indices if 0 <= int(i) < n}
            if len(beams) < self.min_expected_beams:
                return self._unavailable('insufficient_camera_sector_beams',
                                         return_coverage_candidate_beams=len(beams))
            if self.scan_geometry is None:
                self.scan_geometry = geometry
            self.samples.append((beams, valid.copy()))
            if len(self.samples) >= self.baseline_scans:
                counter = Counter(i for indices, _ in self.samples for i in indices)
                min_count = max(2, int(np.ceil(self.min_presence_fraction * len(self.samples))))
                keep = np.array(sorted(i for i, c in counter.items() if c >= min_count), dtype=np.int64)
                if len(keep) < self.min_expected_beams:
                    self.samples.clear()
                    return self._unavailable('unstable_camera_sector_calibration')
                healthy = np.array([float(np.mean(v[keep])) for _, v in self.samples], dtype=float)
                reference = float(np.median(healthy))
                if reference < self.min_healthy_fraction:
                    self.samples.clear()
                    return self._unavailable('unusable_healthy_return_reference')
                self.indices = keep
                self.reference_fraction = reference
                self.samples.clear()
            result = self._unavailable('calibrating' if not self.ready else 'calibration_complete')
        else:
            current = float(np.mean(valid[self.indices]))
            ratio = current / max(self.reference_fraction, 1e-6)
            threshold = self.moving_ratio_threshold if moving else self.static_ratio_threshold
            result = dict(return_coverage_valid=True,
                          return_coverage_bad=bool(ratio < threshold),
                          return_coverage_reason='ok',
                          return_coverage_baseline_ready=True,
                          return_coverage_baseline_scans=self.baseline_scans,
                          return_coverage_baseline_required=self.baseline_scans,
                          return_coverage_expected_beams=int(len(self.indices)),
                          return_coverage_valid_beams=int(valid[self.indices].sum()),
                          return_coverage_observed_fraction=current,
                          return_coverage_reference_fraction=self.reference_fraction,
                          return_coverage_ratio=ratio,
                          return_coverage_threshold_ratio=threshold)
        if stamp_key is not None:
            self.last_key = stamp_key
            self.last_result = dict(result)
        return result
