"""ROS-independent detector core regression checks for the research defaults."""
import ast
from collections import deque
from pathlib import Path
import sys
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/cross_modal_anomaly_detection'))
from cross_modal_anomaly_detection.spatial_irregularity import CrossModalSpatialIrregularity

# The temporal voter is defined in the ROS-dependent node module. Extract
# that one pure-Python class for a unit test without importing rclpy.
source = (ROOT/'src/cross_modal_anomaly_detection/cross_modal_anomaly_detection/cross_modal_anomaly_detection.py').read_text()
tree = ast.parse(source)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ScanUniqueVoter')
code = compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), '<voter>', 'exec')
ns = {'deque':deque, 'time':time}
exec(code, ns)
ScanUniqueVoter = ns['ScanUniqueVoter']

class CoreAlgorithmTests(unittest.TestCase):
    @staticmethod
    def monitor():
        return CrossModalSpatialIrregularity(regions=7,region_overlap_fraction=0.2,
            min_unique_bad_points=8, neighborhood=7, threshold=0.2,
            min_affected_fraction=0.08,min_bad_regions=2, min_fit_samples=45,
            alignment_smooth_sigma=12)

    def test_spatial_irregularity_is_quiet_on_affine_equivalent_profiles(self):
        u=np.arange(180,dtype=float)
        lidar=0.7+0.17*np.sin(u/17)+0.05*np.cos(u/5)
        rgb=(lidar-0.2)/1.3
        result=self.monitor().observe(u,lidar,rgb,stamp_key=1)
        self.assertTrue(result['spatial_irregularity_valid'])
        self.assertFalse(result['spatial_irregularity_bad'])

    def test_spatial_irregularity_detects_distinct_noisy_lidar_measurements(self):
        u=np.arange(180,dtype=float)
        clean=0.7+0.17*np.sin(u/17)+0.05*np.cos(u/5)
        rgb=(clean-0.2)/1.3
        noisy=clean.copy()
        noisy[55:125:2]+=0.35
        result=self.monitor().observe(u,noisy,rgb,stamp_key=1)
        self.assertTrue(result['spatial_irregularity_bad'])
        self.assertGreaterEqual(result['spatial_irregularity_unique_bad_points'],8)
        self.assertGreaterEqual(result['spatial_irregularity_bad_region_count'],2)

    def test_unique_scan_voter_requires_12_of_20(self):
        v=ScanUniqueVoter(window=20,required=12,consecutive=2,
            minimum_seconds=0.25,weak_confirm_window=4,weak_confirm_required=3)
        for i in range(11):
            out=v.observe(valid=True,frame_bad=True,strong_bad=True,
                          stamp_key=(i,0),now=i*0.1)
            self.assertFalse(out['anomaly'])
        out=v.observe(valid=True,frame_bad=True,strong_bad=True,
                      stamp_key=(11,0),now=1.1)
        self.assertTrue(out['anomaly'])
        duplicate=v.observe(valid=True,frame_bad=True,strong_bad=True,
                            stamp_key=(11,0),now=1.2)
        self.assertFalse(duplicate['scan_evidence_added'])
        self.assertEqual(duplicate['bad_frames_in_window'],12)
        for i in range(12,21):
            out=v.observe(valid=True,frame_bad=False,strong_bad=False,
                          stamp_key=(i,0),now=i*0.1)
        self.assertFalse(out['anomaly'])

if __name__=='__main__': unittest.main()
