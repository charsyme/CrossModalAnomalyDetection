from pathlib import Path
import ast
import unittest
import xml.etree.ElementTree as ET
import re

ROOT=Path(__file__).resolve().parents[1]

class PackageChecks(unittest.TestCase):
    def test_no_python_syntax_errors(self):
        for path in (ROOT / 'src').rglob('*.py'):
            with self.subTest(path=str(path)):
                ast.parse(path.read_text(), filename=str(path))

    def test_required_files(self):
        for name in ('rgb_preprocessing','depth_estimation','lidar_projection',
                     'cross_modal_anomaly_detection','motion_status_rviz','sensor_fault_simulation'):
            self.assertTrue((ROOT/'src'/name/'package.xml').is_file())
            manifest = ET.parse(ROOT/'src'/name/'package.xml').getroot()
            self.assertEqual(manifest.findtext('name'), name)

    def test_python_entrypoints(self):
        for setup in (ROOT/'src').glob('*/setup.py'):
            for command, module in re.findall(r"'([^'\n]+) = ([^'\n]+):main'", setup.read_text()):
                module_file = setup.parent.joinpath(*module.split('.')).with_suffix('.py')
                self.assertTrue(module_file.is_file(), f'{command} points to a missing module')
                definitions = [item.name for item in ast.parse(module_file.read_text()).body
                               if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
                self.assertIn('main', definitions, f'{command} is missing main()')

    def test_simulator_assets(self):
        assets = ROOT/'src/sensor_fault_simulation/sensor_fault_simulation/camera_fault_assets'
        self.assertEqual(len(list(assets.glob('*.png'))), 2)

    def test_detector_defaults(self):
        cfg=(ROOT/'src/cross_modal_anomaly_detection/config/detector.yaml').read_text()
        for entry in ('spatial_irregularity_threshold: 0.2',
                      'spatial_irregularity_min_unique_points: 8',
                      'spatial_irregularity_neighborhood: 7',
                      'anomaly_window: 20','required_bad_frames: 12',
                      'spatial_irregularity_vote_enabled: true'):
            self.assertIn(entry,cfg)

if __name__ == '__main__': unittest.main()
