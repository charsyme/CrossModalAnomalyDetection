"""Check source distribution can be safely committed without build products."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class PublicRepositoryTests(unittest.TestCase):
    def test_no_generated_colcon_directories(self):
        for name in ('build', 'install', 'log', '.git', '__pycache__'):
            self.assertFalse((ROOT/name).exists(), name)

    def test_readme_has_both_workflows(self):
        text=(ROOT/'README.md').read_text()
        for heading in ('Run with real sensor measurements',
                        'Run with optional simulated RGB/LiDAR faults'):
            self.assertIn(heading, text)

    def test_scripts_are_executable_and_use_common_helper(self):
        for script in (ROOT/'scripts').glob('run_*.sh'):
            self.assertTrue(script.stat().st_mode & 0o111, script.name)
            self.assertIn('_setup_env.sh', script.read_text())

if __name__=='__main__':
    unittest.main()
