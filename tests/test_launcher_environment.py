"""Regression checks for sourcing a colcon overlay under bash `set -u`."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]


class LauncherEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        scripts = self.root / "scripts"
        shutil.copytree(PROJECT / "scripts", scripts)
        local_setup = self.root / "install" / "local_setup.bash"
        local_setup.parent.mkdir(parents=True)
        # A minimal stand-in for colcon's setup script: without the fix,
        # referencing the unset variable terminates every wrapper on `set -u`.
        local_setup.write_text('''if [[ -z "$COLCON_CURRENT_PREFIX" ]]; then
  export TEST_COLCON_OVERLAY_SOURCED=yes
fi
''')
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        ros2 = bin_dir / "ros2"
        ros2.write_text('''#!/usr/bin/env bash
if [[ "$1" == pkg && "$2" == prefix ]]; then
  printf '%s\\n' "$MOCK_ROS_PREFIX"
else
  printf 'EXEC:%s;OVERLAY:%s\\n' "$1" "${TEST_COLCON_OVERLAY_SOURCED:-no}"
fi
''')
        ros2.chmod(0o755)
        self.env = os.environ.copy()
        self.env.pop("COLCON_CURRENT_PREFIX", None)
        self.env["PATH"] = str(bin_dir) + os.pathsep + self.env["PATH"]
        self.env["MOCK_ROS_PREFIX"] = str(self.root / "install" / "rgb_preprocessing")

    def test_every_launcher_sources_overlay_under_nounset(self):
        for script in sorted((self.root / "scripts").glob("run_*.sh")):
            with self.subTest(script=script.name):
                result = subprocess.run(["bash", str(script)], env=self.env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("OVERLAY:yes", result.stdout)

    def test_launchers_do_not_enable_nounset_while_sourcing_ros(self):
        for script in (self.root / "scripts").glob("run_*.sh"):
            self.assertIn("set -eo pipefail", script.read_text())
            self.assertNotIn("set -euo pipefail", script.read_text())

    def test_rejects_old_workspace_prefix(self):
        env = self.env.copy()
        env["MOCK_ROS_PREFIX"] = "/home/example/.local/share/Trash/files/old_workspace/install/cross_modal_anomaly_detection"
        result = subprocess.run(["bash", str(self.root / "scripts" / "run_anomaly_detection.sh")],
                                env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("different workspace", result.stderr)
        self.assertIn("Trash", result.stderr)

    def test_nounset_is_restored_after_sourcing(self):
        cmd = 'set -euo pipefail; source "$1" rgb_preprocessing; [[ "$-" == *u* ]]; printf "nounset-restored\\n"'
        result = subprocess.run(["bash", "-c", cmd, "bash", str(self.root / "scripts" / "_setup_env.sh")],
                                env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nounset-restored", result.stdout)


if __name__ == "__main__":
    unittest.main()
