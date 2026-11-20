#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_setup_env.sh" rgb_preprocessing
exec ros2 launch rgb_preprocessing rgb_preprocessing.launch.py "$@"
