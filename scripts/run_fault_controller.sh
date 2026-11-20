#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_setup_env.sh" sensor_fault_simulation
exec ros2 run sensor_fault_simulation fault_controller --ros-args \
  -p keys_topic:="${FAULT_KEYS_TOPIC:-/faults/keys}" "$@"
