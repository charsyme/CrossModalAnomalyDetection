#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_setup_env.sh" lidar_projection
config="$(ros2 pkg prefix lidar_projection)/share/lidar_projection/config/projector.yaml"
exec ros2 run lidar_projection lidar_projection_node --ros-args --params-file "$config" "$@"
