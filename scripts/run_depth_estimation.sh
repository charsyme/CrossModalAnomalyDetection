#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_setup_env.sh" depth_estimation
exec ros2 run depth_estimation depth_node --ros-args \
  -p input_topic:="${RGB_INPUT:-/oakd/rgb/preview/image_rect_clean}" \
  -p output_topic:="${DEPTH_TOPIC:-/depth_anything/depth}" \
  -p mode:=relative -p encoder:=vits "$@"
