#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_setup_env.sh" sensor_fault_simulation
exec ros2 run sensor_fault_simulation fault_injector --ros-args \
  -p rgb_in:="${RGB_INPUT:-/oakd/rgb/preview/image_rect_clean}" \
  -p scan_in:="${SCAN_INPUT:-/scan}" \
  -p rgb_out:="${RGB_OUTPUT:-/faults/rgb/image_rect}" \
  -p scan_out:="${SCAN_OUTPUT:-/faults/scan}" \
  -p keys_topic:="${FAULT_KEYS_TOPIC:-/faults/keys}" \
  -p realistic_severity:="${FAULT_SEVERITY:-0.65}" \
  -p lidar_sector_center_deg:="${LIDAR_SECTOR_CENTER_DEG:-0.0}" \
  -p rgb_delay_frames:=4 -p delayed_rgb_stamp_mode:=original \
  -p lidar_replay_stamp_mode:=original -p stale_keys_timeout_sec:=1.0 "$@"
