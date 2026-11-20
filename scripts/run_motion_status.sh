#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_setup_env.sh" motion_status_rviz
exec ros2 run motion_status_rviz motion_status_rviz --topic /odom \
  --anomaly-topic /cross_modal_anomaly/status --anomaly-type json \
  --linear-threshold 0.035 --angular-threshold 0.10 --stop-delay 0.75 "$@"
