#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_setup_env.sh" cross_modal_anomaly_detection
config="$(ros2 pkg prefix cross_modal_anomaly_detection)/share/cross_modal_anomaly_detection/config/detector.yaml"
exec ros2 run cross_modal_anomaly_detection cross_modal_anomaly_detection --ros-args --params-file "$config" "$@"
