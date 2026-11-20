# CrossModalAnomalyDetection

**Cross-Modal Anomaly Detection Between RGB Camera and LiDAR Using Monocular Depth Estimation**

ROS 2 Jazzy packages for monitoring geometric agreement between a planar LiDAR and monocular depth estimated from an RGB camera. Developed around a TurtleBot 4 with an OAK-D Pro camera and 2D LiDAR; sensor topic names, camera calibration and TF frame names can be configured for other platforms. An optional fault injector publishes *separate* modified copies of RGB images and LiDAR scans for controlled experiments.

The pipeline comprises RGB rectification and border cleaning, relative monocular depth inference, LiDAR-to-camera projection, cross-modal anomaly detection, and a motion/anomaly status image for RViz. The fault injector and its keyboard/GUI controller are optional. This is experimental research software, **not a certified safety system**.

## Repository layout

```text
CrossModalAnomalyDetection/
├── src/
│   ├── rgb_preprocessing/             # ROS image rectification + border cleaner
│   ├── depth_estimation/              # Depth Anything V2 adapter + node
│   ├── lidar_projection/              # camera/LiDAR geometry, expected points (C++)
│   ├── cross_modal_anomaly_detection/ # geometric/irregularity/coverage checks
│   ├── motion_status_rviz/            # status image for RViz
│   └── sensor_fault_simulation/       # OPTIONAL injector and GUI controller
├── scripts/                           # launch wrappers; ROS topic overrides accepted
├── tests/                             # source/static and controller regression tests
├── .gitignore
└── README.md
```

Build products (`build/`, `install/`, `log/`), trained model weights, rosbag recordings and previous experimental versions are not stored in this repository.

## Requirements

- **ROS 2 Jazzy** on Ubuntu (or a compatible Jazzy installation), `colcon`, and a functioning ROS graph.
- Sensor publishers for RGB `sensor_msgs/Image`, `sensor_msgs/CameraInfo`, planar `sensor_msgs/LaserScan`, odometry `nav_msgs/Odometry`, and TF (`/tf`, `/tf_static`).
- OpenCV, NumPy, `cv_bridge`, `image_proc`, `image_geometry`, `laser_geometry`, and ROS 2 TF packages. The fault GUI additionally needs Tkinter and an interactive desktop.
- **Depth Anything V2** source code and the **Small (ViT-S) relative-depth** checkpoint. Download these separately; checkpoints are not distributed here. A compatible `torch` installation is also required.

Example ROS-dependent packages on Ubuntu:

```bash
sudo apt update
sudo apt install -y python3-colcon-common-extensions python3-opencv python3-numpy python3-tk \
  ros-jazzy-image-proc ros-jazzy-cv-bridge ros-jazzy-vision-msgs \
  ros-jazzy-sensor-msgs-py ros-jazzy-image-geometry ros-jazzy-laser-geometry \
  ros-jazzy-tf2-geometry-msgs
```

### Clone and build

Clone the repository into a ROS 2 workspace of your choice, or use the repository as its own standalone `colcon` workspace:

```bash
# Clone the repository from GitHub once it is published:
git clone https://github.com/<YOUR-USERNAME>/CrossModalAnomalyDetection.git
cd CrossModalAnomalyDetection

# Source a compatible ROS 2 Jazzy installation.
source /opt/ros/jazzy/setup.bash
# For a custom Jazzy source build, source that installation instead.

colcon build --symlink-install
source install/local_setup.bash
ros2 pkg prefix cross_modal_anomaly_detection
ros2 pkg prefix rgb_preprocessing
```

In each new terminal, activate your ROS 2 / Python environment before running a stage. The launch scripts source this project's local overlay after the initial build. Keep the repository's `src/` and `scripts/` directories intact; generated `build/`, `install/`, and `log/` directories are excluded from version control.

### Depth Anything V2 setup

Clone the [Depth Anything V2 implementation](https://github.com/DepthAnything/Depth-Anything-V2), install the compatible Python dependencies, and obtain its relative-depth ViT-S checkpoint from the upstream instructions. One possible layout is:

```text
~/models/Depth-Anything-V2/
├── depth_anything_v2/
└── checkpoints/
    └── depth_anything_v2_vits.pth
```

Make the source package importable in the **same Python environment** as ROS 2:

```bash
export PYTHONPATH="$HOME/models/Depth-Anything-V2:${PYTHONPATH:-}"
python3 -c 'from depth_anything_v2.dpt import DepthAnythingV2; print("Depth Anything V2 import OK")'
```

Pass `-p checkpoint_dir:="$HOME/models/Depth-Anything-V2/checkpoints"` to the depth launcher. The inference node does not automatically download weights. If you already have a functional ROS 2/PyTorch virtual environment, reuse it; do not install a second conflicting interpreter.

## Check TurtleBot 4 sensor data before starting

The examples below assume these topics and TF frames; adjust them to match your driver:

| Data | Example topic or frame |
|---|---|
| OAK-D raw RGB | `/oakd/rgb/preview/image_raw` |
| RGB camera calibration | `/oakd/rgb/preview/camera_info` |
| LiDAR scan | `/scan` |
| Odometry | `/odom` |
| Camera optical frame | `oakd_rgb_camera_optical_frame` |
| LiDAR frame | `rplidar_link` |

```bash
ros2 topic list | grep -E '(oakd|scan|odom|tf)'
ros2 topic info /scan -v
ros2 topic info /oakd/rgb/preview/camera_info -v
ros2 run tf2_ros tf2_echo oakd_rgb_camera_optical_frame rplidar_link
```

The camera-to-LiDAR TF and camera intrinsics must be valid. The example projector configuration uses a 2.5° yaw correction derived from the original experimental platform; review this setting for another robot.

## Run with real sensor measurements (no simulation)

Start the robot's camera, LiDAR, odometry and TF publishers first. Open **five terminals**, with `cd ~/CrossModalAnomalyDetection` and your ROS/Python environment activated in each. Run the following stages in order, one per terminal.

**1. Rectify and clean RGB** (both nodes in one launch):

```bash
./scripts/run_rgb_preprocessing.sh \
  rectify:=true \
  raw_image:=/oakd/rgb/preview/image_raw \
  camera_info:=/oakd/rgb/preview/camera_info \
  rect_image:=/oakd/rgb/preview/image_rect \
  clean_image:=/oakd/rgb/preview/image_rect_clean \
  valid_mask:=/rgb_rectification/valid_mask \
  edge_guard_px:=8
```

If your camera *already publishes a correctly rectified image*, set `rectify:=false` and point `rect_image:=/your/rectified/image` to that stream, to avoid rectifying twice.

**2. Relative monocular depth:**

```bash
./scripts/run_depth_estimation.sh \
  -p input_topic:=/oakd/rgb/preview/image_rect_clean \
  -p output_topic:=/depth_anything/depth \
  -p checkpoint_dir:="$HOME/models/Depth-Anything-V2/checkpoints"
```

**3. LiDAR projection and expected-depth points:**

```bash
./scripts/run_lidar_projection.sh \
  -p image_topic:=/oakd/rgb/preview/image_rect_clean \
  -p scan_topic:=/scan \
  -p camera_info_topic:=/oakd/rgb/preview/camera_info \
  -p relative_depth_topic:=/depth_anything/depth \
  -p expected_max_residual:=0.5 \
  -p show_rejected_expected_overlay:=false
```

The projector publishes `/cross_modal/lidar_projected_points` and `/cross_modal/lidar_expected_projection`; defaults are in `src/lidar_projection/config/projector.yaml`.

**4. Cross-modal anomaly detection:**

```bash
./scripts/run_anomaly_detection.sh \
  -p projected_points_topic:=/cross_modal/lidar_projected_points \
  -p estimated_points_topic:=/cross_modal/lidar_expected_projection \
  -p depth_topic:=/depth_anything/depth \
  -p return_coverage_scan_topic:=/scan \
  -p valid_mask_topic:=/rgb_rectification/valid_mask \
  -p odom_topic:=/odom
```

Outputs: `/cross_modal_anomaly/status` (`std_msgs/String`, JSON) and `/cross_modal_anomaly/anomaly` (`std_msgs/Bool`); the detector configuration lives in `src/cross_modal_anomaly_detection/config/detector.yaml`.

**5. Motion/anomaly status for RViz:**

```bash
./scripts/run_motion_status.sh \
  --topic /odom \
  --anomaly-topic /cross_modal_anomaly/status \
  --anomaly-type json \
  --linear-threshold 0.035 \
  --angular-threshold 0.10 \
  --stop-delay 0.75
```

In RViz, add an **Image** display with topic `/turtlebot/motion_status/image`. The fault simulator is **not needed** in this workflow.

## Run with optional simulated RGB/LiDAR faults

This mode modifies copies of sensor messages on `/faults/...` topics. **Keep navigation, control, and safety-related consumers on the original unmodified sensor topics.** For a repeatable experiment, start sensor publishers and RGB preprocessing as in stage 1 above, then launch the following nodes (separate terminals) in the shown order.

**A. Fault injector** (reads clean RGB and the *real* scan, publishes separate corrupted streams):

```bash
./scripts/run_fault_injector.sh \
  -p rgb_in:=/oakd/rgb/preview/image_rect_clean \
  -p scan_in:=/scan \
  -p rgb_out:=/faults/rgb/image_rect \
  -p scan_out:=/faults/scan \
  -p keys_topic:=/faults/keys
```

**B. Depth on faulted RGB:**

```bash
./scripts/run_depth_estimation.sh \
  -p input_topic:=/faults/rgb/image_rect \
  -p output_topic:=/depth_anything/depth \
  -p checkpoint_dir:="$HOME/models/Depth-Anything-V2/checkpoints"
```

**C. Project faulted RGB, faulted LiDAR and corresponding depth:**

```bash
./scripts/run_lidar_projection.sh \
  -p image_topic:=/faults/rgb/image_rect \
  -p scan_topic:=/faults/scan \
  -p camera_info_topic:=/oakd/rgb/preview/camera_info \
  -p relative_depth_topic:=/depth_anything/depth \
  -p expected_max_residual:=0.5 \
  -p show_rejected_expected_overlay:=false
```

**D. Anomaly detection using the *faulted* scan:**

```bash
./scripts/run_anomaly_detection.sh \
  -p projected_points_topic:=/cross_modal/lidar_projected_points \
  -p estimated_points_topic:=/cross_modal/lidar_expected_projection \
  -p depth_topic:=/depth_anything/depth \
  -p return_coverage_scan_topic:=/faults/scan \
  -p valid_mask_topic:=/rgb_rectification/valid_mask \
  -p odom_topic:=/odom
```

**E. RViz status overlay:**

```bash
./scripts/run_motion_status.sh --topic /odom \
  --anomaly-topic /cross_modal_anomaly/status --anomaly-type json \
  --linear-threshold 0.035 --angular-threshold 0.10 --stop-delay 0.75
```

**F. Keyboard/GUI fault controller** (optional; requires desktop GUI):

```bash
./scripts/run_fault_controller.sh -p keys_topic:=/faults/keys
```

**Hold-to-activate:** Press and hold a keyboard key (with the controller window focused) or a GUI button to keep its simulated fault active, then release to deactivate. The GUI also offers **Toggle GUI buttons instead of holding**, for persistent mouse-click activation. `Esc` or `Space` clears all faults. Different Linux display servers can handle keyboard auto-repeat differently; for verification or longer experiments prefer GUI toggle mode.

The controller publishes `/faults/keys` (`sensor_msgs/Joy`) at 20 Hz while open. The injector periodically publishes `/faults/status` and changes the output sensor stream for as long as its current command state remains active. The final detector can require multiple distinct LiDAR scans before it raises an alarm.

## Runtime topic and parameter overrides

- RGB preprocessing uses ROS launch arguments: `rectify:=false rect_image:=/my/rectified/image`.
- Other ROS nodes use `-p parameter:=value` appended to their `run_*.sh` wrapper; these override the installed YAML/script defaults.
- RViz status uses command-line flags, e.g. `--topic /robot/odom --anomaly-topic /cross_modal_anomaly/status`.

For example, on a robot using `/lidar/scan` instead of `/scan`:

```bash
./scripts/run_lidar_projection.sh -p scan_topic:=/lidar/scan
./scripts/run_anomaly_detection.sh -p return_coverage_scan_topic:=/lidar/scan
```

To monitor an alternative anomaly output, use `-p status_topic:=/your/status` for the detector and `--anomaly-topic /your/status` for RViz.

### Selected detector parameters

The default research profile (`src/cross_modal_anomaly_detection/config/detector.yaml`) includes:

| Parameter | Default |
|---|---:|
| Spatial irregularity threshold | `0.20` |
| Minimum distinct abnormal measurements | `8` |
| Spatial irregularity neighborhood | `7` samples |
| Irregularity regions / overlap | `7` / `0.20` |
| Temporal voting | `12` of `20` unique scans |
| Weak-evidence confirmation | `3` of `4` |
| Expected-point residual gate | `0.5` |
| RGB edge guard | `8` pixels |

Settings are empirically tuned for the experimental setup. This implementation only detects *potential inconsistencies*, not the fault's cause or a safety-certified failure.
