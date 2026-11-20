#!/usr/bin/env python3
"""TurtleBot 4 motion + Depth/LiDAR anomaly status for RViz2 (ROS 2 Jazzy).

Publishes a *single* sensor_msgs/msg/Image containing two independently
highlighted status cards:

    MOTION STATUS: MOVING (green) / STOPPED (red) / NO ODOMETRY (gray)
    SYSTEM STATUS: NORMAL (green) / ANOMALY (red) / NO DATA (gray)

Movement comes from nav_msgs/msg/Odometry. The default anomaly source is
/cross_modal_anomaly/status (std_msgs/msg/String containing JSON), as published
by cross_modal_anomaly_detection.py. This exposes both the final
anomaly decision and check_valid. Other Bool or plain-String sources are
also supported with --anomaly-type bool / string.

No Tkinter or cv_bridge dependency is required. Add an Image display in RViz2
and select /turtlebot/motion_status/image.

Examples:
    python3 motion_status_rviz.py
    python3 motion_status_rviz.py --anomaly-topic /cross_modal_anomaly/status --anomaly-type json
    python3 motion_status_rviz.py --anomaly-topic /cross_modal_anomaly/anomaly --anomaly-type bool
    python3 motion_status_rviz.py --anomaly-timeout 0  # Disable expiry
"""

import argparse
import json
import math
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


# OpenCV colors use BGR order.
BACKGROUND = (27, 20, 17)
CARD = (42, 33, 27)
WHITE = (238, 232, 224)
MUTED = (186, 180, 173)
GREEN = (109, 238, 68)
RED = (89, 83, 255)
GRAY = (176, 168, 161)

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIDTH = 900
HEIGHT = 480


def centered_text(img, value, baseline_y, scale, color, thickness=2, max_width=None):
    """Center text and reduce font scale to avoid clipping long topic names."""
    if max_width is None:
        max_width = img.shape[1] - 88
    while scale > 0.30:
        (text_width, _), _ = cv2.getTextSize(value, FONT, scale, thickness)
        if text_width <= max_width:
            break
        scale *= 0.95
    x = max(8, (img.shape[1] - text_width) // 2)
    cv2.putText(img, value, (x, baseline_y), FONT, scale,
                color, thickness, cv2.LINE_AA)


def draw_card(img, rect, title, label, info, color):
    """Paint a status card with a bright border and colored status LED."""
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), CARD, -1, cv2.LINE_AA)

    # Highlight around each card is independent: movement and anomaly can
    # simultaneously have different colors.
    for padding, opacity in ((7, 0.11), (4, 0.16), (1, 0.19)):
        overlay = img.copy()
        cv2.rectangle(overlay, (x1 - padding, y1 - padding),
                      (x2 + padding, y2 + padding), color, 4, cv2.LINE_AA)
        img[:] = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 5, cv2.LINE_AA)
    cv2.rectangle(img, (x1 + 10, y1 + 10), (x2 - 10, y2 - 10),
                  color, 1, cv2.LINE_AA)

    centered_text(img, title, y1 + 38, 0.73, WHITE, 2)

    scale = 1.65 if len(label) <= 9 else 1.22
    thickness = 4
    (tw, _), _ = cv2.getTextSize(label, FONT, scale, thickness)
    radius = 13
    gap = 18
    overall_width = 2 * radius + gap + tw
    start_x = (WIDTH - overall_width) // 2
    baseline = y1 + 117
    led_x, led_y = start_x + radius, baseline - 19
    cv2.circle(img, (led_x, led_y), radius + 5, color, 1, cv2.LINE_AA)
    cv2.circle(img, (led_x, led_y), radius, color, -1, cv2.LINE_AA)
    cv2.putText(img, label, (start_x + 2 * radius + gap, baseline),
                FONT, scale, color, thickness, cv2.LINE_AA)

    centered_text(img, info, y1 + 156, 0.57, MUTED, 1)


def render_status_image(motion_state, status_state, linear_speed,
                        angular_speed, odom_topic, anomaly_topic, anomaly_type,
                        status_detail=""):
    """Draw the complete two-card RViz image (pure OpenCV, no ROS needed)."""
    img = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    centered_text(img, 'TURTLEBOT 4   |   LIVE ROBOT STATUS',
                  59, 0.88, WHITE, 2)

    motion_color, motion_label = {
        'moving': (GREEN, 'MOVING'),
        'stopped': (RED, 'STOPPED'),
        'unknown': (GRAY, 'NO ODOMETRY'),
    }[motion_state]
    if motion_state == 'unknown':
        motion_info = f'Waiting for odometry: {odom_topic}'
    else:
        motion_info = (f'Linear: {linear_speed:.3f} m/s'
                       f'       |       Angular: {angular_speed:.3f} rad/s')
    draw_card(img, (19, 91, 881, 271), 'MOTION STATUS',
              motion_label, motion_info, motion_color)

    status_color, status_label = {
        'normal': (GREEN, 'NORMAL'),
        'anomaly': (RED, 'ANOMALY'),
        'unknown': (GRAY, 'NO DATA'),
        'invalid': (GRAY, 'NO DATA'),
    }[status_state]
    if status_state == 'unknown':
        status_info = status_detail or f'Waiting for anomaly status: {anomaly_topic} ({anomaly_type})'
    elif status_state == 'invalid':
        status_info = f'Detection unavailable: {status_detail or "invalid comparison"}'
    elif status_state == 'normal':
        status_info = f'No anomaly reported    |    Source: {anomaly_topic}'
    else:
        status_info = f'ANOMALY DETECTED    |    Source: {anomaly_topic}'
    draw_card(img, (19, 288, 881, 468), 'SYSTEM STATUS',
              status_label, status_info, status_color)
    return img


def interpret_detector_status(raw):
    """Decode the detector's JSON status without confusing invalid with healthy.

    The detector publishes `anomaly` as its final, temporally voted decision,
    while `check_valid` indicates whether the comparison was valid. An active
    anomaly takes precedence over check_valid=False (e.g., a missing-stream
    alarm); otherwise invalid comparisons are displayed as NO DATA.
    """
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return 'unknown', 'Invalid JSON from anomaly detector'
    if not isinstance(payload, dict):
        return 'unknown', 'Expected a JSON object from anomaly detector'

    anomaly = payload.get('anomaly')
    valid = payload.get('check_valid')
    if type(anomaly) is not bool or type(valid) is not bool:
        return 'unknown', 'Missing anomaly or check_valid in detector status'
    if anomaly:
        return 'anomaly', ''
    if valid:
        return 'normal', ''
    reason = str(payload.get('reason') or 'comparison not valid')
    return 'invalid', reason[:90]


class MotionStatusImage(Node):
    """Subscribe to odometry/status and publish one image for RViz2."""

    def __init__(self, args):
        super().__init__('turtlebot_motion_status_rviz')
        self.args = args
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.last_odom_at = None
        self.last_moving_at = None
        self.last_anomaly_at = None
        self.anomaly_state = 'unknown'
        self.anomaly_detail = ''
        self.last_motion_state = None
        self.last_system_state = None

        self.odom_subscription = self.create_subscription(
            Odometry, args.topic, self.on_odometry, qos_profile_sensor_data)

        anomaly_class = Bool if args.anomaly_type == 'bool' else String
        self.anomaly_subscription = self.create_subscription(
            anomaly_class, args.anomaly_topic, self.on_anomaly,
            qos_profile_sensor_data)

        self.publisher = self.create_publisher(Image, args.image_topic, 10)
        self.timer = self.create_timer(1.0 / args.rate, self.publish_status_image)
        self.get_logger().info(
            f'Odometry: {args.topic} | Anomaly ({args.anomaly_type}): '
            f'{args.anomaly_topic} | RViz Image: {args.image_topic}')

    def on_odometry(self, msg):
        twist = msg.twist.twist
        self.linear_speed = math.hypot(twist.linear.x, twist.linear.y)
        self.angular_speed = math.sqrt(
            twist.angular.x ** 2 + twist.angular.y ** 2 + twist.angular.z ** 2)
        self.last_odom_at = time.monotonic()

    def on_anomaly(self, msg):
        detail = ''
        if self.args.anomaly_type == 'json':
            state, detail = interpret_detector_status(msg.data)
        elif self.args.anomaly_type == 'bool':
            state = 'anomaly' if msg.data else 'normal'
        else:
            # Only explicitly recognized plain text means NORMAL or ANOMALY.
            value = str(msg.data).strip().lower().replace('-', '_').replace(' ', '_')
            if value in {'anomaly', 'fault', 'alert', 'detected',
                         'anomaly_detected', 'fault_detected', 'true', '1'}:
                state = 'anomaly'
            elif value in {'normal', 'ok', 'healthy', 'clear',
                           'no_anomaly', 'no_fault', 'false', '0'}:
                state = 'normal'
            else:
                state = 'unknown'
                detail = f'Unrecognized anomaly status: {msg.data!s}'
        self.anomaly_state = state
        self.anomaly_detail = detail
        self.last_anomaly_at = time.monotonic()

    def get_motion_state(self, now):
        if (self.last_odom_at is None or
                now - self.last_odom_at > self.args.odom_timeout):
            self.last_moving_at = None
            return 'unknown'

        moving_now = (self.linear_speed > self.args.linear_threshold or
                      self.angular_speed > self.args.angular_threshold)
        if moving_now:
            self.last_moving_at = now
        moving = moving_now or (
            self.last_moving_at is not None and
            now - self.last_moving_at < self.args.stop_delay)
        return 'moving' if moving else 'stopped'

    def get_system_state(self, now):
        if self.last_anomaly_at is None:
            self.anomaly_detail = 'Waiting for first anomaly detector status'
            return 'unknown'
        if (self.args.anomaly_timeout > 0 and
                now - self.last_anomaly_at > self.args.anomaly_timeout):
            self.anomaly_detail = 'Anomaly detector status timed out'
            return 'unknown'
        return self.anomaly_state

    def publish_status_image(self):
        now = time.monotonic()
        motion = self.get_motion_state(now)
        system = self.get_system_state(now)
        if motion != self.last_motion_state:
            self.get_logger().info(f'Motion: {motion.upper()}')
            self.last_motion_state = motion
        if system != self.last_system_state:
            self.get_logger().info(f'System: {system.upper()}')
            self.last_system_state = system

        bgr = render_status_image(
            motion, system, self.linear_speed, self.angular_speed,
            self.args.topic, self.args.anomaly_topic, self.args.anomaly_type,
            self.anomaly_detail)
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.height, msg.width = bgr.shape[:2]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = bgr.tobytes()
        self.publisher.publish(msg)


def nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('Value must be finite and >= 0')
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('Value must be finite and > 0')
    return value


def main():
    parser = argparse.ArgumentParser(
        description='Publish one RViz image with motion and anomaly status')
    parser.add_argument('--topic', default='/odom', help='Odometry topic')
    parser.add_argument('--image-topic', default='/turtlebot/motion_status/image',
                        help='Output image topic (same as previous version)')
    parser.add_argument('--anomaly-topic', default='/cross_modal_anomaly/status',
                        help='ROS topic containing anomaly detection status')
    parser.add_argument('--anomaly-type', choices=('json', 'bool', 'string'), default='json',
                        help='json: final status + validity; bool: final anomaly; string: text status')
    parser.add_argument('--anomaly-timeout', type=nonnegative_float, default=3.0,
                        help='Status missing for this many seconds -> NO DATA; 0 disables')
    parser.add_argument('--linear-threshold', type=nonnegative_float, default=0.035,
                        help='Moving threshold for translation (m/s)')
    parser.add_argument('--angular-threshold', type=nonnegative_float, default=0.10,
                        help='Moving threshold for rotation (rad/s)')
    parser.add_argument('--stop-delay', type=nonnegative_float, default=0.75,
                        help='Briefly preserve MOVING when speed drops (s)')
    parser.add_argument('--odom-timeout', type=positive_float, default=1.50,
                        help='Missing odometry -> NO ODOMETRY (s)')
    parser.add_argument('--rate', type=positive_float, default=5.0,
                        help='Image publishing frequency (Hz)')
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = MotionStatusImage(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
