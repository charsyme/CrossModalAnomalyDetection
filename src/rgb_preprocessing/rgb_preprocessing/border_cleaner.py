#!/usr/bin/env python3
"""Clean geometric rectification padding and publish a shared safe-pixel mask.

Does NOT crop, resize, change CameraInfo, or alter RGB pixel coordinates.
The geometric mask is derived from CameraInfo K/D/R/P and a white-image remap,
not from black-colored scene content. Only actual geometric invalid regions are
inpainted for model input. A separate optional horizontal edge guard excludes
valid but unreliable edge columns from both projection point clouds and anomaly
metrics, without altering RGB pixels in those columns.
"""
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def compute_safe_mask(info, margin=3):
    """Mask of fully sourced rectified RGB pixels (uint8, 255=valid)."""
    w, h = int(info.width), int(info.height)
    if w <= 0 or h <= 0:
        raise ValueError('CameraInfo must have positive dimensions')
    K = np.asarray(info.k, np.float64).reshape(3, 3)
    D = np.asarray(info.d, np.float64)
    R = np.asarray(info.r, np.float64).reshape(3, 3)
    P = np.asarray(info.p, np.float64).reshape(3, 4)
    if not all(np.all(np.isfinite(x)) for x in [K, D, R, P]):
        raise ValueError('Non-finite camera calibration')
    if info.distortion_model not in ('plumb_bob', 'rational_polynomial'):
        raise ValueError(f'Unsupported CameraInfo distortion_model: {info.distortion_model}')
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, R, P[:, :3],
                                               (w, h), cv2.CV_32FC1)
    coverage = cv2.remap(np.full((h, w), 255, dtype=np.uint8), mapx, mapy,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    safe = (coverage == 255).astype(np.uint8)
    margin = max(0, int(margin))
    if margin:
        # BORDER_CONSTANT treats pixels outside the image as unsafe, so
        # the model won't use a narrow artificial edge after inpainting.
        kernel = np.ones((2*margin+1, 2*margin+1), dtype=np.uint8)
        safe = cv2.erode(safe, kernel, borderType=cv2.BORDER_CONSTANT,
                         borderValue=0)
    return safe * 255


class BorderCleaner(Node):
    def __init__(self):
        super().__init__('rgb_border_cleaner')
        from types import SimpleNamespace
        p = lambda key, default: self.declare_parameter(key, default).value
        args = SimpleNamespace(
            input=str(p('input_topic', '/oakd/rgb/preview/image_rect')),
            camera_info=str(p('camera_info_topic', '/oakd/rgb/preview/camera_info')),
            output=str(p('output_topic', '/oakd/rgb/preview/image_rect_clean')),
            mask=str(p('valid_mask_topic', '/rgb_rectification/valid_mask')),
            margin=int(p('margin_px', 3)),
            edge_guard_px=int(p('edge_guard_px', 8)),
            radius=float(p('inpaint_radius', 3.0)),
        )
        self.bridge = CvBridge()
        self.args = args
        self.safe_mask = None
        self.projection_mask = None
        self.calibration_key = None
        self.rgb_pub = self.create_publisher(Image, args.output, 10)
        self.mask_pub = self.create_publisher(Image, args.mask, 10)
        self.create_subscription(CameraInfo, args.camera_info, self.on_info,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, args.input, self.on_rgb,
                                 qos_profile_sensor_data)
        self.get_logger().info('RGB border cleaning: %s -> %s; mask=%s' %
                               (args.input, args.output, args.mask))

    def on_info(self, info):
        key = (int(info.width), int(info.height), info.distortion_model,
               tuple(info.k), tuple(info.d), tuple(info.r), tuple(info.p))
        if key == self.calibration_key:
            return
        try:
            mask = compute_safe_mask(info, self.args.margin)
        except Exception as exc:
            self.safe_mask = None
            self.projection_mask = None
            self.calibration_key = None
            self.get_logger().error('Cannot build rectification mask: %s' % exc)
            return
        # Keep the cleaning/inpainting mask separate from the stricter mask
        # used for projection and anomaly scoring. Do not inpaint legitimate
        # RGB content merely because its x-coordinate is near the image edge.
        projection_mask = mask.copy()
        edge = max(0, int(self.args.edge_guard_px))
        if 2 * edge >= int(info.width):
            self.safe_mask = None
            self.projection_mask = None
            self.calibration_key = None
            self.get_logger().error(
                'edge-guard-px=%d is too large for image width=%d' %
                (edge, info.width))
            return
        if edge:
            projection_mask[:, :edge] = 0
            projection_mask[:, -edge:] = 0
        self.safe_mask = mask
        self.projection_mask = projection_mask
        self.calibration_key = key
        self.get_logger().info(
            'Rectification coverage: %.2f%%; projection/anomaly safe pixels: %.2f%% '
            '(rectification margin %dpx, horizontal edge guard %dpx)' %
            (100 * np.mean(mask != 0), 100 * np.mean(projection_mask != 0),
             self.args.margin, edge))

    def on_rgb(self, msg):
        mask = self.safe_mask
        projection_mask = self.projection_mask
        if mask is None or projection_mask is None:
            self.get_logger().warn('Waiting for camera calibration before publishing',
                                   throttle_duration_sec=5.0)
            return
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error('RGB decode failed: %s' % exc)
            return
        if rgb.shape[:2] != mask.shape:
            self.get_logger().error('RGB %s != CameraInfo mask %s' %
                                    (rgb.shape[:2], mask.shape))
            return
        # Inpaint invalid pixels only. The resulting pixels remain INVALID
        # for any geometric measurement or anomaly decision.
        invalid = np.where(mask == 0, 255, 0).astype(np.uint8)
        clean = cv2.inpaint(rgb, invalid, self.args.radius, cv2.INPAINT_TELEA) \
                if np.any(invalid) else rgb.copy()
        clean_msg = self.bridge.cv2_to_imgmsg(clean, encoding='bgr8')
        clean_msg.header = msg.header
        mask_msg = self.bridge.cv2_to_imgmsg(projection_mask, encoding='mono8')
        mask_msg.header = msg.header
        self.mask_pub.publish(mask_msg)
        self.rgb_pub.publish(clean_msg)


def main():
    rclpy.init()
    node = BorderCleaner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
