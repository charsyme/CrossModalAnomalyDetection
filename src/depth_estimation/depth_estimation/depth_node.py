import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from sensor_msgs.msg import (
    CompressedImage,
    Image,
)

from cv_bridge import CvBridge

from depth_estimation.depth_anything_wrapper import (
    DepthAnythingV2Estimator,
)


class DepthAnythingNode(Node):

    def __init__(self):

        super().__init__('depth_anything_node')

        # ====================================================
        # Topic parameters
        # ====================================================

        self.input_topic = self.declare_parameter(
            'input_topic',
            '/oakd/rgb/preview/image_rect_clean',
        ).value

        self.output_topic = self.declare_parameter(
            'output_topic',
            '/depth_anything/depth',
        ).value

        # "raw"        -> sensor_msgs/msg/Image
        # "compressed" -> sensor_msgs/msg/CompressedImage
        self.input_transport = str(
            self.declare_parameter(
                'input_transport',
                'raw',
            ).value
        ).strip().lower()

        if self.input_transport not in (
            'raw',
            'compressed',
        ):
            raise ValueError(
                'input_transport must be '
                '"raw" or "compressed".'
            )

        # ====================================================
        # Depth Anything parameters
        # ====================================================

        self.depth_mode = self.declare_parameter(
            'mode',
            'relative',
        ).value

        self.depth_encoder = self.declare_parameter(
            'encoder',
            'vits',
        ).value

        self.metric_dataset = self.declare_parameter(
            'metric_dataset',
            'hypersim',
        ).value

        # Empty string selects depth_estimation/checkpoints beside the Python module.
        self.checkpoint_dir = str(self.declare_parameter(
            'checkpoint_dir',
            '',
        ).value)

        self.depth_input_size = int(
            self.declare_parameter(
                'input_size',
                518,
            ).value
        )

        self.print_every = max(
            1,
            int(
                self.declare_parameter(
                    'print_every',
                    30,
                ).value
            ),
        )

        # ====================================================
        # Temporal stabilization
        # ====================================================

        self.temporal_smoothing = bool(
            self.declare_parameter(
                'temporal_smoothing',
                True,
            ).value
        )

        # EMA weight of the CURRENT frame.
        # Smaller -> stronger smoothing, but more lag.
        self.temporal_alpha = float(
            self.declare_parameter(
                'temporal_alpha',
                0.30,
            ).value
        )
        self.temporal_alpha = float(
            np.clip(self.temporal_alpha, 0.0, 1.0)
        )

        # Optional light spatial denoising before temporal EMA.
        # 0 disables it. Use 3 or 5 if desired.
        self.spatial_median_kernel = int(
            self.declare_parameter(
                'spatial_median_kernel',
                0,
            ).value
        )
        if self.spatial_median_kernel not in (0, 3, 5):
            raise ValueError(
                'spatial_median_kernel must be 0, 3, or 5.'
            )

        self.prev_depth = None

        # ====================================================
        # QoS
        # ====================================================

        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ====================================================
        # CV bridge
        # ====================================================

        self.bridge = CvBridge()

        # ====================================================
        # Depth Anything V2
        # ====================================================

        self.get_logger().info(
            'Loading Depth Anything V2'
        )

        self.get_logger().info(
            f'mode={self.depth_mode} | '
            f'encoder={self.depth_encoder} | '
            f'metric_dataset={self.metric_dataset} | '
            f'input_size={self.depth_input_size}'
        )

        self.depth_estimator = (
            DepthAnythingV2Estimator(
                mode=self.depth_mode,
                encoder=self.depth_encoder,
                metric_dataset=self.metric_dataset,
                input_size=self.depth_input_size,
                checkpoint_dir=self.checkpoint_dir or None,
            )
        )

        if self.depth_estimator.is_metric:

            self.get_logger().info(
                'Depth output: METRIC depth [metres]'
            )

        else:

            self.get_logger().info(
                'Depth output: RELATIVE depth'
            )

        # ====================================================
        # RGB subscriber
        #
        # IMPORTANT:
        # For the current LiDAR projection pipeline, use the
        # rectified RGB image by default:
        #
        #   /oakd/rgb/preview/image_rect_clean
        #
        # so that the depth map uses the same pixel geometry as
        # /lidar_projected_points.
        # ====================================================

        if self.input_transport == 'raw':

            self.image_sub = self.create_subscription(
                Image,
                self.input_topic,
                self.raw_image_callback,
                camera_qos,
            )

        else:

            self.image_sub = self.create_subscription(
                CompressedImage,
                self.input_topic,
                self.compressed_image_callback,
                camera_qos,
            )

        # ====================================================
        # Depth publisher
        #
        # sensor_msgs/Image
        # encoding = 32FC1
        #
        # relative:
        #     relative depth values
        #
        # metric:
        #     depth in metres
        # ====================================================

        self.depth_pub = self.create_publisher(
            Image,
            self.output_topic,
            10,
        )

        self.frame_counter = 0

        # ====================================================
        # Startup information
        # ====================================================

        self.get_logger().info(
            f'RGB input: {self.input_topic}'
        )

        self.get_logger().info(
            f'Input transport: {self.input_transport}'
        )

        self.get_logger().info(
            f'Depth output: {self.output_topic}'
        )

        self.get_logger().info(
            f'Temporal smoothing: {self.temporal_smoothing} | '
            f'alpha={self.temporal_alpha:.2f} | '
            f'median_kernel={self.spatial_median_kernel}'
        )

    # ========================================================
    # Raw RGB callback
    # ========================================================

    def raw_image_callback(
        self,
        msg,
    ):

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8',
            )

        except Exception as exc:

            self.get_logger().error(
                'Could not decode raw RGB image: '
                f'{type(exc).__name__}: {exc}'
            )

            return

        self.process_frame(
            frame,
            msg.header,
        )

    # ========================================================
    # Compressed RGB callback
    # ========================================================

    def compressed_image_callback(
        self,
        msg,
    ):

        try:

            compressed = np.frombuffer(
                msg.data,
                dtype=np.uint8,
            )

            frame = cv2.imdecode(
                compressed,
                cv2.IMREAD_COLOR,
            )

            if frame is None:

                self.get_logger().warning(
                    'Failed to decode compressed RGB frame'
                )

                return

        except Exception as exc:

            self.get_logger().error(
                'Could not decode compressed RGB image: '
                f'{type(exc).__name__}: {exc}'
            )

            return

        self.process_frame(
            frame,
            msg.header,
        )

    # ========================================================
    # Common inference path
    # ========================================================

    def process_frame(
        self,
        frame,
        header,
    ):

        try:

            frame = np.asarray(
                frame,
            )

            if (
                frame.ndim != 3
                or frame.shape[2] != 3
            ):

                self.get_logger().warning(
                    f'Expected HxWx3 RGB image, '
                    f'got {frame.shape}'
                )

                return

            image_height = int(
                frame.shape[0]
            )

            image_width = int(
                frame.shape[1]
            )

            # =================================================
            # Depth Anything inference
            # =================================================

            start_time = (
                time.perf_counter()
            )

            depth = self.depth_estimator.predict(
                frame,
                input_format='bgr',
            )

            inference_ms = (
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000.0
            )

            # =================================================
            # Validate depth map
            # =================================================

            depth = np.asarray(
                depth,
                dtype=np.float32,
            )

            if depth.ndim != 2:

                self.get_logger().error(
                    f'Expected HxW depth map, '
                    f'got {depth.shape}'
                )

                return

            # =================================================
            # Keep exact input-image pixel geometry
            #
            # This is important because downstream:
            #
            #   depth[v, u]
            #
            # is compared with the u/v coordinates from
            # /lidar_projected_points.
            # =================================================

            if (
                depth.shape[0]
                != image_height
                or depth.shape[1]
                != image_width
            ):

                self.get_logger().warning(
                    'Depth resolution '
                    f'{depth.shape[1]}x'
                    f'{depth.shape[0]} differs '
                    'from RGB resolution '
                    f'{image_width}x'
                    f'{image_height}; resizing.'
                )

                depth = cv2.resize(
                    depth,
                    (
                        image_width,
                        image_height,
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )

            # =================================================
            # Optional spatial + temporal stabilization
            # =================================================

            if self.spatial_median_kernel > 0:
                depth = cv2.medianBlur(
                    depth.astype(np.float32),
                    self.spatial_median_kernel,
                )

            if self.temporal_smoothing:
                if (
                    self.prev_depth is None
                    or self.prev_depth.shape != depth.shape
                ):
                    self.prev_depth = depth.copy()
                else:
                    valid_now = np.isfinite(depth)
                    valid_prev = np.isfinite(self.prev_depth)
                    both = valid_now & valid_prev

                    filtered = depth.copy()
                    filtered[both] = (
                        self.temporal_alpha * depth[both]
                        + (1.0 - self.temporal_alpha)
                        * self.prev_depth[both]
                    )

                    # If the current frame is invalid at a pixel,
                    # retain the previous valid value.
                    use_prev = (~valid_now) & valid_prev
                    filtered[use_prev] = self.prev_depth[use_prev]

                    depth = filtered
                    self.prev_depth = depth.copy()

            depth = np.ascontiguousarray(
                depth,
                dtype=np.float32,
            )

            # =================================================
            # Publish depth
            # =================================================

            depth_msg = self.bridge.cv2_to_imgmsg(
                depth,
                encoding='32FC1',
            )

            # Preserve timestamp and frame of the exact RGB
            # image used for inference.
            depth_msg.header = header

            self.depth_pub.publish(
                depth_msg
            )

            # =================================================
            # Logging
            # =================================================

            self.frame_counter += 1

            if (
                self.frame_counter
                % self.print_every
                == 0
            ):

                finite = depth[
                    np.isfinite(depth)
                ]

                if finite.size > 0:

                    if self.depth_estimator.is_metric:
                        unit = 'm'
                    else:
                        unit = 'relative'

                    self.get_logger().info(
                        f'depth={depth.shape} | '
                        f'min={finite.min():.3f} | '
                        f'max={finite.max():.3f} | '
                        f'unit={unit} | '
                        f'inference={inference_ms:.1f} ms'
                    )

                else:

                    self.get_logger().warning(
                        'Published depth map contains '
                        'no finite values'
                    )

        except Exception as exc:

            self.get_logger().error(
                'Depth inference failed: '
                f'{type(exc).__name__}: {exc}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = DepthAnythingNode()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':

    main()
