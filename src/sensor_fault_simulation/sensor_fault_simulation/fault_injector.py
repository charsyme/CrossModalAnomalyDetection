#!/usr/bin/env python3
"""ROS 2 Jazzy simulated RGB/LiDAR malfunctions (no hardware writes).

Inputs:  RGB /oakd/rgb/preview/image_rect_clean, /scan,
         sensor_msgs/Joy /faults/keys from keyboard_multimodal_controller.py
Outputs: /faults/rgb/image_rect, /faults/scan
         /faults/status and /faults/ground_truth (std_msgs/String JSON)

Only corrupted OUTPUT topics are published. Original topics remain unchanged.
Not a certified safety system; never connect fault outputs to robot navigation.
"""
import copy
import json
import time
from collections import Counter, deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, LaserScan, Joy
from std_msgs.msg import String
from cv_bridge import CvBridge

from .fault_effects import KEYS, FAULTS, corrupt_rgb, corrupt_scan

RGB_IMAGE_KEYS = set('1235690bsxv')
LIDAR_SCAN_KEYS = set('qwrtgxv')


def stamp_as_float(stamp):
    return float(stamp.sec) + float(stamp.nanosec)*1e-9


class FaultInjector(Node):
    def __init__(self):
        super().__init__('fault_injector')
        p = lambda name, default: self.declare_parameter(name, default).value
        self.rgb_in = str(p('rgb_in','/oakd/rgb/preview/image_rect'))
        self.scan_in = str(p('scan_in','/scan'))
        self.rgb_out = str(p('rgb_out','/faults/rgb/image_rect'))
        self.scan_out = str(p('scan_out','/faults/scan'))
        self.keys_topic = str(p('keys_topic','/faults/keys'))
        inputs = {self.rgb_in, self.scan_in}
        outputs = [self.rgb_out, self.scan_out]
        if any(out in inputs for out in outputs) or len(set(outputs)) != 2:
            raise ValueError('Fault outputs must be unique and different from every original sensor input topic')
        if self.scan_out == '/scan':
            raise ValueError('Refusing to publish simulated LiDAR measurements onto /scan')
        self.region_index = int(p('region_index',2))  # 0..4, 2 = middle
        self.realistic_severity = float(np.clip(p('realistic_severity',0.65),0.,1.))
        self.rgb_delay_frames = max(1, int(p('rgb_delay_frames',4)))
        self.lidar_sector_center_deg = float(p('lidar_sector_center_deg',0.0))
        self.delayed_rgb_stamp_mode = str(p('delayed_rgb_stamp_mode','original')).lower()
        self.lidar_replay_stamp_mode = str(p('lidar_replay_stamp_mode','original')).lower()
        for mode in (self.delayed_rgb_stamp_mode,self.lidar_replay_stamp_mode):
            if mode not in ('original','current'):
                raise ValueError('frame replay stamp mode must be original or current')
        self.rgb_history = deque(maxlen=max(3,self.rgb_delay_frames+2))
        self.previous_scan = None
        self.frame_index = Counter()
        self.started_at = {}
        self.last_rgb_fault_stats = {}
        # ROS-configurable stronger faults. The original topics stay untouched.
        self.rgb_overexposure_gain=float(p('rgb_overexposure_gain',3.2))
        self.rgb_overexposure_offset=float(p('rgb_overexposure_offset',105.0))
        self.crack_seed=int(p('crack_seed',137))
        self.crack_intensity=float(p('crack_intensity',1.0))
        self.lidar_dropout_sector_deg=float(p('lidar_dropout_sector_deg',360.0))
        self.lidar_range_bias_factor=float(p('lidar_range_bias_factor',1.55))
        self.lidar_range_warp_amplitude=float(p('lidar_range_warp_amplitude',1.20))
        self.lidar_fake_distance_m=float(p('lidar_fake_distance_m',0.65))
        self.lidar_random_dropout_fraction=float(p('lidar_random_dropout_fraction',0.70))
        self.lidar_range_noise_fraction=float(p('lidar_range_noise_fraction',0.30))
        self.stale_keys_timeout_sec = max(.2,float(p('stale_keys_timeout_sec',1.0)))
        self.stamp_mode = str(p('freeze_timestamp_mode','stale')).lower()
        if self.stamp_mode not in ('stale','current'):
            raise ValueError('freeze_timestamp_mode must be stale or current')
        self.rng = np.random.default_rng(int(p('seed',42)))
        self.held = set()
        self.last_keys = None
        self.last_cmd_rx = None
        self.frozen_rgb = None
        self.counts = Counter()
        self.last_lidar_stats = {}
        self.bridge = CvBridge()

        sensor_in = QoSProfile(history=HistoryPolicy.KEEP_LAST,depth=5,
                               reliability=ReliabilityPolicy.BEST_EFFORT,
                               durability=DurabilityPolicy.VOLATILE)
        # Reliable publishers are compatible with either reliable or best-effort readers.
        sensor_out = QoSProfile(history=HistoryPolicy.KEEP_LAST,depth=5,
                                reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.VOLATILE)
        self.rgb_pub = self.create_publisher(Image, self.rgb_out, sensor_out)
        self.scan_pub = self.create_publisher(LaserScan,self.scan_out,sensor_out)
        self.create_subscription(Image,self.rgb_in,self.rgb_cb,sensor_in)
        self.create_subscription(LaserScan,self.scan_in,self.scan_cb,sensor_in)
        self.create_subscription(Joy,self.keys_topic,self.keys_cb,10)
        self.status_pub = self.create_publisher(String,'/faults/status',10)
        self.truth_pub = self.create_publisher(String,'/faults/ground_truth',10)
        self.create_timer(.2,self.timer_cb)
        self.get_logger().info(f'RGB: {self.rgb_in} -> {self.rgb_out}')
        self.get_logger().info(f'LiDAR: {self.scan_in} -> {self.scan_out}')
        self.get_logger().info(f'Joy: {self.keys_topic}; indices: {" ".join(KEYS)}')
        self.get_logger().info(f'Realistic RGB/LiDAR fault keys enabled; severity={self.realistic_severity:.2f}')
        self.get_logger().warning('FAULT OUTPUTS ARE FOR OFFLINE/CONTROLLED EXPERIMENTS ONLY. '
                                  'Do not use them as inputs to robot navigation or safety controllers.')

    def publish_truth(self, reason='changed'):
        payload = dict(stamp=stamp_as_float(self.get_clock().now().to_msg()),
                       monotonic=time.monotonic(), reason=reason,
                       active_keys=[k for k in KEYS if k in self.held],
                       active_faults={k:FAULTS[k][1] for k in KEYS if k in self.held},
                       realistic_severity=self.realistic_severity)
        self.truth_pub.publish(String(data=json.dumps(payload)))
        self.get_logger().info(f'Fault state: {payload["active_keys"]} ({reason})')

    def keys_cb(self,msg):
        self.last_cmd_rx = time.monotonic()
        # Keep historical button indices but reject unselected faults even if
        # an older publisher sets a removed button to 1.
        new = {key for i,key in enumerate(KEYS)
               if key in FAULTS and i<len(msg.buttons) and int(msg.buttons[i]) > 0}
        if new != self.held:
            now = time.monotonic()
            for key in new - self.held:
                self.started_at[key] = now
            for key in self.held - new:
                self.started_at.pop(key,None)
            self.held = new
            # The 16-mode allowlist excludes replay and freeze faults.
            self.frozen_rgb = None
            self.publish_truth()

    def timer_cb(self):
        if (self.last_cmd_rx is not None and
              time.monotonic()-self.last_cmd_rx>self.stale_keys_timeout_sec):
            self.last_cmd_rx = None
            if self.held:
                self.held.clear()
                self.started_at.clear()
                self.rgb_history.clear()
                self.frozen_rgb = None
                self.publish_truth('control_watchdog_timeout')
        payload = dict(active_keys=[k for k in KEYS if k in self.held],
                       active_faults={k:FAULTS[k][1] for k in KEYS if k in self.held},
                       input_counts=dict(self.counts),
                       control_connected=self.last_cmd_rx is not None,
                       frozen_rgb=self.frozen_rgb is not None,
                       realistic_severity=self.realistic_severity,
                       rgb_delay_frames=self.rgb_delay_frames,
                       rgb_fault_stats=self.last_rgb_fault_stats,
                       lidar_output_stats=self.last_lidar_stats)
        self.status_pub.publish(String(data=json.dumps(payload)))

    def active_elapsed(self, keys):
        starts = [self.started_at[k] for k in keys if k in self.started_at]
        return max(0.,time.monotonic()-min(starts)) if starts else 0.

    def rgb_cb(self, msg):
        self.counts['rgb'] += 1
        self.frame_index['rgb'] += 1
        a = self.held.intersection(FAULTS)
        freeze = bool({'4','c'} & a)
        if not freeze:
            self.frozen_rgb = None
        if freeze and self.frozen_rgb is not None:
            out = copy.deepcopy(self.frozen_rgb)
            if self.stamp_mode == 'current':
                out.header = copy.deepcopy(msg.header)
            self.rgb_pub.publish(out)
            return

        source_msg = msg
        if 'y' in a:
            self.rgb_history.append(copy.deepcopy(msg))
            if len(self.rgb_history) <= self.rgb_delay_frames:
                # Pipeline buffering: first frames not delivered until delay is filled.
                self.last_rgb_fault_stats = {'latency_buffering':True,
                    'buffered':len(self.rgb_history)}
                return
            source_msg = copy.deepcopy(self.rgb_history[-(self.rgb_delay_frames+1)])
        else:
            self.rgb_history.clear()

        if not (a & RGB_IMAGE_KEYS):
            out = source_msg
            self.last_rgb_fault_stats = dict(
                rgb_delay_active=('y' in a), delay_frames=self.rgb_delay_frames if 'y' in a else 0,
                delayed_timestamp_mode=self.delayed_rgb_stamp_mode if 'y' in a else 'none')
        else:
            try:
                bgr = self.bridge.imgmsg_to_cv2(source_msg,desired_encoding='bgr8')
                result = corrupt_rgb(
                    bgr,a,self.rng,self.region_index,
                    overexposure_gain=self.rgb_overexposure_gain,
                    overexposure_offset=self.rgb_overexposure_offset,
                    crack_seed=self.crack_seed,
                    crack_intensity=self.crack_intensity,
                    severity=self.realistic_severity,
                    frame_index=self.frame_index['rgb'],
                    elapsed_s=self.active_elapsed(a & RGB_IMAGE_KEYS))
                out = self.bridge.cv2_to_imgmsg(result,encoding='bgr8')
                out.header = copy.deepcopy(source_msg.header)
                self.last_rgb_fault_stats = {
                    'changed_fraction_gt_10':round(float(np.mean(np.max(abs(result.astype(np.int16)-bgr.astype(np.int16)),axis=2)>10)),3),
                    'mean_abs_change':round(float(np.mean(abs(result.astype(np.float32)-bgr.astype(np.float32)))),2),
                    'rgb_delay_active':'y' in a,
                }
            except Exception as exc:
                self.get_logger().error(f'RGB corruption failed: {exc}')
                out = source_msg
        if 'y' in a and self.delayed_rgb_stamp_mode == 'current':
            out = copy.deepcopy(out)
            out.header = copy.deepcopy(msg.header)
        if freeze:
            self.frozen_rgb = copy.deepcopy(out)
        self.rgb_pub.publish(out)

    @staticmethod
    def scan_change_stats(original, corrupted):
        """Simple quantitative check that a simulated LiDAR fault reached /faults/scan."""
        a=np.asarray(original.ranges,dtype=np.float64)
        b=np.asarray(corrupted.ranges,dtype=np.float64)
        if not a.size or a.shape!=b.shape:
            return {'n':int(a.size)}
        av=np.isfinite(a)&(a>=float(original.range_min))&(a<=float(original.range_max))
        bv=np.isfinite(b)&(b>=float(corrupted.range_min))&(b<=float(corrupted.range_max))
        mismatch=av^bv
        both=av&bv
        mismatch[both]=np.abs(a[both]-b[both])>0.02
        return {'n':int(a.size),
                'input_valid_fraction':round(float(av.mean()),3),
                'output_valid_fraction':round(float(bv.mean()),3),
                'changed_fraction':round(float(mismatch.mean()),3)}

    def scan_cb(self,msg):
        self.counts['lidar'] += 1
        self.frame_index['lidar'] += 1
        a = self.held.intersection(FAULTS)
        source_msg = msg
        # Simulate short processing stalls: replay a recent physical scan.
        if 'h' in a and self.previous_scan is not None and self.frame_index['lidar'] % 5 in (0,1,2):
            source_msg = copy.deepcopy(self.previous_scan)
            if self.lidar_replay_stamp_mode == 'current':
                source_msg.header = copy.deepcopy(msg.header)
        self.previous_scan = copy.deepcopy(msg)
        out = source_msg
        if a & LIDAR_SCAN_KEYS:
            try:
                out = corrupt_scan(
                    copy.deepcopy(source_msg),a,self.rng,
                    sector_deg=self.lidar_dropout_sector_deg,
                    bias_factor=self.lidar_range_bias_factor,
                    range_warp_amplitude=self.lidar_range_warp_amplitude,
                    fake_distance_m=self.lidar_fake_distance_m,
                    random_dropout_fraction=self.lidar_random_dropout_fraction,
                    range_noise_fraction=self.lidar_range_noise_fraction,
                    severity=self.realistic_severity,
                    frame_index=self.frame_index['lidar'],
                    elapsed_s=self.active_elapsed(a & LIDAR_SCAN_KEYS),
                    lidar_sector_center_deg=self.lidar_sector_center_deg)
            except Exception as exc:
                self.get_logger().error(f'LiDAR corruption failed: {exc}')
                out = source_msg
        self.last_lidar_stats=self.scan_change_stats(msg,out)
        self.last_lidar_stats['replay_active']='h' in a
        self.scan_pub.publish(out)


def main():
    rclpy.init()
    node = FaultInjector()
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
