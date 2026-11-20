#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "rclcpp/rclcpp.hpp"

#include "builtin_interfaces/msg/time.hpp"

#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "sensor_msgs/image_encodings.hpp"

#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "cv_bridge/cv_bridge.hpp"
#include "image_geometry/pinhole_camera_model.hpp"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Vector3.h"

using namespace std::chrono_literals;


class LidarProjectionNode : public rclcpp::Node
{
public:
  LidarProjectionNode()
  : Node("lidar_projection")
  {
    // ============================================================
    // Topics / calibration
    // ============================================================

    camera_info_topic_ = declare_parameter<std::string>(
      "camera_info_topic",
      "/oakd/rgb/preview/camera_info");

    scan_topic_ = declare_parameter<std::string>(
      "scan_topic",
      "/scan");

    projected_points_topic_ = declare_parameter<std::string>(
      "projected_points_topic",
      "/lidar_projected_points");

    expected_points_topic_ = declare_parameter<std::string>(
      "expected_points_topic",
      "/lidar_expected_projection");
    // Diagnostic-only: never feed this rejected cloud to NCC/irregularity.
    rejected_expected_points_topic_ = declare_parameter<std::string>(
      "rejected_expected_points_topic",
      "/lidar_expected_projection_rejected");
    // Suppress orange rejected candidates in the main visualization unless
    // diagnostic overlay is requested. Rejected candidate cloud remains separate.
    show_rejected_expected_overlay_ = declare_parameter<bool>(
      "show_rejected_expected_overlay", false);

    publish_expected_points_ = declare_parameter<bool>(
      "publish_expected_points",
      true);

    expected_publish_rate_hz_ = declare_parameter<double>(
      "expected_publish_rate_hz",
      10.0);

    relative_depth_topic_ = declare_parameter<std::string>(
      "relative_depth_topic",
      "/depth_anything/depth");

    expected_min_range_ = declare_parameter<double>(
      "expected_min_range",
      0.25);

    expected_max_range_ = declare_parameter<double>(
      "expected_max_range",
      10.0);

    expected_range_samples_ = declare_parameter<int>(
      "expected_range_samples",
      120);

    // Do not publish an expected point if even its BEST candidate poorly
    // matches the depth map. The earlier default (0.0) disabled this test,
    // causing artificial vertical columns at masked left/right boundaries.
    // The threshold is in native (model-dependent) relative depth units.
    expected_max_residual_ = declare_parameter<double>(
      "expected_max_residual",
      0.5);
    if (!std::isfinite(expected_max_residual_) || expected_max_residual_ <= 0.0) {
      throw std::invalid_argument(
        "expected_max_residual must be a finite positive value; "
        "0 disables rejection and is prohibited in the checked build");
    }

    relative_depth_patch_radius_ = declare_parameter<int>(
      "relative_depth_patch_radius",
      2);

    calibration_min_points_ = declare_parameter<int>(
      "calibration_min_points",
      10);

    // Faster update for WELL-SUPPORTED normal measurements only.
    calibration_alpha_ = declare_parameter<double>(
      "calibration_alpha",
      0.35);

    // Never refit from nearly black/white featureless RGB patches: the
    // projected LiDAR samples may have valid depth but corrupt RGB content.
    calibration_rgb_mask_enabled_ = declare_parameter<bool>(
      "calibration_rgb_mask_enabled", true);
    calibration_freeze_on_obstruction_ = declare_parameter<bool>(
      "calibration_freeze_on_obstruction", true);
    calibration_rgb_max_time_difference_ = declare_parameter<double>(
      "calibration_rgb_max_time_difference", 0.20);
    calibration_min_inlier_fraction_ = declare_parameter<double>(
      "calibration_min_inlier_fraction", 0.55);
    // Maximum allowed fit change relative to the current RGB-depth IQR.
    calibration_max_update_fraction_ = declare_parameter<double>(
      "calibration_max_update_fraction", 0.20);

    max_relative_depth_time_difference_ = declare_parameter<double>(
      "max_relative_depth_time_difference",
      0.15);

    camera_frame_ = declare_parameter<std::string>(
      "camera_frame",
      "oakd_rgb_camera_optical_frame");

    max_range_ = declare_parameter<double>(
      "max_range",
      -1.0);

    yaw_correction_deg_ = declare_parameter<double>(
      "yaw_correction_deg",
      2.1);

    // ============================================================
    // LiDAR filtering
    //
    // For a static-scene test:
    //   median window = 3
    //   range EMA alpha = 0.25
    //   UV EMA alpha = 0.25
    //
    // alpha = 1.0 means no temporal smoothing.
    // Smaller alpha means stronger smoothing.
    // ============================================================

    lidar_median_window_ = declare_parameter<int>(
      "lidar_median_window",
      3);

    lidar_temporal_filter_ = declare_parameter<bool>(
      "lidar_temporal_filter",
      true);

    lidar_temporal_alpha_ = declare_parameter<double>(
      "lidar_temporal_alpha",
      0.25);

    uv_temporal_filter_ = declare_parameter<bool>(
      "uv_temporal_filter",
      true);

    uv_temporal_alpha_ = declare_parameter<double>(
      "uv_temporal_alpha",
      0.25);

    // ============================================================
    // Fixed beam selection
    //
    // -1 / -1 => use all beams.
    //
    // Example:
    //   fixed_beam_start = 420
    //   fixed_beam_end   = 440
    //
    // Then exactly those LaserScan beam indices are considered on
    // every frame. This is more stable than reselecting points
    // based only on their current RGB pixel coordinates.
    // ============================================================

    fixed_beam_start_ = declare_parameter<int>(
      "fixed_beam_start",
      -1);

    fixed_beam_end_ = declare_parameter<int>(
      "fixed_beam_end",
      -1);

    // ============================================================
    // Visualization
    // ============================================================

    visualize_ = declare_parameter<bool>(
      "visualize",
      false);

    image_topic_ = declare_parameter<std::string>(
      "image_topic",
      "/oakd/rgb/preview/image_rect_clean");

    visualization_topic_ = declare_parameter<std::string>(
      "visualization_topic",
      "/lidar_rgb_projection_rect");

    expected_visualization_topic_ = declare_parameter<std::string>(
      "expected_visualization_topic",
      "/lidar_expected_projection_rect");

    point_radius_ = declare_parameter<int>(
      "point_radius",
      2);

    // Geometrically valid rectification pixels. 0=invalid, 255=valid.
    valid_mask_topic_ = declare_parameter<std::string>(
      "valid_mask_topic", "/rgb_rectification/valid_mask");
    require_valid_mask_ = declare_parameter<bool>(
      "require_valid_mask", true);

    // ============================================================
    // TF
    // ============================================================

    tf_buffer_ =
      std::make_unique<tf2_ros::Buffer>(get_clock());

    tf_listener_ =
      std::make_shared<tf2_ros::TransformListener>(
        *tf_buffer_);

    const auto qos = rclcpp::SensorDataQoS();

    // ============================================================
    // Subscribers
    // ============================================================

    camera_info_sub_ =
      create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_,
        qos,
        std::bind(
          &LidarProjectionNode::cameraInfoCallback,
          this,
          std::placeholders::_1));

    valid_mask_sub_ =
      create_subscription<sensor_msgs::msg::Image>(
        valid_mask_topic_, qos,
        std::bind(&LidarProjectionNode::validMaskCallback,
                  this, std::placeholders::_1));

    scan_sub_ =
      create_subscription<sensor_msgs::msg::LaserScan>(
        scan_topic_,
        qos,
        std::bind(
          &LidarProjectionNode::scanCallback,
          this,
          std::placeholders::_1));

    relative_depth_sub_ =
      create_subscription<sensor_msgs::msg::Image>(
        relative_depth_topic_,
        qos,
        std::bind(
          &LidarProjectionNode::relativeDepthCallback,
          this,
          std::placeholders::_1));

    // ============================================================
    // Publishers
    // ============================================================

    projected_points_pub_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(
        projected_points_topic_,
        10);

    if (publish_expected_points_) {
      expected_points_pub_ =
        create_publisher<sensor_msgs::msg::PointCloud2>(
          expected_points_topic_,
          10);
      rejected_expected_points_pub_ =
        create_publisher<sensor_msgs::msg::PointCloud2>(
          rejected_expected_points_topic_, 10);

      const double safe_rate =
        std::max(
          0.1,
          expected_publish_rate_hz_);

      const auto expected_period =
        std::chrono::milliseconds(
          static_cast<int>(
            std::lround(
              1000.0 / safe_rate)));

      expected_timer_ =
        create_wall_timer(
          expected_period,
          std::bind(
            &LidarProjectionNode::expectedProjectionTimerCallback,
            this));
    }

    // The RGB stream is needed by the calibration validity check even if
    // overlay rendering is disabled.
    if (visualize_ || calibration_rgb_mask_enabled_) {
      image_sub_ =
        create_subscription<sensor_msgs::msg::Image>(
          image_topic_,
          qos,
          std::bind(
            &LidarProjectionNode::imageCallback,
            this,
            std::placeholders::_1));
    }

    if (visualize_) {
      image_pub_ =
        create_publisher<sensor_msgs::msg::Image>(
          visualization_topic_,
          10);

      expected_image_pub_ =
        create_publisher<sensor_msgs::msg::Image>(
          expected_visualization_topic_,
          10);
    }

    // ============================================================
    // Startup log
    // ============================================================

    RCLCPP_INFO(
      get_logger(),
      "RGB_LIDAR_RESIDUAL_GATE_20261117 + RGB_FAULT_GUARD_20261118: confidence rejection ACTIVE; "
      "expected_max_residual=%.4f", expected_max_residual_);

    RCLCPP_INFO(
      get_logger(),
      "LiDAR -> RGB projection with temporal stabilization");

    RCLCPP_INFO(
      get_logger(),
      "scan: %s",
      scan_topic_.c_str());

    RCLCPP_INFO(
      get_logger(),
      "camera_info: %s",
      camera_info_topic_.c_str());

    RCLCPP_INFO(
      get_logger(),
      "projected points: %s",
      projected_points_topic_.c_str());

    if (publish_expected_points_) {
      RCLCPP_INFO(
        get_logger(),
        "estimated beam pixels: %s at %.2f Hz",
        expected_points_topic_.c_str(),
        expected_publish_rate_hz_);

      RCLCPP_INFO(
        get_logger(),
        "relative depth input: %s",
        relative_depth_topic_.c_str());

      RCLCPP_INFO(
        get_logger(),
        "estimated range search: %.2f..%.2f m, samples=%d",
        expected_min_range_,
        expected_max_range_,
        expected_range_samples_);

      RCLCPP_INFO(
        get_logger(),
        "expected max residual: %.4f (positive threshold, always enabled)",
        expected_max_residual_);
    }

    RCLCPP_INFO(
      get_logger(),
      "camera frame: %s",
      camera_frame_.c_str());

    RCLCPP_INFO(
      get_logger(),
      "yaw correction: %.3f deg",
      yaw_correction_deg_);

    RCLCPP_INFO(
      get_logger(),
      "LiDAR median window: %d",
      lidar_median_window_);

    RCLCPP_INFO(
      get_logger(),
      "LiDAR EMA: %s, alpha=%.3f",
      lidar_temporal_filter_ ? "on" : "off",
      lidar_temporal_alpha_);

    RCLCPP_INFO(
      get_logger(),
      "UV EMA: %s, alpha=%.3f",
      uv_temporal_filter_ ? "on" : "off",
      uv_temporal_alpha_);

    RCLCPP_INFO(
      get_logger(),
      "fixed beams: start=%d end=%d",
      fixed_beam_start_,
      fixed_beam_end_);

    if (visualize_ && publish_expected_points_) {
      RCLCPP_INFO(
        get_logger(),
        "RGB_LIDAR_DEPTH_OVERLAY_20261117: magenta expected points over "
        "relative DEPTH image (not RGB): %s",
        expected_visualization_topic_.c_str());
    }
  }

 private:

  struct ProjectedPoint
  {
    float u;
    float v;
    float range;

    float x;
    float y;
    float z;

    float camera_x;
    float camera_y;
    float camera_z;

    std::int32_t scan_index;
  };


  struct ExpectedPoint
  {
    float u;
    float v;
    float angle;
    float estimated_range;
    float camera_z;
    float residual;
    std::int32_t scan_index;
  };


  // ============================================================
  // Helpers
  // ============================================================

  bool validRange(
    float r,
    const sensor_msgs::msg::LaserScan & scan) const
  {
    if (!std::isfinite(r)) {
      return false;
    }

    if (r < scan.range_min ||
        r > scan.range_max)
    {
      return false;
    }

    if (max_range_ > 0.0 &&
        r > max_range_)
    {
      return false;
    }

    return true;
  }


  std::vector<float> spatialMedianFilter(
    const sensor_msgs::msg::LaserScan & scan) const
  {
    std::vector<float> output = scan.ranges;

    int window = std::max(
      1,
      lidar_median_window_);

    if ((window % 2) == 0) {
      window += 1;
    }

    if (window <= 1) {
      return output;
    }

    const int half = window / 2;

    for (std::size_t i = 0;
         i < scan.ranges.size();
         ++i)
    {
      const float centre = scan.ranges[i];

      // Do not create a point when the centre beam is invalid.
      if (!validRange(centre, scan)) {
        output[i] = centre;
        continue;
      }

      const int begin =
        std::max(
          0,
          static_cast<int>(i) - half);

      const int end =
        std::min(
          static_cast<int>(scan.ranges.size()) - 1,
          static_cast<int>(i) + half);

      std::vector<float> values;
      values.reserve(
        static_cast<std::size_t>(window));

      for (int j = begin; j <= end; ++j) {
        const float r =
          scan.ranges[
            static_cast<std::size_t>(j)];

        if (validRange(r, scan)) {
          values.push_back(r);
        }
      }

      if (!values.empty()) {
        std::sort(
          values.begin(),
          values.end());

        output[i] =
          values[
            values.size() / 2];
      }
    }

    return output;
  }


  void temporalRangeFilter(
    std::vector<float> & ranges,
    const sensor_msgs::msg::LaserScan & scan)
  {
    if (!lidar_temporal_filter_) {
      previous_ranges_ = ranges;
      return;
    }

    const double alpha =
      std::clamp(
        lidar_temporal_alpha_,
        0.0,
        1.0);

    if (previous_ranges_.size() != ranges.size()) {
      previous_ranges_ = ranges;
      return;
    }

    for (std::size_t i = 0;
         i < ranges.size();
         ++i)
    {
      const float current =
        ranges[i];

      const float previous =
        previous_ranges_[i];

      // Important:
      // if the current beam is invalid, keep it invalid.
      // We do not carry old geometry into a missing current beam.
      if (!validRange(current, scan)) {
        continue;
      }

      if (validRange(previous, scan)) {
        ranges[i] =
          static_cast<float>(
            alpha *
              static_cast<double>(current)
            +
            (1.0 - alpha) *
              static_cast<double>(previous));
      }
    }

    previous_ranges_ = ranges;
  }


  void ensureUvStateSize(
    std::size_t count)
  {
    if (previous_u_.size() == count) {
      return;
    }

    previous_u_.assign(
      count,
      std::numeric_limits<float>::quiet_NaN());

    previous_v_.assign(
      count,
      std::numeric_limits<float>::quiet_NaN());

    previous_uv_valid_.assign(
      count,
      false);
  }


  void temporalUvFilter(
    std::size_t scan_index,
    float & u,
    float & v)
  {
    if (!uv_temporal_filter_) {
      previous_u_[scan_index] = u;
      previous_v_[scan_index] = v;
      previous_uv_valid_[scan_index] = true;
      return;
    }

    const double alpha =
      std::clamp(
        uv_temporal_alpha_,
        0.0,
        1.0);

    if (previous_uv_valid_[scan_index]) {
      u =
        static_cast<float>(
          alpha * static_cast<double>(u) +
          (1.0 - alpha) *
            static_cast<double>(
              previous_u_[scan_index]));

      v =
        static_cast<float>(
          alpha * static_cast<double>(v) +
          (1.0 - alpha) *
            static_cast<double>(
              previous_v_[scan_index]));
    }

    previous_u_[scan_index] = u;
    previous_v_[scan_index] = v;
    previous_uv_valid_[scan_index] = true;
  }


  // ============================================================
  // Shared rectification valid-pixel mask
  // ============================================================

  void validMaskCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    try {
      const auto img = cv_bridge::toCvCopy(
        msg, sensor_msgs::image_encodings::MONO8);
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (have_camera_info_ &&
          (img->image.cols != image_width_ || img->image.rows != image_height_)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "Rectification mask size does not match CameraInfo; ignoring");
        valid_mask_.release();
        return;
      }
      valid_mask_ = img->image.clone();
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
        "Rectification mask decode failed: %s", e.what());
    }
  }

  static bool pixelInValidRegion(const cv::Mat & mask,
                                  double u, double v,
                                  int width, int height,
                                  bool mask_required)
  {
    if (!std::isfinite(u) || !std::isfinite(v) ||
        u < 0 || v < 0 || u >= width || v >= height) return false;
    if (mask.empty()) return !mask_required;
    if (mask.type() != CV_8UC1 || mask.cols != width || mask.rows != height)
      return false;
    const int ix = static_cast<int>(std::lround(u));
    const int iy = static_cast<int>(std::lround(v));
    return ix >= 0 && iy >= 0 && ix < width && iy < height &&
           mask.at<std::uint8_t>(iy, ix) != 0;
  }

  // ============================================================
  // Relative-depth input
  // ============================================================

  void relativeDepthCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    try {
      auto cv_ptr =
        cv_bridge::toCvCopy(
          msg,
          sensor_msgs::image_encodings::TYPE_32FC1);

      std::lock_guard<std::mutex>
        lock(data_mutex_);

      latest_relative_depth_ =
        cv_ptr->image.clone();

      latest_relative_depth_header_ =
        msg->header;

      have_relative_depth_ = true;
    }
    catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Relative-depth cv_bridge error: %s",
        e.what());
    }
  }


  static double stampSeconds(
    const builtin_interfaces::msg::Time & stamp)
  {
    return
      static_cast<double>(stamp.sec)
      + 1e-9 *
        static_cast<double>(stamp.nanosec);
  }


  bool sampleRelativeDepth(
    const cv::Mat & depth,
    double u,
    double v,
    int camera_width,
    int camera_height,
    const cv::Mat & valid_mask,
    float & value) const
  {
    if (depth.empty() ||
        depth.type() != CV_32FC1 ||
        camera_width <= 0 ||
        camera_height <= 0)
    {
      return false;
    }

    const double sx =
      static_cast<double>(depth.cols)
      / static_cast<double>(camera_width);

    const double sy =
      static_cast<double>(depth.rows)
      / static_cast<double>(camera_height);

    const int cx =
      static_cast<int>(
        std::lround(u * sx));

    const int cy =
      static_cast<int>(
        std::lround(v * sy));

    if (cx < 0 ||
        cx >= depth.cols ||
        cy < 0 ||
        cy >= depth.rows ||
        !pixelInValidRegion(valid_mask, u, v,
                            camera_width, camera_height,
                            require_valid_mask_))
    {
      return false;
    }

    const int radius =
      std::max(
        0,
        relative_depth_patch_radius_);

    const int x0 =
      std::max(0, cx - radius);

    const int x1 =
      std::min(
        depth.cols - 1,
        cx + radius);

    const int y0 =
      std::max(0, cy - radius);

    const int y1 =
      std::min(
        depth.rows - 1,
        cy + radius);

    std::vector<float> values;

    values.reserve(
      static_cast<std::size_t>(
        (x1 - x0 + 1)
        * (y1 - y0 + 1)));

    for (int yy = y0; yy <= y1; ++yy) {
      for (int xx = x0; xx <= x1; ++xx) {
        // Depth image may differ in resolution from CameraInfo. Use the
        // corresponding rectified-image pixel when testing the valid mask.
        if (!pixelInValidRegion(valid_mask,
                                (static_cast<double>(xx) + 0.5) / sx - 0.5,
                                (static_cast<double>(yy) + 0.5) / sy - 0.5,
                                camera_width, camera_height,
                                require_valid_mask_))
          continue;
        const float d =
          depth.at<float>(yy, xx);

        if (std::isfinite(d)) {
          values.push_back(d);
        }
      }
    }

    if (values.empty()) {
      return false;
    }

    const std::size_t middle =
      values.size() / 2;

    std::nth_element(
      values.begin(),
      values.begin() +
        static_cast<std::ptrdiff_t>(middle),
      values.end());

    value = values[middle];

    return std::isfinite(value);
  }


  bool fitAffineInverseDepth(
    const std::vector<double> & x,
    const std::vector<double> & y,
    double & a,
    double & b) const
  {
    if (x.size() != y.size() ||
        x.size() < 2)
    {
      return false;
    }

    double mean_x = 0.0;
    double mean_y = 0.0;

    for (std::size_t i = 0;
         i < x.size();
         ++i)
    {
      mean_x += x[i];
      mean_y += y[i];
    }

    mean_x /=
      static_cast<double>(x.size());

    mean_y /=
      static_cast<double>(y.size());

    double var_x = 0.0;
    double cov_xy = 0.0;

    for (std::size_t i = 0;
         i < x.size();
         ++i)
    {
      const double dx =
        x[i] - mean_x;

      const double dy =
        y[i] - mean_y;

      var_x += dx * dx;
      cov_xy += dx * dy;
    }

    if (!std::isfinite(var_x) ||
        var_x <= 1e-12)
    {
      return false;
    }

    a = cov_xy / var_x;
    b = mean_y - a * mean_x;

    return
      std::isfinite(a)
      && std::isfinite(b)
      && std::abs(a) > 1e-10;
  }


  // Coarse detection of wide, full-height black/white lens occlusions.
  // This is intentionally conservative: calibration is temporarily HELD,
  // not replaced by values learned from a corrupted image. Image dimensions
  // can differ from the depth raster; sampling is performed in RGB pixels.
  bool severeRgbVerticalObstruction(const cv::Mat & rgb) const
  {
    if (!calibration_freeze_on_obstruction_ || rgb.empty() ||
        rgb.type() != CV_8UC3 || rgb.cols < 40 || rgb.rows < 20)
      return false;
    const int sample_columns = 80;
    const int sample_rows = 20;
    int longest = 0, current = 0;
    for (int cx = 0; cx < sample_columns; ++cx) {
      const int x = std::min(rgb.cols-1,
        (cx * rgb.cols + rgb.cols/2) / sample_columns);
      int dark = 0, bright = 0;
      for (int cy = 0; cy < sample_rows; ++cy) {
        const int y = std::min(rgb.rows-1,
          (cy * rgb.rows + rgb.rows/2) / sample_rows);
        const cv::Vec3b pixel = rgb.at<cv::Vec3b>(y,x);
        const int sum = int(pixel[0])+int(pixel[1])+int(pixel[2]);
        dark += (sum <= 27);
        bright += (sum >= 738);
      }
      if (dark >= 19 || bright >= 19) {
        ++current;
        longest = std::max(longest,current);
      } else {
        current = 0;
      }
    }
    return longest >= 5;  // a contiguous >= 6.25%-wide suspect stripe
  }

  // Reject fully black/white and essentially textureless patches created
  // by simulated lens obstruction/overexposure. Do NOT reject ordinary dark
  // textured surfaces. This affects calibration samples, not point geometry.
  bool isObscuredRgbPatch(const cv::Mat & image, double u, double v,
                          int width, int height) const
  {
    if (!calibration_rgb_mask_enabled_ || image.empty() ||
        image.type() != CV_8UC3 || width <= 0 || height <= 0)
      return false;
    const int x = static_cast<int>(std::lround(u * image.cols / width));
    const int y = static_cast<int>(std::lround(v * image.rows / height));
    if (x < 0 || y < 0 || x >= image.cols || y >= image.rows)
      return true;
    const int r = 3;
    const cv::Rect roi(std::max(0,x-r), std::max(0,y-r),
      std::min(image.cols-1,x+r) - std::max(0,x-r) + 1,
      std::min(image.rows-1,y+r) - std::max(0,y-r) + 1);
    cv::Scalar mean, sd;
    cv::meanStdDev(image(roi), mean, sd);
    const double brightness = (mean[0]+mean[1]+mean[2]) / 3.0;
    const double texture = (sd[0]+sd[1]+sd[2]) / 3.0;
    return (brightness <= 9.0 || brightness >= 246.0) && texture <= 4.0;
  }

  void updateRelativeDepthCalibration(
    const std::vector<ProjectedPoint> & projected_points,
    const cv::Mat & relative_depth,
    int camera_width,
    int camera_height,
    const cv::Mat & valid_mask,
    const cv::Mat & rgb_image)
  {
    if (relative_depth.empty()) {
      return;
    }
    // Explicitly do not absorb a simulated partial black/white stripe into
    // the globally fitted relative-depth mapping. Preserve last reliable
    // parameters; candidate residual rejection remains active as before.
    if (severeRgbVerticalObstruction(rgb_image)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1500,
        "Calibration HELD: broad dark/saturated RGB obstruction; "
        "magenta residuals remain diagnostic");
      return;
    }

    std::vector<double> inverse_camera_depth;
    std::vector<double> relative_values;

    inverse_camera_depth.reserve(
      projected_points.size());

    relative_values.reserve(
      projected_points.size());

    for (const auto & point :
         projected_points)
    {
      if (!std::isfinite(point.camera_z) ||
          point.camera_z <= 0.05f)
      {
        continue;
      }
      if (isObscuredRgbPatch(rgb_image, point.u, point.v,
                             camera_width, camera_height))
        continue;

      float relative_value = 0.0f;

      if (!sampleRelativeDepth(
            relative_depth,
            point.u,
            point.v,
            camera_width,
            camera_height,
            valid_mask,
            relative_value))
      {
        continue;
      }

      inverse_camera_depth.push_back(
        1.0 /
        static_cast<double>(
          point.camera_z));

      relative_values.push_back(
        static_cast<double>(
          relative_value));
    }

    const std::size_t min_points =
      static_cast<std::size_t>(
        std::max(
          3,
          calibration_min_points_));

    if (inverse_camera_depth.size() <
        min_points)
    {
      return;
    }

    double a0 = 0.0;
    double b0 = 0.0;

    if (!fitAffineInverseDepth(
          inverse_camera_depth,
          relative_values,
          a0,
          b0))
    {
      return;
    }

    // ----------------------------------------------------------
    // One robust refit:
    // remove points with very large residuals according to MAD.
    // ----------------------------------------------------------

    std::vector<double> residuals;

    residuals.reserve(
      inverse_camera_depth.size());

    for (std::size_t i = 0;
         i < inverse_camera_depth.size();
         ++i)
    {
      residuals.push_back(
        std::abs(
          relative_values[i]
          - (
              a0 *
              inverse_camera_depth[i]
              + b0
            )));
    }

    std::vector<double>
      residual_copy = residuals;

    const std::size_t mid =
      residual_copy.size() / 2;

    std::nth_element(
      residual_copy.begin(),
      residual_copy.begin() +
        static_cast<std::ptrdiff_t>(mid),
      residual_copy.end());

    const double median_residual =
      residual_copy[mid];

    const double threshold =
      std::max(
        1e-6,
        3.0 * median_residual);

    std::vector<double> x_inliers;
    std::vector<double> y_inliers;

    x_inliers.reserve(
      inverse_camera_depth.size());

    y_inliers.reserve(
      inverse_camera_depth.size());

    for (std::size_t i = 0;
         i < residuals.size();
         ++i)
    {
      if (residuals[i] <= threshold) {
        x_inliers.push_back(
          inverse_camera_depth[i]);

        y_inliers.push_back(
          relative_values[i]);
      }
    }

    double a_fit = a0;
    double b_fit = b0;

    if (x_inliers.size() < min_points ||
        static_cast<double>(x_inliers.size()) /
          static_cast<double>(inverse_camera_depth.size()) <
          calibration_min_inlier_fraction_)
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "Calibration held: insufficient reliable inliers (%zu/%zu)",
        x_inliers.size(), inverse_camera_depth.size());
      return;
    }
    if (!fitAffineInverseDepth(x_inliers, y_inliers, a_fit, b_fit))
      return;

    const double alpha =
      std::clamp(
        calibration_alpha_,
        0.0,
        1.0);

    std::lock_guard<std::mutex>
      lock(data_mutex_);

    if (!have_relative_depth_calibration_) {
      relative_depth_a_ = a_fit;
      relative_depth_b_ = b_fit;
      have_relative_depth_calibration_ = true;
    }
    else {
      // Guard against global recalibration caused by a local image fault.
      // Evaluate the proposed mapping change at CURRENT valid correspondences;
      // there is no startup baseline or learned reference period.
      std::vector<double> current_values = y_inliers;
      const auto half = current_values.begin() +
        static_cast<std::ptrdiff_t>(current_values.size()/2);
      std::nth_element(current_values.begin(), half, current_values.end());
      const double median_y = *half;
      std::vector<double> deviations;
      deviations.reserve(y_inliers.size());
      for (double y : y_inliers) deviations.push_back(std::abs(y-median_y));
      const auto half_dev = deviations.begin() +
        static_cast<std::ptrdiff_t>(deviations.size()/2);
      std::nth_element(deviations.begin(), half_dev, deviations.end());
      const double robust_spread = std::max(0.05, 1.4826*(*half_dev));
      std::vector<double> changes;
      changes.reserve(x_inliers.size());
      for (double x : x_inliers)
        changes.push_back(std::abs((a_fit-relative_depth_a_)*x +
                                    (b_fit-relative_depth_b_)));
      const auto half_change = changes.begin() +
        static_cast<std::ptrdiff_t>(changes.size()/2);
      std::nth_element(changes.begin(), half_change, changes.end());
      if (*half_change > calibration_max_update_fraction_*robust_spread) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "Calibration held: abrupt affine change %.4f > %.4f; "
          "preserving reliable previous fit",
          *half_change, calibration_max_update_fraction_*robust_spread);
        return;
      }
      relative_depth_a_ =
        alpha * a_fit
        + (1.0 - alpha) *
          relative_depth_a_;

      relative_depth_b_ =
        alpha * b_fit
        + (1.0 - alpha) *
          relative_depth_b_;
    }

    relative_depth_fit_error_ =
      median_residual;

    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      3000,
      "Relative-depth calibration: Drel = %.5f*(1/Z) + %.5f | points=%zu | med_res=%.5f",
      relative_depth_a_,
      relative_depth_b_,
      x_inliers.size(),
      relative_depth_fit_error_);
  }


  // ============================================================
  // Diagnostic visualization ONLY: grayscale relative depth background.
  //
  // The same float depth image is used for expected-range estimation and
  // this overlay.  Percentile normalization changes DISPLAY INTENSITY ONLY,
  // never the floating-point depth data or the LiDAR projection coordinates.
  // Depth is resized for visualization when its raster differs from the
  // rectified RGB / CameraInfo raster; magenta u,v stay camera pixels.
  // Invalid rectification-mask pixels remain dark in the visualization.
  // ============================================================

  cv::Mat relativeDepthVisualization(
    const cv::Mat & depth,
    const cv::Mat & valid_mask,
    int width,
    int height) const
  {
    if (depth.empty() || depth.type() != CV_32FC1 ||
        width <= 0 || height <= 0) {
      return cv::Mat();
    }

    cv::Mat resampled;
    if (depth.cols == width && depth.rows == height) {
      resampled = depth;
    } else {
      cv::resize(depth, resampled, cv::Size(width, height),
                 0.0, 0.0, cv::INTER_LINEAR);
    }

    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (!pixelInValidRegion(valid_mask, x, y,
                                width, height, require_valid_mask_)) {
          continue;
        }
        const float d = resampled.at<float>(y, x);
        if (std::isfinite(d)) values.push_back(d);
      }
    }
    if (values.empty()) return cv::Mat();

    std::sort(values.begin(), values.end());
    const auto lower_index = static_cast<std::size_t>(
      std::floor(0.02 * static_cast<double>(values.size() - 1)));
    const auto upper_index = static_cast<std::size_t>(
      std::floor(0.98 * static_cast<double>(values.size() - 1)));
    const float low = values[lower_index];
    const float high = values[upper_index];
    const float span = high - low;

    cv::Mat display(height, width, CV_8UC3, cv::Scalar(15, 15, 15));
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (!pixelInValidRegion(valid_mask, x, y,
                                width, height, require_valid_mask_)) {
          continue;
        }
        const float d = resampled.at<float>(y, x);
        if (!std::isfinite(d)) continue;
        const double t = span > 1e-6f
          ? std::clamp(static_cast<double>((d - low) / span), 0.0, 1.0)
          : 0.5;
        const std::uint8_t gray = static_cast<std::uint8_t>(
          std::lround(255.0 * t));
        display.at<cv::Vec3b>(y, x) = cv::Vec3b(gray, gray, gray);
      }
    }
    return display;
  }

  // ============================================================
  // Estimated LiDAR beam projection using relative depth
  //
  // This projection deliberately DOES NOT use a LiDAR range.
  //
  // For each LaserScan index:
  //
  //   d_L = [cos(theta), sin(theta), 0]
  //
  // Each candidate LiDAR range is transformed using the FULL TF,
  // including translation, and projected with the calibrated camera
  // model. Candidate (u,v) samples relative depth. This is an estimated
  // image-space beam location and can differ from actual LiDAR pixels
  // because of model errors and camera/LiDAR parallax.
  //
  // The LaserScan angular geometry is cached from the latest scan.
  // A timer republishes these expected pixels even if /scan later stops.
  // ============================================================

  void expectedProjectionTimerCallback()
  {
    if (!publish_expected_points_ ||
        !expected_points_pub_)
    {
      return;
    }

    image_geometry::PinholeCameraModel
      camera_model;

    std::string info_frame;
    std::string scan_frame;

    int image_width = 0;
    int image_height = 0;

    double angle_min = 0.0;
    double angle_increment = 0.0;
    std::size_t beam_count = 0;

    cv::Mat relative_depth;
    std_msgs::msg::Header relative_depth_header;
    cv::Mat valid_mask;

    double calibration_a = 0.0;
    double calibration_b = 0.0;
    bool have_calibration = false;

    {
      std::lock_guard<std::mutex>
        lock(data_mutex_);

      if (!have_camera_info_ ||
          !have_scan_geometry_ ||
          !have_relative_depth_)
      {
        return;
      }

      camera_model =
        camera_model_;

      info_frame =
        camera_frame_from_info_;

      image_width =
        image_width_;

      image_height =
        image_height_;

      angle_min =
        scan_angle_min_;

      angle_increment =
        scan_angle_increment_;

      beam_count =
        scan_beam_count_;

      scan_frame =
        scan_frame_;

      relative_depth =
        latest_relative_depth_.clone();
      relative_depth_header = latest_relative_depth_header_;
      valid_mask = valid_mask_.clone();

      have_calibration =
        have_relative_depth_calibration_;

      calibration_a =
        relative_depth_a_;

      calibration_b =
        relative_depth_b_;
    }

    if (beam_count == 0 ||
        scan_frame.empty() ||
        relative_depth.empty() ||
        (require_valid_mask_ && valid_mask.empty()) ||
        !have_calibration)
    {
      return;
    }

    const std::string target_frame =
      camera_frame_.empty()
      ? info_frame
      : camera_frame_;

    geometry_msgs::msg::TransformStamped tf;

    try {
      tf =
        tf_buffer_->lookupTransform(
          target_frame,
          scan_frame,
          tf2::TimePointZero,
          200ms);
    }
    catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Estimated projection TF %s -> %s unavailable: %s",
        scan_frame.c_str(),
        target_frame.c_str(),
        ex.what());

      return;
    }

    publishExpectedProjection(
      camera_model,
      image_width,
      image_height,
      angle_min,
      angle_increment,
      beam_count,
      target_frame,
      tf,
      relative_depth,
      relative_depth_header,
      valid_mask,
      calibration_a,
      calibration_b);
  }


  void publishExpectedProjection(
    const image_geometry::PinholeCameraModel & camera_model,
    int image_width,
    int image_height,
    double angle_min,
    double angle_increment,
    std::size_t beam_count,
    const std::string & target_frame,
    const geometry_msgs::msg::TransformStamped & tf,
    const cv::Mat & relative_depth,
    const std_msgs::msg::Header & relative_depth_header,
    const cv::Mat & valid_mask,
    double calibration_a,
    double calibration_b)
  {
    if (beam_count == 0 ||
        relative_depth.empty())
    {
      return;
    }

    int first_beam = 0;

    int last_beam =
      static_cast<int>(
        beam_count) - 1;

    if (fixed_beam_start_ >= 0) {
      first_beam =
        std::clamp(
          fixed_beam_start_,
          0,
          last_beam);
    }

    if (fixed_beam_end_ >= 0) {
      last_beam =
        std::clamp(
          fixed_beam_end_,
          0,
          static_cast<int>(
            beam_count) - 1);
    }

    if (first_beam > last_beam) {
      std::swap(
        first_beam,
        last_beam);
    }

    const double min_range =
      std::max(
        0.05,
        expected_min_range_);

    const double max_range =
      std::max(
        min_range + 0.01,
        expected_max_range_);

    const int range_samples =
      std::max(
        2,
        expected_range_samples_);

    const double log_min =
      std::log(min_range);

    const double log_max =
      std::log(max_range);

    const double yaw =
      yaw_correction_deg_
      * M_PI / 180.0;

    const double cos_yaw =
      std::cos(yaw);

    const double sin_yaw =
      std::sin(yaw);

    // Count beams rejected by the confidence gate for diagnostics.
    std::size_t rejected_high_residual = 0;

    std::vector<ExpectedPoint>
      expected_points;
    std::vector<ExpectedPoint> rejected_points;

    expected_points.reserve(
      static_cast<std::size_t>(
        last_beam
        - first_beam
        + 1));

    // Magenta points are visualized on the SAME relative depth map used
    // for expected-range estimation, not on the RGB image. The RGB image
    // subscription is retained only for the green measured-LiDAR overlay.
    cv::Mat expected_image;
    bool have_expected_image = false;
    if (visualize_ && expected_image_pub_) {
      expected_image = relativeDepthVisualization(
        relative_depth, valid_mask, image_width, image_height);
      have_expected_image = !expected_image.empty();
    }

    std_msgs::msg::Header expected_image_header = relative_depth_header;
    expected_image_header.frame_id = target_frame;

    for (int index = first_beam;
         index <= last_beam;
         ++index)
    {
      const double angle =
        angle_min
        + static_cast<double>(index)
        * angle_increment;

      bool found = false;

      ExpectedPoint best_point;
      double best_score =
        std::numeric_limits<double>::infinity();

      // --------------------------------------------------------
      // Search plausible ranges along this LiDAR beam.
      //
      // Log spacing gives more resolution at short ranges, where
      // parallax changes the image v coordinate most strongly.
      // --------------------------------------------------------

      for (int sample_index = 0;
           sample_index < range_samples;
           ++sample_index)
      {
        const double t =
          static_cast<double>(
            sample_index)
          / static_cast<double>(
            range_samples - 1);

        const double range =
          std::exp(
            log_min
            + t *
              (log_max - log_min));

        geometry_msgs::msg::Point
          p_lidar;

        p_lidar.x =
          range * std::cos(angle);

        p_lidar.y =
          range * std::sin(angle);

        p_lidar.z = 0.0;

        geometry_msgs::msg::Point
          p_cam;

        // Full transform, INCLUDING translation.
        tf2::doTransform(
          p_lidar,
          p_cam,
          tf);

        // Same additional calibrated yaw as the real projection.
        const double x0 =
          p_cam.x;

        const double z0 =
          p_cam.z;

        p_cam.x =
          cos_yaw * x0
          + sin_yaw * z0;

        p_cam.z =
          -sin_yaw * x0
          + cos_yaw * z0;

        if (!std::isfinite(p_cam.x) ||
            !std::isfinite(p_cam.y) ||
            !std::isfinite(p_cam.z) ||
            p_cam.z <= 0.05)
        {
          continue;
        }

        const cv::Point2d uv =
          camera_model.project3dToPixel(
            cv::Point3d(
              p_cam.x,
              p_cam.y,
              p_cam.z));

        if (!std::isfinite(uv.x) ||
            !std::isfinite(uv.y) ||
            uv.x < 0.0 ||
            uv.x >=
              static_cast<double>(
                image_width) ||
            uv.y < 0.0 ||
            uv.y >=
              static_cast<double>(
                image_height))
        {
          continue;
        }

        // Reject invalid rectification padding BEFORE sampling depth,
        // scoring candidate ranges, or emitting magenta points.
        if (!pixelInValidRegion(valid_mask, uv.x, uv.y,
                                image_width, image_height, require_valid_mask_))
          continue;

        float observed_relative = 0.0f;

        if (!sampleRelativeDepth(
              relative_depth,
              uv.x,
              uv.y,
              image_width,
              image_height,
              valid_mask,
              observed_relative))
        {
          continue;
        }

        const double expected_relative =
          calibration_a *
          (
            1.0 / p_cam.z
          )
          + calibration_b;

        const double score =
          std::abs(
            static_cast<double>(
              observed_relative)
            - expected_relative);

        if (score < best_score) {
          found = true;
          best_score = score;

          best_point.u =
            static_cast<float>(
              uv.x);

          best_point.v =
            static_cast<float>(
              uv.y);

          best_point.angle =
            static_cast<float>(
              angle);

          best_point.estimated_range =
            static_cast<float>(
              range);

          best_point.camera_z =
            static_cast<float>(
              p_cam.z);

          best_point.residual =
            static_cast<float>(
              score);

          best_point.scan_index =
            static_cast<std::int32_t>(
              index);
        }
      }

      if (!found || !std::isfinite(best_score)) {
        continue;
      }

      // Being the best candidate is not sufficient: near the left/right
      // validity-mask boundaries only short, geometrically valid candidates
      // may remain, and all can be extremely poor depth-consistency matches.
      // Do NOT publish those candidates as trustworthy magenta points.
      if (best_score > expected_max_residual_) {
        ++rejected_high_residual;
        // Maintain this candidate ONLY on the diagnostic topic and overlay.
        // Do not put it into /faults/lidar_expected_projection: downstream
        // NCC and spatial irregularity must never treat it as trustworthy.
        rejected_points.push_back(best_point);
        if (have_expected_image && show_rejected_expected_overlay_) {
          cv::circle(expected_image,
            cv::Point(static_cast<int>(std::lround(best_point.u)),
                      static_cast<int>(std::lround(best_point.v))),
            std::max(1,point_radius_), cv::Scalar(0,165,255),
            1, cv::LINE_AA);  // optional diagnostic orange outline
        }
        continue;
      }

      expected_points.push_back(
        best_point);

      if (have_expected_image) {
        const int draw_u =
          static_cast<int>(
            std::lround(
              best_point.u));

        const int draw_v =
          static_cast<int>(
            std::lround(
              best_point.v));

        cv::circle(
          expected_image,
          cv::Point(
            draw_u,
            draw_v),
          std::max(
            1,
            point_radius_),
          cv::Scalar(
            255,
            0,
            255),
          -1,
          cv::LINE_AA);
      }
    }

    // ----------------------------------------------------------
    // Publish estimated image-space beam locations.
    // ----------------------------------------------------------

    sensor_msgs::msg::PointCloud2
      output;

    output.header.stamp = now();

    output.header.frame_id =
      target_frame;

    sensor_msgs::PointCloud2Modifier
      modifier(output);

    modifier.setPointCloud2Fields(
      7,

      "u",
      1,
      sensor_msgs::msg::PointField::FLOAT32,

      "v",
      1,
      sensor_msgs::msg::PointField::FLOAT32,

      "angle",
      1,
      sensor_msgs::msg::PointField::FLOAT32,

      "estimated_range",
      1,
      sensor_msgs::msg::PointField::FLOAT32,

      "camera_z",
      1,
      sensor_msgs::msg::PointField::FLOAT32,

      "residual",
      1,
      sensor_msgs::msg::PointField::FLOAT32,

      "scan_index",
      1,
      sensor_msgs::msg::PointField::INT32);

    modifier.resize(
      expected_points.size());

    sensor_msgs::PointCloud2Iterator<float>
      out_u(output, "u");

    sensor_msgs::PointCloud2Iterator<float>
      out_v(output, "v");

    sensor_msgs::PointCloud2Iterator<float>
      out_angle(output, "angle");

    sensor_msgs::PointCloud2Iterator<float>
      out_estimated_range(
        output,
        "estimated_range");

    sensor_msgs::PointCloud2Iterator<float>
      out_camera_z(
        output,
        "camera_z");

    sensor_msgs::PointCloud2Iterator<float>
      out_residual(
        output,
        "residual");

    sensor_msgs::PointCloud2Iterator<
      std::int32_t>
      out_scan_index(
        output,
        "scan_index");

    for (const auto & point :
         expected_points)
    {
      *out_u = point.u;
      *out_v = point.v;
      *out_angle = point.angle;
      *out_estimated_range =
        point.estimated_range;
      *out_camera_z =
        point.camera_z;
      *out_residual =
        point.residual;
      *out_scan_index =
        point.scan_index;

      ++out_u;
      ++out_v;
      ++out_angle;
      ++out_estimated_range;
      ++out_camera_z;
      ++out_residual;
      ++out_scan_index;
    }

    expected_points_pub_->publish(output);

    if (rejected_expected_points_pub_) {
      // Copy the exact PointCloud2 schema and header. This is a SEPARATE
      // diagnostic stream and is not consumed by the anomaly detector.
      sensor_msgs::msg::PointCloud2 rejected_output = output;
      sensor_msgs::PointCloud2Modifier reject_modifier(rejected_output);
      reject_modifier.resize(rejected_points.size());
      sensor_msgs::PointCloud2Iterator<float> ru(rejected_output,"u");
      sensor_msgs::PointCloud2Iterator<float> rv(rejected_output,"v");
      sensor_msgs::PointCloud2Iterator<float> ra(rejected_output,"angle");
      sensor_msgs::PointCloud2Iterator<float> rr(rejected_output,"estimated_range");
      sensor_msgs::PointCloud2Iterator<float> rz(rejected_output,"camera_z");
      sensor_msgs::PointCloud2Iterator<float> re(rejected_output,"residual");
      sensor_msgs::PointCloud2Iterator<std::int32_t> ri(rejected_output,"scan_index");
      for (const auto &p : rejected_points) {
        *ru=p.u; *rv=p.v; *ra=p.angle; *rr=p.estimated_range;
        *rz=p.camera_z; *re=p.residual; *ri=p.scan_index;
        ++ru; ++rv; ++ra; ++rr; ++rz; ++re; ++ri;
      }
      rejected_expected_points_pub_->publish(rejected_output);
    }

    if (have_expected_image) {
      cv::putText(
        expected_image,
        "Expected LiDAR on relative depth map",
        cv::Point(
          8,
          18),
        cv::FONT_HERSHEY_SIMPLEX,
        0.38,
        cv::Scalar(
          255,
          255,
          255),
        1,
        cv::LINE_AA);

      cv::putText(
        expected_image,
        "points=" +
          std::to_string(
            expected_points.size()),
        cv::Point(
          8,
          36),
        cv::FONT_HERSHEY_SIMPLEX,
        0.38,
        cv::Scalar(
          255,
          255,
          255),
        1,
        cv::LINE_AA);

      cv::putText(
        expected_image,
        show_rejected_expected_overlay_
          ? "magenta=accepted; orange=rejected"
          : "magenta=accepted only",
        cv::Point(
          8,
          54),
        cv::FONT_HERSHEY_SIMPLEX,
        0.38,
        cv::Scalar(
          255,
          255,
          255),
        1,
        cv::LINE_AA);

      auto expected_out_image =
        cv_bridge::CvImage(
          expected_image_header,
          sensor_msgs::image_encodings::BGR8,
          expected_image)
        .toImageMsg();

      expected_image_pub_->publish(
        *expected_out_image);
    }

    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      3000,
      "Published %zu relative-depth-estimated beam pixels | rejected residual=%zu | Drel=%.4f*(1/Z)+%.4f",
      expected_points.size(),
      rejected_high_residual,
      calibration_a,
      calibration_b);
  }


  // ============================================================
  // RGB callback
  // ============================================================

  void imageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    try {
      auto cv_ptr =
        cv_bridge::toCvCopy(
          msg,
          sensor_msgs::image_encodings::BGR8);

      std::lock_guard<std::mutex>
        lock(data_mutex_);

      latest_image_ =
        cv_ptr->image.clone();

      latest_image_header_ =
        msg->header;

      have_image_ = true;
    }
    catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "cv_bridge error: %s",
        e.what());
    }
  }


  // ============================================================
  // Camera calibration callback
  // ============================================================

  void cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
  {
    std::lock_guard<std::mutex>
      lock(data_mutex_);

    if (have_camera_info_ &&
        (image_width_ != static_cast<int>(msg->width) ||
         image_height_ != static_cast<int>(msg->height))) {
      valid_mask_.release();
    }
    camera_model_.fromCameraInfo(msg);

    camera_frame_from_info_ =
      msg->header.frame_id;

    image_width_ =
      static_cast<int>(msg->width);

    image_height_ =
      static_cast<int>(msg->height);

    have_camera_info_ = true;
  }


  // ============================================================
  // LiDAR callback
  // ============================================================

  void scanCallback(
    const sensor_msgs::msg::LaserScan::ConstSharedPtr scan)
  {
    image_geometry::PinholeCameraModel
      camera_model;

    std::string info_frame;

    int image_width = 0;
    int image_height = 0;

    cv::Mat image;
    cv::Mat valid_mask;
    std_msgs::msg::Header image_header;
    cv::Mat calibration_rgb_image;
    std_msgs::msg::Header calibration_rgb_header;

    cv::Mat relative_depth_for_calibration;
    std_msgs::msg::Header relative_depth_header;
    bool have_relative_depth_for_calibration = false;

    // ----------------------------------------------------------
    // Copy camera / optional image state
    // ----------------------------------------------------------

    {
      std::lock_guard<std::mutex>
        lock(data_mutex_);

      if (!have_camera_info_) {
        RCLCPP_WARN_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "Waiting for camera_info...");

        return;
      }

      camera_model =
        camera_model_;

      info_frame =
        camera_frame_from_info_;

      image_width =
        image_width_;

      image_height =
        image_height_;

      valid_mask = valid_mask_.clone();
      if (require_valid_mask_ && valid_mask.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "Waiting for /rgb_rectification/valid_mask before projecting LiDAR");
        return;
      }

      if (visualize_) {
        if (!have_image_) {
          RCLCPP_WARN_THROTTLE(
            get_logger(),
            *get_clock(),
            2000,
            "Visualization enabled: waiting for image_rect...");

          return;
        }

        image =
          latest_image_.clone();

        image_header =
          latest_image_header_;


      }

      if (calibration_rgb_mask_enabled_ && have_image_) {
        calibration_rgb_image = latest_image_.clone();
        calibration_rgb_header = latest_image_header_;
      }

      if (have_relative_depth_) {
        relative_depth_for_calibration =
          latest_relative_depth_.clone();

        relative_depth_header =
          latest_relative_depth_header_;

        have_relative_depth_for_calibration =
          true;
      }
    }

    if (scan->ranges.empty()) {
      return;
    }

    // Cache angular geometry independently from range validity.
    // The timer can continue publishing nominal beam pixels if /scan
    // subsequently stops.
    {
      std::lock_guard<std::mutex>
        lock(data_mutex_);

      scan_angle_min_ =
        static_cast<double>(
          scan->angle_min);

      scan_angle_increment_ =
        static_cast<double>(
          scan->angle_increment);

      scan_beam_count_ =
        scan->ranges.size();

      scan_frame_ =
        scan->header.frame_id;

      have_scan_geometry_ = true;
    }

    const std::string target_frame =
      camera_frame_.empty()
      ? info_frame
      : camera_frame_;

    // ----------------------------------------------------------
    // LiDAR stabilization
    // ----------------------------------------------------------

    std::vector<float> ranges =
      spatialMedianFilter(*scan);

    temporalRangeFilter(
      ranges,
      *scan);

    ensureUvStateSize(
      ranges.size());

    // ----------------------------------------------------------
    // Fixed beam bounds
    // ----------------------------------------------------------

    int first_beam = 0;

    int last_beam =
      static_cast<int>(
        ranges.size()) - 1;

    if (fixed_beam_start_ >= 0) {
      first_beam =
        std::clamp(
          fixed_beam_start_,
          0,
          last_beam);
    }

    if (fixed_beam_end_ >= 0) {
      last_beam =
        std::clamp(
          fixed_beam_end_,
          0,
          static_cast<int>(
            ranges.size()) - 1);
    }

    if (first_beam > last_beam) {
      std::swap(
        first_beam,
        last_beam);
    }

    // ----------------------------------------------------------
    // TF LiDAR -> camera
    // ----------------------------------------------------------

    geometry_msgs::msg::TransformStamped tf;

    try {
      tf =
        tf_buffer_->lookupTransform(
          target_frame,
          scan->header.frame_id,
          rclcpp::Time(
            scan->header.stamp),
          200ms);
    }
    catch (const tf2::TransformException &) {
      try {
        tf =
          tf_buffer_->lookupTransform(
            target_frame,
            scan->header.frame_id,
            tf2::TimePointZero,
            200ms);
      }
      catch (const tf2::TransformException & ex2) {
        RCLCPP_WARN_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "TF %s -> %s unavailable: %s",
          scan->header.frame_id.c_str(),
          target_frame.c_str(),
          ex2.what());

        return;
      }
    }

    // ----------------------------------------------------------
    // Manual camera-Y rotation
    // ----------------------------------------------------------

    const double yaw =
      yaw_correction_deg_ *
      M_PI / 180.0;

    const double cos_yaw =
      std::cos(yaw);

    const double sin_yaw =
      std::sin(yaw);

    std::vector<ProjectedPoint>
      projected_points;

    projected_points.reserve(
      static_cast<std::size_t>(
        last_beam - first_beam + 1));

    // ----------------------------------------------------------
    // Directly iterate original LaserScan indices.
    //
    // This is intentional: each published point retains the
    // original beam index and fixed angular direction.
    // ----------------------------------------------------------

    for (int index = first_beam;
         index <= last_beam;
         ++index)
    {
      const std::size_t i =
        static_cast<std::size_t>(
          index);

      const float range =
        ranges[i];

      if (!validRange(
            range,
            *scan))
      {
        previous_uv_valid_[i] = false;
        continue;
      }

      const double angle =
        static_cast<double>(
          scan->angle_min)
        +
        static_cast<double>(index) *
        static_cast<double>(
          scan->angle_increment);

      // 2D LaserScan lies in the LiDAR XY plane.
      geometry_msgs::msg::Point
        p_lidar;

      p_lidar.x =
        static_cast<double>(range) *
        std::cos(angle);

      p_lidar.y =
        static_cast<double>(range) *
        std::sin(angle);

      p_lidar.z = 0.0;

      // LiDAR -> camera optical frame.
      geometry_msgs::msg::Point
        p_cam;

      tf2::doTransform(
        p_lidar,
        p_cam,
        tf);

      // Additional manually calibrated yaw correction.
      const double x0 =
        p_cam.x;

      const double z0 =
        p_cam.z;

      p_cam.x =
        cos_yaw * x0 +
        sin_yaw * z0;

      p_cam.z =
        -sin_yaw * x0 +
        cos_yaw * z0;

      if (!std::isfinite(p_cam.x) ||
          !std::isfinite(p_cam.y) ||
          !std::isfinite(p_cam.z) ||
          p_cam.z <= 0.05)
      {
        previous_uv_valid_[i] = false;
        continue;
      }

      const cv::Point2d uv =
        camera_model.project3dToPixel(
          cv::Point3d(
            p_cam.x,
            p_cam.y,
            p_cam.z));

      if (!std::isfinite(uv.x) ||
          !std::isfinite(uv.y))
      {
        previous_uv_valid_[i] = false;
        continue;
      }

      float u =
        static_cast<float>(
          uv.x);

      float v =
        static_cast<float>(
          uv.y);

      // Stabilize pixel location for the same scan beam.
      temporalUvFilter(
        i,
        u,
        v);

      if (u < 0.0f ||
          u >= static_cast<float>(
            image_width) ||
          v < 0.0f ||
          v >= static_cast<float>(
            image_height))
      {
        continue;
      }

      // Omit green points outside true rectification support.
      // This also removes those beams from the published PointCloud2.
      if (!pixelInValidRegion(valid_mask, u, v,
                              image_width, image_height, require_valid_mask_)) {
        previous_uv_valid_[i] = false;
        continue;
      }

      ProjectedPoint point;

      point.u = u;
      point.v = v;
      point.range = range;

      point.x =
        static_cast<float>(
          p_lidar.x);

      point.y =
        static_cast<float>(
          p_lidar.y);

      point.z =
        static_cast<float>(
          p_lidar.z);

      point.camera_x =
        static_cast<float>(
          p_cam.x);

      point.camera_y =
        static_cast<float>(
          p_cam.y);

      point.camera_z =
        static_cast<float>(
          p_cam.z);

      point.scan_index =
        static_cast<std::int32_t>(
          index);

      projected_points.push_back(
        point);

      if (visualize_) {
        const int draw_u =
          static_cast<int>(
            std::lround(u));

        const int draw_v =
          static_cast<int>(
            std::lround(v));

        const int radius =
          std::max(
            1,
            point_radius_ +
              (range < 1.0f ? 1 : 0));

        cv::circle(
          image,
          cv::Point(
            draw_u,
            draw_v),
          radius,
          cv::Scalar(
            0,
            255,
            0),
          -1,
          cv::LINE_AA);
      }

    }

    // ----------------------------------------------------------
    // Learn relative-depth <-> inverse metric-depth mapping while
    // real LiDAR measurements are available.
    // ----------------------------------------------------------

    if (have_relative_depth_for_calibration &&
        !relative_depth_for_calibration.empty())
    {
      const double scan_time =
        stampSeconds(
          scan->header.stamp);

      const double depth_time =
        stampSeconds(
          relative_depth_header.stamp);

      const double dt =
        std::abs(
          scan_time - depth_time);

      if (max_relative_depth_time_difference_ <= 0.0 ||
          dt <= max_relative_depth_time_difference_)
      {
        // Image and depth must be close in capture time for RGB masking.
        // Otherwise the RGB mask is omitted rather than masking stale pixels.
        const bool rgb_fresh = !calibration_rgb_image.empty() &&
          std::abs(stampSeconds(calibration_rgb_header.stamp) - depth_time)
          <= calibration_rgb_max_time_difference_;
        if (calibration_rgb_mask_enabled_ && !rgb_fresh) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
            "Calibration HELD: no RGB frame synchronized to depth; "
            "preserving last reliable fit");
        } else {
          updateRelativeDepthCalibration(
            projected_points,
            relative_depth_for_calibration,
            image_width,
            image_height,
            valid_mask,
            rgb_fresh ? calibration_rgb_image : cv::Mat());
        }
      }
    }

    // ==========================================================
    // Publish:
    //
    // u, v, range,
    // x, y, z,
    // camera_x, camera_y, camera_z,
    // scan_index
    // ==========================================================

    sensor_msgs::msg::PointCloud2
      output;

    output.header.stamp =
      scan->header.stamp;

    output.header.frame_id =
      target_frame;

    sensor_msgs::PointCloud2Modifier
      modifier(output);

    modifier.setPointCloud2Fields(
      10,

      "u",
      1,
      sensor_msgs::msg::PointField::FLOAT32,

      "v",
      1,
      sensor_msgs::msg::PointField::FLOAT32,

      "range",
      1,
      sensor_msgs::msg::PointField::FLOAT32,

      "x",
      1,
      sensor_msgs::msg::PointField::FLOAT32,

      "y",
      1,
      sensor_msgs::msg::PointField::FLOAT32,

      "z",
      1,
      sensor_msgs::msg::PointField::FLOAT32,

      "camera_x",
      1,
      sensor_msgs::msg::PointField::FLOAT32,

      "camera_y",
      1,
      sensor_msgs::msg::PointField::FLOAT32,

      "camera_z",
      1,
      sensor_msgs::msg::PointField::FLOAT32,

      "scan_index",
      1,
      sensor_msgs::msg::PointField::INT32);

    modifier.resize(
      projected_points.size());

    sensor_msgs::PointCloud2Iterator<float>
      out_u(output, "u");

    sensor_msgs::PointCloud2Iterator<float>
      out_v(output, "v");

    sensor_msgs::PointCloud2Iterator<float>
      out_range(output, "range");

    sensor_msgs::PointCloud2Iterator<float>
      out_x(output, "x");

    sensor_msgs::PointCloud2Iterator<float>
      out_y(output, "y");

    sensor_msgs::PointCloud2Iterator<float>
      out_z(output, "z");

    sensor_msgs::PointCloud2Iterator<float>
      out_camera_x(output, "camera_x");

    sensor_msgs::PointCloud2Iterator<float>
      out_camera_y(output, "camera_y");

    sensor_msgs::PointCloud2Iterator<float>
      out_camera_z(output, "camera_z");

    sensor_msgs::PointCloud2Iterator<std::int32_t>
      out_scan_index(
        output,
        "scan_index");

    for (const auto & point :
         projected_points)
    {
      *out_u = point.u;
      *out_v = point.v;
      *out_range = point.range;

      *out_x = point.x;
      *out_y = point.y;
      *out_z = point.z;

      *out_camera_x =
        point.camera_x;

      *out_camera_y =
        point.camera_y;

      *out_camera_z =
        point.camera_z;

      *out_scan_index =
        point.scan_index;

      ++out_u;
      ++out_v;
      ++out_range;

      ++out_x;
      ++out_y;
      ++out_z;

      ++out_camera_x;
      ++out_camera_y;
      ++out_camera_z;

      ++out_scan_index;
    }

    projected_points_pub_->publish(
      output);

    // ----------------------------------------------------------
    // Visualization
    // ----------------------------------------------------------

    if (visualize_) {
      cv::putText(
        image,
        "LiDAR projected points",
        cv::Point(
          8,
          18),
        cv::FONT_HERSHEY_SIMPLEX,
        0.38,
        cv::Scalar(
          255,
          255,
          255),
        1,
        cv::LINE_AA);

      cv::putText(
        image,
        "points=" +
          std::to_string(
            projected_points.size()),
        cv::Point(
          8,
          36),
        cv::FONT_HERSHEY_SIMPLEX,
        0.38,
        cv::Scalar(
          255,
          255,
          255),
        1,
        cv::LINE_AA);

      cv::putText(
        image,
        "beams=" +
          std::to_string(
            first_beam) +
          ".." +
          std::to_string(
            last_beam),
        cv::Point(
          8,
          54),
        cv::FONT_HERSHEY_SIMPLEX,
        0.38,
        cv::Scalar(
          255,
          255,
          255),
        1,
        cv::LINE_AA);

      auto out_image =
        cv_bridge::CvImage(
          image_header,
          sensor_msgs::image_encodings::BGR8,
          image)
        .toImageMsg();

      image_pub_->publish(
        *out_image);
    }

    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Published %zu projected points | beams=%d..%d | LiDAR=%s | camera=%s",
      projected_points.size(),
      first_beam,
      last_beam,
      scan->header.frame_id.c_str(),
      target_frame.c_str());
  }


  // ============================================================
  // Parameters
  // ============================================================

  std::string camera_info_topic_;
  std::string scan_topic_;
  std::string projected_points_topic_;
  std::string expected_points_topic_;
  std::string rejected_expected_points_topic_;
  bool show_rejected_expected_overlay_{false};
  std::string relative_depth_topic_;
  std::string camera_frame_;
  std::string valid_mask_topic_;
  bool require_valid_mask_{true};

  std::string image_topic_;
  std::string visualization_topic_;
  std::string expected_visualization_topic_;
  double max_range_{-1.0};
  double yaw_correction_deg_{2.1};

  bool publish_expected_points_{true};
  double expected_publish_rate_hz_{10.0};

  double expected_min_range_{0.25};
  double expected_max_range_{10.0};
  int expected_range_samples_{120};
  double expected_max_residual_{0.5};

  int relative_depth_patch_radius_{2};
  int calibration_min_points_{10};
  double calibration_alpha_{0.35};
  bool calibration_rgb_mask_enabled_{true};
  bool calibration_freeze_on_obstruction_{true};
  double calibration_rgb_max_time_difference_{0.20};
  double calibration_min_inlier_fraction_{0.55};
  double calibration_max_update_fraction_{0.20};
  double max_relative_depth_time_difference_{0.15};

  int lidar_median_window_{3};

  bool lidar_temporal_filter_{true};
  double lidar_temporal_alpha_{0.25};

  bool uv_temporal_filter_{true};
  double uv_temporal_alpha_{0.25};

  int fixed_beam_start_{-1};
  int fixed_beam_end_{-1};

  bool visualize_{false};
  int point_radius_{2};

  // ============================================================
  // Filter state
  // ============================================================

  std::vector<float>
    previous_ranges_;

  std::vector<float>
    previous_u_;

  std::vector<float>
    previous_v_;

  std::vector<bool>
    previous_uv_valid_;

  // ============================================================
  // Subscribers
  // ============================================================

  rclcpp::Subscription<
    sensor_msgs::msg::CameraInfo>::SharedPtr
    camera_info_sub_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
    valid_mask_sub_;

  rclcpp::Subscription<
    sensor_msgs::msg::LaserScan>::SharedPtr
    scan_sub_;

  rclcpp::Subscription<
    sensor_msgs::msg::Image>::SharedPtr
    relative_depth_sub_;

  rclcpp::Subscription<
    sensor_msgs::msg::Image>::SharedPtr
    image_sub_;

  // ============================================================
  // Publishers
  // ============================================================

  rclcpp::Publisher<
    sensor_msgs::msg::PointCloud2>::SharedPtr
    projected_points_pub_;

  rclcpp::Publisher<
    sensor_msgs::msg::PointCloud2>::SharedPtr
    expected_points_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
    rejected_expected_points_pub_;

  rclcpp::TimerBase::SharedPtr
    expected_timer_;

  rclcpp::Publisher<
    sensor_msgs::msg::Image>::SharedPtr
    image_pub_;

  rclcpp::Publisher<
    sensor_msgs::msg::Image>::SharedPtr
    expected_image_pub_;

  // ============================================================
  // TF
  // ============================================================

  std::unique_ptr<tf2_ros::Buffer>
    tf_buffer_;

  std::shared_ptr<
    tf2_ros::TransformListener>
    tf_listener_;

  // ============================================================
  // Camera state
  // ============================================================

  std::mutex data_mutex_;

  image_geometry::PinholeCameraModel
    camera_model_;

  std::string
    camera_frame_from_info_;

  int image_width_{0};
  int image_height_{0};

  bool have_camera_info_{false};
  cv::Mat valid_mask_;

  cv::Mat latest_relative_depth_;

  std_msgs::msg::Header
    latest_relative_depth_header_;

  bool have_relative_depth_{false};

  // Learned mapping:
  //   relative_depth ~= a * (1 / camera_z) + b
  bool have_relative_depth_calibration_{false};
  double relative_depth_a_{0.0};
  double relative_depth_b_{0.0};
  double relative_depth_fit_error_{0.0};

  // Cached LaserScan geometry used for expected projection.
  double scan_angle_min_{0.0};
  double scan_angle_increment_{0.0};
  std::size_t scan_beam_count_{0};
  std::string scan_frame_;
  bool have_scan_geometry_{false};

  cv::Mat latest_image_;

  std_msgs::msg::Header
    latest_image_header_;

  bool have_image_{false};


};


// ================================================================
// Main
// ================================================================

int main(
  int argc,
  char ** argv)
{
  rclcpp::init(
    argc,
    argv);

  rclcpp::spin(
    std::make_shared<
      LidarProjectionNode>());

  rclcpp::shutdown();

  return 0;
}
