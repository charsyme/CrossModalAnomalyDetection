cmake_minimum_required(VERSION 3.8)
project(lidar_projection)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_geometry REQUIRED)
find_package(laser_geometry REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(lidar_projection_node
  src/lidar_projection_node.cpp
)

ament_target_dependencies(lidar_projection_node
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  image_geometry
  laser_geometry
  tf2
  tf2_ros
  tf2_geometry_msgs
)

target_include_directories(lidar_projection_node PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(lidar_projection_node
  ${OpenCV_LIBS}
)

install(TARGETS
  lidar_projection_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
