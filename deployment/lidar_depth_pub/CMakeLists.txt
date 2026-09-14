cmake_minimum_required(VERSION 3.16)
project(lidar_depth_pub)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(livox_ros_driver2 REQUIRED)

# libzmq — try pkg-config first, fall back to direct linking
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ZMQ IMPORTED_TARGET libzmq)
endif()

# ── Target ────────────────────────────────────────────────────────────────────
add_executable(lidar_depth_pub lidar_depth_pub.cpp)
add_executable(lidar_depth_pub_occ lidar_depth_pub_occ.cpp)

ament_target_dependencies(lidar_depth_pub
    rclcpp
    livox_ros_driver2
)

ament_target_dependencies(lidar_depth_pub_occ
    rclcpp
    livox_ros_driver2
)

target_link_libraries(lidar_depth_pub
    $<IF:$<TARGET_EXISTS:PkgConfig::ZMQ>,PkgConfig::ZMQ,zmq>
    pthread
)

target_link_libraries(lidar_depth_pub_occ
    $<IF:$<TARGET_EXISTS:PkgConfig::ZMQ>,PkgConfig::ZMQ,zmq>
    pthread
)

target_compile_options(lidar_depth_pub PRIVATE -Wall -Wextra -O2)
target_compile_options(lidar_depth_pub_occ PRIVATE -Wall -Wextra -O2)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS lidar_depth_pub lidar_depth_pub_occ DESTINATION lib/${PROJECT_NAME})

ament_package()
