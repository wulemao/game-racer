cmake_minimum_required(VERSION 3.0.2)
project(car_ros)

# Find catkin and any dependencies
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  tf2_ros
)

# Declare the catkin package
catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs nav_msgs visualization_msgs tf2_ros
)

# Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Install targets, if applicable
install(
  DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# Testing (if needed)
if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  # Add test configuration here, if any
endif()
