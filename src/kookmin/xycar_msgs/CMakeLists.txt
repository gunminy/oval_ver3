cmake_minimum_required(VERSION 2.8.3)
project(xycar_msgs)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  sensor_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  XycarMotor.msg
  XycarUltrasonic.msg
)

## Generate added messages and services with any dependencies
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs sensor_msgs
)

## Build
include_directories(
  ${catkin_INCLUDE_DIRS}
)
