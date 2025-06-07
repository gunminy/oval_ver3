#!/usr/bin/env python
# -*- coding: utf-8 -*-
#=============================================
# 본 프로그램은 2025 제8회 국민대 자율주행 경진대회에서
# 예선과제를 수행하기 위한 파일입니다. 
# 예선과제 수행 용도로만 사용가능하며 외부유출은 금지됩니다.
#=============================================
import numpy as np
import cv2, rospy, time, os, math
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
image = np.empty(shape=[0])
ranges = None
motor = None
motor_msg = XycarMotor()
Fix_Speed = 30
MAX_ANGLE = 50

# Stanley 제어 게인
K_SOFT = 1.0
K_STANLEY = 3.0        # 더 강한 CTE 반응
K_HEADING = 2.5        # 더 큰 방향 보정


# 차선 검출 파라미터
MIN_AREA = 50
DEFAULT_LANE_WIDTH = 160
MIN_POINTS = 2

# 현재 차선 상태
current_lane = "UNKNOWN"

# OpenCV 브릿지 객체
bridge = CvBridge()

# 라이다 시각화용 플롯 설정
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-120, 120)
ax.set_ylim(-120, 120)
ax.set_aspect('equal')
lidar_points, = ax.plot([], [], 'bo')

#=============================================
# 콜백함수 - 카메라 토픽을 처리하는 콜백함수
#=============================================
def usbcam_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

#=============================================
# 콜백함수 - 라이다 토픽을 받아서 처리하는 콜백함수
#=============================================
def lidar_callback(data):
    global ranges
    ranges = data.ranges[0:360]

#=============================================
# 모터로 토픽을 발행하는 함수
#=============================================
def drive(angle, speed):
    motor_msg.angle = int(angle)
    motor_msg.speed = float(speed)
    motor.publish(motor_msg)

#=============================================
# 차선 검출 함수 (HSV 이미지 입력)
#=============================================
def detect_lanes(hsv_image):
    # 흰색 (Solid Line)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    # 노란색 (Dashed Center)
    lower_yellow = np.array([20, 150, 150])
    upper_yellow = np.array([40, 255, 255])

    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    return white_mask, yellow_mask

#=============================================
# 두 차선의 중심 좌표로부터 차선 폭 계산
#=============================================
def calculate_lane_width(left_points, right_points):
    if not left_points or not right_points:
        return DEFAULT_LANE_WIDTH
    left_mean = np.mean(left_points)
    right_mean = np.mean(right_points)
    return (right_mean - left_mean) / 2

#=============================================
# 카메라 중심과 차선 위치로 현재 차선 구간 판단
#=============================================
def determine_current_lane(left_solid, center_dashed, right_solid, cx):
    global current_lane
    left_pos = np.mean(left_solid) if left_solid else None
    center_pos = np.mean(center_dashed) if center_dashed else None
    right_pos = np.mean(right_solid) if right_solid else None
    if left_pos is not None and center_pos is not None:
        if abs(cx - ((left_pos + center_pos) / 2)) < 50:
            current_lane = "LEFT"
    elif center_pos is not None and right_pos is not None:
        if abs(cx - ((center_pos + right_pos) / 2)) < 50:
            current_lane = "RIGHT"
    return current_lane

#=============================================
# contour 중심점들을 기반으로 차선의 기울기 계산
#=============================================
def calculate_slope(points):
    if len(points) < 2:
        return 0
    y_coords = np.arange(len(points))
    x_coords = np.array(points)
    try:
        slope, _ = np.polyfit(x_coords, y_coords, 1)
        return -slope
    except Exception:
        return 0

#=============================================
# Stanley 제어기 구현 - cte, heading error 기반 조향각 계산
#=============================================
def stanley_control(cte, heading_error, speed, k=K_STANLEY, k_heading=K_HEADING):
    angle = np.rad2deg(np.arctan2(k * cte, K_SOFT + speed))
    angle += k_heading * heading_error
    return np.clip(angle, -MAX_ANGLE, MAX_ANGLE)

#=============================================
# 이미지에서 차선을 검출하고 중심선과 heading error 계산
#=============================================
def get_lane_info(image):
    global current_lane
    height, width, _ = image.shape
    roi = image[int(height*0.6):, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white_mask, yellow_mask = detect_lanes(hsv)
    debug_image = image.copy()
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_contours = [cnt for cnt in white_contours if cv2.contourArea(cnt) > MIN_AREA]
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours = [cnt for cnt in yellow_contours if cv2.contourArea(cnt) > MIN_AREA]
    left_solid, right_solid, center_dashed = [], [], []
    cx = width // 2
    for cnt in white_contours:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cX = int(M['m10'] / M['m00'])
        if cX < cx:
            left_solid.append(cX)
            cv2.drawContours(debug_image[int(height*0.6):, :], [cnt], -1, (255, 0, 0), 2)
        else:
            right_solid.append(cX)
            cv2.drawContours(debug_image[int(height*0.6):, :], [cnt], -1, (0, 0, 255), 2)
    for cnt in yellow_contours:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cX = int(M['m10'] / M['m00'])
        center_dashed.append(cX)
        cv2.drawContours(debug_image[int(height*0.6):, :], [cnt], -1, (255, 255, 255), 2)
    current_lane = determine_current_lane(left_solid, center_dashed, right_solid, cx)
    heading_error = 0
    lane_width = None
    if current_lane == "LEFT":
        if len(left_solid) > 0 and len(center_dashed) > 0:
            left_mean = np.mean(left_solid)
            center_mean = np.mean(center_dashed)
            left_slope = calculate_slope(left_solid)
            center_slope = calculate_slope(center_dashed)
            if abs(left_slope) > 0.1 or abs(center_slope) > 0.1:
                lane_center = (left_mean + center_mean) / 2
                lane_center += -20 if left_slope > 0 else 20
            else:
                lane_center = (left_mean + center_mean) / 2
            lane_width = calculate_lane_width(left_solid, center_dashed)
            heading_error = np.rad2deg(np.arctan2(left_slope + center_slope, 2))
        elif len(left_solid) > 0:
            left_mean = np.mean(left_solid)
            left_slope = calculate_slope(left_solid)
            if abs(left_slope) > 0.1:
                lane_center = left_mean + DEFAULT_LANE_WIDTH * (1.2 if left_slope > 0 else 0.8)
            else:
                lane_center = left_mean + DEFAULT_LANE_WIDTH
            heading_error = np.rad2deg(np.arctan2(left_slope, 1))
        else:
            return 0.0, 0.0, debug_image
    elif current_lane == "RIGHT":
        if len(right_solid) > 0 and len(center_dashed) > 0:
            right_mean = np.mean(right_solid)
            center_mean = np.mean(center_dashed)
            right_slope = calculate_slope(right_solid)
            center_slope = calculate_slope(center_dashed)
            if abs(right_slope) > 0.1 or abs(center_slope) > 0.1:
                lane_center = (center_mean + right_mean) / 2
                lane_center += -25 if right_slope > 0 else 25
            else:
                lane_center = (center_mean + right_mean) / 2
            lane_width = calculate_lane_width(center_dashed, right_solid)
            heading_error = np.rad2deg(np.arctan2(right_slope + center_slope, 2))
        elif len(right_solid) > 0:
            right_mean = np.mean(right_solid)
            right_slope = calculate_slope(right_solid)
            if abs(right_slope) > 0.1:
                lane_center = right_mean - DEFAULT_LANE_WIDTH * (1.2 if right_slope < 0 else 0.8)
            else:
                lane_center = right_mean - DEFAULT_LANE_WIDTH
            heading_error = np.rad2deg(np.arctan2(right_slope, 1))
        else:
            return 0.0, 0.0, debug_image
    cte = (lane_center - cx) / (width / 2)
    cte = np.clip(cte, -1.0, 1.0)
    if abs(heading_error) < 1.0 and abs(cte) > 0.2:
        heading_error = -25.0 if cte < 0 else 25.0
    cv2.circle(debug_image, (int(lane_center), height - 50), 5, (0, 255, 0), -1)
    cv2.line(debug_image, (cx, 0), (cx, height), (255, 255, 0), 1)
    cv2.putText(debug_image, f'Lane: {current_lane}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if lane_width:
        cv2.putText(debug_image, f'Width: {lane_width:.1f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(debug_image, f'Heading: {heading_error:.1f}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    print(f"왼쪽 차선 점 개수: {len(left_solid)}")
    print(f"중앙 점선 점 개수: {len(center_dashed)}")
    print(f"오른쪽 차선 점 개수: {len(right_solid)}")
    return cte, heading_error, debug_image

#=============================================
# 메인 루프 함수 - 센서 입력 기반 자율주행 수행
#=============================================
def start():
    global motor, image, ranges
    print("Start program --------------")
    rospy.init_node('Track_Driver')
    rospy.Subscriber("/usb_cam/image_raw/", Image, usbcam_callback, queue_size=1)
    rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1)
    motor = rospy.Publisher('xycar_motor', XycarMotor, queue_size=1)
    rospy.wait_for_message("/usb_cam/image_raw/", Image)
    print("Camera Ready --------------")
    rospy.wait_for_message("/scan", LaserScan)
    print("Lidar Ready ----------")
    plt.ion()
    plt.show()
    print("Lidar Visualizer Ready ----------")
    print("======================================")
    print(" S T A R T    D R I V I N G ...")
    print("======================================")
    while not rospy.is_shutdown():
        if image is None:
            continue
        try:
            current_time = rospy.Time.now()
            cte, heading_error, debug_image = get_lane_info(image)
            angle = stanley_control(cte, heading_error, Fix_Speed)
            cv2.putText(debug_image, f'CTE: {cte:.3f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(debug_image, f'Angle: {angle:.1f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            drive(angle=angle, speed=Fix_Speed)
            cv2.imshow("Lane Detection", debug_image)
            cv2.waitKey(1)
            if ranges is not None:
                angles = np.linspace(0, 2 * np.pi, len(ranges)) + np.pi / 2
                x = ranges * np.cos(angles)
                y = ranges * np.sin(angles)
                lidar_points.set_data(x, y)
                fig.canvas.draw_idle()
                plt.pause(0.01)
        except Exception as e:
            rospy.logerr(f"오류 발생: {e}")
            drive(angle=0, speed=Fix_Speed)
        time.sleep(0.1)

#=============================================
# 메인 함수 호출
#=============================================
if __name__ == '__main__':
    try:
        start()
    except rospy.ROSInterruptException:
        pass