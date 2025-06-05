#!/usr/bin/env python
# -*- coding: utf-8 -*- 2
#=============================================
# 본 프로그램은 2025 제8회 국민대 자율주행 경진대회에서
# 예선과제를 수행하기 위한 파일입니다. 
# 예선과제 수행 용도로만 사용가능하며 외부유출은 금지됩니다.
#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import numpy as np
import cv2, rospy, time, os, math
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from scipy.optimize import linear_sum_assignment


##### THIS CODE: CONE IMPLEMENTATION DONE (NOT PERFECT) #####
##### THIS CODE: MODE SWITCHING (CONE MODE and NORMAL MODE) #####

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
# Driving modes
NORMAL_MODE = 0  # Normal road driving mode
CONE_MODE = 1    # Cone navigation mode
current_mode = NORMAL_MODE

# Lane detection parameters
LANE_ROI_HEIGHT = 120  # Height of the ROI for lane detection
LANE_ROI_WIDTH = 640   # Width of the ROI for lane detection
LANE_THRESHOLD = 100   # Threshold for lane detection

image = np.empty(shape=[0])  # 카메라 이미지를 담을 변수
ranges = None  # 라이다 데이터를 담을 변수
motor = None  # 모터노드
motor_msg = XycarMotor()  # 모터 토픽 메시지
Fix_Speed = 10  # 모터 속도 고정 상수값 
new_angle = 0  # 모터 조향각 초기값
new_speed = Fix_Speed  # 모터 속도 초기값
bridge = CvBridge()  # OpenCV 함수를 사용하기 위한 브릿지 

#=============================================
# 라이다 스캔정보로 그림을 그리기 위한 변수
#=============================================
fig, ax = plt.subplots(figsize=(4, 4))
ax.set_xlim(-15, 15)
ax.set_ylim(-13, 17)
ax.set_aspect('equal')
left_cones_plot, = ax.plot([], [], 'o', color='red', markersize=6, label='Left Cone')
right_cones_plot, = ax.plot([], [], 'o', color='green', markersize=6, label='Right Cone')
centerline_plot, = ax.plot([], [], 'o-', color='blue', linewidth=2, label='Center Path')
raw_points_plot, = ax.plot([], [], '.', color='gray', markersize=2, alpha=0.4, label='Raw Lidar')

ax.legend()

#=============================================
# PID 컨트롤러 클래스
#=============================================
class SteeringPID:
    def __init__(self, kp, ki=0.05, kd=0.2, max_i=1.0, dt=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_i = max_i  # Anti-windup
        self.dt = dt
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error):
        # Proportional term
        p = self.kp * error
        
        # Integral term with clamping
        self.integral += error * self.dt
        self.integral = max(-self.max_i, min(self.max_i, self.integral))
        i = self.ki * self.integral
        
        # Derivative term
        d = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        
        return p + i + d

steering_pid = SteeringPID(kp=60.0, ki=0.05, kd=0.2)

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
    motor_msg.angle = float(angle)
    motor_msg.speed = float(speed)
    motor.publish(motor_msg)

def detect_cones(ranges):
    # Convert polar to cartesian coordinates
    angles = np.linspace(0, 2*np.pi, len(ranges)) + np.pi/2
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)

    # Filter out invalid points (inf or zero distance)
    points = np.array([[x[i], y[i]] for i in range(len(ranges)) if 0.02 < ranges[i] < 13.0])

    if len(points) == 0:
        return []

    # DBSCAN clustering
    dbscan = DBSCAN(eps=1.0, min_samples=1)  # eps and min_samples may need tuning
    labels = dbscan.fit_predict(points)

    # Extract clusters and filter small ones (likely cones)
    cone_centers = []
    for label in set(labels):
        if label == -1:
            continue  # noise
        cluster = points[labels == label]
        if 1 <= len(cluster) <= 3:  # likely a small object like a cone
            center = np.mean(cluster, axis=0)
            cone_centers.append(center)

    return cone_centers

#=============================================
# 콘 분류 함수
#=============================================
def hungarian_classify(cone_centers, max_pair_distance=2.0):
    if len(cone_centers) < 2:
        return np.empty((0,2)), np.empty((0,2))
    
    # Cost matrix: Absolute lateral (x) distance between cone pairs
    cost_matrix = np.abs(cone_centers[:,0] - cone_centers[:,0][:, np.newaxis])
    
    # Apply Hungarian algorithm for optimal pairing
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Filter valid pairs and assign left/right
    left, right = [], []
    paired_indices = set()
    for i, j in zip(row_ind, col_ind):
        if i != j and cost_matrix[i,j] < max_pair_distance and i not in paired_indices and j not in paired_indices:
            if cone_centers[i,0] < cone_centers[j,0]:
                left.append(cone_centers[i])
                right.append(cone_centers[j])
            else:
                left.append(cone_centers[j])
                right.append(cone_centers[i])
            paired_indices.update([i,j])
    
    # Handle unpaired cones
    unpaired = [cone_centers[k] for k in range(len(cone_centers)) if k not in paired_indices]
    for cone in unpaired:
        if cone[0] < np.mean([c[0] for c in left]) if left else cone[0] < 0:
            left.append(cone)
        else:
            right.append(cone)
    
    return np.array(left), np.array(right)

def classify_cones(cone_centers, max_pair_distance=2.0):
    # cone_centers: Nx2 numpy array (x, y)
    if len(cone_centers) == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    cone_centers = np.array(cone_centers)
    sorted_indices = np.argsort(cone_centers[:,1])
    cones_sorted = cone_centers[sorted_indices]

    left_cones = []
    right_cones = []
    
    window_size = 1.5
    overlap = 0.5  # Critical for S-curve continuity
    y_current = cones_sorted[0,1]

    while y_current <= cones_sorted[-1,1]:
        window_mask = (cones_sorted[:,1] >= y_current) & (cones_sorted[:,1] < y_current + window_size)
        in_window = cones_sorted[window_mask]
        
        if len(in_window) >= 2:
            window_left, window_right = hungarian_classify(in_window)
            left_cones.extend(window_left)
            right_cones.extend(window_right)
        elif len(in_window) == 1:
            # Fallback to median split
            if in_window[0,0] < np.median([c[0] for c in left_cones]) if left_cones else in_window[0,0] < 0:
                left_cones.append(in_window[0])
            else:
                right_cones.append(in_window[0])
        
        y_current += (window_size - overlap)

    return np.array(left_cones), np.array(right_cones)

#=============================================
# 조향각 계산 함수
#=============================================
def compute_steering_angle(left_cones, right_cones, max_steer=45.0):
    global steering_pid
    if len(left_cones) == 0 and len(right_cones) > 4:
        return 60  # Hard left turn
    elif len(right_cones) == 0 and len(left_cones) > 4:
        return -60  # Hard right turn
    elif len(left_cones) == 0 and len(right_cones) == 0:
        return 0.0  # No cones, maintain course

    if left_cones.size == 0 or right_cones.size == 1:
        return 0.0
        
    left_cones = np.array(left_cones, dtype=np.float64).reshape(-1, 2)
    right_cones = np.array(right_cones, dtype=np.float64).reshape(-1, 2)
    

    # Match each left cone with its nearest right cone based on y-axis (forward)
    center_points = []

    for l in left_cones:
        if right_cones.shape[0] == 0:  # Extra protection
            continue
            
        dy = np.abs(right_cones[:, 1] - l[1])
        
        # Skip if no valid right cones
        if dy.size == 0:
            continue
            
        min_idx = np.argmin(dy)
        
        if dy[min_idx] < 1.5:
            paired_right = right_cones[min_idx]
            center_points.append([(l[0]+paired_right[0])/2, 
                                 (l[1]+paired_right[1])/2])


    # Enhanced fallback logic
    if not center_points:
        if left_cones.size > 0 and right_cones.size == 0:
            return -max_steer * 0.9
        elif right_cones.size > 0 and left_cones.size == 0:
            return max_steer * 0.7
        else:
            return 0.0


    center_points = np.array(center_points) 

    # Select target point 2m ahead for better stability
    lookahead_distance = 3.6
    distances = np.abs(center_points[:,1] - lookahead_distance)
    target_idx = np.argmin(distances)
    target_x = center_points[target_idx, 0]
    
    # Calculate PID control
    error = target_x  # Lateral deviation from center
    steer_angle = steering_pid.compute(error)
    
    # Clamp steering angle
    return max(-max_steer, min(max_steer, steer_angle))

def detect_lanes(image):
    """
    Detect lanes in the image and return steering angle for normal road driving.
    Road structure: Two solid white outer lanes and dotted center line.
    Distance between outer lane and center line is slightly wider than car width.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define ROI
    height = image.shape[0]
    width = image.shape[1]
    roi = gray[height-LANE_ROI_HEIGHT:height, :]
    
    # Apply threshold with a slightly lower value to better detect the dotted line
    _, binary = cv2.threshold(roi, LANE_THRESHOLD-20, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to better detect dotted lines
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Expected lane positions (in pixels)
    CAR_WIDTH_PX = 200  # Approximate car width in pixels
    LANE_TO_CENTER = int(CAR_WIDTH_PX * 1.2)  # Distance from outer lane to center line
    
    # Lists to store detected lane positions
    left_outer = []
    right_outer = []
    center_dots = []
    
    mid_x = width // 2
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:  # Filter small contours
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        center_x = x + w//2
        
        # Classify based on position and shape
        aspect_ratio = float(h)/w if w > 0 else 0
        
        if aspect_ratio > 2.0:  # Likely a solid line
            if center_x < mid_x - LANE_TO_CENTER//2:
                left_outer.append(center_x)
            elif center_x > mid_x + LANE_TO_CENTER//2:
                right_outer.append(center_x)
        else:  # Likely a dotted line
            if abs(center_x - mid_x) < LANE_TO_CENTER//2:
                center_dots.append(center_x)
    
    # Calculate steering angle based on all detected lanes
    if left_outer and right_outer:
        # Both outer lanes detected
        left_avg = sum(left_outer) / len(left_outer)
        right_avg = sum(right_outer) / len(right_outer)
        center = (left_avg + right_avg) / 2
        error = center - mid_x
        return -error / mid_x * 50.0
    elif left_outer and center_dots:
        # Left outer and center line detected
        left_avg = sum(left_outer) / len(left_outer)
        center_avg = sum(center_dots) / len(center_dots)
        expected_right = center_avg + LANE_TO_CENTER
        center = (left_avg + expected_right) / 2
        error = center - mid_x
        return -error / mid_x * 40.0
    elif right_outer and center_dots:
        # Right outer and center line detected
        right_avg = sum(right_outer) / len(right_outer)
        center_avg = sum(center_dots) / len(center_dots)
        expected_left = center_avg - LANE_TO_CENTER
        center = (expected_left + right_avg) / 2
        error = center - mid_x
        return -error / mid_x * 40.0
    elif center_dots:
        # Only center line detected
        center_avg = sum(center_dots) / len(center_dots)
        error = center_avg - mid_x
        return -error / mid_x * 30.0
    elif left_outer:
        # Only left outer lane detected
        left_avg = sum(left_outer) / len(left_outer)
        expected_center = left_avg + LANE_TO_CENTER
        error = expected_center - mid_x
        return -error / mid_x * 30.0
    elif right_outer:
        # Only right outer lane detected
        right_avg = sum(right_outer) / len(right_outer)
        expected_center = right_avg - LANE_TO_CENTER
        error = expected_center - mid_x
        return -error / mid_x * 30.0
    
    return 0.0  # No lanes detected

def check_mode(ranges, image):
    """
    Determine which driving mode to use based on sensor data
    """
    global current_mode
    
    # Detect cones using LiDAR
    cone_centers = detect_cones(ranges)
    
    # If we detect more than 2 cones, switch to cone mode
    if len(cone_centers) >= 2:
        if current_mode != CONE_MODE:
            print("Switching to CONE mode")
            current_mode = CONE_MODE
    else:
        # Check for lane markers in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height = image.shape[0]
        roi = gray[height-LANE_ROI_HEIGHT:height, :]
        _, binary = cv2.threshold(roi, LANE_THRESHOLD, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(binary)
        
        # If we see enough lane markers, switch to normal mode
        if white_pixels > 1000:  # Threshold can be adjusted
            if current_mode != NORMAL_MODE:
                print("Switching to NORMAL mode")
                current_mode = NORMAL_MODE

def process_normal_mode(image):
    """
    Process normal road driving mode with enhanced visualization
    Shows outer solid lanes and center dotted line separately
    """
    # Get original steering angle
    steering_angle = detect_lanes(image)
    speed = Fix_Speed
    
    # Create visualization image
    height = image.shape[0]
    width = image.shape[1]
    
    # Extract ROI
    roi = image[height-LANE_ROI_HEIGHT:height, :].copy()
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_roi, LANE_THRESHOLD-20, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a color visualization image
    vis_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # Expected lane positions
    CAR_WIDTH_PX = 200
    LANE_TO_CENTER = int(CAR_WIDTH_PX * 1.2)
    mid_x = width // 2
    
    # Lists to store detected lane positions
    left_outer = []
    right_outer = []
    center_dots = []
    
    # Draw detected lane markers with different colors for each type
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        center_x = x + w//2
        aspect_ratio = float(h)/w if w > 0 else 0
        
        if aspect_ratio > 2.0:  # Solid lines
            if center_x < mid_x - LANE_TO_CENTER//2:
                left_outer.append(center_x)
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red for left outer
            elif center_x > mid_x + LANE_TO_CENTER//2:
                right_outer.append(center_x)
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue for right outer
        else:  # Dotted center line
            if abs(center_x - mid_x) < LANE_TO_CENTER//2:
                center_dots.append(center_x)
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 255), 2)  # Yellow for center
    
    # Draw reference lines
    cv2.line(vis_image, (mid_x, 0), (mid_x, LANE_ROI_HEIGHT), (0, 255, 0), 1)  # Center reference
    cv2.line(vis_image, (mid_x - LANE_TO_CENTER, 0), (mid_x - LANE_TO_CENTER, LANE_ROI_HEIGHT), (128, 128, 128), 1)  # Left reference
    cv2.line(vis_image, (mid_x + LANE_TO_CENTER, 0), (mid_x + LANE_TO_CENTER, LANE_ROI_HEIGHT), (128, 128, 128), 1)  # Right reference
    
    # Draw detected lane averages
    if left_outer:
        left_avg = int(sum(left_outer) / len(left_outer))
        cv2.line(vis_image, (left_avg, 0), (left_avg, LANE_ROI_HEIGHT), (0, 0, 255), 2)
    
    if right_outer:
        right_avg = int(sum(right_outer) / len(right_outer))
        cv2.line(vis_image, (right_avg, 0), (right_avg, LANE_ROI_HEIGHT), (255, 0, 0), 2)
    
    if center_dots:
        center_avg = int(sum(center_dots) / len(center_dots))
        cv2.line(vis_image, (center_avg, 0), (center_avg, LANE_ROI_HEIGHT), (0, 255, 255), 2)
    
    # Draw steering indicator
    center_y = LANE_ROI_HEIGHT - 30
    steering_x = mid_x + int(steering_angle * width / 100)
    cv2.arrowedLine(vis_image, (mid_x, center_y), (steering_x, center_y), (0, 255, 0), 2)
    
    # Add text information
    cv2.putText(vis_image, f"Steering: {steering_angle:.1f}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(vis_image, "Left Outer" if left_outer else "No Left", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if left_outer else (128, 128, 128), 2)
    cv2.putText(vis_image, "Center Line" if center_dots else "No Center", (mid_x-40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if center_dots else (128, 128, 128), 2)
    cv2.putText(vis_image, "Right Outer" if right_outer else "No Right", (width-120, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0) if right_outer else (128, 128, 128), 2)
    
    # Show both original ROI and processed image side by side
    combined_vis = np.hstack((roi, vis_image))
    cv2.imshow("Lane Detection", combined_vis)
    
    return steering_angle, speed

def process_cone_mode(ranges):
    """
    Process cone navigation mode
    """
    # Detect and classify cones
    cone_centers = detect_cones(ranges)
    left_cones, right_cones = classify_cones(cone_centers, 1.0)
    
    # Compute steering angle using existing cone navigation logic
    steering_angle = compute_steering_angle(left_cones, right_cones, max_steer=55.0)
    speed = Fix_Speed
    
    return steering_angle, speed

#=============================================
# 실질적인 메인 함수 
#=============================================
def start():

    global motor, image, ranges
    
    print("Start program --------------")

    #=========================================
    # 노드를 생성하고, 구독/발행할 토픽들을 선언합니다.
    #=========================================
    rospy.init_node('Track_Driver')
    rospy.Subscriber("/usb_cam/image_raw/",Image,usbcam_callback, queue_size=1)
    rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1)
    motor = rospy.Publisher('xycar_motor', XycarMotor, queue_size=1)
        
    #=========================================
    # 노드들로부터 첫번째 토픽들이 도착할 때까지 기다립니다.
    #=========================================
    rospy.wait_for_message("/usb_cam/image_raw/", Image)
    print("Camera Ready --------------")
    rospy.wait_for_message("/scan", LaserScan)
    print("Lidar Ready ----------")
    
    #=========================================
    # 라이다 스캔정보에 대한 시각화 준비를 합니다.
    #=========================================
    plt.ion()
    plt.show()
    print("Lidar Visualizer Ready ----------")
    
    print("======================================")
    print(" S T A R T    D R I V I N G ...")
    print("======================================")

    #=========================================
    # 메인 루프 
    #=========================================
    while not rospy.is_shutdown():
        if image.size == 0 or ranges is None:
            continue

        # Check and update driving mode
        check_mode(ranges, image)

        # Process current mode
        if current_mode == NORMAL_MODE:
            steering_angle, speed = process_normal_mode(image)
        else:  # CONE_MODE
            steering_angle, speed = process_cone_mode(ranges)

            # Show raw LiDAR points and cone visualization
            angles = np.linspace(0, 2*np.pi, len(ranges)) + np.pi/2
            x = ranges * np.cos(angles)
            y = ranges * np.sin(angles)

            raw_x, raw_y = [], []
            for i in range(len(ranges)):
                if 0.1 < ranges[i] < 15.0:
                    raw_x.append(x[i])
                    raw_y.append(y[i])
            raw_points_plot.set_data(raw_x, raw_y)

            # Detect and visualize cones
            cone_centers = detect_cones(ranges)
            left_cones, right_cones = classify_cones(cone_centers, 1.0)

            if len(left_cones) > 0:
                left_cones_plot.set_data(left_cones[:, 0], left_cones[:, 1])
            else:
                left_cones_plot.set_data([], [])

            if len(right_cones) > 0:
                right_cones_plot.set_data(right_cones[:, 0], right_cones[:, 1])
            else:
                right_cones_plot.set_data([], [])

            # Center path calculation
            center_points = []
            for l in left_cones:
                closest_r = min(right_cones, key=lambda r: abs(r[1] - l[1]), default=None)
                if closest_r is not None and abs(l[1] - closest_r[1]) < 1.0:
                    center = (l + closest_r) / 2
                    center_points.append(center)

            if center_points:
                center_points = np.array(center_points)
                centerline_plot.set_data(center_points[:, 0], center_points[:, 1])
            else:
                centerline_plot.set_data([], [])

            # Update plot
            fig.canvas.draw_idle()
            plt.pause(0.1)

        # Show mode status
        cv2.putText(image, f"Mode: {'NORMAL' if current_mode == NORMAL_MODE else 'CONE'}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Steering: {steering_angle:.1f}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("original", image)

        print(f"Mode: {'NORMAL' if current_mode == NORMAL_MODE else 'CONE'}, "
              f"Steering: {steering_angle:.1f}, Speed: {speed}")
        drive(angle=steering_angle, speed=speed)
        time.sleep(0.1)
        
        cv2.waitKey(1)

#=============================================
# 메인함수를 호출합니다.
# start() 함수가 실질적인 메인함수입니다.
#=============================================
if __name__ == '__main__':
    start()
