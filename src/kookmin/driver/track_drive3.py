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


#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
image = np.empty(shape=[0])  # 카메라 이미지를 담을 변수
ranges = None  # 라이다 데이터를 담을 변수
motor = None  # 모터노드
motor_msg = XycarMotor()  # 모터 토픽 메시지
Fix_Speed = 20  # 모터 속도 고정 상수값 
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
from scipy.optimize import linear_sum_assignment

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

    # center_points = np.array(center_points)

    # # Find center point closest to the car (smallest y)
    # nearest_idx = np.argmin(center_points[:, 1])
    # target_x = center_points[nearest_idx, 0]
    # target_y = center_points[nearest_idx, 1]

    # # Steering proportional to lateral offset (x) at target point
    # # Optionally, use atan2 for angle to point
    # angle_rad = math.atan2(target_x, target_y)
    # angle_deg = math.degrees(angle_rad)

    # # Clamp to steering limits
    # angle_deg = max(-max_steer, min(max_steer, angle_deg))

    # return angle_deg

def classify_cones_kmeans(cone_centers):
    if len(cone_centers) < 2:
        return np.array(cone_centers), np.empty((0, 2))

    # Convert to NumPy array
    cone_centers = np.array(cone_centers)

    # x좌표에 3배 가중치 부여
    weighted_data = cone_centers.copy()
    weighted_data[:, 0] *= 3  # x좌표 강조

    # KMeans 클러스터링
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init='auto').fit(weighted_data)
    labels = kmeans.labels_

    # 클러스터 분리
    left_cluster = cone_centers[labels == 0]
    right_cluster = cone_centers[labels == 1]

    # 왼쪽/오른쪽 재정렬
    if np.mean(left_cluster[:, 0]) > np.mean(right_cluster[:, 0]):
        left_cluster, right_cluster = right_cluster, left_cluster

    return left_cluster, right_cluster



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

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("original", image)
        #cv2.imshow("gray", gray)

        if ranges is not None:
            # === Show raw LiDAR points for debugging ===
            angles = np.linspace(0, 2*np.pi, len(ranges)) + np.pi/2
            x = ranges * np.cos(angles)
            y = ranges * np.sin(angles)

            raw_x, raw_y = [], []
            for i in range(len(ranges)):
                if 0.1 < ranges[i] < 15.0:
                    raw_x.append(x[i])
                    raw_y.append(y[i])
            raw_points_plot.set_data(raw_x, raw_y)

            # === Detect cones ===
            cone_centers = detect_cones(ranges)

            # Classify left and right cones
            left_cones, right_cones = classify_cones(cone_centers, 1.0)

            steering_angle = compute_steering_angle(left_cones, right_cones, max_steer=55.0)
            speed = Fix_Speed

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


        print(steering_angle, speed)
        drive(angle=steering_angle, speed=speed)
        time.sleep(0.1)
        
        cv2.waitKey(1)

#=============================================
# 메인함수를 호출합니다.
# start() 함수가 실질적인 메인함수입니다.
#=============================================
if __name__ == '__main__':
    start()