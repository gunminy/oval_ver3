#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import numpy as np
import cv2, rospy, time
from sensor_msgs.msg import Image, LaserScan
from xycar_msgs.msg import XycarMotor
from cv_bridge import CvBridge
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment

#---------------------------------------------
# 전역 변수 및 설정
#---------------------------------------------
bridge     = CvBridge()
motor      = None
motor_msg  = XycarMotor()


# 최신 센서 데이터 저장용
latest_img  = None
latest_scan = None

# 주행 모드
DRIVE_MODE = 0          # 0:라바콘 1:차선 
LANE_CHANGE_MODE = False
# 현재/이전 차로 위치 (0:왼쪽, 1:오른쪽)
CURRENT_LANE = 0
PREV_LANE    = 0

# 목표 CTE (차선 변경 시점)
target_cte = 0.0

# 주행 파라미터
Fix_Speed   = 50       # 주행 속도
MAX_ANGLE   = 100      # 내부 제어 ±100
K_SOFT      = 1.0
K_STANLEY   = 110.0
K_HEADING   = 10.0

# Lidar 비주얼 파라미터
NUM_LIDAR   = 360
angles      = np.linspace(0, 2*np.pi, NUM_LIDAR, endpoint=False)
IMG_SIZE    = 500
CENTER      = np.array([IMG_SIZE//2, IMG_SIZE//2])
SCALE       = 100.0   # 1m → 100px

# 차선 검출
MIN_AREA    = 50

# 신호등 검출 상태
started     = True

#마지막 차선변경 시간
last_lane_change_time = 0.0

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
        p = self.kp * error
        self.integral += error * self.dt
        self.integral = max(-self.max_i, min(self.max_i, self.integral))
        i = self.ki * self.integral
        d = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        return p + i + d

# Stanley 및 차선 변경 PID 인스턴스
steering_pid = SteeringPID(kp=50.0, ki=0.0, kd=2.0)
lane_change_pid = SteeringPID(kp=1.025, ki=0.1, kd=1.1)

#=============================================
# 조향각 계산 함수
#=============================================
def compute_steering_angle(left_cones, right_cones, max_steer=100.0):
    global steering_pid
        
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
        
        if dy[min_idx] < 2.0:
            paired_right = right_cones[min_idx]
            center_points.append([(l[0]+paired_right[0])/2, 
                                 (l[1]+paired_right[1])/2])


    # Enhanced fallback logic
    if not center_points:
        return 0.0
            


    center_points = np.array(center_points) 

    # Select target point 2m ahead for better stability
    lookahead_distance = 2.0
    distances = np.abs(center_points[:,1] - lookahead_distance)
    target_idx = np.argmin(distances)
    target_x = center_points[target_idx, 0]
    
    # Calculate PID control
    error = target_x  # Lateral deviation from center
    steer_angle = steering_pid.compute(error)
    
    # Clamp steering angle
    return max(-max_steer, min(max_steer, steer_angle))

#=============================================
# 간단 좌/우 분류: x 좌표 중앙값 기준
#=============================================
def classify_cones(cone_centers):
    """
    cone_centers: 리스트 또는 Nx2 numpy array of (x,y)
    returns: left_cones, right_cones as numpy arrays
    """
    pts = np.array(cone_centers, dtype=np.float64)
    if pts.size == 0:
        return np.empty((0,2)), np.empty((0,2))

    # x 좌표 중앙값을 기준으로 좌/우 분리
    median_x = np.median(pts[:,0])
    left_cones  = pts[pts[:,0] < median_x]
    right_cones = pts[pts[:,0] >= median_x]

    return left_cones, right_cones


#---------------------------------------------
# 센서 데이터 처리 함수
#---------------------------------------------
def detect_cones(ranges):
    # Convert polar to cartesian coordinates
    angles = np.linspace(0, 2*np.pi, len(ranges)) + np.pi/2
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)

    # Filter out invalid points (inf or zero distance)
    points = np.array([[x[i], y[i]] for i in range(len(ranges)) if 0.02 < ranges[i] < 5.5])

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

#---------------------------------------------
# 모터 제어 퍼블리시
#---------------------------------------------
def drive(angle, speed):
    motor_msg.angle = float(angle)
    motor_msg.speed = float(speed)
    motor.publish(motor_msg)
    rospy.loginfo(f"[Drive] angle={angle:.1f}, speed={speed:.1f}")

#---------------------------------------------
# 신호등 인식 (초록만)
#---------------------------------------------
def detect_green_light(image):
    roi = image[30:110, 180:450]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([50,100,100]), np.array([80,255,255]))
    return cv2.countNonZero(mask) > 500

#---------------------------------------------
# 차선 검출 및 CTE 계산
#---------------------------------------------
def detect_lanes(hsv):
    w_mask = cv2.morphologyEx(cv2.inRange(hsv, np.array([0,0,200]), np.array([180,30,255])),
                               cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    y_mask = cv2.morphologyEx(cv2.inRange(hsv, np.array([20,150,150]), np.array([40,255,255])),
                               cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    return w_mask, y_mask


#---------------------------------------------
# 차로 정보 추출
#---------------------------------------------
def get_lane_info(img):
    global CURRENT_LANE
    h, w = img.shape[:2]
    roi = img[int(h*0.6):]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    w_mask, y_mask = detect_lanes(hsv)
    cx = w//2
    pts_left, pts_right, pts_center = [], [], []

    # 흰 차선(왼/오른쪽)
    for cnt in cv2.findContours(w_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(cnt) < MIN_AREA: 
            continue
        x = int(cv2.moments(cnt)['m10'] / cv2.moments(cnt)['m00'])
        (pts_left if x < cx else pts_right).append(x)

    # 노란 차선(센터라인)
    for cnt in cv2.findContours(y_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        x = int(cv2.moments(cnt)['m10'] / cv2.moments(cnt)['m00'])
        pts_center.append(x)

    # 1) 노란선이 있으면, 화면 기준으로 왼/오른쪽 판정
    if pts_center:
        mean_c = np.mean(pts_center)
        if mean_c > cx:
            # 노란선이 오른쪽
            CURRENT_LANE = 0
            # 왼쪽 흰선이 있으면 평균값 이용, 없으면 노란선만 사용
            lane_center = (mean_c + np.mean(pts_left)) / 2 if pts_left else mean_c
        else:
            # 노란선이 왼쪽
            CURRENT_LANE = 1
            lane_center = (mean_c + np.mean(pts_right)) / 2 if pts_right else mean_c

    # 2) 노란선 없고 흰선 양쪽 다 있으면 양쪽 흰선 평균
    elif pts_left and pts_right:
        lane_center = (np.mean(pts_left) + np.mean(pts_right)) / 2

    # 3) 그 외(검출 불안정)에는 화면 중앙
    else:
        lane_center = cx

    # CTE 계산
    cte = (lane_center - cx) / (w / 2)
    return cte, 0.0, roi  # heading error는 필요시 추가 계산


#---------------------------------------------
# Stanley 제어
#---------------------------------------------
def stanley_control(cte, heading_error, speed):
    ang = np.rad2deg(np.arctan2(K_STANLEY*cte, K_SOFT+speed)) + K_HEADING*heading_error
    return np.clip(ang, -MAX_ANGLE, MAX_ANGLE)

#---------------------------------------------
# 차선 변경 시작
#---------------------------------------------
def start_lane_change():
    global LANE_CHANGE_MODE, PREV_LANE, target_cte, lane_change_pid, last_lane_change_time

    # 쿨다운 체크: 마지막 차선변경 완료 후 3초가 지나야 다시 시작
    if time.time() - last_lane_change_time < 3.0:
        return
    
    if not LANE_CHANGE_MODE:
        LANE_CHANGE_MODE = True
        PREV_LANE = CURRENT_LANE
        target_cte =  30 if CURRENT_LANE==0 else -30
        lane_change_pid.integral = 0.0
        lane_change_pid.prev_error = 0.0
        rospy.loginfo(f"[LaneChange] start: {PREV_LANE}-> {1-PREV_LANE}")

#---------------------------------------------
# 차선 변경 제어
#---------------------------------------------
def lane_change_control(cte):
    global LANE_CHANGE_MODE, target_cte
    global last_lane_change_time
    error = target_cte - cte
    ang = lane_change_pid.compute(error)
    angle = np.clip(ang, -MAX_ANGLE, MAX_ANGLE)
    if CURRENT_LANE != PREV_LANE:
        LANE_CHANGE_MODE = False
        last_lane_change_time = time.time()    # 차선변경 완료 시각 기록
        rospy.loginfo(f"[LaneChange] completed: now lane {CURRENT_LANE} (cooldown start)")
    return angle

#---------------------------------------------
# 동기화 없이, 최신 메시지를 이용해 처리
#---------------------------------------------
def synced_cb(event):
    global started, DRIVE_MODE
    global LANE_CHANGE_MODE
    
    # 초기 메시지 수신 전엔 스킵
    if latest_img is None or latest_scan is None:
        return

    img_msg  = latest_img
    scan_msg = latest_scan

    img = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
    if not started:
        if detect_green_light(img):
            rospy.loginfo("🟢 Green light! Start driving")
            started = True
            drive(0.0, Fix_Speed)
        else:
            drive(0.0, 0.0)
        cv2.waitKey(1)
        return

    if DRIVE_MODE == 0:
        ranges = scan_msg.ranges[0:360]
        cone_centers     = detect_cones(ranges)

        # --- 전방 12시 방향 ±15° (총 30°) 구간에 장애물(콘 포함)이 없으면 차선 모드 전환 ---
        forward_thresh = 100.0  # m 단위 조정
        # 인덱스 0이 12시, 배열 길이 360 기준 ±15° → 인덱스 -20~+20
        # ranges[-15:] + ranges[:16] 로 추출
        fwd_ranges = list(ranges[-20:]) + list(ranges[:20])
        has_obstacle = any(0.02 < r < forward_thresh for r in fwd_ranges)
        if not has_obstacle:
            DRIVE_MODE = 1
            return
        # --------------------------------------------------------------
        left_cones, right_cones = classify_cones(cone_centers)
        steering_angle = compute_steering_angle(left_cones, right_cones)
        drive(steering_angle, 25)

    else:
        cte, heading, lane_img = get_lane_info(img)
        if LANE_CHANGE_MODE:
            angle = lane_change_control(cte)
        else:
            angle = stanley_control(cte, heading, Fix_Speed)
        
        ranges = scan_msg.ranges[0:360]
        # --- 전방 12시 방향 ±15° (총 30°) 구간에 장애물(콘 포함)이 없으면 차선 모드 전환 ---
        forward_thresh = 8.0  # m 단위 조정
        # 인덱스 0이 12시, 배열 길이 360 기준 ±15° → 인덱스 -20~+20
        # ranges[-15:] + ranges[:16] 로 추출
        fwd_ranges = list(ranges[-20:]) + list(ranges[:20])
        has_obstacle = any(0.02 < r < forward_thresh for r in fwd_ranges)

        if has_obstacle:
            start_lane_change()
            drive(angle, 60)
        # --------------------------------------------------------------
        else:
            drive(angle, Fix_Speed)

    # 화면 표시
    # cv2.imshow('original', img)
    # if DRIVE_MODE == 0:
    #     cv2.imshow('cones', img)
    # else:
    #     cv2.imshow('lane', lane_img)

    # # 여기선 최소값으로 키 이벤트만 처리
    # cv2.waitKey(1)

#---------------------------------------------
# 콜백: 센서 데이터 업데이트
#---------------------------------------------
def img_cb(msg):
    global latest_img
    latest_img = msg

def scan_cb(msg):
    global latest_scan
    latest_scan = msg

#---------------------------------------------
# 노드 초기화 및 실행 (20 Hz 타이머)
#---------------------------------------------
def start():
    global motor
    rospy.init_node('Track_Driver_Lane_TL')

    motor = rospy.Publisher('xycar_motor', XycarMotor, queue_size=1)

    rospy.Subscriber('/usb_cam/image_raw/', Image, img_cb)
    rospy.Subscriber('/scan',           LaserScan, scan_cb)

    # 20 Hz 고정
    rospy.Timer(rospy.Duration(1.0/20.0), synced_cb)

    rospy.loginfo("Waiting for first messages…")
    rospy.wait_for_message('/usb_cam/image_raw/', Image)
    rospy.wait_for_message('/scan',           LaserScan)

    rospy.loginfo("Ready. Running at 10 Hz")
    rospy.spin()

if __name__ == '__main__':
    start()