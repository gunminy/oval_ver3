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

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
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
    motor_msg.angle = float(angle)
    motor_msg.speed = float(speed)
    motor.publish(motor_msg)
             
#=============================================
# 신호등 인식 함수 (초록불만 체크)
#=============================================
def detect_green_light(image):
    h, w, _ = image.shape
    roi = image[30:110, 180:450]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    cv2.imshow("traffic_light_roi", roi)

    center_pixel = hsv[hsv.shape[0]//2, hsv.shape[1]//2]
    print("ROI center HSV:", center_pixel)

    lower_green = np.array([50, 100, 100])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    green_count = cv2.countNonZero(green_mask)
    print("green:", green_count)

    threshold = 500

    if green_count > threshold:
        return True
    else:
        return False

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
    
    started = False  # 출발 여부

    #=========================================
    # 메인 루프 
    #=========================================
    while not rospy.is_shutdown():
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("original", image)
        cv2.imshow("gray", gray)

        if not started:
            # 신호등(초록불) 체크는 처음에만!
            if detect_green_light(image):
                print("green light! start driving.")
                started = True
                drive(angle=0.0, speed=50.0)
            else:
                print("not green light! stop driving.")
                drive(angle=0.0, speed=0.0)
            time.sleep(0.1)
            cv2.waitKey(1)
            continue  # 아래 코드 실행하지 않고 루프 반복

        # started == True 이후에는 신호등 체크 없이 계속 주행
        drive(angle=0.0, speed=50.0)
        time.sleep(0.1)
        cv2.waitKey(1)

#=============================================
# 메인함수를 호출합니다.
# start() 함수가 실질적인 메인함수입니다.
#=============================================
if __name__ == '__main__':
    start()
