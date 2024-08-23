#!/usr/bin/env python
# -*- coding: utf-8 -*- 1
#=============================================
# 본 프로그램은 자이트론에서 제작한 것입니다.
# 상업라이센스에 의해 제공되므로 무단배포 및 상업적 이용을 금합니다.
# 교육과 실습 용도로만 사용가능하며 외부유출은 금지됩니다.
#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import numpy as np
import cv2, rospy, time, math, os
from ControlCar import *
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from xycar_msgs.msg import xycar_motor
from sensor_msgs.msg import LaserScan
from cv_bridge import CvBridge
from ar_track_alvar_msgs.msg import AlvarMarkers
from LaneDetect import *
import importlib.util

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
motor = None  # 모터 노드 변수
Fix_Speed = 5  # 모터 속도 고정 상수값 
new_angle = 0  # 모터 조향각 초기값
new_speed = Fix_Speed  # 모터 속도 초기값
bridge = CvBridge()  # OpenCV 함수를 사용하기 위한 브릿지 
lidar_points = None  # 라이다 데이터를 담을 변수
image = np.empty(shape=[0])  # 카메라 이미지를 담을 변수
motor_msg = xycar_motor()  # 모터 토픽 메시지
WIDTH, HEIGHT = 640, 480  # 카메라 이미지 가로x세로 크기
Blue =  (255,0,0) # 파란색
Green = (0,255,0) # 녹색
Red =   (0,0,255) # 빨간색
Yellow = (0,255,255) # 노란색
stopline_num = 1 # 정지선 발견때마다 1씩 증가
View_Center = WIDTH//2  # 화면의 중앙값 = 카메라 위치
ar_msg = {"ID":[],"DX":[],"DZ":[]}  # AR태그 토픽을 담을 변수

#=============================================
# 학습결과 파일의 위치 지정
#=============================================
PATH_TO_CKPT = '/home/pi/xycar_ws/src/study/track_drive/src/detect.tflite'
PATH_TO_LABELS = '/home/pi/xycar_ws/src/study/track_drive/src/labelmap.txt'

#=============================================
# 차선인식 프로그램에서 사용할 상수 선언부
#=============================================
CAM_FPS = 30  # 카메라 FPS 초당 30장의 사진을 보냄
WIDTH, HEIGHT = 640, 480  # 카메라 이미지 가로x세로 크기
ROI_START_ROW = 300  # 차선을 찾을 ROI 영역의 시작 Row값
ROI_END_ROW = 380  # 차선을 찾을 ROT 영역의 끝 Row값
ROI_HEIGHT = ROI_END_ROW - ROI_START_ROW  # ROI 영역의 세로 크기  
L_ROW = 40  # 차선의 위치를 찾기 위한 ROI 안에서의 기준 Row값 

#=============================================
# 프로그램에서 사용할 이동평균필터 클래스
#=============================================
class MovingAverage:

    # 클래스 생성과 초기화 함수 (데이터의 개수를 지정)
    def __init__(self, n):
        self.samples = n
        self.data = []
        self.weights = list(range(1, n + 1))

    # 새로운 샘플 데이터를 추가하는 함수
    def add_sample(self, new_sample):
        if len(self.data) < self.samples:
            self.data.append(new_sample)
        else:
            self.data.pop(0)  # 가장 오래된 샘플 제거
            self.data.append(new_sample)

    # 저장된 샘플 데이터의 갯수를 구하는 함수
    def get_sample_count(self):
        return len(self.data)

    # 이동평균값을 구하는 함수
    def get_mavg(self):
        if not self.data:
            return 0.0
        return float(sum(self.data)) / len(self.data)

    # 중앙값을 사용해서 이동평균값을 구하는 함수
    def get_mmed(self):
        if not self.data:
            return 0.0
        return float(np.median(self.data))

    # 가중치를 적용하여 이동평균값을 구하는 함수        
    def get_wmavg(self):
        if not self.data:
            return 0.0
        s = sum(x * w for x, w in zip(self.data, self.weights[:len(self.data)]))
        return float(s) / sum(self.weights[:len(self.data)])

    # 샘플 데이터 중에서 제일 작은 값을 반환하는 함수
    def get_min(self):
        if not self.data:
            return 0.0
        return float(min(self.data))
    
    # 샘플 데이터 중에서 제일 큰 값을 반환하는 함수
    def get_max(self):
        if not self.data:
            return 0.0
        return float(max(self.data))
        
#=============================================
#=============================================
# 초음파 거리정보에 대해서 이동평균필터를 적용하기 위한 선언
#=============================================
avg_count = 10  # 이동평균값을 계산할 데이터 갯수 지정
ultra_data = [MovingAverage(avg_count) for i in range(8)]

#=============================================
# 조향각에 대해서 이동평균필터를 적용하기 위한 선언
#=============================================
angle_avg_count = 10  # 이동평균값을 계산할 데이터 갯수 지정
angle_avg = MovingAverage(angle_avg_count)

#=============================================

#=============================================
# 콜백함수 - USB 카메라 토픽을 받아서 처리하는 콜백함수
#=============================================
def usbcam_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

#=============================================
# 콜백함수 - 라이다 토픽을 받아서 처리하는 콜백함수
#=============================================
def lidar_callback(data):
    global lidar_points
    lidar_points = data.ranges

#=============================================
# 콜백함수 - 초음파 토픽을 받아서 처리하는 콜백함수
#=============================================
def ultra_callback(data):
    global ultra_msg
    ultra_msg = data.data

    # 초음파센서로부터 받은 데이터를 필터링 처리함.
    #ultra_filtering()

#=============================================
# 초음파 거리정보에 이동평균값을 적용하는 필터링 함수
#=============================================
def ultra_filtering():
    global ultra_msg

    # 이동평균필터를 적용해서 튀는 값을 제거하는 필터링 작업 수행
    for i in range(8):
        ultra_data[i].add_sample(float(ultra_msg[i]))
        
    # 여기서는 중앙값(Median)을 이용 - 평균값 또는 가중평균값을 이용하는 것도 가능 
    ultra_list = [int(ultra_data[i].get_mmed()) for i in range(8)]
    
    # 평균값(Average)을 이용 
    #ultra_list = [int(ultra_data[i].get_mavg()) for i in range(8)]
    
    # 가중평균값(Weighted Average)을 이용 
    #ultra_list = [int(ultra_data[i].get_wmavg()) for i in range(8)]
        
    # 최소값(Min Value)을 이용 
    #ultra_list = [int(ultra_data[i].get_min()) for i in range(8)]
    
    # 최대값(Max Value)을 이용 
    #ultra_list = [int(ultra_data[i].get_max()) for i in range(8)]
    
    ultra_msg = tuple(ultra_list)

#=============================================
# 콜백함수 - AR태그 토픽을 받아서 처리하는 콜백함수
#=============================================
def ar_callback(data):
    global ar_msg

    # AR태그의 ID값, X 위치값, Z 위치값을 담을 빈 리스트 준비
    ar_msg["ID"] = []
    ar_msg["DX"] = []
    ar_msg["DZ"] = []

    # 발견된 모두 AR태그에 대해서 정보 수집하여 ar_msg 리스트에 담음
    for i in data.markers:
        ar_msg["ID"].append(i.id) # AR태그의 ID값을 리스트에 추가
        ar_msg["DX"].append(int(i.pose.pose.position.x*100)) # X값을 cm로 바꿔서 리스트에 추가
        ar_msg["DZ"].append(int(i.pose.pose.position.z*100)) # Z값을 cm로 바꿔서 리스트에 추가
    
#=============================================
# 모터 토픽을 발행하는 함수 
#=============================================
def drive(angle, speed):
    motor_msg.angle = angle
    motor_msg.speed = speed
    motor.publish(motor_msg)
    
#=============================================
# 차량을 정차시키는 함수  
# 입력으로 지속시간을 받아 그 시간동안 속도=0 토픽을 모터로 보냄.
# 지속시간은 0.1초 단위임. 만약 15이면 1.5초가 됨.
#=============================================
def stop_car(duration):
    for i in range(int(duration)): 
        drive(angle=0, speed=0)
        time.sleep(0.1)
    
#=============================================
# 차량을 이동시키는 함수 
# 입력으로 조향각과 속도, 지속시간을 받아 차량을 이동시킴.
# 지속시간은 0.1초 단위임. 만약 15이면 1.5초가 됨. 
#=============================================
def move_car(move_angle, move_speed, duration):
    for i in range(int(duration)): 
        drive(move_angle, move_speed)
        time.sleep(0.1)
		
#=============================================
# 카메라의 Exposure 값을 변경하는 함수 
# 입력으로 0~255 값을 받는다.
#=============================================
def cam_exposure(value):
    command = 'v4l2-ctl -d /dev/videoCAM -c exposure_time_absolute=' + str(value)
    os.system(command)
    
#=============================================
# 특정 ROS 노드를 중단시키고 삭제하는 함수 
# 더 이상 사용할 필요가 없는 ROS 노드를 삭제할 때 사용한다.
#=============================================
def kill_node(node_name):
    try:
        # rosnode kill 명령어를 사용하여 노드를 종료
        result = os.system(f"rosnode kill {node_name}")
        if result == 0:
            rospy.loginfo(f"Node {node_name} has been killed successfully.")
        else:
            rospy.logwarn(f"Failed to kill node {node_name}. It may not exist.")
    except Exception as e:
        rospy.logerr(f"Failed to kill node {node_name}: {e}")

#=============================================
# AR 패지키지가 발행하는 토픽을 받아서 
# 제일 가까이 있는 AR Tag에 적힌 ID 값을 반환하는 함수
# 거리값과 좌우치우침값을 함께 반환
#=============================================
def check_AR():

    ar_data = ar_msg
    id_value = 99

    if (len(ar_msg["ID"]) == 0):
        # 발견된 AR태그가 없으면 
        # ID값 99, Z위치값 500cm, X위치값 500cm로 리턴
        return 99, 500, 500  

    # 새로 도착한 AR태그에 대해서 아래 작업 수행
    z_pos = 500  # Z위치값을 500cm로 초기화
    x_pos = 500  # X위치값을 500cm로 초기화
    
    for i in range(len(ar_msg["ID"])):
        # 발견된 AR태그 모두에 대해서 조사

        if(ar_msg["DZ"][i] < z_pos):
            # 더 가까운 거리에 AR태그가 있으면 그걸 사용
            id_value = ar_msg["ID"][i]
            z_pos = ar_msg["DZ"][i]
            x_pos = ar_msg["DX"][i]

    # ID번호, 거리값(미터), 좌우치우침값(미터) 리턴
    return id_value, round(z_pos,2), round(x_pos,2)
    
#=============================================
# 신호등의 파란불을 체크해서 True/False 값을 반환하는 함수
#=============================================
def check_traffic_sign():
    MIN_RADIUS, MAX_RADIUS = 15, 25
    
    # 원본이미지를 복제한 후에 특정영역(ROI Area)을 잘라내기
    cimg = image.copy()
    Center_X, Center_Y = 320, 100  # ROI 영역의 중심위치 좌표 
    XX, YY = 220, 80  # 위 중심 좌표에서 좌우로 XX, 상하로 YY만큼씩 벌려서 ROI 영역을 잘라냄   

    # ROI 영역에 녹색 사각형으로 테두리를 쳐서 표시함 
    cv2.rectangle(cimg, (Center_X-XX, Center_Y-YY), (Center_X+XX, Center_Y+YY) , Green, 2)
	
    # 원본 이미지에서 ROI 영역만큼 잘라서 roi_img에 담음 
    roi_img = cimg[Center_Y-YY:Center_Y+YY, Center_X-XX:Center_X+XX]

    # roi_img 칼라 이미지를 회색 이미지로 바꾸고 노이즈 제거를 위해 블러링 처리를 함  
    img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Hough Circle 함수를 이용해서 이미지에서 원을 (여러개) 찾음 
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20,
                  param1=40, param2=20, 
                  minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)

    # 디버깅을 위해서 Canny 처리를 했을때의 모습을 화면에 표시함
    # 위 HoughCircles에서 param1, param2에 사용했던 값을 아래 canny에서 똑같이 적용해야 함. 순서 조심.
    canny = cv2.Canny(blur, 20, 40)
    cv2.imshow('Canny image used by HoughCircles', canny)
    cv2.waitKey(1)

    if circles is not None:
    
        # 정수값으로 바꾸고 발견된 원의 개수를 출력
        circles = np.round(circles[0, :]).astype("int")
        print("\nFound",len(circles),"circles")
        
        # 중심의 Y좌표값 순서대로 소팅해서 따로 저장
        y_circles = sorted(circles, key=lambda circle: circle[1])
 
        # 중심의 X좌표값 순서대로 소팅해서 circles에 다시 저장
        circles = sorted(circles, key=lambda circle: circle[0])
         
        # 발견된 원들에 대해서 루프를 돌면서 하나씩 녹색으로 그리기 
        for i, (x, y, r) in enumerate(circles):
            cv2.circle(cimg, (x+Center_X-XX, y+Center_Y-YY), r, Green, 2)
 
    # 이미지에서 정확하게 3개의 원이 발견됐다면 신호등 찾는 작업을 진행  
    if (circles is not None) and (len(circles)==3):
            
        # 가장 밝은 원을 찾을 때 사용할 변수 선언
        max_mean_value = 0
        max_mean_value_circle = None
        max_mean_value_index = None

        # 발견된 원들에 대해서 루프를 돌면서 하나씩 처리 
 	    # 원의 중심좌표, 반지름. 내부밝기 정보를 구해서 화면에 출력
        for i, (x, y, r) in enumerate(circles):
            roi = img[y-(r//2):y+(r//2),x-(r//2):x+(r//2)]
            # 밝기 값은 반올림해서 10의 자리수로 만들어 사용
            mean_value = round(np.mean(roi),-1)
            print(f"Circle {i} at ({x},{y}), radius={r}: brightness={mean_value}")
			
            # 이번 원의 밝기가 기존 max원보다 밝으면 이번 원을 max원으로 지정  
            if mean_value > max_mean_value:
                max_mean_value = mean_value
                max_mean_value_circle = (x, y, r)
                max_mean_value_index = i
                
            # 원의 밝기를 계산했던 사각형 영역을 빨간색으로 그리기 
            cv2.rectangle(cimg, ((x-(r//2))+Center_X-XX, (y-(r//2))+Center_Y-YY),
                ((x+(r//2))+Center_X-XX, (y+(r//2))+Center_Y-YY), Red, 2)

        # 가장 밝은 원을 찾았으면 그 원의 정보를 출력하고 노란색으로 그리기 
        if max_mean_value_circle is not None:
            (x, y, r) = max_mean_value_circle
            print(f" --- Circle {max_mean_value_index} is the brightest.")
            cv2.circle(cimg, (x+Center_X-XX, y+Center_Y-YY), r, Yellow, 2)
            
        # 신호등 찾기 결과가 표시된 이미지를 화면에 출력
        cv2.imshow('Circles Detected', cimg)
        
        # 제일 위와 제일 아래에 있는 2개 원의 Y좌표값 차이가 크면 안됨 
        vertical_diff = MAX_RADIUS * 2
        if (y_circles[-1][1] - y_circles[0][1]) > vertical_diff:
            print("Circles are scattered vertically!")
            return False
        
        # 제일 왼쪽과 제일 오른쪽에 있는 2개 원의 X좌표값 차이가 크면 안됨 
        horizontal_diff = MAX_RADIUS * 8
        if (circles[-1][0] - circles[0][0]) > horizontal_diff:
            print("Circles are scattered horizontally!")
            return False      
            
        # 원들이 좌우로 너무 붙어 있으면 안됨 
        min_distance = MIN_RADIUS * 3
        for i in range(len(circles) - 1):
            if (circles[i+1][0] - circles[i][0]) < min_distance:
                print("Circles are too close horizontally!")
                return False 
            
        # 3개 중에서 세번째 원이 가장 밝으면 (파란색 신호등) True 리턴 
        if (max_mean_value_index == 2):
            print("Traffic Sign is Blue...!")
            return True
        
        # 첫번째나 두번째 원이 가장 밝으면 (파란색 신호등이 아니면) False 반환 
        else:
            print("Traffic Sign is NOT Blue...!")
            return False

    # 신호등 찾기 결과가 표시된 이미지를 화면에 출력
    cv2.imshow('Circles Detected', cimg)
    
    # 원본 이미지에서 원이 발견되지 않았다면 False 리턴   
    #print("Can't find Traffic Sign...!")
    return False
       
#=============================================
# 라이다 센서를 이용해서 벽까지의 거리를 알아내서
# 벽과 충돌하지 않으며 주행하도록 모터로 토픽을 보내는 함수
#=============================================
def sensor_drive():
    global new_angle, new_speed

    # 왼쪽 벽이 오른쪽 벽보다 멀리 있으면, 왼쪽으로 주행
    if (lidar_points[45]*100 - 10 > lidar_points[460]*100):
        new_angle = -50

    # 왼쪽 벽보다 오른쪽 벽이 멀리 있으면, 오른쪽으로 주행
    elif (lidar_points[45]*100 < lidar_points[460]*100 - 10):
        new_angle = 50

    # 위 조건들에 해당하지 않는 경우라면 직진 주행
    else:
        new_angle = 0

    # 모터에 주행명령 토픽을 보낸다
    print(lidar_points[45]*100, lidar_points[460]*100)
    
    new_speed = Fix_Speed
    drive(new_angle, new_speed)       

#=============================================
# 정지선이 있는지 체크해서 True/False 값을 반환하는 함수
#=============================================
def check_stopline():
    global stopline_num

    # 원본 영상을 화면에 표시
    #cv2.imshow("Original Image", image)
    
    # image(원본이미지)의 특정영역(ROI Area)을 잘라내기
    roi_img = image[300:480, 0:640]
    cv2.imshow("ROI Image", roi_img)

    # HSV 포맷으로 변환하고 V채널에 대해 범위를 정해서 흑백이진화 이미지로 변환
    hsv_image = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV) 
    upper_white = np.array([255, 255, 255])
    lower_white = np.array([0, 0, 180])
    binary_img = cv2.inRange(hsv_image, lower_white, upper_white)
    #cv2.imshow("Black&White Binary Image", binary_img)

    # 흑백이진화 이미지에서 특정영역을 잘라내서 정지선 체크용 이미지로 만들기
    stopline_check_img = binary_img[100:120, 200:440] 
    #cv2.imshow("Stopline Check Image", stopline_check_img)
    
    # 흑백이진화 이미지를 칼라이미지로 바꾸고 정지선 체크용 이미지 영역을 녹색사각형으로 표시
    img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (200,100),(440,120),Green,3)
    cv2.imshow('Stopline Check', img)
    cv2.waitKey(1)
    
    # 정지선 체크용 이미지에서 흰색 점의 개수 카운트하기
    stopline_count = cv2.countNonZero(stopline_check_img)
    
    # 사각형 안의 흰색 점이 기준치 이상이면 정지선을 발견한 것으로 한다
    if stopline_count > 2500:
        print("Stopline Found...! -", stopline_num)
        stopline_num = stopline_num + 1
        #cv2.destroyWindow("ROI Image")
        return True
    
    else:
        return False

#=========================================
#  사다리꼴 모양으로 이미지 자르기
#=========================================
def apply_trapezoid_mask(img):
    mask = np.zeros_like(img)
    height, width = img.shape[:2]

    # 사다리꼴 모양의 좌표 정의
    trapezoid = np.array([

        [width * 2//6, height * 1//6],
        [width * 4//6, height * 1//6],
        [width,        height * 5//6],
        [0,            height * 5//6]

    ], np.int32)

    # 사다리꼴 모양으로 마스크 이미지 채우기
    cv2.fillPoly(mask, [trapezoid], 255)

    cv2.imshow("Mask", mask)
    
    # 마스크를 사용하여 이미지 자르기
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

#=============================================
# 카메라 이미지에서 차선을 찾아 그 위치를 반환하는 함수
#=============================================
def lane_detect():
    lanedetector = LaneDetector()
    return lanedetector.forward(image)

def lane_drive():
    
# 카메라 이미지에서 차선의 위치를 알아냅니다. 
    found, x_midpoint = lane_detect()	
    if found:
        # 차선인식이 됐으면 차선의 위치정보를 이용해서 핸들 조향각을 결정합니다. 
        new_angle = get_control_value(x_midpoint, Fix_Speed + 5)
                             
        if abs(new_angle) < 10:
            new_speed =  Fix_Speed + 5
        else:  
            new_speed = Fix_Speed

        drive(new_angle, new_speed)  
            
    else:
        # 차선인식이 안됐으면 기존 핸들값을 사용하여 주행합니다. 	
        drive(new_angle, new_speed)


#=============================================
# 실질적인 메인 함수 
#=============================================
def start():

    global motor, ultra_msg, image 
    global new_angle, new_speed
   
    STARTING_LINE = 1
    TRAFFIC_SIGN = 2
    SENSOR_DRIVE = 3
    LANE_DRIVE = 4
    AR_DRIVE = 5
    PARKING = 7
    FINISH = 9
		
    # 처음에 어떤 미션부터 수행할 것인지 여기서 결정합니다. 
    drive_mode = STARTING_LINE
    cam_exposure(0)  # 카메라의 Exposure 값을 변경
    
    #=========================================
    # 노드를 생성하고, 구독/발행할 토픽들을 선언합니다.
    #=========================================
    rospy.init_node('Track_Driver')
    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    rospy.Subscriber("xycar_ultrasonic", Int32MultiArray, ultra_callback, queue_size=1)
    rospy.Subscriber("/usb_cam/image_raw/",Image,usbcam_callback, queue_size=1)
    rospy.Subscriber('ar_pose_marker', AlvarMarkers, ar_callback, queue_size=1 )
    rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1)

    #=========================================
    # 발행자 노드들로부터 첫번째 토픽들이 도착할 때까지 기다립니다.
    #=========================================
    rospy.wait_for_message("/usb_cam/image_raw/", Image)
    print("Camera Ready --------------")
    rospy.wait_for_message("xycar_ultrasonic", Int32MultiArray)
    print("UltraSonic Ready ----------")
    rospy.wait_for_message("/scan", LaserScan)
    print("Lidar Ready ----------")
    rospy.wait_for_message("ar_pose_marker", AlvarMarkers)
    print("AR detector Ready ----------")

    print("======================================")
    print(" S T A R T    D R I V I N G ...")
    print("======================================")
	
    # 일단 차량이 움직이지 않도록 정지상태로 만듭니다.  
    stop_car(10) # 1초 동안 정차
    cam_exposure(20)
    while not rospy.is_shutdown():
        lane_drive()
	
    #=========================================
    # 메인 루프 
    #=========================================
    """
    while not rospy.is_shutdown():

        # ======================================
        # 출발선으로 차량을 이동시킵니다. 
        # AR태그 인식을 통해 AR태그 바로 앞에 가서 멈춥니다.
        # 신호등인식 TRAFFIC_SIGN 모드로 넘어갑니다.  
        # ======================================
        while drive_mode == STARTING_LINE:
		
            # 전방에 AR태그가 보이는지 체크합니다.             
            ar_ID, z_pos, x_pos = check_AR()
            cv2.imshow('AR Detecting', image)
            cv2.waitKey(1)
            
            if (ar_ID == 99):
            # AR태그가 안 보이면 AR태그를 계속 찾습니다.  
                print("Finding AR")
                continue  # while 블럭 처음으로 되돌아 갑니다.
                
            print("Distance :", z_pos)
            new_angle = 0
            new_speed = Fix_Speed
            drive(new_angle, new_speed)
            
            if (z_pos < 60):  
                # AR태그가 가까워지면 다음 미션으로 넘어갑니다.
                drive_mode = TRAFFIC_SIGN     
                cam_exposure(20)  # 카메라의 Exposure 값을 변경
                stop_car(10)
                print("----- Traffic Sign Detecting... -----")
                # 열려 있는 모든 OpenCV 윈도우 창을 닫습니다. 
                #cv2.destroyAllWindows()
                
        # ======================================
        # 출발선에서 신호등을 찾습니다. 
        # 일단 정차해 있다가 파란색 불이 켜지면 출발합니다.
        # AR_DRIVE 모드로 넘어갑니다.  
        # ======================================
        while drive_mode == TRAFFIC_SIGN:
		
            # 앞에 있는 신호등에 파란색 불이 켜졌는지 체크합니다.  
            result = check_traffic_sign()
			
            if (result == True):
                # 신호등이 파란불이면 AR_DRIVE 모드로 넘어갑니다.
                drive_mode = AR_DRIVE
                cam_exposure(100)  # 카메라의 Exposure 값을 변경
                print ("----- AR Following Start... -----")
                # 열려 있는 모든 OpenCV 윈도우 창을 닫습니다. 
                #cv2.destroyAllWindows()
                
        # ======================================
        # AR태그를 찾고 AR 주행을 시작합니다.
        # ======================================
        retry_count = 0
        while drive_mode == AR_DRIVE:
        
            # 전방에 AR태그가 보이는지 체크합니다.   
            ar_ID, z_pos, x_pos = check_AR()
            cv2.imshow('AR following', image)
            cv2.waitKey(1)
            
            if (ar_ID != 99):  # AR태그가 발견되면
                # AR태그가 있는 곳으로 주행합니다. 
                print("ID=", ar_ID," Z_pos=",z_pos," X_pos=",x_pos)
                retry_count = 0

                distance = math.sqrt(z_pos**2 + x_pos**2)
                if( distance > 100):
                    x_pos = x_pos + 0
                    new_angle = x_pos * 1
                elif( distance > 70):
                    x_pos = x_pos + 10
                    new_angle = x_pos * 2
                elif( distance > 30):
                    x_pos = x_pos + 20
                    new_angle = x_pos * 2.5
                else:
                    x_pos = x_pos + 30
                    new_angle = x_pos * 3

                new_speed = Fix_Speed
                print ("Following AR...", new_angle)

            else:
                # AR태그가 안 보이면 지정된 회수만큼 원래 방향으로 주행하기를 반복합니다.   
                retry_count = retry_count + 1
                if (retry_count < 5):
                    print("Keep going...", new_angle)
                
                else:                    
                    # 지정된 회수가 지나면 더 이상 AR태그가 없는 것으로 판정하고 
                    # AR주행을 끝내고 다음 미션으로 넘어갑니다.                     
                    
                    # 더 이상 사용하지 않는 AR인식노드를 Kill 한다.
                    # AR인식노드의 이름은 launch 파일에서 찾을 수 있다.
                    #kill_node("ar_track_alvar")
                    
                    drive_mode = SENSOR_DRIVE  
                    cam_exposure(100)  # 카메라의 Exposure 값을 변경
                    stop_car(10)
                    print ("----- Sensor driving Start... -----")
                    # 열려 있는 모든 OpenCV 윈도우 창을 닫습니다. 
                    #cv2.destroyAllWindows()
                    
            drive(new_angle, new_speed)
       
        # ======================================
        # 센서로 미로주행을 진행합니다.
        # AR이 보이면 차선인식주행 LANE_DRIVE 모드로 넘어갑니다. 
        # ======================================

        while drive_mode == SENSOR_DRIVE:

            # 라이다센서를 이용해서 미로주행을 합니다. 
            sensor_drive() 

            # AR태그를 발견하면 차선주행 모드로 변경합니다. 
            result = False
            ar_ID, z_pos, x_pos = check_AR()            
            if (ar_ID != 99) and (z_pos < 90):
                result = True 

            if (result == True):
                # AR이 발견되면 차량을 정차시키고 LANE_DRIVE 모드로 넘어갑니다.
                stop_car(20) # 2초 정차
                drive_mode = LANE_DRIVE  
                cam_exposure(100)  # 카메라의 Exposure 값을 변경
                print ("----- Lane driving Start... -----")
                # 열려 있는 모든 OpenCV 윈도우 창을 닫습니다. 
                #cv2.destroyAllWindows()
         
                # 더 이상 사용하지 않는 AR인식노드를 Kill 한다.
                #kill_node("ar_track_alvar")
         
        # ======================================
        # 차선을 보고 주행합니다. 
        # ======================================
    
        while drive_mode == LANE_DRIVE:
		
            # 카메라 이미지에서 차선의 위치를 알아냅니다. 
            found, x_left, x_right = lane_detect()
			
            if found:
                # 차선인식이 됐으면 차선의 위치정보를 이용해서 핸들 조향각을 결정합니다. 
                x_midpoint = (x_left + x_right) // 2 
                new_angle = (x_midpoint - View_Center) / 2
                
                
                '''                
                #=========================================
                # new_angle에 이동평균값을 적용
                #=========================================
                angle_avg.add_sample(new_angle)
                new_angle = angle_avg.get_mmed()  
                
                #=========================================
                # PID 제어 적용
                #=========================================
                new_angle = pid.pid_control(new_angle)                
                
                '''
                
                print("New angle is", new_angle)

                if abs(new_angle) < 10:
                    new_speed =  Fix_Speed + 5
                else:  
                    new_speed = Fix_Speed

                drive(new_angle, new_speed)  
				
            else:
                # 차선인식이 안됐으면 기존 핸들값을 사용하여 주행합니다. 	
                drive(new_angle, new_speed)
    
                  
        # ======================================
        # 주차합니다. 
        # AR 표지판 바로 앞에 가면 주행종료 모드로 변경합니다.
        # ======================================
        while drive_mode == PARKING:
       
            # 전방에 AR태그가 보이는지 체크합니다.   
            ar_ID, z_pos, x_pos = check_AR()
            #print(" ID=",ar_ID," Z_pos=",z_pos," X_pos=",x_pos)
            cv2.imshow('Parking with AR', image)
            cv2.waitKey(1)

            if (ar_ID != 99):
                # AR태그가 있는 곳으로 주행합니다. 
                if (z_pos > 30):
                    # Z값이 30센치 이상이면 핸들 각도를 조종하면서 주행합니다.  
                    new_angle = x_pos * 3
                    new_speed = Fix_Speed       
         
                else:
                    # Z값이 30센치 이하이면 차량을 세우고 FINISH 모드로 넘어갑니다.     
                    new_angle = 0
                    new_speed = 0        
                    cv2.destroyAllWindows()              
                    drive_mode = FINISH  
                    print ("----- Parking completed... -----")

            else:
                new_speed = 0  # 차량을 정지시키기
                
            drive(new_angle,new_speed)  

        # ======================================
        # 주행을 끝냅니다. 
        # ======================================
        if drive_mode == FINISH:
           
            # 차량을 정지시키고 모든 작업을 끝냅니다.
            stop_car(10) # 1초간 정차 
            print ("----- Bye~! -----")
            return            
    """

    stop_car(10) # 정차 

#=============================================
# 메인함수를 호출합니다.
# start() 함수가 실질적인 메인함수입니다.
#=============================================
if __name__ == '__main__':
    start()
