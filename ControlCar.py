#! /usr/bin/env python3



import cv2 # opencv
import numpy as np # 넘파이...
from constants import * # 상수 정의
#---사용자 라이브러리
from PID import *

def is_similar(a, b):
    #---a,b가 유사한지 확인하는 함수
    epsilon = 0.3 # 유사한 정도 허용 오차
    diff = abs(a - b)
    return diff < epsilon

def u_multiplier(speed, mul, min_, diff, threshold):
    if diff > threshold:
        #like corner
        #rospy.loginfo(f"!!!!!!!!!!!!!!!!!!!!!huge action!!!!!!!!!!!!!!!!!!")

        return 1.4
    else:
        x = speed * mul
        return max([min_, -abs(x) + 1])
#-----------------데이터 처리 및 실행--------------------------------------

def get_control_value(middle_point, max_speed):
    #---PID 제어값 계산
    u = get_u(WIDTH // 2, middle_point)
    speed = max_speed
    threshold = 100
    diff = abs(WIDTH // 2 - middle_point)
    u_w = u * u_multiplier(speed, 0.75, 0.6, diff, threshold)
    #---제어값 반환
    return u_w

