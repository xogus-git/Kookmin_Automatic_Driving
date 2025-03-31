import numpy as np

# 사용할 상수 및 변수 선언
Kp = 0.00085
Ki = 0.000000001
Kd = 0.002

accuE = 0
bef_error = 99999


def P(Kp, error):
    return Kp * error

def I(Ki, error, accuE):
    return Ki * (accuE + error)

def D(Kd, error, bef_error):
    return Kd * (error - bef_error)


def get_u(current_x, pred_x):
    global accuE, bef_error
    
    error = current_x - pred_x
    
    if bef_error == 99999: # 시작 시에는 이전 error 값을 현재 error값으로 사용
        bef_error = error
    
    p = P(Kp, error)
    i = I(Ki, error, accuE)
    d = D(Kd, error, bef_error)

    u = p + i + d

    # 업데이트
    accuE += error
    bef_error = error

    return -u # 에러를 상쇄하는 방향으로