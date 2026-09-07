# 25 04 25
# Without Arduino Board
# With PCA9685 for servo motor control (New library applied)

import time
import math
import odrive
import threading
import busio
import os
import numpy as np
from adafruit_pca9685 import PCA9685
import socket
import select 
import struct
import cv2
import ast
import board
from odrive.enums import *
from pymavlink import mavutil
import sys  # 파일 상단에 추가
import queue

# Jetson 환경변수 설정 (I2C Bus 7)
os.environ["BLINKA_I2C_BUS_DEFAULT"] = "7"

last_cmd_time = 0.0          
last_log_time = 0.0          

# 전역 변수 선언
max_speed = 10  # Motor Max Speed [round/sec]
# delta_max_angle = 15  # (deg)
# delta_max_change = 5  # (deg/loop)
current_left_angle = 0
current_right_angle = 0

VIDEO_PORT = 8000
CMD_PORT = 8001

# 차체 파라미터 (설정 필요)
L = 425.617  # Wheelbase [mm] (축간 거리)
T = 187.752 * 2  # Track width [mm] (트레드, 좌우 바퀴 거리)
W = T  # W는 T와 동일 (Differential 함수에서 사용)
R = 1500  # 회전 반경 [mm]

# Differential steering 관련 파라미터
delta_max_angle = 20  # 최대 조향각 (deg)
DIFF_GAIN = 1.4   # 조향시 바퀴 속도차 증폭 계수

# angle 값 직접 계산 (R만 조정하면 됨)
delta_out = math.degrees(math.atan(L / (R + T/2)))
delta_in = math.degrees(math.atan(L / (R - T/2)))

# Servo Motor 초기 설정
PWM_FREQ = 50  # 50Hz
SERVO_LEFT_CHANNEL = 14  # Left Wheel
SERVO_RIGHT_CHANNEL = 0  # Right Wheel

# =========================
# PCA9685 초기화 (새 방식)
# =========================
i2c = busio.I2C(board.SCL, board.SDA)
pca9685 = PCA9685(i2c, address=0x40)
pca9685.frequency = PWM_FREQ

# Servo Motor 각도 → duty_cycle 변환 함수
def angle_to_duty_cycle(angle):
    pulse_us = 500 + (angle / 180.0) * 2000  # µs
    duty_cycle = int((pulse_us / 20000.0) * 0xFFFF)  # 20ms 주기 기준
    return duty_cycle

# rc_car_base2.py의 서보모터 각도 계산 함수 추가
def compute_servo_angles(steering_angle):
    """
    이론적 조향각을 실제 서보모터 각도로 변환
    steering_angle: 조향각 (degrees, + = 우회전, - = 좌회전)
    return: (servo_left_angle, servo_right_angle)
    """
    DEG2RAD = math.pi / 180
    RAD2DEG = 180 / math.pi
    c, b, a, d = 23.72, 89.232, 21.071, 105.111
    k1, k2, k3 = d/a, d/c, (a**2 - b**2 + c**2 + d**2) / (2 * a * c)
    T, L = 187.752 * 2, 425.617

    Z = abs(steering_angle)
    if Z == 0:
        # 직진 시 중앙값
        return 90, 90
        
    r = L / math.tan(DEG2RAD * Z)
    Zo = RAD2DEG * math.atan(L / (r + T / 2))
    Zi = RAD2DEG * math.atan(L / (r - T / 2))

    def th4(theta2):
        A = math.cos(DEG2RAD * theta2) - k1 - k2 * math.cos(DEG2RAD * theta2) + k3
        B = -2 * math.sin(DEG2RAD * theta2)
        C = k1 - (k2 + 1) * math.cos(DEG2RAD * theta2) + k3
        return RAD2DEG * (2 * math.atan((-B + math.sqrt(B**2 - 4*A*C)) / (2*A)))

    th4_Zo = th4(th2_Zo := +Zo + 57.304)
    th4_Zi = th4(th2_Zi := -Zi + 57.304)

    if steering_angle < 0:  # 좌회전
        th4_servo_Left = th4_Zi
        th4_servo_Right = th4_Zo
    else:  # 우회전
        th4_servo_Left = th4_Zo
        th4_servo_Right = th4_Zi

    servo_left_angle = -32.786 - th4_servo_Left
    servo_right_angle = th4_servo_Right + 212.786
    
    print(f"[SERVO] 조향각: {steering_angle:.2f}도 -> 서보각도 L: {servo_left_angle:.2f}도, R: {servo_right_angle:.2f}도")
    
    return servo_left_angle, servo_right_angle

# ===== Differential Steering 함수 =====
def Differential(delta, speed):
    """
    조향각과 속도에 따라 각 바퀴의 속도 비율을 계산
    delta: 조향각 (degrees)
    speed: 기준 속도
    return: (Vdiff_fl, Vdiff_fr, Vdiff_rl, Vdiff_rr) - 각 바퀴의 속도 비율
    """
    if delta != 0 and speed != 0:
        R_road = -L / math.tan(math.radians(delta))
        delta_l = math.atan(L / (R_road - W / 2))
        delta_r = math.atan(L / (R_road + W / 2))
        R_road_fl = L / math.sin(delta_l)
        R_road_fr = L / math.sin(delta_r)
        R_road_rl = R_road - W / 2
        R_road_rr = R_road + W / 2
        Vdiffs = np.array([R_road_fl, R_road_fr, R_road_rl, R_road_rr]) / R_road

        # --- 여기서 속도차 증폭 ---
        k = 1 + (abs(delta)/delta_max_angle) * (DIFF_GAIN - 1)
        mean_V = np.mean(Vdiffs)
        Vdiffs = (Vdiffs - mean_V) * k + mean_V
        Vdiff_fl, Vdiff_fr, Vdiff_rl, Vdiff_rr = Vdiffs
    else:
        Vdiff_fl = Vdiff_fr = Vdiff_rl = Vdiff_rr = 1
    return Vdiff_fl, Vdiff_fr, Vdiff_rl, Vdiff_rr

# ODrive 초기 설정
def calibrate_odrive_axis(odrv, axis, axis_name):
    print(f"Starting full calibration for ODrive {axis_name}...")
    axis.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
    while axis.current_state != AXIS_STATE_IDLE:
        time.sleep(0.1)
    axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
    axis.controller.config.vel_ramp_rate = 25
    axis.controller.config.input_mode = INPUT_MODE_VEL_RAMP
    axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
    print(f"Calibration ready for {axis_name}.")

def control_motor_speed(axis, speed, axis_name):
    print(f"[MOTOR] {axis_name} 속도 설정: {speed} (현재 상태: {axis.current_state})")
    axis.controller.input_vel = speed

def set_motors_idle():
    print("Setting all motors to stop...")
    odrv0.axis0.controller.input_vel = 0
    odrv0.axis1.controller.input_vel = 0
    odrv1.axis0.controller.input_vel = 0
    odrv1.axis1.controller.input_vel = 0
    # IDLE 상태로 변경하지 않고 CLOSED_LOOP_CONTROL 상태 유지
    print("Motors are now stopped.")

def execute_parallel(*functions):
    threads = []
    for func in functions:
        thread = threading.Thread(target=func)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

def send_frame(frame):
    ret, enc = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ret:
        return
    data = enc.tobytes()
    pkt = struct.pack('<L', len(data)) + data
    video_client.sendall(pkt)

def execute_forward(value):
    print(f"[MOTOR] execute_forward 호출됨 - value: {value}")
    
    # 직진 시 서보모터 각도 계산 (조향각 0도)
    servo_left_angle, servo_right_angle = compute_servo_angles(0)
    
    # Differential steering 적용 (직진)
    base_speed = abs(value) * max_speed/2
    print(f"[MOTOR] 계산된 base_speed: {base_speed}")
    delta_angle = 0  # 직진 시 조향각 0
    Vdiff_fl, Vdiff_fr, Vdiff_rl, Vdiff_rr = Differential(delta_angle, base_speed)
    
    # 각 바퀴에 다른 속도 적용
    speed_fl = base_speed * Vdiff_fl  # Front Left
    speed_fr = base_speed * Vdiff_fr  # Front Right  
    speed_rl = base_speed * Vdiff_rl  # Rear Left
    speed_rr = base_speed * Vdiff_rr  # Rear Right
    
    execute_parallel(
        lambda: control_motor_speed(odrv0.axis0, speed_fr, "0 Axis0"),  # Front Right
        lambda: control_motor_speed(odrv0.axis1, speed_fl, "0 Axis1"),  # Front Left
        lambda: control_motor_speed(odrv1.axis0, speed_rr, "1 Axis0"),  # Rear Right
        lambda: control_motor_speed(odrv1.axis1, speed_rl, "1 Axis1")   # Rear Left
    )
    pca9685.channels[SERVO_LEFT_CHANNEL].duty_cycle = angle_to_duty_cycle(servo_left_angle)
    pca9685.channels[SERVO_RIGHT_CHANNEL].duty_cycle = angle_to_duty_cycle(servo_right_angle)
    global current_left_angle, current_right_angle
    current_left_angle = 0
    current_right_angle = 0

def execute_backward(value):
    # 직진 시 서보모터 각도 계산 (조향각 0도)
    servo_left_angle, servo_right_angle = compute_servo_angles(0)
    
    # Differential steering 적용 (후진 직진)
    base_speed = abs(value) * max_speed/2
    delta_angle = 0  # 직진 시 조향각 0
    Vdiff_fl, Vdiff_fr, Vdiff_rl, Vdiff_rr = Differential(delta_angle, base_speed)
    
    # 각 바퀴에 다른 속도 적용 (후진이므로 음수)
    speed_fl = -base_speed * Vdiff_fl  # Front Left
    speed_fr = -base_speed * Vdiff_fr  # Front Right  
    speed_rl = -base_speed * Vdiff_rl  # Rear Left
    speed_rr = -base_speed * Vdiff_rr  # Rear Right
    
    execute_parallel(
        lambda: control_motor_speed(odrv0.axis0, speed_fr, "0 Axis0"),  # Front Right
        lambda: control_motor_speed(odrv0.axis1, speed_fl, "0 Axis1"),  # Front Left
        lambda: control_motor_speed(odrv1.axis0, speed_rr, "1 Axis0"),  # Rear Right
        lambda: control_motor_speed(odrv1.axis1, speed_rl, "1 Axis1")   # Rear Left
    )
    pca9685.channels[SERVO_LEFT_CHANNEL].duty_cycle = angle_to_duty_cycle(servo_left_angle)
    pca9685.channels[SERVO_RIGHT_CHANNEL].duty_cycle = angle_to_duty_cycle(servo_right_angle)
    global current_left_angle, current_right_angle
    current_left_angle = 0
    current_right_angle = 0

def execute_right(value):
    print(f"[SERVO] 우회전 실행 - value: {value}")
    
    # 우회전 시 서보모터 각도 계산 (양수 조향각)
    steering_angle = delta_out * 1.5  # 이론적 조향각 사용
    servo_left_angle, servo_right_angle = compute_servo_angles(steering_angle)
    
    pca9685.channels[SERVO_LEFT_CHANNEL].duty_cycle = angle_to_duty_cycle(servo_left_angle)
    pca9685.channels[SERVO_RIGHT_CHANNEL].duty_cycle = angle_to_duty_cycle(servo_right_angle)
    print(f"[SERVO] 우회전 서보모터 설정 완료")
    
    # Differential steering 적용
    base_speed = abs(value) * max_speed/2
    delta_angle = delta_out*1.5  # 우회전 시 조향각
    Vdiff_fl, Vdiff_fr, Vdiff_rl, Vdiff_rr = Differential(delta_angle, base_speed)
    
    # 각 바퀴에 다른 속도 적용
    speed_fl = base_speed * Vdiff_fl  # Front Left
    speed_fr = base_speed * Vdiff_fr  # Front Right  
    speed_rl = base_speed * Vdiff_rl  # Rear Left
    speed_rr = base_speed * Vdiff_rr  # Rear Right
    
    execute_parallel(
        lambda: control_motor_speed(odrv0.axis0, speed_fr, "0 Axis0"),  # Front Right
        lambda: control_motor_speed(odrv0.axis1, speed_fl, "0 Axis1"),  # Front Left
        lambda: control_motor_speed(odrv1.axis0, speed_rr, "1 Axis0"),  # Rear Right
        lambda: control_motor_speed(odrv1.axis1, speed_rl, "1 Axis1")   # Rear Left
    )
    global current_left_angle, current_right_angle
    current_left_angle = delta_out
    current_right_angle = delta_in

def execute_left(value):
    print(f"[SERVO] 좌회전 실행 - value: {value}")
    
    # 좌회전 시 서보모터 각도 계산 (음수 조향각)
    steering_angle = -delta_out*1.5  # 이론적 조향각의 음수 사용
    servo_left_angle, servo_right_angle = compute_servo_angles(steering_angle)
    
    pca9685.channels[SERVO_LEFT_CHANNEL].duty_cycle = angle_to_duty_cycle(servo_left_angle)
    pca9685.channels[SERVO_RIGHT_CHANNEL].duty_cycle = angle_to_duty_cycle(servo_right_angle)
    print(f"[SERVO] 좌회전 서보모터 설정 완료")
    
    # Differential steering 적용
    base_speed = abs(value) * max_speed/2
    delta_angle = -delta_out*1.5  # 좌회전 시 조향각 (음수)
    Vdiff_fl, Vdiff_fr, Vdiff_rl, Vdiff_rr = Differential(delta_angle, base_speed)
    
    # 각 바퀴에 다른 속도 적용
    speed_fl = base_speed * Vdiff_fl  # Front Left
    speed_fr = base_speed * Vdiff_fr  # Front Right  
    speed_rl = base_speed * Vdiff_rl  # Rear Left
    speed_rr = base_speed * Vdiff_rr  # Rear Right
    
    execute_parallel(
        lambda: control_motor_speed(odrv0.axis0, speed_fr, "0 Axis0"),  # Front Right
        lambda: control_motor_speed(odrv0.axis1, speed_fl, "0 Axis1"),  # Front Left
        lambda: control_motor_speed(odrv1.axis0, speed_rr, "1 Axis0"),  # Rear Right
        lambda: control_motor_speed(odrv1.axis1, speed_rl, "1 Axis1")   # Rear Left
    )
    global current_left_angle, current_right_angle
    current_left_angle = delta_in
    current_right_angle = delta_out

def execute_backward_left(value):
    left_angle = delta_in
    right_angle = delta_out
    pca9685.channels[SERVO_LEFT_CHANNEL].duty_cycle = angle_to_duty_cycle(left_angle)
    pca9685.channels[SERVO_RIGHT_CHANNEL].duty_cycle = angle_to_duty_cycle(right_angle)
    
    # Differential steering 적용 (후진 좌회전)
    base_speed = abs(value) * max_speed
    delta_angle = -delta_out  # 좌회전 시 조향각 (음수)
    Vdiff_fl, Vdiff_fr, Vdiff_rl, Vdiff_rr = Differential(delta_angle, base_speed)
    
    # 각 바퀴에 다른 속도 적용 (후진이므로 음수)
    speed_fl = -base_speed * Vdiff_fl  # Front Left
    speed_fr = -base_speed * Vdiff_fr  # Front Right  
    speed_rl = -base_speed * Vdiff_rl  # Rear Left
    speed_rr = -base_speed * Vdiff_rr  # Rear Right
    
    execute_parallel(
        lambda: control_motor_speed(odrv0.axis0, speed_fr, "0 Axis0"),  # Front Right
        lambda: control_motor_speed(odrv0.axis1, speed_fl, "0 Axis1"),  # Front Left
        lambda: control_motor_speed(odrv1.axis0, speed_rr, "1 Axis0"),  # Rear Right
        lambda: control_motor_speed(odrv1.axis1, speed_rl, "1 Axis1")   # Rear Left
    )
    global current_left_angle, current_right_angle
    current_left_angle = left_angle
    current_right_angle = right_angle

def execute_backward_right(value):
    left_angle = delta_out
    right_angle = delta_in
    pca9685.channels[SERVO_LEFT_CHANNEL].duty_cycle = angle_to_duty_cycle(left_angle)
    pca9685.channels[SERVO_RIGHT_CHANNEL].duty_cycle = angle_to_duty_cycle(right_angle)
    
    # Differential steering 적용 (후진 우회전)
    base_speed = abs(value) * max_speed
    delta_angle = delta_out  # 우회전 시 조향각
    Vdiff_fl, Vdiff_fr, Vdiff_rl, Vdiff_rr = Differential(delta_angle, base_speed)
    
    # 각 바퀴에 다른 속도 적용 (후진이므로 음수)
    speed_fl = -base_speed * Vdiff_fl  # Front Left
    speed_fr = -base_speed * Vdiff_fr  # Front Right  
    speed_rl = -base_speed * Vdiff_rl  # Rear Left
    speed_rr = -base_speed * Vdiff_rr  # Rear Right
    
    execute_parallel(
        lambda: control_motor_speed(odrv0.axis0, speed_fr, "0 Axis0"),  # Front Right
        lambda: control_motor_speed(odrv0.axis1, speed_fl, "0 Axis1"),  # Front Left
        lambda: control_motor_speed(odrv1.axis0, speed_rr, "1 Axis0"),  # Rear Right
        lambda: control_motor_speed(odrv1.axis1, speed_rl, "1 Axis1")   # Rear Left
    )
    global current_left_angle, current_right_angle
    current_left_angle = left_angle
    current_right_angle = right_angle

def execute_up(value):
    left_angle = 0
    right_angle = 0
    pca9685.channels[SERVO_LEFT_CHANNEL].duty_cycle = angle_to_duty_cycle(left_angle)
    pca9685.channels[SERVO_RIGHT_CHANNEL].duty_cycle = angle_to_duty_cycle(right_angle)
    
    # 현재 속도 기반으로 증가
    current_speed = odrv0.axis0.controller.input_vel
    base_speed = current_speed + 1
    
    # Differential steering 적용
    delta_angle = 0  # 직진 시 조향각 0
    Vdiff_fl, Vdiff_fr, Vdiff_rl, Vdiff_rr = Differential(delta_angle, abs(base_speed))
    
    # 각 바퀴에 다른 속도 적용 (속도 방향 유지)
    speed_fl = base_speed * Vdiff_fl if base_speed >= 0 else -abs(base_speed) * Vdiff_fl
    speed_fr = base_speed * Vdiff_fr if base_speed >= 0 else -abs(base_speed) * Vdiff_fr
    speed_rl = base_speed * Vdiff_rl if base_speed >= 0 else -abs(base_speed) * Vdiff_rl
    speed_rr = base_speed * Vdiff_rr if base_speed >= 0 else -abs(base_speed) * Vdiff_rr
    
    execute_parallel(
        lambda: control_motor_speed(odrv0.axis0, speed_fr, "0 Axis0"),  # Front Right
        lambda: control_motor_speed(odrv0.axis1, speed_fl, "0 Axis1"),  # Front Left
        lambda: control_motor_speed(odrv1.axis0, speed_rr, "1 Axis0"),  # Rear Right
        lambda: control_motor_speed(odrv1.axis1, speed_rl, "1 Axis1")   # Rear Left
    )
    global current_left_angle, current_right_angle
    current_left_angle = left_angle
    current_right_angle = right_angle

def execute_down(value):
    left_angle = 0
    right_angle = 0
    pca9685.channels[SERVO_LEFT_CHANNEL].duty_cycle = angle_to_duty_cycle(left_angle)
    pca9685.channels[SERVO_RIGHT_CHANNEL].duty_cycle = angle_to_duty_cycle(right_angle)
    
    # 현재 속도 기반으로 감소
    current_speed = odrv0.axis0.controller.input_vel
    base_speed = current_speed - 1
    
    # Differential steering 적용
    delta_angle = 0  # 직진 시 조향각 0
    Vdiff_fl, Vdiff_fr, Vdiff_rl, Vdiff_rr = Differential(delta_angle, abs(base_speed))
    
    # 각 바퀴에 다른 속도 적용 (속도 방향 유지)
    speed_fl = base_speed * Vdiff_fl if base_speed >= 0 else -abs(base_speed) * Vdiff_fl
    speed_fr = base_speed * Vdiff_fr if base_speed >= 0 else -abs(base_speed) * Vdiff_fr
    speed_rl = base_speed * Vdiff_rl if base_speed >= 0 else -abs(base_speed) * Vdiff_rl
    speed_rr = base_speed * Vdiff_rr if base_speed >= 0 else -abs(base_speed) * Vdiff_rr
    
    execute_parallel(
        lambda: control_motor_speed(odrv0.axis0, speed_fr, "0 Axis0"),  # Front Right
        lambda: control_motor_speed(odrv0.axis1, speed_fl, "0 Axis1"),  # Front Left
        lambda: control_motor_speed(odrv1.axis0, speed_rr, "1 Axis0"),  # Rear Right
        lambda: control_motor_speed(odrv1.axis1, speed_rl, "1 Axis1")   # Rear Left
    )
    global current_left_angle, current_right_angle
    current_left_angle = left_angle
    current_right_angle = right_angle

def execute_fast(value):
    if current_left_angle == 0 and current_right_angle == 0: # 멈춰있거나 전진 or 후진중이면 무시
        return
    
    # 현재 속도 기반으로 증가
    current_speed = odrv0.axis0.controller.input_vel
    base_speed = current_speed + 1
    
    # 현재 조향각에 따른 Differential steering 적용
    if current_left_angle != 0 or current_right_angle != 0:
        # 조향 중일 때
        if current_left_angle == delta_out:  # 우회전
            delta_angle = delta_out
        elif current_left_angle == delta_in:  # 좌회전
            delta_angle = -delta_out
        else:
            delta_angle = 0
    else:
        delta_angle = 0
    
    Vdiff_fl, Vdiff_fr, Vdiff_rl, Vdiff_rr = Differential(delta_angle, abs(base_speed))
    
    # 각 바퀴에 다른 속도 적용 (속도 방향 유지)
    speed_fl = base_speed * Vdiff_fl if base_speed >= 0 else -abs(base_speed) * Vdiff_fl
    speed_fr = base_speed * Vdiff_fr if base_speed >= 0 else -abs(base_speed) * Vdiff_fr
    speed_rl = base_speed * Vdiff_rl if base_speed >= 0 else -abs(base_speed) * Vdiff_rl
    speed_rr = base_speed * Vdiff_rr if base_speed >= 0 else -abs(base_speed) * Vdiff_rr
    
    execute_parallel(
        lambda: control_motor_speed(odrv0.axis0, speed_fr, "0 Axis0"),  # Front Right
        lambda: control_motor_speed(odrv0.axis1, speed_fl, "0 Axis1"),  # Front Left
        lambda: control_motor_speed(odrv1.axis0, speed_rr, "1 Axis0"),  # Rear Right
        lambda: control_motor_speed(odrv1.axis1, speed_rl, "1 Axis1")   # Rear Left
    )

def execute_slow(value):
    if current_left_angle == 0 and current_right_angle == 0: # 멈춰있거나 전진 or 후진중이면 무시
        return
    
    # 현재 속도 기반으로 감소
    current_speed = odrv0.axis0.controller.input_vel
    base_speed = current_speed - 1
    
    # 현재 조향각에 따른 Differential steering 적용
    if current_left_angle != 0 or current_right_angle != 0:
        # 조향 중일 때
        if current_left_angle == delta_out:  # 우회전
            delta_angle = delta_out
        elif current_left_angle == delta_in:  # 좌회전
            delta_angle = -delta_out
        else:
            delta_angle = 0
    else:
        delta_angle = 0
    
    Vdiff_fl, Vdiff_fr, Vdiff_rl, Vdiff_rr = Differential(delta_angle, abs(base_speed))
    
    # 각 바퀴에 다른 속도 적용 (속도 방향 유지)
    speed_fl = base_speed * Vdiff_fl if base_speed >= 0 else -abs(base_speed) * Vdiff_fl
    speed_fr = base_speed * Vdiff_fr if base_speed >= 0 else -abs(base_speed) * Vdiff_fr
    speed_rl = base_speed * Vdiff_rl if base_speed >= 0 else -abs(base_speed) * Vdiff_rl
    speed_rr = base_speed * Vdiff_rr if base_speed >= 0 else -abs(base_speed) * Vdiff_rr
    
    execute_parallel(
        lambda: control_motor_speed(odrv0.axis0, speed_fr, "0 Axis0"),  # Front Right
        lambda: control_motor_speed(odrv0.axis1, speed_fl, "0 Axis1"),  # Front Left
        lambda: control_motor_speed(odrv1.axis0, speed_rr, "1 Axis0"),  # Rear Right
        lambda: control_motor_speed(odrv1.axis1, speed_rl, "1 Axis1")   # Rear Left
    )

def execute_setting(value):
    if current_left_angle != 0 and current_right_angle != 0:
        return
    
    # 현재 속도 방향에 따라 새로운 속도 설정
    current_speed = odrv0.axis0.controller.input_vel
    if current_speed > 0:
        base_speed = (value/30) * max_speed
    else:
        base_speed = -(value/30) * max_speed
    
    # Differential steering 적용 (직진)
    delta_angle = 0  # 직진 시 조향각 0
    Vdiff_fl, Vdiff_fr, Vdiff_rl, Vdiff_rr = Differential(delta_angle, abs(base_speed))
    
    # 각 바퀴에 다른 속도 적용 (속도 방향 유지)
    speed_fl = base_speed * Vdiff_fl if base_speed >= 0 else -abs(base_speed) * Vdiff_fl
    speed_fr = base_speed * Vdiff_fr if base_speed >= 0 else -abs(base_speed) * Vdiff_fr
    speed_rl = base_speed * Vdiff_rl if base_speed >= 0 else -abs(base_speed) * Vdiff_rl
    speed_rr = base_speed * Vdiff_rr if base_speed >= 0 else -abs(base_speed) * Vdiff_rr
    
    execute_parallel(
        lambda: control_motor_speed(odrv0.axis0, speed_fr, "0 Axis0"),  # Front Right
        lambda: control_motor_speed(odrv0.axis1, speed_fl, "0 Axis1"),  # Front Left
        lambda: control_motor_speed(odrv1.axis0, speed_rr, "1 Axis0"),  # Rear Right
        lambda: control_motor_speed(odrv1.axis1, speed_rl, "1 Axis1")   # Rear Left
    )

def execute_quit(value):
    current_speed = odrv0.axis0.controller.input_vel # 현재 모터 속도를 읽어서 0이면 종료, 0이 아니면 무시
    if current_speed != 0:
        return False
    return True

# 명령어와 실행 함수를 매핑하는 딕셔너리
command_functions = {
    "forward": execute_forward,
    "backward": execute_backward,
    "right": execute_right,
    "left": execute_left,
    "stop": set_motors_idle,
    "backward left": execute_backward_left,
    "backward right": execute_backward_right,
    "up": execute_up,
    "down": execute_down,
    "fast": execute_fast,
    "slow": execute_slow,
    "setting": execute_setting,
    "quit": execute_quit
}

# 명령 실행을 위한 스레드 풀과 큐 추가
command_queue = queue.Queue()
command_executor_running = True

def command_executor():
    """명령을 별도 스레드에서 실행하는 함수"""
    global command_executor_running
    
    while command_executor_running:
        try:
            # 큐에서 명령 가져오기 (1초 타임아웃)
            commands = command_queue.get(timeout=1.0)
            
            # 단일 명령 처리
            if len(commands) == 1:
                command = commands[0]
                command_id = command.get("command_id")
                value = command.get("value", 0)
                print(f"단일 명령 수신: {command_id}, 값: {value}")
                
                if command_id in command_functions:
                    if command_id == "quit":
                        if command_functions[command_id](value):
                            print("프로그램을 종료합니다...")
                            set_motors_idle()  # 모터 정지
                            sys.exit(0)  # 프로그램 종료
                    elif command_id == "stop":
                        command_functions[command_id]()
                    else:
                        command_functions[command_id](value)
                else:
                    print("알 수 없는 명령:", command_id)
                
            # 복합 명령 처리
            else:
                print(f"복합 명령 수신: {len(commands)}개의 명령")
                for i, command in enumerate(commands):
                    command_id = command.get("command_id")
                    value = command.get("value", 0)
                    duration = command.get("duration", -1)
                    print(f"[{i+1}/{len(commands)}] 명령 실행: {command_id}, 값: {value}, 지속시간: {duration}")
                    
                    if command_id in command_functions:
                        if command_id == "quit":
                            if command_functions[command_id](value):
                                print("프로그램을 종료합니다...")
                                sys.exit(0)  # 프로그램 종료
                        elif command_id == "stop":
                            command_functions[command_id]()
                            print(f"정지 명령 실행 완료")
                            if duration > 0:
                                print(f"{duration}초 대기 중...")
                                time.sleep(duration)
                                print(f"{duration}초 대기 완료")
                        else:
                            command_functions[command_id](value)
                            print(f"{command_id} 명령 실행 완료")
                            if duration > 0:
                                print(f"{duration}초 동안 {command_id} 실행 중...")
                                time.sleep(duration)
                                print(f"{duration}초 실행 완료")
                    else:
                        print("알 수 없는 명령:", command_id)
                print("모든 복합 명령 실행 완료")
            
            command_queue.task_done()
            
        except queue.Empty:
            # 타임아웃 발생 시 계속 진행
            continue
        except Exception as e:
            print(f"명령 실행 오류: {e}")

if __name__ == "__main__":
    try:
        print("Finding ODrives...")
        odrv0 = odrive.find_any(serial_number="3663387E3333")
        odrv1 = odrive.find_any(serial_number="3687387E3333")
        print("ODrives found.")

        threads = [
            threading.Thread(target=calibrate_odrive_axis, args=(odrv0, odrv0.axis0, "0 Axis0")),
            threading.Thread(target=calibrate_odrive_axis, args=(odrv0, odrv0.axis1, "0 Axis1")),
            threading.Thread(target=calibrate_odrive_axis, args=(odrv1, odrv1.axis0, "1 Axis0")),
            threading.Thread(target=calibrate_odrive_axis, args=(odrv1, odrv1.axis1, "1 Axis1")),
        ]
        for thread in threads: thread.start()
        for thread in threads: thread.join()

        # 명령 실행 스레드 시작
        command_thread = threading.Thread(target=command_executor)
        command_thread.daemon = True
        command_thread.start()
        print("[CMD] Command executor thread started")

        video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        video_socket.bind(('0.0.0.0', VIDEO_PORT))
        video_socket.listen(1)
        print(f"[VIDEO] Waiting on port {VIDEO_PORT}...")
        video_client, _ = video_socket.accept()
        print("[VIDEO] Client connected")

        command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        command_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        command_socket.bind(('0.0.0.0', CMD_PORT))
        command_socket.listen(1)
        print(f"[CMD] Waiting on port {CMD_PORT}...")
        
        command_client = None
        cmd_buffer = ''

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")

        while True:
            ret, frame = cap.read()
            if ret:
                send_frame(frame)

                # 명령 클라이언트가 없거나 연결이 끊어진 경우 새 연결 대기
                if command_client is None:
                    try:
                        command_socket.settimeout(0.1)  # 0.1초 타임아웃
                        command_client, addr = command_socket.accept()
                        print(f"[CMD] Connected: {addr}")
                        command_client.setblocking(False)
                        cmd_buffer = ''  # 버퍼 초기화
                    except socket.timeout:
                        continue  # 타임아웃이면 다음 루프로
                    except Exception as e:
                        print(f"[CMD] Accept error: {e}")
                        continue

                # 명령 데이터 수신
                if command_client:
                    try:
                        ready, _, _ = select.select([command_client], [], [], 0)
                        if command_client in ready:
                            try:
                                chunk = command_client.recv(1024)
                                if not chunk:
                                    print("CMD socket closed by peer")
                                    command_client.close()
                                    command_client = None
                                    continue
                                cmd_buffer += chunk.decode('utf-8')
                            except ConnectionResetError:
                                print("CMD connection reset")
                                command_client.close()
                                command_client = None
                                continue
                    except Exception as e:
                        print(f"CMD receive error: {e}")
                        command_client.close()
                        command_client = None
                        continue

                while '\n' in cmd_buffer:
                    line, cmd_buffer = cmd_buffer.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    print(f"[DEBUG] 수신된 명령 라인: {line}")
                    try:
                        cmd_dict = ast.literal_eval(line)
                        print(f"[DEBUG] 파싱된 명령: {cmd_dict}")
                    except (ValueError, SyntaxError) as e:
                        print("error:", e, "Line:", line)
                        continue
                
                    # 새로운 명령 포맷 처리 - 큐에 추가
                    if "commands" in cmd_dict:
                        commands = cmd_dict.get("commands", [])
                        # 명령을 큐에 추가 (비동기 실행)
                        command_queue.put(commands)
                        print(f"명령이 큐에 추가됨: {len(commands)}개")
                        continue
                    
                    print("알 수 없는 명령 형식:", line)
                    continue

    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        # 명령 실행 스레드 종료
        command_executor_running = False
        
        # 프로그램 종료 시에만 모터를 완전히 정지
        print("프로그램 종료 - 모터를 IDLE 상태로 변경...")
        odrv0.axis0.controller.input_vel = 0
        odrv0.axis1.controller.input_vel = 0
        odrv1.axis0.controller.input_vel = 0
        odrv1.axis1.controller.input_vel = 0
        odrv0.axis0.requested_state = AXIS_STATE_IDLE
        odrv0.axis1.requested_state = AXIS_STATE_IDLE
        odrv1.axis0.requested_state = AXIS_STATE_IDLE
        odrv1.axis1.requested_state = AXIS_STATE_IDLE
        
        cap.release()
        if command_client:
            command_client.close()
        command_socket.close()
        video_client.close()
        video_socket.close()
