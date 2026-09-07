from flask import Flask, request, jsonify
from flask_cors import CORS
import socket
import json
import logging
import cv2
import struct
import numpy as np
import threading
import time
import errno
import datetime

app = Flask(__name__)
CORS(app)

# VPN 서버에서 할당받은 IP 주소와 포트 설정
LOCAL_HOST = '0.0.0.0'  # 모든 인터페이스에서 수신
LOCAL_PORT = 5000

# Jetson Orin Nano 소켓 설정
JETSON_IP = '100.104.103.98'  # Jetson Orin Nano의 IP
JETSON_PORT = 8001

VIDEO_PORT = 8000  # Jetson Orin Nano의 비디오 포트

# command_id를 한글로 매핑하는 딕셔너리
# cv 화면 표시용
COMMAND_ID_TO_KR = {
    "forward": "직진",
    "backward": "후진",
    "right": "우회전",
    "left": "좌회전",
    "stop": "멈춰",
    "backward left": "후진좌회전",
    "backward right": "후진우회전",
    "up": "속도올려",
    "down": "속도내려",
    "fast": "빨리회전",
    "slow": "천천히회전",
    "setting": "속도설정",
    "quit": "종료"
}

# 표시할 명령어 리스트와 락 (스레드 안전)
display_commands = []  # [(명령어, duration), ...]
display_lock = threading.Lock()

def create_jetson_socket():
    """Jetson Orin Nano로 연결하는 소켓 생성 함수"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            print(f"Jetson 명령 소켓 연결 시도 {attempt + 1}/{max_retries}...")
            jetson_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            jetson_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            jetson_socket.settimeout(5)  # 5초 타임아웃 설정
            jetson_socket.connect((JETSON_IP, JETSON_PORT))
            print(f"Jetson Orin Nano 연결 성공: {JETSON_IP}:{JETSON_PORT}")
            return jetson_socket
        except Exception as e:
            print(f"Jetson Orin Nano 연결 실패 (시도 {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    print("Jetson 명령 소켓 연결을 포기합니다.")
    return None

@app.route('/test', methods=['POST'])
def receive_data():
    if not request.is_json:
        print("JSON 형식이 아닙니다.")
        return jsonify({"error": "JSON 형식이 아닙니다."}), 400
        
    data = request.json
    print("받은 데이터:", data)  # 디버깅용
    
    if 'llm_output' not in data:
        print("llm_output 필드가 없습니다.")
        return jsonify({"error": "llm_output 필드가 없습니다."}), 400
        
    llm_output = data.get('llm_output', '')
    print("llm_output 에 저장된 값:", llm_output)
    client_ip = request.remote_addr
    
    # 매 요청마다 새로운 소켓 생성
    jetson_socket = create_jetson_socket()
    if jetson_socket is None:
        return jsonify({"error": "Jetson Orin Nano 연결 실패"}), 500
    
    try:
        # Jetson Orin Nano로 LLM 출력 전송 (개행문자 추가)
        message_with_newline = llm_output + '\n'
        jetson_socket.sendall(message_with_newline.encode('utf-8'))
        print(f"Jetson Orin Nano로 전송 완료: {llm_output}")

        # 화면 표시용 명령어 파싱 및 저장
        try:
            print(f"[DEBUG] 원본 llm_output: {llm_output}")
            print(f"[DEBUG] llm_output 타입: {type(llm_output)}")
            
            # 작은따옴표로 감싸진 경우를 위해 큰따옴표로 변환
            converted_output = llm_output.replace("'", '"')
            print(f"[DEBUG] 변환된 출력: {converted_output}")
            
            llm_dict = json.loads(converted_output)
            print(f"[DEBUG] 파싱된 딕셔너리: {llm_dict}")
            
            commands = llm_dict.get("commands", [])
            temp_display = []
            
            print(f"[DEBUG] 파싱된 명령 개수: {len(commands)}")
            
            for i, cmd in enumerate(commands):
                cmd_id = cmd.get("command_id", "unknown")
                duration = cmd.get("duration", -1)
                value = cmd.get("value", 0)
                
                print(f"[DEBUG] 명령 {i+1}: command_id='{cmd_id}', duration={duration}, value={value}")
                
                # 한글 명령어로 변환
                kr_cmd = COMMAND_ID_TO_KR.get(cmd_id, f"알수없음({cmd_id})")
                print(f"[DEBUG] 한글 변환: {cmd_id} -> {kr_cmd}")
                
                # 복합 명령인 경우 순서 번호 추가
                if len(commands) > 1:
                    if duration > 0:
                        temp_display.append(f"{i+1}. {kr_cmd} ({duration}초)")
                    else:
                        temp_display.append(f"{i+1}. {kr_cmd}")
                else:
                    # 단일 명령인 경우
                    if duration > 0:
                        temp_display.append(f"{kr_cmd} ({duration}초)")
                    else:
                        temp_display.append(f"{kr_cmd}")
            
            print(f"[DEBUG] 화면 표시 명령어: {temp_display}")
            
            with display_lock:
                display_commands = temp_display
                print(f"[DEBUG] display_commands 업데이트 완료: {display_commands}")
                
        except Exception as e:
            print(f"명령어 파싱 오류: {e}")
            print(f"원본 데이터: {llm_output}")
            # 파싱 실패 시 원본 데이터 표시
            with display_lock:
                display_commands = [f"파싱오류: {llm_output[:50]}..."]

        return jsonify({
            "status": "success",
            "received_llm_output": llm_output,
            "client_ip": client_ip
        })
    except Exception as e:
        print(f"Jetson Orin Nano 전송 에러: {e}")
        return jsonify({"error": "Jetson Orin Nano 전송 실패"}), 500
    finally:
        # 전송 후 소켓 닫기
        jetson_socket.close()

def video_stream():
    video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    video_socket.connect((JETSON_IP, VIDEO_PORT))
    video_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    video_socket.setblocking(False)

    data = b""
    payload_size = struct.calcsize('<L')
    prev_time = time.time()
    last_frame = None

    while True:
        try:
            while True:
                packet = video_socket.recv(4096)
                if not packet:
                    break
                data += packet
        except socket.error as e:
            if e.errno != errno.EWOULDBLOCK and e.errno != errno.EAGAIN:
                raise

        while True:
            if len(data) < payload_size:
                break
            packed_msg_size = data[:payload_size]
            msg_size = struct.unpack('<L', packed_msg_size)[0]
            if len(data) < payload_size + msg_size:
                break

            frame_data = data[payload_size:payload_size+msg_size]
            data = data[payload_size+msg_size:]
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

            if frame is not None:
                last_frame = frame

        if last_frame is not None:
            display_frame = last_frame.copy()
            current_time = time.time()
            dt = current_time - prev_time
            fps = 1.0 / dt if dt > 0 else 0
            prev_time = current_time

            # FPS 표시
            cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 현재 시각 표시
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, now_str, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 명령어 표시
            # y0 = 110
            # with display_lock:
            #     current_commands = display_commands.copy()  # 복사본 생성
            
            
            # if current_commands:
            #     # 명령어 개수에 따라 제목 표시
            #     if len(current_commands) > 1:
            #         # 복합 명령
            #         title_text = f"Command: Multiple {len(current_commands)}"
            #         cv2.putText(display_frame, title_text, (10, y0),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            #         y_start = y0 + 35
            #     else:
            #         # 단일 명령
            #         cv2.putText(display_frame, "Command: Single", (10, y0),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            #         y_start = y0 + 35
                
            #     # 각 명령어 표시 (일단 영어로 테스트)
            #     for i, cmd in enumerate(current_commands):
            #         y_pos = y_start + i * 35
            #         # 화면 높이를 벗어나지 않도록 제한
            #         if y_pos < display_frame.shape[0] - 20:
            #             # 한글 대신 영어로 임시 표시
            #             cmd_text = str(cmd).encode('ascii', 'ignore').decode('ascii')
            #             if not cmd_text.strip():  # 한글만 있어서 빈 문자열이 된 경우
            #                 cmd_text = f"Korean_Command_{i+1}"
            #             cv2.putText(display_frame, cmd_text, (10, y_pos),
            #                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # else:
            #     # 명령어가 없을 때
            #     cv2.putText(display_frame, "Command: Waiting", (10, y0),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

            cv2.imshow("RC Car Video Stream", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video_socket.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 비디오 스트리밍 스레드 시작
    video_thread = threading.Thread(target=video_stream)
    video_thread.daemon = True
    video_thread.start()

    print(f"로컬 서버 시작... (VPN IP: {LOCAL_HOST}, 포트: {LOCAL_PORT})")
    app.run(host=LOCAL_HOST, port=LOCAL_PORT) 