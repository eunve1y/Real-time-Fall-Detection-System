import cv2
import mediapipe as mp
import time
import threading
import pygame  # 소리 재생용 모듈


class FallDetection:
    def __init__(self):
        self.alert_active = False
        self.fall_start_time = None  # 쓰러짐 감지 시작 시간
        self.sound_lock = threading.Lock()  # 소리 재생 관리용 Lock

        # pygame 초기화
        pygame.mixer.init()
        self.alert_sound = "C:/Users/User/Desktop/sw_project/alert_sound.mp3"

    # 경고음 재생 함수
    def playalert(self):
        with self.sound_lock:
            if not self.alert_active:
                self.alert_active = True
                pygame.mixer.music.load(self.alert_sound)
                pygame.mixer.music.play(-1)  # 반복 재생

    # 경고음 중지 함수
    def stopalert(self):
        with self.sound_lock:
            if self.alert_active:
                pygame.mixer.music.stop()
                self.alert_active = False

    # 쓰러짐 판단 함수
    def is_fallen(self, landmarks):
        try:
            # 필요한 관절 좌표 추출
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

            # 엉덩이와 어깨의 Y 좌표 중심 계산
            hip_center_y = (left_hip.y + right_hip.y) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2

            # 어깨와 엉덩이의 수평 차이 계산 (쓰러짐 판단 기준)
            hip_shoulder_delta = abs(shoulder_center_y - hip_center_y)

            # 어깨와 엉덩이의 X 축 차이로 평평한 상태 확인
            shoulder_x_diff = abs(left_shoulder.x - right_shoulder.x)
            hip_x_diff = abs(left_hip.x - right_hip.x)
            is_flat = shoulder_x_diff < 0.1 and hip_x_diff < 0.1

            print(f"Hip-Shoulder Y Delta: {hip_shoulder_delta}, Flatness: {is_flat}")

            # 쓰러짐 판단 조건
            if hip_shoulder_delta < 0.2 and is_flat:
                return True
            return False
        except Exception as e:
            # 관절 인식 실패 시 쓰러짐으로 간주하지 않음
            print(f"쓰러짐 판단 오류: {e}")
            return False

    # 메인 루프
    def run(self):
        cap = cv2.VideoCapture(0)  # 웹캠 사용
        # cap = cv2.VideoCapture("data/test_videos/test_v1.mp4")  # 비디오 사용

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("비디오 끝 또는 읽기 실패!")
                    break

                # BGR 이미지를 RGB로 변환
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Pose 추론
                results = pose.process(image)

                # 이미지를 다시 BGR로 변환
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 스켈레톤 시각화
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

                    # 쓰러짐 감지 및 상태 변경
                    if self.is_fallen(results.pose_landmarks.landmark):
                        if self.fall_start_time is None:  # 쓰러짐 감지 시작 시간 기록
                            self.fall_start_time = time.time()
                            print("쓰러짐 상태 시작")
                        else:
                            elapsed_time = time.time() - self.fall_start_time
                            print(f"쓰러짐 유지 시간: {elapsed_time:.2f}s")
                            if elapsed_time >= 8:  # 쓰러짐이 8초 이상 지속되면
                                # 경고 문구 표시
                                cv2.putText(image, "WARNING: Fall Detected!", (50, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

                                # 소리 재생
                                self.playalert()
                    else:
                        # 쓰러짐 해제 즉시 경고음과 상태 초기화
                        if self.fall_start_time is not None:
                            print("쓰러짐 상태 해제")
                        self.fall_start_time = None
                        self.stopalert()
                else:
                    # 관절 인식이 되지 않는 경우도 쓰러짐 해제 처리
                    print("관절 인식 실패: 쓰러짐 해제 처리")
                    self.fall_start_time = None
                    self.stopalert()

                # 쓰러짐 해제 시 화면에서 문구 제거
                if self.fall_start_time is None:
                    cv2.putText(image, "", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

                # 결과 화면 출력
                cv2.imshow('Fall Detection', image)

                # ESC 키로 종료
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()


# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# FallDetection 실행
fall_detection = FallDetection()
fall_detection.run()
