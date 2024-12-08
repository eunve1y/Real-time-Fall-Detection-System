import cv2
import mediapipe as mp

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 웹캠/비디오 캡처
cap = cv2.VideoCapture(0)  # 웹캠
# cap = cv2.VideoCapture("data/test_videos/test_v1.mp4")  # 비디오 파일

# Pose 추론
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("비디오 끝 또는 읽기 실패!")
            break

        # BGR 이미지를 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 관절 추론
        results = pose.process(image)

        # 이미지를 다시 BGR로 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 관절 및 스켈레톤 시각화
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # 결과 출력
        cv2.imshow('Skeleton Detection', image)

        # 종료 키 설정
        if cv2.waitKey(5) & 0xFF == 27:  # ESC 키로 종료
            break

cap.release()
cv2.destroyAllWindows()
