import cv2
from yolov5 import DetectMultiBackend
import torch

# 모델 로드
model_path = "runs/train/exp/weights/best.pt"
model = DetectMultiBackend(model_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# 실시간 웹캠 탐지
cap = cv2.VideoCapture(0)  # 웹캠 사용
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5 추론
    results = model(frame)
    results.render()  # 탐지 결과를 원본 프레임에 표시

    # 결과 화면에 출력
    cv2.imshow("Real-Time Detection", results.imgs[0])

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
