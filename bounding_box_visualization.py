import torch
import cv2
import os

# YOLOv5 모델 로드
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='C:/Users/User/Desktop/sw_project/scripts/yolov5/runs/train/exp6/weights/best.pt',
    force_reload=True
)

# 클래스 이름 영어로 덮어쓰기
model.names = ["Person", "Sofa", "Bed", "Chair"]  # 한글 대신 영어로 설정

# 테스트 이미지 경로
test_images_path = "C:/Users/User/Desktop/sw_project/data/test_images/"
output_path = "runs/detect/bounding_box_visualization/"
os.makedirs(output_path, exist_ok=True)

# 테스트 이미지 파일 리스트
test_images = [f for f in os.listdir(test_images_path) if f.endswith(('.jpg', '.png'))]

for img_file in test_images:
    img_path = os.path.join(test_images_path, img_file)
    img = cv2.imread(img_path)

    # YOLOv5 모델 예측
    results = model(img)

    # 바운딩 박스 시각화
    for *box, conf, cls in results.xyxy[0]:  # bbox 좌표, confidence, class
        x_min, y_min, x_max, y_max = map(int, box)
        label = f"{model.names[int(cls)]} {conf:.2f}"  # 영어 라벨 적용
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 저장
    output_file = os.path.join(output_path, img_file)
    cv2.imwrite(output_file, img)
    print(f"Processed {img_file} saved to {output_file}")

print("Bounding box visualization complete!")
