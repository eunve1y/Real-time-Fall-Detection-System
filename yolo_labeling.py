import os
import torch
from PIL import Image

# YOLO 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 클래스 매핑 (COCO 클래스 → 커스텀 클래스, 테이블 제외)
coco_to_custom = {
    0: "사람",       # COCO 클래스 'person' → '사람'
    56: "쇼파",      # COCO 클래스 'couch' → '쇼파'
    57: "침대",      # COCO 클래스 'bed' → '침대'
    63: "의자"       # COCO 클래스 'chair' → '의자'
}

custom_class_ids = list(coco_to_custom.keys())  # 사용할 COCO 클래스 ID

# 경로 설정
image_path = "C:/Users/User/Desktop/sw_project/data/furniture_img"  # 원본 이미지 경로
yolo_images_path = "C:/Users/User/Desktop/sw_project/data/yolo_dataset/images/"  # YOLO 이미지 저장 경로
yolo_labels_path = "C:/Users/User/Desktop/sw_project/data/yolo_dataset/labels/"  # YOLO 라벨 저장 경로
os.makedirs(yolo_images_path, exist_ok=True)
os.makedirs(yolo_labels_path, exist_ok=True)

# 이미지 파일 가져오기
image_files = [f for f in os.listdir(image_path) if f.endswith(('.jpg', '.png'))]

# 자동 라벨링
for img_file in image_files:
    img_path = os.path.join(image_path, img_file)

    # 이미지 복사 (YOLO 이미지 경로에 저장)
    img_output_path = os.path.join(yolo_images_path, img_file)
    Image.open(img_path).save(img_output_path)

    # 모델 추론
    results = model(img_path)

    # 입력 이미지 크기 가져오기
    img_width, img_height = Image.open(img_path).size

    # YOLO 형식으로 바운딩 박스 저장
    yolo_data = []
    for *bbox, conf, class_id in results.xyxy[0].tolist():
        if int(class_id) not in custom_class_ids:
            continue  # 커스텀 클래스에 없는 경우 건너뜀

        # 클래스 ID 변환
        class_name = coco_to_custom[int(class_id)]
        class_index = list(coco_to_custom.values()).index(class_name)

        # 바운딩 박스 좌표
        x_min, y_min, x_max, y_max = bbox
        x_center = (x_min + x_max) / 2 / img_width   # 정규화된 x 중심
        y_center = (y_min + y_max) / 2 / img_height  # 정규화된 y 중심
        bbox_width = (x_max - x_min) / img_width     # 정규화된 너비
        bbox_height = (y_max - y_min) / img_height   # 정규화된 높이

        # YOLO 형식 데이터 추가
        yolo_data.append(f"{class_index} {x_center} {y_center} {bbox_width} {bbox_height}")

    # 라벨 파일 저장
    label_file = os.path.join(yolo_labels_path, f"{os.path.splitext(img_file)[0]}.txt")
    with open(label_file, "w") as f:
        f.write("\n".join(yolo_data))

print("자동 라벨링 완료!")
