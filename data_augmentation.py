import albumentations as A
import cv2
import os

# 데이터 증강 파이프라인 설정
augmentation = A.Compose(
    [
        A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.2),
)

# YOLO 라벨 저장 함수
def save_yolo_labels(label_path, bboxes, class_labels):
    with open(label_path, "w") as f:
        for bbox, label in zip(bboxes, class_labels):
            bbox_line = f"{label} " + " ".join(map(str, bbox))
            f.write(bbox_line + "\n")

# 경로 설정
input_img_path = "data/yolo_dataset/images/"
input_label_path = "data/yolo_dataset/labels/"
output_img_path = "data/augmented_images/images/"
output_label_path = "data/augmented_images/labels/"
os.makedirs(output_img_path, exist_ok=True)
os.makedirs(output_label_path, exist_ok=True)

# 증강 데이터 생성
for img_file in os.listdir(input_img_path):
    img_path = os.path.join(input_img_path, img_file)
    label_file = os.path.join(input_label_path, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))

    # 이미지 로드
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지를 로드할 수 없습니다: {img_path}")
        continue

    # 라벨 파일 확인
    if not os.path.exists(label_file):
        print(f"라벨 파일이 없습니다: {label_file}")
        continue

    with open(label_file, "r") as f:
        label_data = f.readlines()

    bboxes = []
    class_labels = []
    for line in label_data:
        parts = line.strip().split()
        class_labels.append(int(parts[0]))
        bboxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

    # 증강 적용
    try:
        augmented = augmentation(image=img, bboxes=bboxes, class_labels=class_labels)
        aug_img = augmented["image"]
        aug_bboxes = augmented["bboxes"]
        aug_labels = augmented["class_labels"]

        # 증강된 데이터 저장
        aug_img_file = os.path.join(output_img_path, f"aug_{img_file}")
        aug_label_file = os.path.join(output_label_path, f"aug_{img_file.replace('.jpg', '.txt').replace('.png', '.txt')}")
        cv2.imwrite(aug_img_file, aug_img)
        save_yolo_labels(aug_label_file, aug_bboxes, aug_labels)
    except Exception as e:
        print(f"증강 처리 중 오류가 발생했습니다: {img_file}, {e}")

print("데이터 증강 완료!")
