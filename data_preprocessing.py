import cv2
import os

def resize_with_padding(img, target_size=(640, 640), pad_color=(114, 114, 114)):
    h, w = img.shape[:2]
    target_w, target_h = target_size

    # 스케일 계산
    scale = min(target_w / w, target_h / h)
    resized_w, resized_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    # 패딩 추가
    pad_w = (target_w - resized_w) // 2
    pad_h = (target_h - resized_h) // 2
    padded_img = cv2.copyMakeBorder(resized_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=pad_color)

    return padded_img

# 이미지 경로 설정
input_path = "data/furniture_img/"  # 원본 이미지 디렉토리
output_path = "data/processed_images/"  # 전처리된 이미지 디렉토리
os.makedirs(output_path, exist_ok=True)

# 이미지 전처리
for img_file in os.listdir(input_path):
    img_path = os.path.join(input_path, img_file)
    img = cv2.imread(img_path)

    if img is not None:
        processed_img = resize_with_padding(img)
        cv2.imwrite(os.path.join(output_path, img_file), processed_img)
        print(f"처리 완료: {img_file}")
    else:
        print(f"이미지를 로드할 수 없습니다: {img_path}")

print("이미지 전처리 완료!")
