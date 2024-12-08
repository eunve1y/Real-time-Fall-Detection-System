from PIL import Image
import os

def check_and_fill_transparency(image_path, fill_color=(255, 255, 255)):
    img = Image.open(image_path).convert("RGBA")
    alpha = img.split()[-1]

    # 투명도 확인
    if alpha.getextrema()[1] < 255:  # 투명도가 존재한다면
        # 투명 부분을 단색으로 채우기
        bg = Image.new("RGBA", img.size, fill_color + (255,))
        img = Image.alpha_composite(bg, img)

    return img.convert("RGB")  # RGB로 변환

# 이미지 처리
input_path = "data/furniture_img/"
output_path = "data/processed_images_no_transparency/"
os.makedirs(output_path, exist_ok=True)

for img_file in os.listdir(input_path):
    img_path = os.path.join(input_path, img_file)
    processed_img = check_and_fill_transparency(img_path)
    processed_img.save(os.path.join(output_path, img_file))

print("누끼 여부 처리 완료!")
