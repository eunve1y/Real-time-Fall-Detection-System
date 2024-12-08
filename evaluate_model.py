import sys
import os

# yolov5 경로 추가
sys.path.append(os.path.join(os.getcwd(), "yolov5"))

from yolov5 import val

def main():
    # 경로 설정
    model_path = "C:/Users/User/Desktop/sw_project/scripts/yolov5/runs/train/exp6/weights/best.pt"  # 학습된 모델 경로
    data_path = "C:/Users/User/Desktop/sw_project/scripts/yolov5/data/custom_data.yaml"  # 데이터 설정 파일 경로

    # YOLOv5 성능 평가
    results = val.run(
        data=data_path,
        weights=model_path,
        imgsz=640,
        batch_size=16,
        task="val",  # 검증 실행
        device=""  # CPU/GPU 자동 감지
    )

    # 결과 구조 확인
    print(f"Results structure: {results}")

    # 적절히 결과 추출
    if isinstance(results, tuple):
        metrics = results[0]  # 예: 결과 metric 추출
        print(f"Metrics: {metrics}")
    else:
        print("Unexpected results format.")

if __name__ == '__main__':
    main()
