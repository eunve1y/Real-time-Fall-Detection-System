from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# 모델 설계
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # Fall, Normal
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 데이터 로드
from sklearn.model_selection import train_test_split
import numpy as np

# 이미지와 레이블 데이터
images = np.load("data/augmented_frames/images.npy")  # numpy로 저장된 이미지 배열
labels = np.load("data/augmented_frames/labels.npy")  # 레이블 배열

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 학습
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# 모델 저장
model.save("models/fall_detection_model.h5")
