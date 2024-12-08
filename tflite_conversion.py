import tensorflow as tf

# 모델 로드 및 변환
model = tf.keras.models.load_model("models/fall_detection_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 저장
with open("models/fall_detection_model.tflite", "wb") as f:
    f.write(tflite_model)
