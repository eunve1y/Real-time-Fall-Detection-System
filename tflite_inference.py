import tensorflow as tf
import cv2
import numpy as np

# TensorFlow Lite 모델 로드
interpreter = tf.lite.Interpreter(model_path="models/fall_detection_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 실시간 카메라
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (224, 224))
    input_data = np.expand_dims(resized_frame / 255.0, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    label = "Fall" if np.argmax(output_data) == 1 else "Normal"
    color = (0, 0, 255) if label == "Fall" else (0, 255, 0)
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
