import cv2
from keras.models import load_model
import numpy as np

def predict_emotion_with_display(image_path):
    # Đọc ảnh và chuyển sang ảnh xám
    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc ảnh. Hãy kiểm tra đường dẫn ảnh.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt bằng Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("Không phát hiện khuôn mặt trong ảnh.")
        cv2.imshow('Input Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Tải mô hình đã huấn luyện
    model = load_model('models/emotion_model.h5')
    emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprised', 6: 'Neutral'}

    for (x, y, w, h) in faces:
        # Cắt và xử lý khuôn mặt
        face = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = face_normalized.reshape(1, 48, 48, 1)

        # Dự đoán cảm xúc
        prediction = model.predict(face_input)
        emotion = emotion_dict[np.argmax(prediction)]

        # Vẽ khung và hiển thị cảm xúc trên ảnh
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Hiển thị ảnh với khuôn mặt và cảm xúc
    cv2.namedWindow('Emotion Recognition', cv2.WINDOW_NORMAL) 
    cv2.imshow('Emotion Recognition', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_emotion_with_display('D:/22DTHD5/TriTueNhanTao/Emotion/surprised.jpg')
