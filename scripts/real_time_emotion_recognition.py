import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Tải mô hình đã huấn luyện
model = load_model('models/emotion_model.h5')  # Đảm bảo rằng mô hình đã huấn luyện ở đúng đường dẫn

# Định nghĩa các nhãn cảm xúc
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Khởi tạo webcam
cap = cv2.VideoCapture(0)  # Mở webcam

# Kiểm tra nếu không mở được webcam
if not cap.isOpened():
    print("Không thể mở webcam")
    exit()

# Tải Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Không thể tải cascade classifier.")
    exit()

while True:
    # Đọc một frame từ webcam
    ret, frame = cap.read()

    # Nếu không đọc được frame, thoát khỏi vòng lặp
    if not ret:
        print("Không thể đọc webcam")
        break

    # Chuyển đổi ảnh từ BGR (mặc định của OpenCV) sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong ảnh
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Nếu phát hiện khuôn mặt
    for (x, y, w, h) in faces:
        # Cắt khuôn mặt từ ảnh
        face = gray[y:y+h, x:x+w]

        # Thay đổi kích thước khuôn mặt để phù hợp với đầu vào của mô hình
        face_resized = cv2.resize(face, (48, 48))
        face_resized = face_resized.astype('float32') / 255.0  # Chuẩn hóa
        face_resized = np.expand_dims(face_resized, axis=-1)  # Thêm chiều kênh màu
        face_resized = np.expand_dims(face_resized, axis=0)  # Thêm chiều batch

        # Dự đoán cảm xúc từ khuôn mặt
        emotion_prediction = model.predict(face_resized)
        emotion_index = np.argmax(emotion_prediction)  # Lấy cảm xúc có xác suất cao nhất
        emotion = emotion_labels[emotion_index]
        emotion_probability = np.max(emotion_prediction)

        # Vẽ một hình chữ nhật quanh khuôn mặt và hiển thị cảm xúc
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{emotion} ({emotion_probability*100:.2f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị ảnh webcam
    cv2.imshow('Emotion Recognition', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp và đóng các cửa sổ OpenCV
cap.release()
cv2.destroyAllWindows()
