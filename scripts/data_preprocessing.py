import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def preprocess_data_from_folders(data_dir):
    image_list = []
    label_list = []

    emotion_dict = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
    
    for emotion in os.listdir(data_dir):
        emotion_folder = os.path.join(data_dir, emotion)
        if os.path.isdir(emotion_folder):
            # Lấy tất cả các ảnh trong thư mục của cảm xúc
            for img_name in os.listdir(emotion_folder):
                img_path = os.path.join(emotion_folder, img_name)
                if img_path.endswith(".jpg") or img_path.endswith(".png"):
                    # Đọc ảnh
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (48, 48))
                    img = img / 255.0  # Chuẩn hóa ảnh
                    image_list.append(img)
                    label_list.append(emotion_dict[emotion])

    # Chuyển ảnh và nhãn thành numpy array
    X = np.array(image_list)
    y = np.array(label_list)

    # Thêm chiều cho ảnh (batch_size, height, width, channels)
    X = X.reshape(X.shape[0], 48, 48, 1)
    
    # Chuyển nhãn thành dạng one-hot encoding
    y = to_categorical(y, num_classes=7)

    # Chia dữ liệu thành train và test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data_from_folders('data')
    print("Dữ liệu đã được tiền xử lý thành công!")
