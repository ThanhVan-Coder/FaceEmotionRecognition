import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Đường dẫn đến thư mục test
test_dir = 'D:\\22DTHD5\\TriTueNhanTao\\face_emotion_recognition\\test'  # Thay đổi đường dẫn đến thư mục chứa ảnh kiểm tra

# Tiền xử lý dữ liệu kiểm tra
def preprocess_test_data(test_dir, img_size=(48, 48)):
    test_data = []
    test_labels = []
    class_names = os.listdir(test_dir)  # Danh sách các lớp cảm xúc

    for label in class_names:
        class_folder = os.path.join(test_dir, label)
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển thành ảnh xám
                img = cv2.resize(img, img_size)  # Thay đổi kích thước ảnh
                img = img.astype('float32') / 255.0  # Chuẩn hóa ảnh
                test_data.append(img)
                test_labels.append(label)

    # Chuyển đổi thành numpy array
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    # Chuyển đổi nhãn thành số (từ lớp tên thành chỉ số lớp)
    label_map = {class_names[i]: i for i in range(len(class_names))}
    test_labels = np.array([label_map[label] for label in test_labels])

    # Thêm chiều cho kênh màu (1)
    test_data = np.expand_dims(test_data, axis=-1)

    return test_data, test_labels

# Tiền xử lý dữ liệu kiểm tra
X_test, y_test = preprocess_test_data(test_dir)

# Kiểm tra dữ l
