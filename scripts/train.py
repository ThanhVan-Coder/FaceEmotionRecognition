from scripts.data_preprocessing import preprocess_data_from_folders
from scripts.model import build_model

def train_model():
    # Tiền xử lý dữ liệu
    X_train, X_test, y_train, y_test = preprocess_data_from_folders('data')
    
    # Xây dựng mô hình
    model = build_model()
    
    # Huấn luyện mô hình
    model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test))
    
    # Lưu mô hình
    model.save('models/emotion_model.h5')
    print("Mô hình đã được lưu thành công!")

if __name__ == "__main__":
    train_model()
