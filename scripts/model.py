from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model():
    model = Sequential()

    # Thêm lớp Convolutional
    model.add(Conv2D(64, (3, 3), input_shape=(48, 48, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Thêm lớp Dense
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(7, activation='softmax'))  # 7 cảm xúc

    # Biên dịch mô hình
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
