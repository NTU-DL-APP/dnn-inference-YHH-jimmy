import tensorflow as tf
from tensorflow import keras

# 載入 Fashion-MNIST 資料集
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# 正規化像素值
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 建立神經網路模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 編譯和訓練
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=15, validation_split=0.2)

# 儲存為 .h5 檔案
model.save('fashion_mnist.h5')

