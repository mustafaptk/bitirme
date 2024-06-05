import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Veri seti dizin yolu
data_dir = "C:/Users/musta/Desktop/data"

# Veri setindeki sınıflar
classes = ["cloudy", "desert", "green_area", "water"]

# Verileri depolamak için boş listeler oluşturalım
images = []
labels = []

# Veri setini yükleme ve boyutları uygun hale getirme
for cls in classes:
    cls_path = os.path.join(data_dir, cls)
    for img in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img)
        img_arr = cv2.imread(img_path)  # cv2 kullanarak resmi oku
        img_arr = cv2.resize(img_arr, (256, 256))  # 256x256 boyutuna yeniden boyutlandır
        images.append(img_arr)
        labels.append(classes.index(cls))

# Verileri Numpy dizilerine dönüştürme
images = np.array(images)
labels = np.array(labels)

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Verileri 0-1 aralığına ölçekleme
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Etiketleri kategorik hale getirme
y_train = to_categorical(y_train, len(classes))
y_test = to_categorical(y_test, len(classes))

# Veri artırma ve modeli eğitme
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

# CNN modeli oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),  # Regularizasyon ekleme
    Dropout(0.6),  # Dropout oranını artırma
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=15, validation_data=(X_test, y_test))

# Model performansını değerlendirme
model.evaluate(X_test, y_test)

# Eğitim sırasında elde edilen doğruluk ve kayıp değerleri
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Doğruluk grafiği
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Kayıp grafiği
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
