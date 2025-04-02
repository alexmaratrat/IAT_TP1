import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Charger et préparer les données
path="obstacles_dataset"
train_dir = f'{path}/train'
val_dir = f'{path}/test'
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=32,class_mode='categorical')
val_generator = datagen.flow_from_directory(val_dir, target_size=(64, 64), batch_size=32,class_mode='categorical')
# Définir le modèle CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax') ])
# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
# Entraînement du modèle
model.fit(train_generator, validation_data=val_generator, epochs=10)
# Évaluation
loss, accuracy = model.evaluate(val_generator)
print(f'Loss: {loss}, Accuracy: {accuracy}')