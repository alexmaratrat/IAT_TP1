import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Charger et préparer les données
path = "obstacles_dataset"
train_dir = f'{path}/train'
val_dir = f'{path}/test'
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')
val_generator = datagen.flow_from_directory(val_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')

# Définir le modèle CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle avec historique
history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# Évaluation
loss, accuracy = model.evaluate(val_generator)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Visualisation des métriques d'entraînement
plt.figure(figsize=(12, 5))

# Courbe d'accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Train vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Courbe de loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Train vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300)
plt.show()

# Calcul et exportation de la matrice de confusion
val_generator.reset()  # Réinitialise le générateur pour une prédiction cohérente
pred_probs = model.predict(val_generator)
y_pred = np.argmax(pred_probs, axis=1)  # Prédiction de la classe avec la plus haute probabilité

# Vraies étiquettes
y_true = val_generator.classes
class_names = list(val_generator.class_indices.keys())

# Calcul de la matrice de confusion
cm = confusion_matrix(y_true, y_pred)

# Affichage de la matrice de confusion
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Prédictions')
plt.ylabel('Véritables Labels')
plt.title('Matrice de Confusion')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()
