import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

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

# Évaluation sur le jeu de validation
loss, accuracy = model.evaluate(val_generator)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Visualisation des métriques d'entraînement
plt.figure(figsize=(12, 5))

# Graphique Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Train vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Graphique Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Train vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('metrics/training_metrics.png', dpi=300)
plt.show()

# Calcul des prédictions sur le jeu de validation
val_generator.reset()  # Pour s'assurer de repartir du début
pred_probs = model.predict(val_generator)
y_pred = np.argmax(pred_probs, axis=1)  # Classe prédite pour chaque image

# Récupérer les vraies étiquettes et les noms de classes
y_true = val_generator.classes
class_names = list(val_generator.class_indices.keys())

# Calcul et affichage de la matrice de confusion
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Prédictions')
plt.ylabel('Véritables Labels')
plt.title('Matrice de Confusion')
plt.tight_layout()
plt.savefig('metrics/confusion_matrix.png', dpi=300)
plt.show()

# Calcul du rapport de classification
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

# Conversion en DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Supprimer la ligne 'accuracy' 
report_df_cleaned = report_df.drop(['accuracy'], errors='ignore')

# Sélectionner uniquement les colonnes pertinentes
metrics_to_plot = ['precision', 'recall', 'f1-score']
report_plot = report_df_cleaned[metrics_to_plot]

# Créer une heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(report_plot, annot=True, cmap='YlGnBu', fmt=".2f", cbar=True)
plt.title('Rapport de Classification - Précision, Rappel, F1-score')
plt.xlabel('Métriques')
plt.ylabel('Classes')
plt.tight_layout()

# Exporter en image PNG
plt.savefig('metrics/classification_report_heatmap.png', dpi=300)