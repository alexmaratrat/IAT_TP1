import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os
import math

# Créer un dossier pour sauvegarder les résultats
os.makedirs('metrics', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Fonction pour créer un modèle avec un nombre variable de couches convolutives et de taille de kernel
def create_model(num_conv_layers, learning_rate=0.001, kernel_size=3):
    model = Sequential()
    
    # Calculer la dimension initiale de l'image
    current_dim = 64
    
    # Première couche convolutive (obligatoire pour définir input_shape)
    model.add(Conv2D(32, (kernel_size, kernel_size), activation='relu', padding="same", input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    current_dim = current_dim // 2  # Réduire de moitié après le pooling
    
    # Ajouter des couches convolutives supplémentaires selon le paramètre
    for i in range(1, num_conv_layers):
        # Augmenter progressivement le nombre de filtres
        filters = 32 * (2 ** min(i, 3))  # Limiter à 256 filtres maximum
        
        model.add(Conv2D(filters, (kernel_size, kernel_size), padding="same", activation='relu'))
        
        # Vérifier si on peut encore appliquer un MaxPooling
        if current_dim > 1:
            model.add(MaxPooling2D(pool_size=(2, 2)))
            current_dim = current_dim // 2
        else:
            # Si la dimension est trop petite, utiliser GlobalAveragePooling2D au lieu de MaxPooling
            model.add(GlobalAveragePooling2D())
            break  # Sortir de la boucle car on a déjà appliqué GlobalAveragePooling
    
    # Si on n'a pas encore appliqué GlobalAveragePooling, ajouter Flatten
    if current_dim > 0:
        model.add(Flatten())
    
    # Couches fully connected
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    # Créer l'optimiseur avec le learning rate spécifié
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compiler le modèle
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Fonction pour entraîner et évaluer un modèle
def train_and_evaluate(num_conv_layers, learning_rate=0.001, epochs=10, kernel_size=3):
    print(f"\n=== Entraînement du modèle avec {num_conv_layers} couches convolutives, lr={learning_rate}, epochs={epochs}, kernel_size={kernel_size} ===\n")
    
    # Charger et préparer les données
    path = "obstacles_dataset"
    train_dir = f'{path}/train'
    val_dir = f'{path}/test'
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')
    val_generator = datagen.flow_from_directory(val_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')
    
    # Créer le modèle
    model = create_model(num_conv_layers, learning_rate, kernel_size)
    model.summary()
    
    # Entraînement du modèle avec historique
    history = model.fit(
        train_generator, 
        validation_data=val_generator, 
        epochs=epochs, 
        verbose=1
    )
    
    # Évaluation sur le jeu de validation
    loss, accuracy = model.evaluate(val_generator)
    print(f'Loss: {loss}, Accuracy: {accuracy}')
    
    # Sauvegarder le modèle
    model.save(f'models/model_conv{num_conv_layers}_lr{learning_rate}_ep{epochs}_k{kernel_size}.h5')
    
    # Visualisation des métriques d'entraînement
    plt.figure(figsize=(12, 5))
    
    # Graphique Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Train vs Validation Accuracy\n({num_conv_layers} conv, lr={learning_rate}, k={kernel_size})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Graphique Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Train vs Validation Loss\n({num_conv_layers} conv, lr={learning_rate}, k={kernel_size})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'metrics/training_metrics_conv{num_conv_layers}_lr{learning_rate}_ep{epochs}_k{kernel_size}.png', dpi=300)
    plt.close()
    
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
    plt.title(f'Matrice de Confusion\n({num_conv_layers} conv, lr={learning_rate}, k={kernel_size})')
    plt.tight_layout()
    plt.savefig(f'metrics/confusion_matrix_conv{num_conv_layers}_lr{learning_rate}_ep{epochs}_k{kernel_size}.png', dpi=300)
    plt.close()
    
    # Calcul du rapport de classification
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Conversion en DataFrame
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Calculer le F1-score moyen pondéré (on peut l'extraire du rapport)
    weighted_f1 = report_dict['weighted avg']['f1-score']
    
    # Supprimer la ligne 'accuracy' 
    report_df_cleaned = report_df.drop(['accuracy'], errors='ignore')
    
    # Sélectionner uniquement les colonnes pertinentes
    metrics_to_plot = ['precision', 'recall', 'f1-score']
    report_plot = report_df_cleaned[metrics_to_plot]
    
    # Créer une heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_plot, annot=True, cmap='YlGnBu', fmt=".2f", cbar=True)
    plt.title(f'Rapport de Classification - Précision, Rappel, F1-score\n({num_conv_layers} conv, lr={learning_rate}, k={kernel_size})')
    plt.xlabel('Métriques')
    plt.ylabel('Classes')
    plt.tight_layout()
    
    # Exporter en image PNG
    plt.savefig(f'metrics/classification_report_conv{num_conv_layers}_lr{learning_rate}_ep{epochs}_k{kernel_size}.png', dpi=300)
    plt.close()
    
    return accuracy, loss, weighted_f1, history

# Comparer les performances pour différents nombres de couches, learning rates, epochs et kernel sizes
def compare_models(conv_layers_list=[1, 2, 3, 4], learning_rates=[0.001], epochs_list=[10], kernel_sizes=[3]):
    results = []
    
    for num_conv in conv_layers_list:
        for lr in learning_rates:
            for ep in epochs_list:
                for k_size in kernel_sizes:
                    accuracy, loss, f1_score, _ = train_and_evaluate(num_conv, lr, ep, k_size)
                    results.append({
                        'num_conv_layers': num_conv,
                        'learning_rate': lr,
                        'epochs': ep,
                        'kernel_size': k_size,
                        'accuracy': accuracy,
                        'loss': loss,
                        'f1_score': f1_score
                    })
    
    # Créer un DataFrame avec les résultats
    results_df = pd.DataFrame(results)
    
    # Sauvegarder les résultats au format CSV
    results_df.to_csv('metrics/model_comparison_results.csv', index=False)
    
    # Visualiser les résultats en fonction du nombre de couches convolutives
    plt.figure(figsize=(15, 10))
    
    # Subplot pour l'accuracy en fonction du nombre de couches
    plt.subplot(2, 2, 1)
    for lr in learning_rates:
        for ep in epochs_list:
            for k_size in kernel_sizes:
                filtered_results = results_df[(results_df['learning_rate'] == lr) & 
                                             (results_df['epochs'] == ep) &
                                             (results_df['kernel_size'] == k_size)]
                plt.plot(filtered_results['num_conv_layers'], filtered_results['accuracy'], 'o-', 
                         label=f'lr={lr}, ep={ep}, k={k_size}')
    
    plt.title('Accuracy en fonction du nombre de couches convolutives')
    plt.xlabel('Nombre de couches convolutives')
    plt.ylabel('Accuracy')
    plt.xticks(conv_layers_list)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Subplot pour la loss
    plt.subplot(2, 2, 2)
    for lr in learning_rates:
        for ep in epochs_list:
            for k_size in kernel_sizes:
                filtered_results = results_df[(results_df['learning_rate'] == lr) & 
                                             (results_df['epochs'] == ep) &
                                             (results_df['kernel_size'] == k_size)]
                plt.plot(filtered_results['num_conv_layers'], filtered_results['loss'], 'o-', 
                         label=f'lr={lr}, ep={ep}, k={k_size}')
    
    plt.title('Loss en fonction du nombre de couches convolutives')
    plt.xlabel('Nombre de couches convolutives')
    plt.ylabel('Loss')
    plt.xticks(conv_layers_list)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Subplot pour le F1-score
    plt.subplot(2, 2, 3)
    for lr in learning_rates:
        for ep in epochs_list:
            for k_size in kernel_sizes:
                filtered_results = results_df[(results_df['learning_rate'] == lr) & 
                                             (results_df['epochs'] == ep) &
                                             (results_df['kernel_size'] == k_size)]
                plt.plot(filtered_results['num_conv_layers'], filtered_results['f1_score'], 'o-', 
                         label=f'lr={lr}, ep={ep}, k={k_size}')
    
    plt.title('F1-Score en fonction du nombre de couches convolutives')
    plt.xlabel('Nombre de couches convolutives')
    plt.ylabel('F1-Score')
    plt.xticks(conv_layers_list)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('metrics/model_comparison_by_layers.png', dpi=300)
    plt.close()
    
    # Si plusieurs tailles de kernel sont testées, visualiser les résultats en fonction de la taille du kernel
    if len(kernel_sizes) > 1:
        plt.figure(figsize=(15, 10))
        
        # Subplot pour l'accuracy en fonction de la taille du kernel
        plt.subplot(2, 2, 1)
        for lr in learning_rates:
            for ep in epochs_list:
                for num_conv in conv_layers_list:
                    filtered_results = results_df[(results_df['learning_rate'] == lr) & 
                                                 (results_df['epochs'] == ep) &
                                                 (results_df['num_conv_layers'] == num_conv)]
                    plt.plot(filtered_results['kernel_size'], filtered_results['accuracy'], 'o-', 
                             label=f'conv={num_conv}, lr={lr}, ep={ep}')
        
        plt.title('Accuracy en fonction de la taille du kernel')
        plt.xlabel('Taille du kernel')
        plt.ylabel('Accuracy')
        plt.xticks(kernel_sizes)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Subplot pour la loss
        plt.subplot(2, 2, 2)
        for lr in learning_rates:
            for ep in epochs_list:
                for num_conv in conv_layers_list:
                    filtered_results = results_df[(results_df['learning_rate'] == lr) & 
                                                 (results_df['epochs'] == ep) &
                                                 (results_df['num_conv_layers'] == num_conv)]
                    plt.plot(filtered_results['kernel_size'], filtered_results['loss'], 'o-', 
                             label=f'conv={num_conv}, lr={lr}, ep={ep}')
        
        plt.title('Loss en fonction de la taille du kernel')
        plt.xlabel('Taille du kernel')
        plt.ylabel('Loss')
        plt.xticks(kernel_sizes)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Subplot pour le F1-score
        plt.subplot(2, 2, 3)
        for lr in learning_rates:
            for ep in epochs_list:
                for num_conv in conv_layers_list:
                    filtered_results = results_df[(results_df['learning_rate'] == lr) & 
                                                 (results_df['epochs'] == ep) &
                                                 (results_df['num_conv_layers'] == num_conv)]
                    plt.plot(filtered_results['kernel_size'], filtered_results['f1_score'], 'o-', 
                             label=f'conv={num_conv}, lr={lr}, ep={ep}')
        
        plt.title('F1-Score en fonction de la taille du kernel')
        plt.xlabel('Taille du kernel')
        plt.ylabel('F1-Score')
        plt.xticks(kernel_sizes)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('metrics/model_comparison_by_kernel.png', dpi=300)
        plt.close()
    
    # Afficher un résumé des résultats
    print("\n=== Résultats de la comparaison ===")
    print(results_df)
    
    print("\nMeilleur modèle selon l'accuracy:")
    best_model_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"Nombre de couches convolutives: {best_model_accuracy['num_conv_layers']}")
    print(f"Learning rate: {best_model_accuracy['learning_rate']}")
    print(f"Epochs: {best_model_accuracy['epochs']}")
    print(f"Taille du kernel: {best_model_accuracy['kernel_size']}")
    print(f"Accuracy: {best_model_accuracy['accuracy']}")
    print(f"F1-Score: {best_model_accuracy['f1_score']}")
    
    print("\nMeilleur modèle selon le F1-Score:")
    best_model_f1 = results_df.loc[results_df['f1_score'].idxmax()]
    print(f"Nombre de couches convolutives: {best_model_f1['num_conv_layers']}")
    print(f"Learning rate: {best_model_f1['learning_rate']}")
    print(f"Epochs: {best_model_f1['epochs']}")
    print(f"Taille du kernel: {best_model_f1['kernel_size']}")
    print(f"Accuracy: {best_model_f1['accuracy']}")
    print(f"F1-Score: {best_model_f1['f1_score']}")
    
    return results_df

# Configuration principale - modifiez ces valeurs selon vos besoins
if __name__ == "__main__":
    # Liste des nombres de couches convolutives à tester
    conv_layers_to_test = [2]
    
    # Liste des learning rates à tester
    learning_rates_to_test = [0.1, 0.01, 0.001, 0.0001, 0.00001 ]
    
    # Liste des nombres d'epochs à tester
    epochs_to_test = [10]
    
    # Liste des tailles de kernel à tester
    kernel_sizes_to_test = [3]
    
    # Lancer la comparaison
    compare_models(conv_layers_to_test, learning_rates_to_test, epochs_to_test, kernel_sizes_to_test)