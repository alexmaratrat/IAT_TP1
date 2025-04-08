# Analyse d'Architectures CNN pour la Classification d'Obstacles

Ce projet permet de tester et comparer différentes architectures de réseaux de neurones convolutifs (CNN) pour la classification d'images d'obstacles. Il permet d'explorer l'impact de plusieurs hyperparamètres sur les performances des modèles.

## Fonctionnalités

- **Variation du nombre de couches convolutives** 
- **Variation du learning rate** 
- **Variation du nombre d'epochs** 
- **Variation de la taille des noyaux convolutifs** 
- **Visualisations automatiques** : Génération de graphiques de métriques d'entraînement, matrices de confusion et rapports de classification pour chaque modèle
- **Analyse comparative** : Exportation des résultats au format CSV pour des analyses approfondies

## Structure du projet

```
.
├── train.py                 # Script principal pour l'entraînement et l'évaluation
├── obstacles_dataset/       # Jeu de données (non inclus dans le repo)
│   ├── train/               # Images d'entraînement organisées en sous-dossiers par classe
│   └── test/                # Images de test organisées en sous-dossiers par classe
├── models/                  # Modèles sauvegardés (générés après exécution)
└── metrics/                 # Visualisations et rapports (générés après exécution)
    ├── training_metrics_*.png           # Graphiques d'accuracy et loss
    ├── confusion_matrix_*.png           # Matrices de confusion
    ├── classification_report_*.png      # Heatmaps des rapports de classification
    ├── model_comparison_by_layers.png   # Comparaison par nombre de couches
    ├── model_comparison_by_kernel.png   # Comparaison par taille de kernel
    └── model_comparison_results.csv     # Résultats complets au format CSV
```

## Prérequis

- Python 3.6 ou supérieur
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn


## Utilisation

1. Organisez votre jeu de données dans le dossier `obstacles_dataset` avec une structure comme suit:
   ```
   obstacles_dataset/
   ├── train/
   │   ├── classe1/
   │   ├── classe2/
   │   └── ...
   └── test/
       ├── classe1/
       ├── classe2/
       └── ...
   ```

2. Modifiez les paramètres de test dans la section principale du script `train.py`:
   ```python
   # Configuration principale
   if __name__ == "__main__":
       # Liste des nombres de couches convolutives à tester
       conv_layers_to_test = [2, 3, 4, 5, 6]
       
       # Liste des learning rates à tester
       learning_rates_to_test = [0.001, 0.0001]
       
       # Liste des nombres d'epochs à tester
       epochs_to_test = [10, 15]
       
       # Liste des tailles de kernel à tester
       kernel_sizes_to_test = [3, 5]
       
       # Lancer la comparaison
       compare_models(conv_layers_to_test, learning_rates_to_test, epochs_to_test, kernel_sizes_to_test)
   ```

3. Exécutez le script:
   ```bash
   python train.py
   ```

4. Analysez les résultats:
   - Consultez les graphiques générés dans le dossier `metrics/`
   - Explorez les données détaillées dans `metrics/model_comparison_results.csv`
   - Les modèles sont sauvegardés dans le dossier `models/` pour une utilisation ultérieure

## Exemple de résultats

Le script fournit automatiquement un résumé des meilleurs modèles dans la console:

```
=== Résultats de la comparaison ===
   num_conv_layers  learning_rate  epochs  kernel_size   accuracy     loss  f1_score
0                2          0.001      10            3  0.932407  0.247514  0.931275
1                2          0.001      10            5  0.940741  0.217669  0.940023
...

Meilleur modèle selon l'accuracy:
Nombre de couches convolutives: 4
Learning rate: 0.0001
Epochs: 15
Taille du kernel: 5
Accuracy: 0.968519
F1-Score: 0.967891

Meilleur modèle selon le F1-Score:
Nombre de couches convolutives: 4
Learning rate: 0.0001
Epochs: 15
Taille du kernel: 5
Accuracy: 0.968519
F1-Score: 0.967891
```

