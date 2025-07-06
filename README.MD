# 🖐️ Reconnaissance de Langue des Signes - Classification d'Actions

## 📋 Table des Matières

1. [Vue d'ensemble du Projet](#vue-densemble-du-projet)
2. [Architecture Technique](#architecture-technique)
3. [Méthodologie et Approche](#méthodologie-et-approche)
4. [Préprocessing des Données](#préprocessing-des-données)
5. [Architecture du Modèle](#architecture-du-modèle)
6. [Résultats et Performance](#résultats-et-performance)
7. [Application Web Flask](#application-web-flask)
8. [Installation et Utilisation](#installation-et-utilisation)
9. [Structure du Projet](#structure-du-projet)
10. [Technologies Utilisées](#technologies-utilisées)

---

## 🎯 Vue d'ensemble du Projet

Ce projet implémente un système de reconnaissance de langue des signes basé sur l'intelligence artificielle, capable de classifier des actions/signes à partir de vidéos. Le système utilise une combinaison de détection d'objets (YOLOv8), d'extraction de points clés (MediaPipe), et d'apprentissage profond (LSTM) pour identifier quatre classes d'objets informatiques : **clavier**, **disque dur**, **ordinateur**, et **souris**.

### 🎯 Objectifs
- Détecter et classifier automatiquement les signes de langue des signes
- Fournir une interface web intuitive pour l'upload et la prédiction
- Atteindre une précision élevée avec un modèle robuste
- Créer un pipeline complet de traitement vidéo

---

## 🏗️ Architecture Technique

### Pipeline de Traitement
```
Vidéo d'entrée → Détection YOLO → Extraction MediaPipe → Séquence LSTM → Prédiction
```

### Composants Principaux
1. **YOLOv8** : Détection des régions d'intérêt (mains/personnes)
2. **MediaPipe Hands** : Extraction de 21 points clés par main
3. **LSTM** : Modèle de séquence pour la classification temporelle
4. **Flask** : Interface web pour l'interaction utilisateur

---

## 🔬 Méthodologie et Approche

### 1. Détection et Localisation
- **YOLOv8n** : Modèle léger pour la détection de personnes/mains
- **Seuil de confiance** : 0.4 pour filtrer les détections faibles
- **ROI (Region of Interest)** : Extraction des zones contenant les mains

### 2. Extraction de Points Clés
- **MediaPipe Hands** : 21 points clés par main (x, y normalisés)
- **Support multi-mains** : Jusqu'à 2 mains simultanément
- **Coordonnées normalisées** : Adaptation à différentes résolutions

### 3. Traitement Temporel
- **Séquence fixe** : 30 frames par vidéo
- **Padding/Truncation** : Standardisation de la longueur
- **Features par frame** : 84 dimensions (2 mains × 21 points × 2 coordonnées)

---

## 📊 Préprocessing des Données

### Configuration des Paramètres
```python
# Paramètres MediaPipe
MAX_NUM_HANDS_MEDIAPIPE = 2
MIN_DETECTION_CONF_MEDIAPIPE = 0.5
MIN_TRACKING_CONF_MEDIAPIPE = 0.5

# Paramètres de séquence
NUM_FRAMES_PER_VIDEO = 30
NUM_KEYPOINTS_PER_HAND = 21
NUM_COORDS_PER_KEYPOINT = 2
INPUT_SIZE = 84  # 2 × 21 × 2
```

### Pipeline de Préprocessing
1. **Extraction vidéo** : Lecture frame par frame
2. **Détection YOLO** : Identification des ROIs
3. **Extraction MediaPipe** : Points clés des mains
4. **Normalisation** : Coordonnées relatives à l'image
5. **Séquençage** : Padding/truncation à 30 frames
6. **Stockage HDF5** : Format optimisé pour l'apprentissage

### Dataset Final
- **199 vidéos** au total
- **4 classes** : clavier, disque dur, ordinateur, souris
- **Répartition** : 139 train / 40 validation / 20 test
- **Format** : HDF5 avec métadonnées JSON

---

## 🧠 Architecture du Modèle

### LSTMClassifier
```python
class LSTMClassifier(L.LightningModule):
    def __init__(self, input_size=84, hidden_size=256, num_layers=2, 
                 num_classes=4, dropout=0.3, learning_rate=1e-3, weight_decay=1e-4):
```

### Caractéristiques du Modèle
- **Type** : LSTM bidirectionnel
- **Couches cachées** : 256 unités
- **Nombre de couches** : 2
- **Dropout** : 0.3 pour la régularisation
- **Paramètres** : 877K paramètres entraînables
- **Taille estimée** : 3.51 MB

### Optimisation
- **Optimiseur** : AdamW avec weight decay
- **Scheduler** : ReduceLROnPlateau
- **Monitoring** : F1-score macro pour la validation
- **Early stopping** : Patience de 10 époques

---

## 📈 Résultats et Performance

### Métriques Finales (Test Set)
```
Test Accuracy: 80.0%
Test F1-Score (Macro): 78.7%
Test Loss: 0.814
```

### Rapport de Classification Détaillé
```
              precision    recall  f1-score   support

     clavier       1.00      1.00      1.00         5
  disque_dur       0.57      0.80      0.67         5
  ordinateur       0.83      1.00      0.91         5
      souris       1.00      0.40      0.57         5

    accuracy                           0.80        20
   macro avg       0.85      0.80      0.79        20
weighted avg       0.85      0.80      0.79        20
```

### Analyse des Performances
- **Clavier** : Performance parfaite (100% precision/recall)
- **Ordinateur** : Excellente performance (91% F1-score)
- **Disque dur** : Performance moyenne (67% F1-score)
- **Souris** : Performance limitée (57% F1-score)

### Modèles Sauvegardés
- **Meilleur modèle** : `best-action-model-epoch=31-val_f1=0.92.ckpt`
- **F1-score validation** : 92%
- **Époques d'entraînement** : 31 (early stopping)

---

## 🌐 Application Web Flask

### Architecture de l'Application
```python
# Pipeline de prédiction dans app.py
def predict_sign(video_path):
    1. process_video_for_inference()  # Extraction des points clés
    2. Conversion en tensor          # Préparation pour le modèle
    3. Inference LSTM               # Prédiction
    4. Softmax                     # Probabilités
    5. Retour des résultats        # Classe + confiance
```

### Fonctionnalités
- **Upload vidéo** : Support MP4, AVI, MOV, MKV, WEBM
- **Taille maximale** : 16 MB
- **Prédiction temps réel** : Traitement immédiat
- **Nettoyage automatique** : Suppression des fichiers temporaires
- **Gestion d'erreurs** : Messages d'erreur détaillés

### Interface Utilisateur
- **Template HTML** : Interface moderne et responsive
- **Feedback visuel** : Affichage des résultats
- **Gestion des erreurs** : Messages utilisateur clairs

---

## 🚀 Installation et Utilisation

### Prérequis
```bash
# Python 3.9+
# CUDA (optionnel, pour GPU)
# 8GB+ RAM recommandé
```

### Installation
```bash
# 1. Cloner le repository
git clone <repository-url>
cd sign-lang-classification

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Vérifier les fichiers requis
python run_app.py
```

### Fichiers Requis
- `yolov8n.pt` : Modèle YOLO pré-entraîné
- `data-yolo/model.h5` : Dataset préprocessé
- `model_yolo/best-action-model-*.ckpt` : Modèle entraîné

### Lancement de l'Application
```bash
# Méthode 1 : Script de lancement
python run_app.py

# Méthode 2 : Direct
python app.py

# Accès : http://localhost:5000
```

---

## 📁 Structure du Projet

```
sign-lang-classification/
├── 📄 app.py                    # Application Flask principale
├── 📄 run_app.py               # Script de lancement avec vérifications
├── 📄 requirements.txt         # Dépendances Python
├── 📄 README.MD               # Documentation (ce fichier)
├── 📄 yolov8n.pt              # Modèle YOLO pré-entraîné
│
├── 📁 SignLanguageDataset/     # Dataset vidéo brut
│   ├── 📁 clavier/            # 50 vidéos
│   ├── 📁 disque_dur/         # 50 vidéos
│   ├── 📁 ordinateur/         # 50 vidéos
│   └── 📁 souris/             # 49 vidéos
│
├── 📁 data-yolo/              # Données préprocessées
│   └── 📄 model.h5            # Dataset HDF5
│
├── 📁 model_yolo/             # Modèles entraînés
│   ├── 📄 best-action-model-epoch=31-val_f1=0.92.ckpt
│   ├── 📄 best-action-model-epoch=13-val_f1=0.88-v3.ckpt
│   └── 📁 action_rec_logs/    # Logs d'entraînement
│
├── 📁 templates/              # Templates Flask
│   └── 📄 index.html          # Interface utilisateur
│
├── 📁 uploads/                # Fichiers temporaires
└── 📄 model - FINAL .ipynb    # Notebook d'entraînement
```

---

## 🛠️ Technologies Utilisées

### Intelligence Artificielle
- **PyTorch** : Framework de deep learning
- **PyTorch Lightning** : Organisation du code d'entraînement
- **Ultralytics YOLO** : Détection d'objets
- **MediaPipe** : Extraction de points clés
- **TorchMetrics** : Métriques d'évaluation

### Traitement de Données
- **OpenCV** : Traitement vidéo et image
- **NumPy** : Calculs numériques
- **H5Py** : Stockage de données optimisé
- **Pandas** : Manipulation de données

### Web et Interface
- **Flask** : Framework web
- **Werkzeug** : Utilitaires web
- **HTML/CSS/JavaScript** : Interface utilisateur

### Utilitaires
- **Scikit-learn** : Métriques et validation
- **Matplotlib** : Visualisation
- **TQDM** : Barres de progression

---

## 🔍 Détails Techniques Avancés

### Optimisations Implémentées
1. **Détection ROI** : YOLO pour réduire la zone de traitement
2. **Multi-mains** : Support simultané de 2 mains
3. **Normalisation** : Coordonnées relatives pour la robustesse
4. **Padding intelligent** : Gestion des vidéos de longueurs variables
5. **Early stopping** : Prévention du surapprentissage
6. **Learning rate scheduling** : Adaptation automatique du taux d'apprentissage

### Gestion de la Mémoire
- **Batch processing** : Traitement par lots pour les vidéos longues
- **Nettoyage automatique** : Suppression des fichiers temporaires
- **Optimisation GPU** : Utilisation efficace de la mémoire CUDA

### Robustesse
- **Gestion d'erreurs** : Traitement des cas d'échec
- **Validation des entrées** : Vérification des formats de fichiers
- **Fallback** : Détection sur image complète si YOLO échoue

---

## 📝 Notes de Développement

### Défis Rencontrés
1. **Synchronisation multi-mains** : Gestion des mains gauche/droite
2. **Variabilité temporelle** : Standardisation des séquences
3. **Performance temps réel** : Optimisation pour l'interface web
4. **Gestion mémoire** : Traitement de vidéos volumineuses

### Améliorations Futures
1. **Augmentation de données** : Plus de classes et d'exemples
2. **Architecture avancée** : Transformer ou attention mechanisms
3. **Temps réel** : Prédiction en streaming
4. **Interface mobile** : Application mobile native

---

## 📞 Support et Contact

Pour toute question ou problème :
- **Issues GitHub** : Signaler les bugs
- **Documentation** : Consulter ce README
- **Exemples** : Voir le notebook d'entraînement

---

*Projet développé avec ❤️ pour la reconnaissance de langue des signes*
