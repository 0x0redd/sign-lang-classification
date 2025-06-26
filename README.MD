# Document : Pipeline de Reconnaissance d'Actions/Signes Vidéo utilisant des Points Clés et des Modèles Séquentiels

**Auteur :** (Dérivé du Notebook `model - final .ipynb`)
**Date :** (Basée sur les comptes d'exécution, probablement récente)

---

**Table des Matières :**

1.  [Idée Générale et Vue d'Ensemble](#1-idée-générale-et-vue-densemble)
    *   [1.1. Énoncé du Problème](#11-énoncé-du-problème)
    *   [1.2. Approche du Pipeline](#12-approche-du-pipeline)
    *   [1.3. Technologies Clés et Définitions](#13-technologies-clés-et-définitions)
2.  [Étape 1 : Installation et Configuration](#2-étape-1--installation-et-configuration)
    *   [2.1. Installation des Bibliothèques et Importations](#21-installation-des-bibliothèques-et-importations)
    *   [2.2. Paramètres de Configuration Globaux](#22-paramètres-de-configuration-globaux)
    *   [2.3. Reproductibilité](#23-reproductibilité)
3.  [Étape 2 : Préparation de l'Ensemble de Données (Optionnel : Données Factices)](#3-étape-2--préparation-de-lensemble-de-données-optionnel--données-factices)
    *   [3.1. Structure Attendue de l'Ensemble de Données](#31-structure-attendue-de-lensemble-de-données)
    *   [3.2. Création d'un Ensemble de Données Vidéo Factice](#32-création-dun-ensemble-de-données-vidéo-factice)
4.  [Étape 3 : Prétraitement - Extraction de Points Clés](#4-étape-3--prétraitement---extraction-de-points-clés)
    *   [4.1. Initialisation des Détecteurs (YOLO & MediaPipe)](#41-initialisation-des-détecteurs-yolo--mediapipe)
    *   [4.2. Fonction Auxiliaire : `extract_normalized_hand_kps`](#42-fonction-auxiliaire--extract_normalized_hand_kps)
    *   [4.3. Fonction Auxiliaire Principale : `process_frame_for_keypoints`](#43-fonction-auxiliaire-principale--process_frame_for_keypoints)
    *   [4.4. Boucle Principale de Prétraitement : `preprocess_videos_to_hdf5`](#44-boucle-principale-de-prétraitement--preprocess_videos_to_hdf5)
    *   [4.5. Stockage des Données HDF5](#45-stockage-des-données-hdf5)
5.  [Étape 4 : Chargement et Préparation des Données pour l'Entraînement](#5-étape-4--chargement-et-préparation-des-données-pour-lentraînement)
    *   [5.1. `KeypointSequenceDataset` (Dataset PyTorch)](#51-keypointsequencedataset-dataset-pytorch)
    *   [5.2. `KeypointDataModule` (DataModule PyTorch Lightning)](#52-keypointdatamodule-datamodule-pytorch-lightning)
    *   [5.3. Répartition des Données](#53-répartition-des-données)
6.  [Étape 5 : Définition du Modèle Séquentiel](#6-étape-5--définition-du-modèle-séquentiel)
    *   [6.1. `LSTMClassifier` (Module PyTorch Lightning)](#61-lstmclassifier-module-pytorch-lightning)
    *   [6.2. Architecture du Modèle (LSTM)](#62-architecture-du-modèle-lstm)
    *   [6.3. Fonction de Perte, Métriques et Optimiseur](#63-fonction-de-perte-métriques-et-optimiseur)
7.  [Étape 6 : Entraînement du Modèle](#7-étape-6--entraînement-du-modèle)
    *   [7.1. Callbacks (Sauvegarde de Points de Contrôle, Arrêt Précoce, Suivi du Taux d'Apprentissage)](#71-callbacks-sauvegarde-de-points-de-contrôle-arrêt-précoce-suivi-du-taux-dapprentissage)
    *   [7.2. Logger (CSVLogger)](#72-logger-csvlogger)
    *   [7.3. Configuration de l'Entraîneur PyTorch Lightning](#73-configuration-de-lentraîneur-pytorch-lightning)
    *   [7.4. Exécution de l'Entraînement](#74-exécution-de-lentraînement)
8.  [Étape 7 : Évaluation](#8-étape-7--évaluation)
    *   [8.1. Chargement du Meilleur Modèle](#81-chargement-du-meilleur-modèle)
    *   [8.2. Tracé des Courbes d'Entraînement](#82-tracé-des-courbes-dentraînement)
    *   [8.3. Évaluation sur l'Ensemble de Test](#83-évaluation-sur-lensemble-de-test)
    *   [8.4. Rapport de Classification et Matrice de Confusion](#84-rapport-de-classification-et-matrice-de-confusion)
9.  [Étape 8 : Inférence sur une Nouvelle Vidéo](#9-étape-8--inférence-sur-une-nouvelle-vidéo)
    *   [9.1. Sélection de Vidéo et Extraction de Points Clés](#91-sélection-de-vidéo-et-extraction-de-points-clés)
    *   [9.2. Prédiction](#92-prédiction)
    *   [9.3. Visualisation](#93-visualisation)
10. [Conclusion](#10-conclusion)

---

## 1. Idée Générale et Vue d'Ensemble

### 1.1. Énoncé du Problème
L'objectif est de reconnaître des actions ou des gestes de la langue des signes à partir d'une entrée vidéo. Cela implique le traitement des images vidéo, l'extraction de caractéristiques pertinentes, puis l'utilisation d'un modèle séquentiel pour classer l'ensemble de la séquence vidéo dans des catégories d'actions/signes prédéfinies.

### 1.2. Approche du Pipeline
Le notebook met en œuvre un pipeline en plusieurs étapes :
1.  **Prétraitement :** C'est une étape cruciale où les données vidéo brutes sont transformées en un format plus gérable et riche en caractéristiques.
    *   **Traitement Image par Image :** Les vidéos sont décomposées en images individuelles.
    *   **Détection de Région d'Intérêt (ROI) (Optionnel) :** YOLOv8 est utilisé pour détecter des personnes ou des mains dans chaque image. Cela aide à concentrer l'extraction de points clés sur les zones pertinentes, améliorant potentiellement la précision et l'efficacité.
    *   **Extraction de Points Clés :** MediaPipe Hands est ensuite appliqué (soit aux ROI, soit à l'image entière) pour extraire des points clés 2D (landmarks) pour un maximum de deux mains.
    *   **Normalisation :** Les points clés sont normalisés pour être indépendants de la taille et de la position de l'image.
    *   **Création de Séquence :** Pour chaque vidéo, un nombre fixe d'images (`NUM_FRAMES_PER_VIDEO`) est sélectionné (par remplissage ou troncature), et leurs vecteurs de points clés correspondants sont concaténés pour former une séquence.
2.  **Création de l'Ensemble de Données :** Les séquences de points clés extraites et leurs étiquettes correspondantes sont stockées dans un fichier HDF5 pour un chargement efficace pendant l'entraînement.
3.  **Modélisation Séquentielle :** Un réseau de neurones basé sur LSTM est entraîné pour apprendre des motifs à partir de ces séquences de points clés et les classer. Le notebook mentionne que des modèles GRU ou Transformer pourraient être substitués.
4.  **Évaluation & Inférence :** Le modèle entraîné est évalué sur un ensemble de test, et ses performances sont démontrées en faisant des prédictions sur une vidéo d'exemple.

### 1.3. Technologies Clés et Définitions
*   **YOLO (You Only Look Once) :** Un système de détection d'objets en temps réel. Dans ce notebook, `yolov8n.pt` (un petit modèle YOLOv8 rapide) est utilisé pour détecter les régions d'intérêt (mains ou personnes).
*   **MediaPipe :** Un framework de Google pour construire des pipelines d'apprentissage automatique appliqué multimodaux (par exemple, vidéo, audio). `mp.solutions.hands` est spécifiquement utilisé ici pour un suivi robuste des mains et l'extraction de 21 points clés (landmarks) par main.
*   **Points Clés (Landmarks) :** Points d'intérêt spécifiques sur un objet (par exemple, articulations des doigts, centre de la paume). Chaque point clé a des coordonnées (x, y, et parfois z et visibilité). Ici, des coordonnées 2D (x, y) sont utilisées.
*   **Normalisation :** Processus de mise à l'échelle des coordonnées des points clés (généralement entre 0 et 1) par rapport aux dimensions de l'image. Cela rend le modèle invariant à la résolution vidéo et à la position absolue des mains dans l'image.
*   **Séquence :** Un ensemble ordonné de points de données. Dans ce contexte, une séquence de vecteurs de points clés, où chaque vecteur représente les points clés des mains d'une image vidéo.
*   **HDF5 (Hierarchical Data Format 5) :** Un format de fichier conçu pour stocker et organiser de grandes quantités de données numériques. Il convient au stockage des séquences de points clés extraites.
*   **LSTM (Long Short-Term Memory) :** Un type d'architecture de Réseau de Neurones Récurrents (RNN) bien adapté à l'apprentissage à partir de données séquentielles, car il peut capturer des dépendances à longue portée.
*   **PyTorch :** Une bibliothèque d'apprentissage automatique open-source.
*   **PyTorch Lightning :** Un wrapper léger pour PyTorch qui organise le code PyTorch, facilitant l'entraînement et la mise à l'échelle des modèles, la gestion du matériel et la mise en œuvre des meilleures pratiques avec moins de code répétitif.
*   **Époque (Epoch) :** Un passage complet à travers l'ensemble des données d'entraînement.
*   **Taille de Lot (Batch Size) :** Le nombre d'exemples d'entraînement utilisés dans une itération (un passage avant/arrière).
*   **Taux d'Apprentissage (Learning Rate) :** Un hyperparamètre qui contrôle la modification du modèle en réponse à l'erreur estimée à chaque mise à jour des poids du modèle.
*   **Fonction de Perte (Loss Function) :** Mesure à quel point les prédictions du modèle correspondent aux vraies étiquettes (par exemple, `CrossEntropyLoss` pour la classification).
*   **Optimiseur (Optimizer) :** Un algorithme qui ajuste les poids du modèle pour minimiser la fonction de perte (par exemple, `AdamW`).
*   **Métriques (Metrics) :** Mesures quantitatives des performances du modèle (par exemple, exactitude (accuracy), score F1).

## 2. Étape 1 : Installation et Configuration

### 2.1. Installation des Bibliothèques et Importations
*   **Action :** La première cellule de code utilise `%pip install` pour s'assurer que toutes les bibliothèques Python nécessaires sont disponibles.
*   **Bibliothèques Installées :** `ultralytics`, `mediapipe`, `opencv-python`, `numpy`, `torch`, `torchvision`, `torchaudio`, `tqdm`, `scikit-learn`, `matplotlib`, `h5py`, `lightning`, `pandas`, `ipywidgets`.
*   **Importations :** Les modules Python standard (`os`, `shutil`, etc.) et les bibliothèques installées sont importés pour être utilisés dans tout le notebook.
*   **Réglage PyTorch :** `torch.set_float32_matmul_precision('medium')` est défini pour des multiplications matricielles potentiellement plus rapides sur du matériel compatible.

### 2.2. Paramètres de Configuration Globaux
*   **Action :** Une cellule de code dédiée définit divers paramètres globaux qui contrôlent le comportement du pipeline.
*   **Définitions des Chemins :**
    *   `RAW_VIDEO_DATA_DIR` : Chemin vers le répertoire contenant les fichiers vidéo bruts, organisés en sous-répertoires par classe.
    *   `PROCESSED_DATA_DIR` : Répertoire pour sauvegarder le fichier HDF5 traité.
    *   `H5_FILENAME` : Nom du fichier HDF5 (par exemple, "model.h5").
    *   `MODEL_SAVE_DIR` : Répertoire pour sauvegarder les points de contrôle du modèle entraîné.
*   **Paramètres de Prétraitement :**
    *   `YOLO_MODEL_NAME` : Spécifie le modèle YOLO à utiliser (par exemple, 'yolov8n.pt').
    *   `YOLO_CONF_THRESHOLD` : Confiance minimale pour les détections YOLO.
    *   `HAND_DETECTION_TARGET_CLASSES` : Index de classe COCO pour "personne" (utilisé comme ROI pour MediaPipe si un détecteur de main dédié n'est pas utilisé).
    *   `MAX_NUM_HANDS_MEDIAPIPE`, `MIN_DETECTION_CONF_MEDIAPIPE`, `MIN_TRACKING_CONF_MEDIAPIPE` : Configuration de MediaPipe Hands.
    *   `NUM_FRAMES_PER_VIDEO` : Nombre fixe d'images par séquence vidéo (remplissage/troncature appliqué).
    *   `NUM_KEYPOINTS_PER_HAND` : 21 pour MediaPipe Hands.
    *   `NUM_COORDS_PER_KEYPOINT` : 2 (pour x, y).
    *   `INPUT_SIZE` : Calculé comme `MAX_NUM_HANDS_MEDIAPIPE * NUM_KEYPOINTS_PER_HAND * NUM_COORDS_PER_KEYPOINT` (par exemple, 2 * 21 * 2 = 84 caractéristiques par image).
*   **Paramètres du Modèle & de l'Entraînement :**
    *   `SEQUENCE_MODEL_TYPE` : Défini sur "LSTM".
    *   Hyperparamètres LSTM : `HIDDEN_SIZE_LSTM`, `NUM_LAYERS_LSTM`, `DROPOUT_LSTM`.
    *   Paramètres d'Entraînement Communs : `BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY`, `EPOCHS`, `PATIENCE_EARLY_STOPPING`.
*   **Création de Répertoires :** Les répertoires spécifiés sont créés s'ils n'existent pas déjà.

### 2.3. Reproductibilité
*   **Action :** `RANDOM_SEED` est défini, et cette graine est appliquée à `random`, `numpy`, `torch`, et `pytorch_lightning.seed_everything()` pour s'assurer que les expériences peuvent être reproduites avec les mêmes résultats.
*   `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` est défini pour des opérations déterministes sur CUDA (GPU) si disponible.

## 3. Étape 2 : Préparation de l'Ensemble de Données (Optionnel : Données Factices)

### 3.1. Structure Attendue de l'Ensemble de Données
Le notebook s'attend à ce que les vidéos soient organisées dans `RAW_VIDEO_DATA_DIR` comme suit :

```
RAW_VIDEO_DATA_DIR /
├── nom_classe1 /
│ ├── video1.mp4
│ ├── video2.avi
├── nom_classe2 /
│ ├── video3.mp4
└── ... 
```


### 3.2. Création d'un Ensemble de Données Vidéo Factice
*   **Objectif :** Si `RAW_VIDEO_DATA_DIR` est vide, cette section crée un petit ensemble de données synthétiques à des fins de démonstration.
*   **Action :** La fonction `create_dummy_video` génère de simples vidéos MP4 contenant du texte en mouvement.
*   Le script vérifie si `RAW_VIDEO_DATA_DIR` contient des sous-répertoires. Si ce n'est pas le cas, il crée `DUMMY_CLASSES` (par exemple, "signe_A", "signe_B") et les remplit avec quelques vidéos factices.
*   **Note Utilisateur :** Les utilisateurs sont explicitement invités à remplacer cela par leur ensemble de données vidéo réel ou à pointer correctement `RAW_VIDEO_DATA_DIR`.

## 4. Étape 3 : Prétraitement - Extraction de Points Clés

### 4.1. Initialisation des Détecteurs (YOLO & MediaPipe)
*   **MediaPipe Hands :** `mp_hands = mp.solutions.hands` et `mp_drawing = mp.solutions.drawing_utils` sont initialisés.
*   **Détecteur YOLO :** `yolo_detector = YOLO(YOLO_MODEL_NAME)` charge le modèle YOLOv8 spécifié. Une gestion des erreurs est incluse au cas où le modèle ne se chargerait pas.

### 4.2. Fonction Auxiliaire : `extract_normalized_hand_kps`
*   **Objectif :** Extraire et aplatir les coordonnées (x, y) d'une liste d'objets landmark MediaPipe.
*   **Entrée :** Une liste d'objets landmark (chacun avec des attributs `.x` et `.y`, déjà normalisés par MediaPipe par rapport aux dimensions de l'image) et la largeur/hauteur de l'image (bien que non strictement utilisées dans cette version car les landmarks sont pré-normalisés).
*   **Sortie :** Un tableau NumPy plat de `[x1, y1, x2, y2, ..., xN, yN]` pour une seule main.

### 4.3. Fonction Auxiliaire Principale : `process_frame_for_keypoints`
Cette fonction (`process_frame_for_keypoints_fixed`, qui écrase l'espace réservé d'origine) est au cœur de l'extraction de caractéristiques par image.
*   **Entrée :** Une seule image vidéo (tableau NumPy), la `hands_solution` MediaPipe initialisée, et optionnellement le `yolo_model`.
*   **Étapes :**
    1.  **Détection ROI YOLO (Optionnel) :**
        *   Si `yolo_model` est fourni, il exécute `yolo_model.predict()` sur l'image pour détecter les objets spécifiés par `HAND_DETECTION_TARGET_CLASSES` (par exemple, "personne").
        *   Les boîtes englobantes (`x1, y1, x2, y2`) pour ces détections sont extraites et stockées comme `yolo_rois`. Ces ROI sont dessinées sur une copie annotée de l'image.
    2.  **Sélection de Région pour MediaPipe :**
        *   Si des `yolo_rois` sont trouvées, l'image est recadrée sur ces ROI. MediaPipe Hands sera exécuté sur ces régions plus petites.
        *   Si pas de ROI YOLO, ou si YOLO n'est pas utilisé, l'image entière est traitée par MediaPipe.
    3.  **Traitement MediaPipe Hands :**
        *   Pour chaque région sélectionnée (ROI recadrée ou image entière) :
            *   La région est convertie en RGB (MediaPipe attend du RGB).
            *   `hands_solution.process(img_region_rgb)` est appelé pour détecter les mains et leurs landmarks dans cette région.
    4.  **Extraction et Normalisation des Points Clés :**
        *   Si des mains sont détectées (`results.multi_hand_landmarks`) :
            *   Pour chaque main détectée dans la région :
                *   Les landmarks (`lm_roi.x`, `lm_roi.y`) sont initialement relatifs à la région traitée.
                *   Ces coordonnées relatives à la région sont converties en coordonnées de pixels absolues dans l'image entière en considérant le décalage de la région (`offset_x`, `offset_y`).
                *   Ces coordonnées de pixels absolues pour l'image entière sont ensuite normalisées en divisant par `image_width` et `image_height` pour obtenir des valeurs entre 0 et 1. Celles-ci sont stockées dans `temp_full_frame_landmarks_for_features`.
                *   `extract_normalized_hand_kps` est appelé sur ces points clés normalisés pour l'image entière pour obtenir le tableau plat (x,y) pour la main actuelle.
            *   Les landmarks sont dessinés sur `frame_annotated`.
    5.  **Assignation des Mains et Remplissage (Padding) :**
        *   Les points clés extraits sont assignés soit au tableau de la main gauche (index 0) soit à celui de la main droite (index 1) dans `keypoints_frame_hands`. Cette liste est pré-remplie de zéros.
        *   `multi_handedness` de MediaPipe est utilisé si disponible pour déterminer 'Gauche' ou 'Droite'. Sinon, ou si une main est déjà remplie, il remplit le prochain emplacement disponible jusqu'à `MAX_NUM_HANDS_MEDIAPIPE`.
        *   Cela garantit que `keypoints_frame_hands` contient toujours des points clés pour `MAX_NUM_HANDS_MEDIAPIPE` (par exemple, 2 mains), avec un remplissage par des zéros si moins de mains sont détectées.
    6.  **Concaténation :** Les tableaux de points clés pour toutes les mains (par exemple, gauche et droite) sont concaténés en un seul tableau NumPy plat (`final_keypoints_for_frame`) de taille `INPUT_SIZE`.
*   **Sortie :** `final_keypoints_for_frame` (le vecteur de caractéristiques plat pour l'image) et `frame_annotated` (l'image avec les dessins).

### 4.4. Boucle Principale de Prétraitement : `preprocess_videos_to_hdf5`
*   **Objectif :** Traiter toutes les vidéos de l'ensemble de données, extraire les séquences de points clés et les sauvegarder dans un fichier HDF5.
*   **Entrée :** Répertoire racine des vidéos (`video_dir_root`), chemin du fichier HDF5 de sortie (`output_h5_path`), nombre cible d'images (`num_frames_target`), et le modèle YOLO.
*   **Étapes :**
    1.  **Découverte des Vidéos et Étiquettes :** Parcourt `video_dir_root`, identifie les noms de classes (sous-répertoires) et crée un `label_map` (par exemple, {"signe_A": 0, "signe_B": 1}).
    2.  **Initialisation de MediaPipe Hands :** Crée une instance `mp_hands.Hands` avec les paramètres configurés. Ceci est fait en dehors de la boucle vidéo pour l'efficacité.
    3.  **Ouverture du Fichier HDF5 :** Ouvre `output_h5_path` en mode écriture ('w'). Le `label_map` est sauvegardé comme un attribut de chaîne JSON dans le fichier HDF5.
    4.  **Itération à Travers les Vidéos (avec barre de progression `tqdm`) :**
        *   Pour chaque `video_path` :
            *   Ouvrir la vidéo en utilisant `cv2.VideoCapture`.
            *   Initialiser `video_keypoints_list` pour stocker les points clés de chaque image de la vidéo actuelle.
            *   **Traitement Image par Image :**
                *   Lire les images jusqu'à la fin de la vidéo.
                *   Pour chaque `frame`, appeler `process_frame_for_keypoints()` pour obtenir `keypoints_single_frame`.
                *   Ajouter `keypoints_single_frame` à `video_keypoints_list_inf`.
            *   Libérer la capture vidéo.
            *   Si aucun point clé n'a été extrait, sauter la vidéo.
            *   Convertir `video_keypoints_list` en un tableau NumPy `video_keypoints_np`.
    5.  **Remplissage/Troncature de Séquence :**
        *   `current_num_frames = video_keypoints_np.shape[0]`.
        *   Si `current_num_frames > num_frames_target` : La séquence est tronquée en prenant une tranche du milieu : `video_keypoints_np[start : start + num_frames_target]`.
        *   Si `current_num_frames < num_frames_target` : La séquence est remplie avec des zéros (tableaux de forme `(padding_frames, INPUT_SIZE)`) à la fin.
        *   Le résultat est `processed_sequence` de forme `(num_frames_target, INPUT_SIZE)`.
    6.  **Sauvegarde en HDF5 :**
        *   Créer un `dataset_name` unique basé sur le chemin relatif de la vidéo.
        *   Créer un ensemble de données dans le groupe "sequences" du fichier HDF5 : `hf.create_dataset(dataset_name, data=processed_sequence)`.
        *   Stocker `label`, `original_path`, et `class_name` comme attributs de cet ensemble de données HDF5.
    7.  Fermer le fichier HDF5.
*   **Exécution Conditionnelle :** Le drapeau `run_preprocessing` et une vérification de l'existence du fichier HDF5 déterminent si cette étape potentiellement longue est exécutée.

### 4.5. Stockage des Données HDF5
Les données traitées sont stockées dans un fichier HDF5 avec la structure suivante :
*   Attributs Racine :
    *   `label_map` : Chaîne JSON mappant les noms de classes aux étiquettes entières.
*   Groupe "sequences" :
    *   Chaque vidéo devient un ensemble de données dans ce groupe.
    *   Nom de l'Ensemble de Données : Dérivé du chemin de la vidéo (par exemple, `nom_classe_nom_video_mp4`).
    *   Contenu de l'Ensemble de Données : Un tableau NumPy de forme `(NUM_FRAMES_PER_VIDEO, INPUT_SIZE)` contenant la séquence de points clés.
    *   Attributs de l'Ensemble de Données :
        *   `label` : Étiquette entière de la classe.
        *   `original_path` : Chemin de chaîne vers le fichier vidéo original.
        *   `class_name` : Nom de chaîne de la classe.

## 5. Étape 4 : Chargement et Préparation des Données pour l'Entraînement

### 5.1. `KeypointSequenceDataset` (Dataset PyTorch)
*   **Objectif :** Un `Dataset` PyTorch personnalisé pour charger des séquences de points clés individuelles et des étiquettes à partir du fichier HDF5.
*   **Héritage :** `torch.utils.data.Dataset`.
*   **`__init__(self, h5_file_path, dataset_keys)` :**
    *   Stocke le chemin du fichier HDF5 et une liste de `dataset_keys` (chaînes correspondant aux noms des ensembles de données dans le fichier HDF5 pour une répartition particulière comme entraînement/validation/test).
*   **`__len__(self)` :** Retourne le nombre total d'échantillons (vidéos) dans cette répartition de l'ensemble de données.
*   **`__getitem__(self, idx)` :**
    *   Ouvre le fichier HDF5 (en mode lecture).
    *   Récupère le `dataset_key` pour l'`idx` donné.
    *   Lit les données de la séquence de points clés (`hf['sequences'][dataset_key][:]`) et l'attribut d'étiquette (`hf['sequences'][dataset_key].attrs['label']`).
    *   Convertit les données de séquence en un `torch.FloatTensor` et l'étiquette en un `torch.LongTensor`.
    *   Retourne le tenseur de séquence et le tenseur d'étiquette.
*   **`__del__`, `__getstate__`, `__setstate__` :** Ceux-ci sont inclus pour gérer le handle du fichier HDF5, en particulier si `num_workers` dans `DataLoader` était supérieur à 0. Avec `num_workers=0`, leur impact est minime.

### 5.2. `KeypointDataModule` (DataModule PyTorch Lightning)
*   **Objectif :** Un `DataModule` PyTorch Lightning pour encapsuler toutes les étapes liées aux données : téléchargement (non utilisé ici), préparation des données, répartition et création des DataLoaders.
*   **Héritage :** `L.LightningDataModule`.
*   **`__init__(...)` :** Stocke le chemin HDF5, la taille du lot, les ratios de répartition validation/test et la graine aléatoire. Initialise `label_map`, `num_classes`, `inv_label_map`.
*   **`prepare_data(self)` :** Une méthode qui peut être utilisée pour une configuration unique (par exemple, téléchargement). Ici, elle vérifie simplement si le fichier HDF5 existe.
*   **`setup(self, stage=None)` :**
    *   Cette méthode est appelée sur chaque GPU en entraînement distribué.
    *   Ouvre le fichier HDF5 pour lire `all_dataset_keys` (tous les noms de vidéos) et le `label_map`.
    *   Détermine `num_classes`.
    *   Crée `inv_label_map` (étiquette entière vers nom de classe).
    *   **Répartition des Données :** Utilise `sklearn.model_selection.train_test_split` pour diviser `all_dataset_keys` en ensembles d'entraînement, de validation et de test.
        *   Il divise d'abord en `train_keys_val` et `test_keys`.
        *   Ensuite, il divise `train_keys_val` en `train_keys` et `val_keys`.
        *   La stratification (`stratify=...`) est utilisée si possible pour maintenir les proportions de classes dans les répartitions.
    *   Affiche les tailles des répartitions de l'ensemble de données.
*   **`train_dataloader(self)`, `val_dataloader(self)`, `test_dataloader(self)` :**
    *   Chaque méthode crée une instance `KeypointSequenceDataset` avec les clés appropriées pour sa répartition.
    *   Ensuite, elle retourne un `torch.utils.data.DataLoader` configuré avec l'ensemble de données, `batch_size`, le statut de mélange (True pour l'entraînement, False pour la validation/test), et `num_workers=0` (le chargement des données se fait dans le processus principal). `pin_memory=True` peut accélérer le transfert de données vers le GPU.
*   **Initialisation :** Une instance de `KeypointDataModule` est créée, et ses méthodes `prepare_data()` et `setup()` sont appelées.

### 5.3. Répartition des Données
Le `KeypointDataModule` gère la répartition des clés de l'ensemble de données (identifiants vidéo de HDF5) en trois ensembles :
*   **Ensemble d'Entraînement :** Utilisé pour entraîner le modèle.
*   **Ensemble de Validation :** Utilisé pour surveiller les performances du modèle pendant l'entraînement, ajuster les hyperparamètres et pour l'arrêt précoce.
*   **Ensemble de Test :** Utilisé pour l'évaluation finale et non biaisée du modèle entraîné.
La répartition est effectuée à l'aide de `train_test_split` de `scikit-learn`, avec une tentative de répartition stratifiée pour s'assurer que chaque classe est représentée proportionnellement dans chaque ensemble.

## 6. Étape 5 : Définition du Modèle Séquentiel

### 6.1. `LSTMClassifier` (Module PyTorch Lightning)
*   **Objectif :** Définit l'architecture du réseau de neurones (basée sur LSTM) et la logique d'entraînement/validation/test dans le framework PyTorch Lightning.
*   **Héritage :** `L.LightningModule`.
*   **`__init__(...)` :**
    *   `self.save_hyperparameters()` : Sauvegarde les arguments du constructeur (comme `input_size`, `hidden_size`) dans `self.hparams`, les rendant accessibles plus tard et les sauvegardant avec les points de contrôle.
    *   Définit les couches du modèle :
        *   `self.lstm = nn.LSTM(...)` : Une couche LSTM. `batch_first=True` signifie que les tenseurs d'entrée auront la forme `(batch, seq_len, features)`. Le dropout est appliqué si `num_layers > 1`.
        *   `self.fc = nn.Linear(hidden_size, num_classes)` : Une couche linéaire entièrement connectée pour mapper la sortie LSTM aux scores de classe.
    *   `self.criterion = nn.CrossEntropyLoss()` : La fonction de perte pour la classification multi-classes.
    *   Initialise `torchmetrics` (Accuracy et F1Score) pour l'entraînement, la validation et le test. `task="multiclass"` et `num_classes` sont spécifiés.
*   **`forward(self, x)` :** Définit le passage avant.
    *   `lstm_out, (hn, cn) = self.lstm(x)` : Passe l'entrée `x` à travers le LSTM. `hn` contient les états cachés du dernier pas de temps pour chaque couche.
    *   `out = self.fc(hn[-1])` : Prend l'état caché de la dernière couche au dernier pas de temps (`hn[-1]`) et le passe à travers la couche entièrement connectée pour obtenir les logits de classe.
*   **`_common_step(self, batch, batch_idx)` :** Une méthode auxiliaire pour éviter la duplication de code dans `training_step`, `validation_step`, et `test_step`.
    *   Désassemble le lot en entrée `x` et vraies étiquettes `y_true`.
    *   Effectue un passage avant : `logits = self(x)`.
    *   Calcule la perte : `loss = self.criterion(logits, y_true)`.
    *   Obtient les prédictions : `preds = torch.argmax(logits, dim=1)`.
    *   Retourne `loss, preds, y_true`.
*   **`training_step(self, batch, batch_idx)` :**
    *   Appelle `_common_step`.
    *   Calcule l'exactitude d'entraînement en utilisant `self.train_acc`.
    *   Enregistre `train_loss` et `train_acc` en utilisant `self.log()`. `prog_bar=True` les affiche sur la barre de progression.
    *   Retourne la perte.
*   **`validation_step(self, batch, batch_idx)` :**
    *   Similaire à `training_step`, mais utilise `self.val_acc` et `self.val_f1`.
    *   Enregistre `val_loss`, `val_acc`, et `val_f1`.
    *   Retourne la perte (souvent utilisée par les callbacks comme `ModelCheckpoint`).
*   **`test_step(self, batch, batch_idx)` :**
    *   Similaire à `validation_step`, en utilisant `self.test_acc` et `self.test_f1`.
    *   Enregistre les métriques de test.
    *   Retourne un dictionnaire contenant la perte, les prédictions et les cibles, qui peut être agrégé par PyTorch Lightning.
*   **`configure_optimizers(self)` :**
    *   Définit l'optimiseur : `torch.optim.AdamW` est utilisé, configuré avec `learning_rate` et `weight_decay` de `self.hparams`.
    *   Définit un planificateur de taux d'apprentissage : `torch.optim.lr_scheduler.ReduceLROnPlateau`. Ce planificateur réduit le taux d'apprentissage si `val_f1` (la métrique surveillée) cesse de s'améliorer pendant un certain nombre d'époques (`patience`).
    *   Retourne un dictionnaire spécifiant l'optimiseur et la configuration du planificateur LR.
*   **Initialisation du Modèle :** Une instance de `LSTMClassifier` est créée en utilisant les paramètres de la configuration globale et `data_module.num_classes`.

### 6.2. Architecture du Modèle (LSTM)
Le cœur du modèle séquentiel est une couche LSTM suivie d'une couche linéaire (entièrement connectée).
*   **Entrée du LSTM :** Un lot de séquences, chacune de forme `(NUM_FRAMES_PER_VIDEO, INPUT_SIZE)`.
*   **Sortie du LSTM :** Le LSTM traite la séquence et sort tous les états cachés pour tous les pas de temps, ainsi que l'état caché final (`hn`) et l'état de cellule (`cn`).
*   **Classification :** L'état caché final de la *dernière couche LSTM* (`hn[-1]`) est considéré comme un résumé de l'ensemble de la séquence d'entrée. Ce vecteur résumé est ensuite transmis à la couche linéaire (`self.fc`) pour produire des scores bruts (logits) pour chaque classe. Une fonction softmax (implicitement gérée par `CrossEntropyLoss` pendant l'entraînement, ou appliquée manuellement pendant l'inférence) convertit ces logits en probabilités.

### 6.3. Fonction de Perte, Métriques et Optimiseur
*   **Fonction de Perte :** `nn.CrossEntropyLoss` est utilisée, standard pour la classification multi-classes.
*   **Métriques :**
    *   `Exactitude (Accuracy)` : La proportion de séquences correctement classées.
    *   `Score F1 (macro)` : Le score F1 moyenné macro, qui est la moyenne non pondérée des scores F1 pour chaque classe. C'est une bonne métrique pour les ensembles de données déséquilibrés.
*   **Optimiseur :** `AdamW` (Adam avec décroissance du poids) est utilisé pour mettre à jour les poids du modèle.
*   **Planificateur de Taux d'Apprentissage :** `ReduceLROnPlateau` réduit de manière adaptative le taux d'apprentissage lorsque le score F1 de validation stagne, aidant le modèle à mieux converger.

## 7. Étape 6 : Entraînement du Modèle

### 7.1. Callbacks (Sauvegarde de Points de Contrôle, Arrêt Précoce, Suivi du Taux d'Apprentissage)
Les callbacks PyTorch Lightning sont utilisés pour ajouter un comportement personnalisé pendant l'entraînement :
*   **`ModelCheckpoint` :**
    *   Sauvegarde les points de contrôle du modèle pendant l'entraînement.
    *   `dirpath=MODEL_SAVE_DIR` : Spécifie où sauvegarder.
    *   `filename='best-action-model-{epoch:02d}-{val_f1:.2f}'` : Modèle de nommage pour les fichiers sauvegardés.
    *   `save_top_k=1` : Ne conserve que le meilleur point de contrôle.
    *   `monitor='val_f1'`, `mode='max'` : Surveille le score F1 de validation et sauvegarde le modèle lorsqu'il est à son maximum.
*   **`EarlyStopping` :**
    *   Arrête l'entraînement si la métrique surveillée (`val_f1`) ne s'améliore pas pendant un nombre spécifié d'époques (`PATIENCE_EARLY_STOPPING`).
    *   `mode='max'` : Indique qu'un `val_f1` plus élevé est meilleur.
*   **`LearningRateMonitor` :** Enregistre le taux d'apprentissage à chaque époque.

### 7.2. Logger (CSVLogger)
*   **`CSVLogger` :** Sauvegarde toutes les métriques enregistrées (perte, exactitude, F1, taux d'apprentissage) dans un fichier CSV (`metrics.csv`) dans un sous-répertoire de `MODEL_SAVE_DIR`. Ceci est utile pour une analyse et un tracé ultérieurs.

### 7.3. Configuration de l'Entraîneur PyTorch Lightning
*   Un objet `L.Trainer` est instancié. Cet objet gère la boucle d'entraînement, la gestion des appareils, les callbacks et l'enregistrement.
*   **Arguments Clés de l'Entraîneur :**
    *   `max_epochs=EPOCHS` : Nombre maximum d'époques d'entraînement.
    *   `accelerator="gpu" if torch.cuda.is_available() else "cpu"` : Utilise automatiquement le GPU s'il est disponible.
    *   `devices=1` : Utilise un GPU ou un CPU.
    *   `logger=csv_logger` : Utilise le logger CSV configuré.
    *   `callbacks=[...]` : Liste des callbacks à utiliser.
    *   `deterministic=True` : Vise des exécutions d'entraînement reproductibles (fonctionne en conjonction avec la définition de la graine).

### 7.4. Exécution de l'Entraînement
*   `trainer.fit(model, datamodule=data_module)` : Cette commande démarre le processus d'entraînement. L'`Trainer` va :
    *   Exécuter la boucle d'entraînement pour `max_epochs`.
    *   Appeler `training_step` et `validation_step` du `model`.
    *   Invoquer les callbacks aux moments appropriés.
    *   Enregistrer les métriques.
*   Après l'entraînement, le chemin vers le meilleur point de contrôle du modèle sauvegardé est affiché.

## 8. Étape 7 : Évaluation

### 8.1. Chargement du Meilleur Modèle
*   Le chemin vers le meilleur point de contrôle du modèle (sauvegardé par `ModelCheckpoint`) est récupéré.
*   `trained_model = LSTMClassifier.load_from_checkpoint(best_model_path)` : Le meilleur modèle est chargé à partir du fichier de point de contrôle.
*   `trained_model.eval()` : Met le modèle en mode évaluation (désactive le dropout, etc.).

### 8.2. Tracé des Courbes d'Entraînement
*   **Fonction `plot_training_curves(log_dir)` :**
    *   Lit `metrics.csv` généré par `CSVLogger`.
    *   Utilise `matplotlib` pour tracer :
        *   La perte d'entraînement et la perte de validation en fonction de l'époque.
        *   L'exactitude d'entraînement, l'exactitude de validation et le score F1 de validation en fonction de l'époque.
    *   Cela aide à visualiser la progression de l'apprentissage du modèle et à identifier des problèmes potentiels comme le surajustement.

### 8.3. Évaluation sur l'Ensemble de Test
*   `test_results = trainer.test(model=trained_model, datamodule=data_module, verbose=False)` :
    *   Exécute la boucle d'évaluation sur l'ensemble de test en utilisant le meilleur modèle chargé.
    *   L'`Trainer` appelle la méthode `test_step` du `model`.
    *   Retourne une liste de dictionnaires contenant les métriques de test (par exemple, `test_loss`, `test_acc`, `test_f1`).

### 8.4. Rapport de Classification et Matrice de Confusion
Comme `trainer.test()` ne retourne que des métriques agrégées, une boucle manuelle est effectuée pour obtenir des prédictions par échantillon pour une analyse détaillée :
1.  Obtenir le `test_dataloader` de `data_module`.
2.  Initialiser les listes `all_preds` et `all_targets`.
3.  Déplacer le `trained_model` vers l'appareil approprié (CPU/GPU).
4.  Itérer à travers `test_loader` (avec `torch.no_grad()` pour désactiver les calculs de gradient) :
    *   Obtenir un lot de données `x` et les vraies étiquettes `y_true`.
    *   Déplacer `x` vers l'appareil.
    *   Obtenir les prédictions du modèle (logits) : `logits = trained_model(x)`.
    *   Obtenir les indices de classe prédits : `preds = torch.argmax(logits, dim=1)`.
    *   Ajouter `preds` et `y_true` aux listes respectives.
5.  **Rapport de Classification :**
    *   `classification_report(all_targets, all_preds, target_names=...)` de `sklearn.metrics` est affiché. Cela montre la précision, le rappel, le score F1 et le support pour chaque classe.
6.  **Matrice de Confusion :**
    *   Fonction `display_confusion_matrix(y_true, y_pred, class_names)` :
        *   Calcule la matrice de confusion en utilisant `confusion_matrix()` de `sklearn.metrics`.
        *   La trace en utilisant `matplotlib.pyplot.imshow()` pour une représentation visuelle des classes vraies par rapport aux classes prédites.

## 9. Étape 8 : Inférence sur une Nouvelle Vidéo

*   **Objectif :** Démontrer comment le modèle entraîné peut être utilisé pour prédire l'action/le signe dans une nouvelle vidéo unique.
*   **Étapes :**
    1.  **Sélection de Vidéo :** Choisit aléatoirement une vidéo dans `RAW_VIDEO_DATA_DIR`.
    2.  **Extraction de Points Clés :**
        *   Ouvre la vidéo avec `cv2.VideoCapture`.
        *   Initialise `mp_hands.Hands` pour cette inférence.
        *   Lit la vidéo image par image. Pour chaque image :
            *   Appelle `process_frame_for_keypoints()` (en utilisant `yolo_detector` si disponible) pour obtenir le vecteur de points clés et une image annotée.
            *   Stocke le vecteur de points clés et l'image annotée.
    3.  **Préparation de Séquence :**
        *   La liste des vecteurs de points clés d'image est convertie en un tableau NumPy.
        *   Cette séquence est remplie ou tronquée à `NUM_FRAMES_PER_VIDEO`, similaire à l'étape de prétraitement.
    4.  **Prédiction :**
        *   La séquence traitée est convertie en un tenseur PyTorch, une dimension de lot est ajoutée (`unsqueeze(0)`).
        *   Le tenseur et le `trained_model` sont déplacés vers l'appareil correct.
        *   Le modèle est mis en mode évaluation (`trained_model.eval()`).
        *   Avec `torch.no_grad()`, le modèle fait une prédiction : `logits_inf = trained_model(sequence_tensor_inf)`.
        *   `torch.softmax` est appliqué aux logits pour obtenir les probabilités de classe.
        *   `torch.max` trouve la classe avec la plus haute probabilité (classe prédite) et sa confiance.
        *   L'index de classe prédit est converti en un nom de classe en utilisant `data_module.inv_label_map`.
    5.  **Sortie et Visualisation :**
        *   Le nom de la classe prédite et la confiance sont affichés.
        *   Quelques-unes des images annotées (avec la prédiction superposée) sont affichées en HTML en utilisant `IPython.display.HTML` et des images encodées en base64.

## 10. Conclusion
Le notebook se termine par une simple instruction d'affichage indiquant que son exécution est terminée. Il démontre avec succès un pipeline de bout en bout pour la reconnaissance d'actions vidéo utilisant YOLO pour la détection de ROI, MediaPipe pour l'extraction de points clés, et un modèle LSTM entraîné avec PyTorch Lightning pour la classification de séquences.
