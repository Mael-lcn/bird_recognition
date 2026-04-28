import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
from features import extract_bag_of_windows



def predict_single_file(file_path, df_metadata, model_dict, le_classes):
    """
    Traite un fichier audio unique de bout en bout : extraction des fenêtres, 
    double inférence (Acoustique puis Stacking Géographique), et agrégation finale.

    Args:
        file_path: Objet Path pointant vers le fichier audio à analyser.
        df_metadata: DataFrame contenant les métadonnées de test (pour récupérer la géographie).
        model_dict: Dictionnaire contenant les modèles entraînés ('base' pour l'audio, 'meta' pour la fusion).
        le_classes: Liste exhaustive des noms de classes (espèces d'oiseaux).

    Returns:
        Un dictionnaire associant chaque nom d'espèce à sa probabilité maximale prédite 
        sur l'ensemble des fenêtres du fichier.
    """
    # Récupération de la ligne de métadonnées correspondant au fichier actuel
    meta_row = df_metadata[df_metadata['filename'] == file_path.name]

    # 1. Extraction Audio (Création du Bag of Windows)
    bag = extract_bag_of_windows(file_path)

    # Sécurité anti-plantage : si l'audio est silencieux, trop court ou illisible
    if not bag: 
        return {c: 0.0 for c in le_classes}

    df_windows = pd.DataFrame(bag)

    # Isolation des caractéristiques purement acoustiques
    # (Aligné avec l'entraînement : mfcc et flatness, on exclut le reste)
    audio_cols = [c for c in df_windows.columns if c.startswith('mfcc') or c == 'flatness']
    X_audio = df_windows[audio_cols]

    # 2. Inférence de Niveau 1 : Génération des probabilités par le modèle de base (Audio-only)
    base_probas = model_dict['base'].predict_proba(X_audio)

    # 3. Préparation pour le Niveau 2 : Injection des variables géographiques
    lat = np.float32(meta_row['latitude'].values[0])
    lon = np.float32(meta_row['longitude'].values[0])

    # Duplication des coordonnées géographiques pour chaque fenêtre extraite
    geo_features = np.array([[lat, lon]] * len(df_windows))

    # Stacking (Late Fusion) : Concaténation temporelle des probabilités et de la position GPS
    X_meta = np.hstack([base_probas, geo_features])

    # 4. Inférence de Niveau 2 : Le Méta-Modèle évalue le contexte global
    final_probas_matrice = model_dict['meta'].predict_proba(X_meta)

    # 5. Agrégation par Max-Pooling
    # On considère que si l'oiseau chante de manière évidente dans une seule fenêtre, 
    # alors il est présent dans le fichier complet.
    max_probas = np.max(final_probas_matrice, axis=0)

    # Mapping des probabilités vers les noms réels des espèces
    return {le_classes[i]: max_probas[i] for i in range(len(le_classes))}


def run_inference(test_audio_dir, test_csv, model_path, encoder_path, output_sub):
    """
    Orchestre la boucle d'inférence globale sur l'intégralité du dossier de test 
    et produit le fichier d'export au format requis (CSV).

    Args:
        test_audio_dir: Chemin du répertoire contenant les fichiers audio d'évaluation.
        test_csv: Chemin vers le fichier CSV contenant les métadonnées (latitudes/longitudes).
        model_path: Chemin du fichier binaire (joblib) contenant le dictionnaire de modèles XGBoost.
        encoder_path: Chemin du fichier binaire contenant le LabelEncoder.
        output_sub: Chemin du fichier CSV de sortie (ex: submission.csv).
    """
    print("[*] Lancement de l'inférence (Inference & Max-Pooling)...")

    # Restauration des artefacts d'entraînement (modèles et décodeur de classes)
    model_dict = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    classes = label_encoder.classes_

    # Chargement de la table de test et balayage des fichiers OGG
    df_test = pd.read_csv(test_csv)
    audio_files = list(Path(test_audio_dir).glob("*.ogg"))

    predictions = []

    # Boucle d'inférence avec barre de progression
    for file_path in tqdm(audio_files):
        preds = predict_single_file(file_path, df_test, model_dict, classes)

        # Ajout de l'identifiant du fichier requis par le format de soumission
        preds['row_id'] = file_path.name
        predictions.append(preds)

    # Compilation de toutes les prédictions dans un DataFrame Pandas
    df_sub = pd.DataFrame(predictions)

    # Réorganisation stricte des colonnes : 'row_id' en premier, suivi des noms d'espèces alphabétisés
    cols = ['row_id'] + list(classes)
    df_sub = df_sub[cols] 

    # Sauvegarde du fichier final sans index
    df_sub.to_csv(output_sub, index=False)
    print(f"[+] Submission générée : {output_sub}")
