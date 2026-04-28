import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from features import worker_wrapper_mil



def build_dataset_mil(df_meta, audio_dir, n_workers):
    """
    Extrait des fenêtres audio en parallèle pour l'apprentissage 
    Multiple Instance Learning (MIL).

    Args:
        df_meta: DataFrame contenant les métadonnées des fichiers audio à traiter.
        audio_dir: Chemin vers le répertoire racine contenant les fichiers audio.
        n_workers: Nombre de coeurs CPU à allouer pour le traitement en parallèle.
        
    Returns:
        Un DataFrame contenant l'ensemble des fenêtres extraites et aplaties, 
        prêt pour l'entraînement ou l'évaluation.
    """
    print(f"[*] Découpage audio (Bag of Windows) sur {n_workers} coeurs...")

    # Préparation des tâches contenant les arguments nécessaires pour chaque worker
    tasks = [(idx, row, audio_dir) for idx, row in df_meta.iterrows()]

    # Lancement de l'exécution parallèle
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(executor.map(worker_wrapper_mil, tasks), total=len(tasks)))

    # Aplatissement de la liste de listes (chaque fichier audio retourne N fenêtres)
    flat_results = [item for sublist in results for item in sublist]
    df_windows = pd.DataFrame(flat_results)

    print(f"[*] Dataset brut généré : {len(df_windows)} fenêtres issues de {df_windows['file_id'].nunique()} fichiers.")

    return df_windows


def prepare_for_mil(df_windows):
    """
    Initialise l'encodeur de labels et prépare la structure cible pour
    l'algorithme MIL.
    
    Args:
        df_windows: DataFrame contenant les fenêtres audio et leurs labels textuels.
            
    Returns:
        Un tuple contenant :
            - Le DataFrame original enrichi de la nouvelle colonne numérique 'target'.
            - L'instance ajustée de l'encodeur (LabelEncoder) pour d'éventuelles 
              transformations inverses.
    """
    print("[*] Encodage des classes pour l'algorithme MIL.")

    # Instanciation et ajustement de l'encodeur sur les labels textuels
    encoder = LabelEncoder()
    df_windows['target'] = encoder.fit_transform(df_windows['label'])

    # Nous laissons l'algorithme MIL nettoyer les données AVANT de faire un K-Fold
    return df_windows, encoder
