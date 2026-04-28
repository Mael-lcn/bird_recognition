import numpy as np
import librosa
from pathlib import Path



# Constantes de configuration pour le traitement du signal
SAMPLE_RATE = 32000
WINDOW_DURATION = 3
STEP_DURATION = 2


def extract_bag_of_windows(file_path):
    """
    Extrait un ensemble de fenêtres à partir d'un fichier audio, 
    en calculant des caractéristiques spectrales robustes pour chaque fenêtre.
    
    Cette fonction utilise le Per-Channel Energy Normalization (PCEN) couplé 
    aux coefficients MFCC pour extraire des signaux bioacoustiques optimaux.
    
    Args:
        file_path: Le chemin d'accès au fichier audio à analyser.
        
    Returns:
        Une liste de dictionnaires, où chaque dictionnaire représente une fenêtre 
        de temps et contient ses caractéristiques acoustiques (MFCC, flatness, énergie).
        Retourne None si l'audio est corrompu ou trop court.
    """
    try:
        # Chargement des 30 premières secondes de l'audio à la fréquence d'échantillonnage cible
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=30)

        # Rejet des fichiers dont la durée est inférieure à la taille d'une fenêtre
        if len(y) < SAMPLE_RATE * WINDOW_DURATION: 
            return None

        # 1. Calcul du spectrogramme de Mel avec des bornes de fréquences adaptées aux oiseaux
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmin=1000, fmax=14000)

        # 2. Application du filtre PCEN
        S_pcen = librosa.pcen(S, sr=sr, gain=0.8, bias=10, power=0.25, time_constant=0.06)

        # Conversion des durées (secondes) en nombre de trames (frames)
        window_size = librosa.samples_to_frames(WINDOW_DURATION * SAMPLE_RATE)
        hop_size = window_size // 2

        bag = []

        # Découpage du spectrogramme en fenêtres glissantes
        for start in range(0, S_pcen.shape[1] - window_size, hop_size):
            window = S_pcen[:, start:start+window_size]

            # Extraction des caractéristiques de base
            mfccs = librosa.feature.mfcc(S=window, n_mfcc=20)

            feat = {
                "window_start": start,
                "flatness": np.float32(np.mean(librosa.feature.spectral_flatness(S=window))),
                "heuristic_score": np.float32(np.max(window)) # L'énergie maximale sert de proxy pour la présence d'un chant
            }

            # Extraction des statistiques sur les coefficients MFCC
            for i in range(20):
                feat[f"mfcc_mean_{i}"] = np.float32(np.mean(mfccs[i]))
                feat[f"mfcc_max_{i}"] = np.float32(np.max(mfccs[i]))
                feat[f"mfcc_std_{i}"] = np.float32(np.std(mfccs[i]))

            bag.append(feat)

        return bag
        
    except Exception as e:
        print(f"Erreur avec {file_path}: {e}")
        return None


def worker_wrapper_mil(args):
    """
    Wrapper facilitant l'exécution parallèle du processus d'extraction.
    Prend en charge un fichier audio, extrait ses fenêtres, et enrichit chaque 
    fenêtre avec les métadonnées globales du fichier d'origine.
    
    Args:
        args: Un tuple contenant 3 éléments :
            - L'index de la ligne (ignoré ici).
            - Un dictionnaire ou objet Series Pandas contenant les métadonnées.
            - Le chemin du répertoire racine contenant les fichiers audio.
            
    Returns:
        Une liste de dictionnaires prêts à être intégrés dans un DataFrame global.
        Retourne une liste vide si le fichier n'a pas pu être traité.
    """
    _, row, audio_dir = args
    file_path = Path(audio_dir) / row['filename']

    # Extraction des caractéristiques brutes
    bag = extract_bag_of_windows(file_path)

    if not bag:
        return []

    # Injection des métadonnées contextuelles (labels et géographie) dans chaque fenêtre
    for window_feat in bag:
        window_feat['file_id'] = row['filename']
        window_feat['geo_latitude'] = np.float32(row['latitude'])
        window_feat['geo_longitude'] = np.float32(row['longitude'])
        window_feat['label'] = row['primary_label']
        
    return bag
