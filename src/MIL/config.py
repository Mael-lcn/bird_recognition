import argparse
import multiprocessing



def parse_arguments():
    """Parser pour les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 : SOTA Tabulaire (MIL Disentangled)")

    parser.add_argument("--feature-mode", type=str, choices=["tabular", "perch"], default="tabular", 
                        help="Choix de l'extracteur : 'tabular' (MFCC, etc.) ou 'perch' (Embeddings 1536d)")

    parser.add_argument("--focal-meta", type=str, default="../../../output/train_focal_cleaned.csv")
    parser.add_argument("--focal-audio", type=str, default="../../../data/birdclef-2026/train_audio/")

    parser.add_argument("--soundscape-labels", type=str, default="../../../output/train_soundscapes_cleaned.csv")
    parser.add_argument("--soundscape-audio", type=str, default="../../../data/birdclef-2026/train_soundscapes/")
    parser.add_argument("--vad-threshold", type=float, default=0.005, help="Seuil RMS pour ignorer le silence (VAD)")

    parser.add_argument('-o', "--output", type=str, default="../../../output/res")

    parser.add_argument("--em-iter", type=int, default=3, help="Itérations Expectation-Maximization")
    parser.add_argument("--window-sec", type=int, default=5, help="Taille du chunk en secondes")
    parser.add_argument("--stride-sec", type=int, default=5, help="Saut temporel (overlap)")
    parser.add_argument('-w', "--workers", type=int, default=max(1, multiprocessing.cpu_count()-1))

    return parser.parse_args()
