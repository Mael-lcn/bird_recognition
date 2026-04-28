import argparse
import multiprocessing



def parse_arguments():
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 : Pipeline MIL et Early Fusion.")
                                                            # "../../../output/train_cleaned.csv"
    parser.add_argument('-i', "--input", type=str, default="../../../data/birdclef-2026/train.csv")
    parser.add_argument('-a', "--audio-dir", type=str, default="../../../data/birdclef-2026/train_audio/")
    parser.add_argument('-o', "--output", type=str, default="../../../outputs")

    # Paramètres MIL et K-Fold
    parser.add_argument('-k', "--n-splits", type=int, default=5, help="Nombre de plis pour la validation croisée finale.")
    parser.add_argument("--em-iter", type=int, default=3, help="Nombre d'itérations pour l'algorithme MIL/EM.")
    parser.add_argument("--top-n", type=int, default=2, help="Nombre de fenêtres à conserver par fichier (Pooling).")
    
    parser.add_argument('-w', "--workers", type=int, default=multiprocessing.cpu_count()-1, help="Coeurs CPU alloués.")
    parser.add_argument("--verbose", action="store_true", default=False)
    
    return parser.parse_args()
