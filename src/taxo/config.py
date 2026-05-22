import argparse
import multiprocessing



def parse_arguments():
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 : SOTA Hierarchical SVM")

    parser.add_argument("--taxonomy-path", type=str, default="../../ressources/taxo.json")
    parser.add_argument("--focal-meta", type=str, default="../../../output/train_focal_cleaned.csv")
    parser.add_argument("--focal-audio", type=str, default="../../../data/birdclef-2026/train_audio/")

    parser.add_argument("--soundscape-labels", type=str, default="../../../output/train_soundscapes_cleaned.csv")
    parser.add_argument("--soundscape-audio", type=str, default="../../../data/birdclef-2026/train_soundscapes/")
    
    parser.add_argument("--vad-threshold", type=float, default=0.005)
    parser.add_argument('-o', "--output", type=str, default="../../../output/res")
    parser.add_argument("--window-sec", type=int, default=5)
    parser.add_argument('-w', "--workers", type=int, default=max(1, multiprocessing.cpu_count()-1))

    return parser.parse_args()
