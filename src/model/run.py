import pandas as pd
import joblib
from pathlib import Path
from config import parse_arguments
from dataset import build_dataset_mil, prepare_for_mil
from models import execute_mil_and_cv
from metrics import print_global_metrics, generate_class_report



def main():
    """
    Exécute le flux de travail complet de préparation et d'entraînement.
    
    Les étapes principales sont :
    1. Analyse des arguments en ligne de commande.
    2. Création de l'architecture des données (Bag of Windows).
    3. Encodage des étiquettes (labels textuels vers entiers).
    4. Entraînement via Expectation-Maximization (MIL) couplé à un méta-modèle.
    5. Génération des rapports de performance globaux et détaillés.
    6. Exportation des artefacts (modèles et décodeur) pour l'inférence.
    """
    # Récupération de la configuration définie par l'utilisateur via le terminal
    args = parse_arguments()

    # Création sécurisée du répertoire de destination pour les artefacts
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Chargement des métadonnées du jeu de données (labels, géographie, etc.)
    df_meta = pd.read_csv(args.input)

    # Étape 1 : Parallélisation de l'extraction des caractéristiques (Audio + Géo)
    # Découpe chaque fichier audio en multiples fenêtres indépendantes
    df_windows = build_dataset_mil(df_meta, args.audio_dir, args.workers)

    # Étape 2 : Numérisation de la variable cible
    # L'encodeur transforme les noms d'espèces en identifiants numériques
    df_prepared, encoder = prepare_for_mil(df_windows)

    # Étape 3 : Apprentissage et Validation
    # Application de l'algorithme MIL (Expectation-Maximization) pour nettoyer 
    # le jeu de données, suivi d'un Stacking avec validation croisée
    final_model, y_clean, oof_preds, oof_proba = execute_mil_and_cv(
        df_prepared, 
        encoder, 
        args.em_iter, 
        args.top_n, 
        args.n_splits
    )

    # Étape 4 : Évaluation et restitution des performances
    # Les métriques sont calculées sur la base des prédictions "Out-Of-Fold"
    # uniquement sur les fenêtres considérées comme valides par le modèle (y_clean)
    print_global_metrics(y_clean, oof_preds, oof_proba, encoder.classes_)

    generate_class_report(
        y_clean, 
        oof_preds, 
        oof_proba, 
        encoder, 
        f"{args.output}/rapport_mil.csv", 
        args.verbose
    )

    # Étape 5 : Préservation des résultats
    # Sauvegarde du méta-modèle et de l'encodeur de labels pour une utilisation 
    # ultérieure dans le script d'inférence (predict.py)
    joblib.dump(final_model, f"{args.output}/final_xgb_model.joblib")
    joblib.dump(encoder, f"{args.output}/label_encoder.joblib")

    print(f"\n[+] Modèle et Encodeur sauvegardés dans {args.output}/")


if __name__ == "__main__":
    main()
