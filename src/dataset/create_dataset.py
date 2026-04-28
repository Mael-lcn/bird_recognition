import pandas as pd
import argparse
import os
import time



def remove_spatial_outliers_by_species(df, multiplier=2.0):
    """
    Filtre les coordonnées géographiques aberrantes en utilisant la méthode de l'écart 
    interquartile (IQR), calculé spécifiquement pour chaque espèce.
    
    Args:
        df: DataFrame Pandas contenant les colonnes 'primary_label', 'latitude' et 'longitude'.
        multiplier: Facteur multiplicatif pour l'IQR. Une valeur de 2.0 cible les valeurs 
            extrêmement aberrantes afin de limiter les faux positifs lors de la suppression.

    Returns:
        Un tuple contenant :
            - Le DataFrame nettoyé.
            - Le nombre d'enregistrements supprimés lors de cette étape.
    """
    # Vérification de la présence des colonnes requises
    if not all(col in df.columns for col in ['primary_label', 'latitude', 'longitude']):
        return df, 0

    initial_len = len(df)

    for col in ['latitude', 'longitude']:
        # Calcul du premier et troisième quartile (Q1, Q3) pour chaque espèce
        Q1 = df.groupby('primary_label')[col].transform(lambda x: x.quantile(0.25))
        Q3 = df.groupby('primary_label')[col].transform(lambda x: x.quantile(0.75))
        IQR = Q3 - Q1

        # Définition des seuils de tolérance pour l'identification des valeurs aberrantes
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # Conservation des données comprises dans les seuils ou ayant des valeurs nulles
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound) | (df[col].isna())]

    return df, initial_len - len(df)


def filter_birdclef_csv(input_path, output_dir):
    """
    Orchestre le processus de nettoyage du jeu de données BirdCLEF. 
    Charge les données, supprime les valeurs manquantes et aberrantes, 
    puis sauvegarde le résultat tout en générant un rapport d'exécution.
    
    Args:
        input_path: Chemin vers le fichier CSV source.
        output_dir: Répertoire de destination pour le fichier nettoyé.
    """
    # Validation de l'existence du fichier d'entrée
    if not os.path.exists(input_path):
        print(f"Erreur : Le fichier d'entrée '{input_path}' n'existe pas.")
        return

    start_time = time.time()
    print(f"Chargement et filtrage de {input_path}...")

    df = pd.read_csv(input_path)
    initial_len = len(df)

    # Suppression des enregistrements ne possédant pas de coordonnées géographiques
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df.dropna(subset=['latitude', 'longitude'])

    # Application du filtre d'aberrations spatiales par espèce
    df, outliers_deleted = remove_spatial_outliers_by_species(df, multiplier=3.0)

    # Calcul des "métriques" pour le rapport final
    final_len = len(df)
    total_deleted = initial_len - final_len
    remaining_classes = df['primary_label'].nunique() if 'primary_label' in df.columns else "Inconnu"
    exec_time = time.time() - start_time

    # Création du répertoire de sortie s'il n'existe pas et sauvegarde du fichier
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "train_cleaned.csv")
    df.to_csv(output_path, index=False)

    # Affichage du rapport de nettoyage
    print("\n" + "="*50)
    print("Rapport de nettoyage")
    print("="*50)
    print(f"Lignes supprimées           : {total_deleted:,}".replace(',', ' '))
    print(f"Dont vrais Outliers GPS     : {outliers_deleted:,}".replace(',', ' '))
    print(f"Taille finale du dataset    : {final_len:,} lignes".replace(',', ' '))
    print(f"Classes restantes           : {remaining_classes}")
    print(f"Temps d'exécution           : {exec_time:.3f} s")
    print(f"Fichier sauvegardé          : {output_path}")
    print("="*50 + "\n")


def main():
    """
    Point d'entrée du script. 
    Gère l'analyse des arguments de la ligne de commande et lance le processus de filtrage.
    """
    parser = argparse.ArgumentParser(description="Nettoyage du dataset BirdCLEF.")
    parser.add_argument('-i', "--input", default="../../../data/birdclef-2026/train.csv")
    parser.add_argument('-o', "--output", default="../../../output")
    args = parser.parse_args()

    filter_birdclef_csv(args.input, args.output)


if __name__ == "__main__":
    main()
