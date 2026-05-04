import pandas as pd
import argparse
import os
import ast



def clean_focal_data(df_focal, df_taxa):
    print("\n[*] Nettoyage des données focales (train_metadata.csv)...")
    initial_len = len(df_focal)
    valid_birds = set(df_taxa['primary_label'].astype(str).unique())


    def parse_all_labels(row):
        labels = [str(row['primary_label'])]

        # Extraction sécurisée des secondary_labels (souvent type string "['bird1', 'bird2']")
        sec_labels = row.get('secondary_labels', '[]')
        if pd.notna(sec_labels) and sec_labels != '[]':
            try:
                parsed_sec = ast.literal_eval(sec_labels)
                if isinstance(parsed_sec, list):
                    labels.extend([str(b) for b in parsed_sec])
            except (ValueError, SyntaxError):
                pass # Si le format est corrompu, on ignore

        # On ne garde que les oiseaux valides de la compétition et on déduplique
        valid_labels = list(set([b for b in labels if b in valid_birds]))
        return ';'.join(valid_labels) if valid_labels else "nocall"

    # Création d'une cible unifiée multi-label identique aux soundscapes
    df_focal['target_multi'] = df_focal.apply(parse_all_labels, axis=1)

    print(f"  -> Lignes conservées : {len(df_focal)} / {initial_len}")
    return df_focal



def clean_soundscape_data(df_snd, df_taxa):
    print("\n[*] Nettoyage des Soundscapes (train_soundscapes_labels.csv)...")
    initial_len = len(df_snd)
    valid_birds = set(df_taxa['primary_label'].astype(str).unique())

    df_snd['primary_label'] = df_snd['primary_label'].fillna("nocall")

    def filter_and_clean_labels(label_string):
        if str(label_string) == "nocall": return "nocall"
        birds = [b.strip() for b in str(label_string).split(';') if b.strip()]
        kept_birds = [b for b in birds if b in valid_birds]
        return ';'.join(kept_birds) if kept_birds else "nocall"

    # On renomme 'primary_label' en 'target_multi' pour matcher le Focal dataset
    df_snd['target_multi'] = df_snd['primary_label'].apply(filter_and_clean_labels)

    print(f"  -> Lignes conservées : {len(df_snd)} / {initial_len}")
    return df_snd



def analyze_rare_classes(df_focal, df_snd):
    print("\n" + "="*50)
    print("ALERTE CLASSES RARES (Basé sur Target Multi-Label)")
    print("="*50)

    all_birds = []

    # On compte tous les oiseaux (primary + secondary) dans le focal
    for labels in df_focal['target_multi']:
        if labels != "nocall": all_birds.extend(labels.split(';'))

    # Et dans les soundscapes
    for labels in df_snd['target_multi']:
        if labels != "nocall": all_birds.extend(labels.split(';'))

    total_counts = pd.Series(all_birds).value_counts()

    rare_birds = total_counts[total_counts <= 5]
    very_rare_birds = total_counts[total_counts == 1]

    print(f"[*] Espèces avec <= 5 occurrences totales : {len(rare_birds)}")
    print(f"[*] Espèces avec EXACTEMENT 1 occurrence  : {len(very_rare_birds)}")



def main():
    parser = argparse.ArgumentParser(description="Unification et Nettoyage des Datasets BirdCLEF")
    parser.add_argument("--focal", type=str, default="../../../data/birdclef-2026/train.csv")
    parser.add_argument("--soundscape", type=str, default="../../../data/birdclef-2026/train_soundscapes_labels.csv")
    parser.add_argument("--taxonomy", type=str, default="../../../data/birdclef-2026/taxonomy.csv")
    parser.add_argument("--out-dir", type=str, default="../../../output/")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_focal = pd.read_csv(args.focal)
    df_snd = pd.read_csv(args.soundscape)
    df_taxa = pd.read_csv(args.taxonomy)

    df_focal_clean = clean_focal_data(df_focal, df_taxa)
    df_snd_clean = clean_soundscape_data(df_snd, df_taxa)

    analyze_rare_classes(df_focal_clean, df_snd_clean)

    out_focal = os.path.join(args.out_dir, "train_focal_cleaned.csv")
    out_snd = os.path.join(args.out_dir, "train_soundscapes_cleaned.csv")
    
    df_focal_clean.to_csv(out_focal, index=False)
    df_snd_clean.to_csv(out_snd, index=False)
    
    print(f"\n[+] Succès ! Datasets sauvegardés dans {args.out_dir}")


if __name__ == "__main__":
    main()
