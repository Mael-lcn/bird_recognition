import argparse
import ast
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



def setup_ds_env(output_dir: Path):
    """Initialisation de l'espace de travail."""
    for sub in ["plots", "tables"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("BirdCLEF_Final_Report")



def load_unified_data(input_dir: Path):
    """Charge et fusionne tout le contexte sonore du challenge."""
    train = pd.read_csv(input_dir / "train.csv")
    taxonomy = pd.read_csv(input_dir / "taxonomy.csv")
    ss = pd.read_csv(input_dir / "train_soundscapes_labels.csv")
    
    tax_map = taxonomy.set_index('primary_label')['common_name'].to_dict()

    def parse_labels(x):
        if pd.isna(x) or str(x).strip() in ["", "[]"]: return []
        try: return ast.literal_eval(x) if "[" in str(x) else str(x).split(';')
        except: return []

    # Unification pour la tâche Landscape
    train_labels = train['primary_label'].tolist() + [l for sub in train['secondary_labels'].apply(parse_labels) for l in sub]
    ss_labels = [l for sub in ss['primary_label'].fillna("").str.split(';') for l in sub]
    
    global_names = [tax_map.get(str(i).strip(), str(i)) for i in (train_labels + ss_labels) if str(i).strip()]
    
    return pd.Series(global_names), ss, taxonomy


def plot_distribution_landscape(series, output_dir, dpi=300):
    """
    PLOT 1: DISTRIBUTION EN LONGUE TRAÎNE (LONG TAIL)
    Visualisation par aire remplie (Area Chart). 
    L'axe X représente les espèces individuelles triées par abondance.
    """
    counts = series.value_counts().sort_values(ascending=False)
    y = counts.values
    x = np.arange(len(y))

    plt.figure(figsize=(16, 7))

    # Rendu par aire pour éviter l'effet de barres illisibles sur 200+ classes
    plt.fill_between(x, y, color="#2c3e50", alpha=0.7, label="Fréquence d'apparition")
    plt.plot(x, y, color="#2c3e50", linewidth=1.5)

    # Échelle logarithmique pour visualiser les classes rares
    plt.yscale("log")

    plt.xlim(0, len(x) - 1)
    plt.ylim(1, y.max() * 1.5)

    # PRÉCISION AXE X : Chaque unité est une espèce
    plt.xlabel(f"Axe X : Espèces détectées (1 unité = 1 espèce unique | Total : {len(y)} espèces)", fontsize=13)
    plt.ylabel("Nombre d'occurrences (Échelle Log)", fontsize=13)
    
    # Indicateurs statistiques
    plt.axhline(y.mean(), color="#e74c3c", linestyle="--", label=f"Moyenne ({int(y.mean())})")
    plt.axhline(np.median(y), color="#f1c40f", linestyle="-", label=f"Médiane ({int(np.median(y))})")

    plt.title("Analyse de la Longue Traîne (Paysage Global)", fontsize=16, fontweight='bold')
    plt.legend(frameon=True, loc='upper right')

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_dir / "plots/01_global_distribution.png", dpi=dpi)
    plt.close()


def plot_polyphony_density(ss, output_dir, dpi=300):
    """
    PLOT 2: DENSITÉ MULTI-LABEL
    Analyse de la polyphonie par segment de 5 secondes.
    """
    ss['n_species'] = ss['primary_label'].fillna("").str.split(';').apply(len)
    counts = ss['n_species'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    # Style épuré et pro
    ax = sns.barplot(x=counts.index, y=counts.values, palette="viridis", hue=counts.index, legend=False)
    
    total = len(ss)
    for p in ax.patches:
        ax.annotate(f'{100 * p.get_height() / total:.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height() + 5), 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color="#2c3e50")

    plt.title("Indice de Polyphonie (Nombre d'espèces simultanées)", fontsize=16, fontweight='bold')
    plt.xlabel("Espèces par segment audio (5s)", fontsize=13)
    plt.ylabel("Nombre de segments", fontsize=13)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_dir / "plots/02_polyphony_analysis.png", dpi=dpi)
    plt.close()

def plot_taxonomy_pie(taxonomy, output_dir, dpi=300):
    """
    PLOT 3: RÉPARTITION TAXONOMIQUE
    Camembert plein, classique et lisible.
    """
    counts = taxonomy['class_name'].value_counts()
    
    plt.figure(figsize=(10, 10))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, 
            colors=sns.color_palette("Set2"), textprops={'fontsize': 14, 'fontweight': 'bold'})
    
    plt.title("Diversité des Classes Biologiques", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "plots/03_taxonomy_pie.png", dpi=dpi)
    plt.close()



def main():
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Strategic EDA")
    parser.add_argument("-i", "--input_dir", default="../../../data/birdclef-2026/", help="Dossier Data")
    parser.add_argument("-o", "--output_dir", default="../../../output/eda/", help="Dossier Output")
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()
    out_path = Path(args.output_dir)
    log = setup_ds_env(out_path)

    log.info("Chargement et préparation des données unifiées...")
    full_series, ss_df, tax_df = load_unified_data(Path(args.input_dir))
    
    log.info("Génération du Plot 1: Area Distribution (Long Tail)...")
    plot_distribution_landscape(full_series, out_path, args.dpi)
    
    log.info("Génération du Plot 2: Polyphony Analysis...")
    plot_polyphony_density(ss_df, out_path, args.dpi)

    log.info("Génération du Plot 3: Full Taxonomy Pie...")
    plot_taxonomy_pie(tax_df, out_path, args.dpi)

    # Export des statistiques
    full_series.value_counts().to_csv(out_path / "tables/global_species_stats.csv")
    
    log.info(f"Livrables générés avec succès dans {out_path.absolute()}")

if __name__ == "__main__":
    main()
