import argparse
import ast
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



class EDAConfig:
    """
    Configuration globale pour le pipeline EDA, feed par le CLI.
    """
    def __init__(self, target_col, lat_min, lat_max, lon_min, lon_max, 
                 min_samples_for_stats, ci_lower_bound, ci_upper_bound, 
                 top_n_classes_plot, top_n_secondary, plot_dpi, 
                 style_theme, style_context):
        """
        Initialise les paramètres de configuration du pipeline.
        """
        self.target_col = target_col
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.min_samples_for_stats = min_samples_for_stats
        self.ci_lower_bound = ci_lower_bound
        self.ci_upper_bound = ci_upper_bound
        self.top_n_classes_plot = top_n_classes_plot
        self.top_n_secondary = top_n_secondary
        self.plot_dpi = plot_dpi
        self.style_theme = style_theme
        self.style_context = style_context


class TeeLogger:
    """
    Intercepte stdout pour l'afficher simultanément dans la console 
    et logger.
    """
    def __init__(self, filepath):
        """
        Initialise le logger.
        
        Args:
            filepath: Le chemin complet vers le fichier de log.
        """
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        """Écrit un message dans la console et dans le fichier."""
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        """Force le vidage des buffers d'écriture."""
        self.terminal.flush()
        self.log.flush()


class BirdCLEF_EDA_Pipeline:
    """
    Orchestre l'EDA pour le dataset BirdCLEF.
    en se concentrant uniquement sur les données d'entraînement (anti sata leak).
    """
    def __init__(self, input_dir, output_dir, config):
        """
        Initialise le pipeline EDA.
        
        Args:
            input_dir: Objet Path pointant vers le dossier des CSV.
            output_dir: Objet Path pointant vers le dossier de destination.
            config: Instance de EDAConfig contenant les paramètres.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = config

        # Préparation des répertoires de sortie
        self.plots_dir = output_dir / "plots"
        self.tables_dir = output_dir / "tables"

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        # Configuration visuelle globale de Seaborn
        sns.set_theme(style=self.config.style_theme, context=self.config.style_context, font_scale=1.2)
        
        self.df_train = None
        self.df_taxonomy = None
        self.taxon_mapping = {}

    def load_data(self):
        """
        Charge les fichiers d'entraînement exclusifs (train.csv et taxonomy.csv) 
        et déclenche le prétraitement initial.
        
        Raises:
            FileNotFoundError: Si le fichier train.csv est introuvable.
        """
        print("\n[*] Chargement des données (Zéro Data Leakage : Entraînement pur)...")

        train_path = self.input_dir / "train.csv"
        taxonomy_path = self.input_dir / "taxonomy.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"Fichier critique manquant : {train_path}")

        self.df_train = pd.read_csv(train_path)
        print(f"  [+] train.csv chargé : {self.df_train.shape[0]} lignes, {self.df_train.shape[1]} colonnes")

        if taxonomy_path.exists():
            self.df_taxonomy = pd.read_csv(taxonomy_path)
            print(f"  [+] taxonomy.csv chargé : {self.df_taxonomy.shape[0]} espèces référencées")
            self._build_taxonomy_mapping()
        else:
            print("  [-] taxonomy.csv introuvable. Les IDs originaux seront conservés.")

        self.df_train = self._apply_bird_names(self.df_train)
        self._preprocess_train()

    def _build_taxonomy_mapping(self):
        """
        Génère un dictionnaire associant les codes identifiants des espèces 
        à leurs noms lisibles via le fichier de taxonomie.
        """
        tax_cols = self.df_taxonomy.columns.str.lower()

        code_col = None
        for col in ["species_code", "primary_label", "id", "species2021"]:
            if col in tax_cols:
                code_col = self.df_taxonomy.columns[tax_cols.get_loc(col)]
                break

        name_col = None
        for col in ["primary_com_name", "common_name", "english_name", "sci_name", "name"]:
            if col in tax_cols:
                name_col = self.df_taxonomy.columns[tax_cols.get_loc(col)]
                break

        if code_col and name_col:
            self.taxon_mapping = dict(zip(self.df_taxonomy[code_col], self.df_taxonomy[name_col]))
            print(f"  [->] Mapping taxinomique configuré : {code_col} -> {name_col}")

    def _apply_bird_names(self, df):
        """
        Traduit les codes d'espèces en noms complets dans le DataFrame.
        
        Args:
            df: Le DataFrame à enrichir.
            
        Returns:
            Le DataFrame avec la colonne cible mise à jour.
        """
        if "primary_label" in df.columns:
            if self.taxon_mapping:
                df[self.config.target_col] = df["primary_label"].map(self.taxon_mapping).fillna(df["primary_label"])
            else:
                df[self.config.target_col] = df.get("common_name", df["primary_label"])
        return df

    def _preprocess_train(self):
        """
        Convertit et mappe les colonnes complexes, comme les listes stockées 
        sous forme de texte dans 'secondary_labels'.
        """
        if "secondary_labels" in self.df_train.columns:
            def parse_and_map_secondary(x):
                if pd.isna(x) or str(x).strip() in {"", "[]", "nan"}:
                    return []
                try:
                    parsed = ast.literal_eval(x)
                    return [self.taxon_mapping.get(str(bird), str(bird)) for bird in parsed]
                except Exception:
                    return []

            self.df_train["secondary_labels_list"] = self.df_train["secondary_labels"].apply(parse_and_map_secondary)
            self.df_train["n_secondary"] = self.df_train["secondary_labels_list"].apply(len)

    def analyze_class_imbalance(self):
        """
        Examine la répartition des classes pour visualiser l'effet de longue traîne 
        et exporte les résultats.
        """
        print("\n[*] Génération : Distribution des classes (Long Tail)...")
        class_counts = self.df_train[self.config.target_col].value_counts()
        
        plt.figure(figsize=(16, 6))
        ax = sns.barplot(x=np.arange(len(class_counts)), y=class_counts.values, color="#2b8cbe")
        
        plt.title(f"Distribution des {len(class_counts)} espèces (Long Tail)", fontsize=14, fontweight="bold")
        plt.xlabel("Espèces (triées de la plus à la moins fréquente)")
        plt.ylabel("Nombre d'enregistrements d'entraînement")

        plt.axhline(class_counts.mean(), color="#e34a33", linestyle="--", label=f"Moyenne ({int(class_counts.mean())})")
        plt.axhline(class_counts.median(), color="orange", linestyle="-", label=f"Médiane ({int(class_counts.median())})")

        ax.set_xticks([])
        plt.legend()
        sns.despine()

        plt.tight_layout()
        plt.savefig(self.plots_dir / "01_class_imbalance_longtail.png", dpi=self.config.plot_dpi)
        plt.close()

        class_counts.to_csv(self.tables_dir / "species_frequencies.csv", header=["count"])

    def analyze_geography(self):
        """
        Trace une carte thermique (hexbin) mettant en évidence les zones de forte 
        densité de collecte de données audio.
        """
        print("[*] Génération : Cartographie de densité d'entraînement...")
        if not {"latitude", "longitude"}.issubset(self.df_train.columns):
            return

        geo_df = self.df_train.dropna(subset=["latitude", "longitude"])

        plt.figure(figsize=(12, 8))

        hb = plt.hexbin(geo_df["longitude"], geo_df["latitude"], gridsize=80, cmap="YlOrRd", bins='log', mincnt=1)
        cb = plt.colorbar(hb, label='Densité logarithmique des enregistrements')

        plt.title("Densité géographique de collecte (Données d'entraînement)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "02_geographical_density.png", dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()

    def generate_detailed_profiles(self):
        """
        Construit un profil synthétique pour chaque espèce comprenant la zone 
        géographique centrale et les métadonnées dominantes.
        """
        print("[*] Génération : Profils Types (Core & Catégories) par oiseau...")
        profiles = []

        cat_cols = ["author", "rating", "license"]
        num_cols = ["latitude", "longitude"]

        for class_name, group in self.df_train.groupby(self.config.target_col):
            n_obs = len(group)
            profile = {"Espèce": class_name, "Volume_Audios": n_obs}

            for col in num_cols:
                if col in group.columns:
                    s = pd.to_numeric(group[col], errors="coerce").dropna()
                    if len(s) >= self.config.min_samples_for_stats:
                        vmin = s.quantile(self.config.ci_lower_bound)
                        vmax = s.quantile(self.config.ci_upper_bound)
                        profile[f"{col}_core_min"] = round(vmin, 3)
                        profile[f"{col}_core_max"] = round(vmax, 3)
                    else:
                        profile[f"{col}_core_min"] = np.nan
                        profile[f"{col}_core_max"] = np.nan

            for col in cat_cols:
                if col in group.columns:
                    s = group[col].dropna()
                    if not s.empty:
                        top_val = s.value_counts().index[0]
                        top_pct = (s.value_counts().iloc[0] / n_obs) * 100
                        profile[f"top_{col}"] = top_val
                        profile[f"top_{col}_pct"] = round(top_pct, 1)
                    else:
                        profile[f"top_{col}"] = "N/A"
                        profile[f"top_{col}_pct"] = np.nan

            profiles.append(profile)

        df_profiles = pd.DataFrame(profiles).sort_values("Volume_Audios", ascending=False)
        df_profiles.to_csv(self.tables_dir / "species_master_profile.csv", index=False)

    def analyze_train_multilabel(self):
        """
        Évalue l'occurrence des espèces présentes en arrière-plan sonore.
        """
        print("[*] Génération : Analyse multi-labels (Bruit de fond Train)...")

        all_secondary = [bird for sublist in self.df_train.get("secondary_labels_list", []) for bird in sublist]
        if not all_secondary:
            return

        top_secondary = pd.Series(all_secondary).value_counts().head(self.config.top_n_secondary)
        top_secondary.to_csv(self.tables_dir / "top_background_species_train.csv", header=["count"])

        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=top_secondary.values[:self.config.top_n_classes_plot], 
            y=top_secondary.index[:self.config.top_n_classes_plot], 
            hue=top_secondary.index[:self.config.top_n_classes_plot],
            palette="viridis",
            legend=False
        )
        plt.title(f"Top {self.config.top_n_classes_plot} des espèces en arrière-plan", fontweight="bold")
        plt.xlabel("Occurrences en tant que bruit de fond")
        plt.ylabel("Espèce")

        plt.tight_layout()
        plt.savefig(self.plots_dir / "03_train_multilabel_background.png", dpi=self.config.plot_dpi)
        plt.close()


    def run(self):
        """
        Exécute séquentiellement toutes les étapes d'analyse de données.
        """
        self.load_data()
        self.analyze_class_imbalance()
        self.analyze_geography()
        self.generate_detailed_profiles()
        self.analyze_train_multilabel()
        print(f"\n[+] Terminé. Tous les artefacts sont disponibles dans : {self.output_dir.absolute()}")


def main():
    """Point d'entrée principal : gestion des arguments et lancement du script."""
    parser = argparse.ArgumentParser(description="Pipeline EDA Rigoureux & Strict (Zéro Data Leakage)")

    parser.add_argument("-i", "--input_dir", default="../../../data/birdclef-2026/", help="Dossier contenant les fichiers CSV")
    parser.add_argument("-o", "--output_dir", default="eda_output/", help="Dossier de destination")

    parser.add_argument("--target", default="bird_name", help="Colonne cible après mapping taxinomique")
    # Zone de record de recording_location
    parser.add_argument("--lat_min", type=float, default=-21.6, help="Latitude minimum")
    parser.add_argument("--lat_max", type=float, default=-16.5, help="Latitude maximum")
    parser.add_argument("--lon_min", type=float, default=-57.6, help="Longitude minimum")
    parser.add_argument("--lon_max", type=float, default=-55.9, help="Longitude maximum")

    parser.add_argument("--min_samples", type=int, default=4, help="Min. obs. pour intervalles")
    parser.add_argument("--ci_lower", type=float, default=0.10, help="Quantile inférieur")
    parser.add_argument("--ci_upper", type=float, default=0.90, help="Quantile supérieur")

    parser.add_argument("--top_classes_plot", type=int, default=10, help="Nombre de classes pour graphiques")
    parser.add_argument("--top_secondary", type=int, default=20, help="Nombre de labels secondaires")
    parser.add_argument("--dpi", type=int, default=300, help="Résolution DPI")
    parser.add_argument("--theme", default="whitegrid", help="Thème Seaborn")
    parser.add_argument("--context", default="paper", help="Contexte Seaborn")

    args = parser.parse_args()

    config = EDAConfig(
        target_col=args.target,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        min_samples_for_stats=args.min_samples,
        ci_lower_bound=args.ci_lower,
        ci_upper_bound=args.ci_upper,
        top_n_classes_plot=args.top_classes_plot,
        top_n_secondary=args.top_secondary,
        plot_dpi=args.dpi,
        style_theme=args.theme,
        style_context=args.context
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.stdout = TeeLogger(output_dir / "eda_execution.log")

    print("=" * 80)
    print("DÉMARRAGE DU PIPELINE EDA PROFESSIONNEL (STRICT ML METHODOLOGY)")
    print("=" * 80)

    pipeline = BirdCLEF_EDA_Pipeline(input_dir, output_dir, config)

    try:
        pipeline.run()
    except Exception as e:
        print(f"\n[!] ERREUR FATALE : {str(e)}")
        logging.exception("Détails de l'erreur :")
        sys.exit(1)


if __name__ == "__main__":
    main()
