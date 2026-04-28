import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, log_loss, f1_score, accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize



def print_global_metrics(y_true, oof_preds, oof_proba, le_classes):
    """
    Calcule et affiche un résumé des principales métriques de performance 
    (Out-Of-Fold) sur l'ensemble du jeu de données.

    Args:
        y_true: Les valeurs cibles réelles (vérité terrain).
        oof_preds: Les prédictions fermes (classes majoritairement prédites).
        oof_proba: Les probabilités continues prédites pour chaque classe.
        le_classes: La liste exhaustive des noms des classes.
    """
    # Création d'un tableau d'indices représentant les classes disponibles
    classes_array = np.arange(len(le_classes))
    
    # Binarisation des cibles réelles, nécessaire pour le calcul de la mAP multiclasses
    y_bin_global = label_binarize(y_true, classes=classes_array)

    # Calcul des métriques globales d'évaluation
    val_log_loss = log_loss(y_true, oof_proba, labels=classes_array)
    val_map = average_precision_score(y_bin_global, oof_proba, average='macro')
    val_f1 = f1_score(y_true, oof_preds, average='macro', zero_division=0)
    val_acc = accuracy_score(y_true, oof_preds)

    # Affichage formaté du rapport de performance
    print("\n" + "="*45)
    print("--- Rapport Synthétique (Out-Of-Fold) ---")
    print(f" Perte Logarithmique (Log Loss) : {val_log_loss:.4f}")
    print(f" Précision Moyenne (mAP)        : {val_map:.4f}")
    print(f" Score F1 (Moyenne Macro)       : {val_f1:.4f}")
    print(f" Exactitude (Accuracy)          : {val_acc:.4f}")
    print("="*45)


def generate_class_report(y_true, y_pred, y_proba, encoder, export_path=None, verbose=False):
    """
    Génère un tableau détaillé des performances pour chaque classe 
    individuelle, avec la possibilité d'exporter les résultats et de les afficher.

    Args:
        y_true: Les valeurs cibles réelles.
        y_pred: Les prédictions fermes.
        y_proba: Les probabilités continues prédites.
        encoder: L'instance de l'encodeur de labels (LabelEncoder) utilisée.
        export_path: Le chemin de destination optionnel pour l'export CSV.
        verbose: Un booléen indiquant s'il faut afficher un résumé dans le terminal.
    """
    print("[*] Compilation du rapport détaillé par entité...")

    # Génération du rapport de classification brut sous forme de dictionnaire
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics_list = []

    # Itération sur chaque classe connue par l'encodeur pour extraire ses métriques
    for class_idx, class_name in enumerate(encoder.classes_):
        idx_string = str(class_idx)

        # Sécurité : on ignore les classes qui n'apparaissent pas dans le rapport généré
        if idx_string not in report_dict: 
            continue

        # Création d'un masque pour isoler les échantillons appartenant réellement à cette classe
        mask_true = (y_true == class_idx)

        # Calcul de la probabilité de confiance moyenne allouée par le modèle aux vrais positifs
        mean_prob = np.mean(y_proba[mask_true, class_idx]) if mask_true.sum() > 0 else 0.0

        current_metrics = report_dict[idx_string]

        # Structuration des données de la classe
        metrics_list.append({
            'identifiant_espece': class_name,
            'score_f1': current_metrics['f1-score'],
            'precision': current_metrics['precision'],
            'rappel': current_metrics['recall'],
            'volume_support': current_metrics['support'],
            'probabilite_moyenne': mean_prob
        })

    # Conversion en DataFrame Pandas et tri descendant basé sur le score F1
    dataframe_report = pd.DataFrame(metrics_list).sort_values(by='score_f1', ascending=False)

    # Exportation optionnelle au format CSV
    if export_path:
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)
        dataframe_report.to_csv(export_path, index=False)
        print(f" > Fichier analytique enregistré sous : {export_path}")

    # Affichage optionnel des extrêmes (les meilleures et les pires classes)
    if verbose:
        print("\n--- Les 10 classes les plus performantes ---")
        print(dataframe_report.head(10).to_string(index=False))
        print("\n--- Les 10 classes nécessitant une optimisation ---")
        print(dataframe_report.tail(10).to_string(index=False))
