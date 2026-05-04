import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, classification_report



def calculate_competition_metric(y_true, y_prob):
    idx = np.where(y_true.sum(0) > 0)[0]
    return roc_auc_score(y_true[:, idx], y_prob[:, idx], average='macro') if len(idx)>0 else 0.0

def optimize_f1_thresholds(y_true, y_prob, classes):
    print("[*] Optimisation des seuils (Sur Validation)...")
    ths = np.ones(len(classes)) * 0.3
    f1s = []
    for i in range(len(classes)):
        if y_true[:, i].sum() == 0: f1s.append(np.nan); continue
        best_th, best_f1 = 0.3, -1.0
        for th in np.arange(0.01, 0.7, 0.02):
            f1 = f1_score(y_true[:, i], (y_prob[:, i]>=th).astype(int), zero_division=0)
            if f1 > best_f1: best_f1, best_th = f1, th
        ths[i] = best_th; f1s.append(best_f1)
    
    valid_mask = ~np.isnan(f1s)
    if np.any(valid_mask): ths[~valid_mask] = np.nanmean(ths[valid_mask])
    return ths

def generate_class_analysis(y_true, y_prob, encoder, output_path=None, thresholds=0.3):
    y_pred = (y_prob >= thresholds).astype(int)
    report = classification_report(y_true, y_pred, target_names=encoder.classes_, output_dict=True, zero_division=0)
    data = []
    for i, name in enumerate(encoder.classes_):
        metrics = report.get(name, {})
        auc = roc_auc_score(y_true[:, i], y_prob[:, i]) if 0 < y_true[:, i].sum() < len(y_true) else np.nan
        data.append({'species': name, 'roc_auc': auc, 'optimal_threshold': thresholds[i] if isinstance(thresholds, np.ndarray) else thresholds, 
                     'f1_score': metrics.get('f1-score', 0), 'support': int(y_true[:, i].sum())})
    df = pd.DataFrame(data).sort_values('f1_score', ascending=False)
    if output_path: df.to_csv(output_path, index=False)
    print("\n[Top 5 Espèces sur TEST invisible]\n", df.head(5).to_string(index=False))
    return df


def print_full_report(y_true, y_prob, phase_name):
    auc = calculate_competition_metric(y_true, y_prob)
    
    idx = np.where(y_true.sum(0) > 0)[0]
    if len(idx) > 0:
        pr = average_precision_score(y_true[:, idx], y_prob[:, idx], average='macro')
    else:
        pr = 0.0
    
    print(f"\n[BILAN {phase_name}] ROC-AUC : {auc:.4f} | PR-AUC : {pr:.4f}")