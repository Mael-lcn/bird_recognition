import numpy as np
import torch
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split



def extract_positive_probas(predict_proba_output):
    res = []
    for p in predict_proba_output:
        if p.shape[1] == 2: res.append(p[:, 1])
        else: res.append(np.zeros(p.shape[0], dtype=np.float32))
    return np.array(res).T


def train_kaggle_pipeline(df_focal_windows, df_soundscapes_feat, encoder, n_iter):
    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if any(c.startswith('perch_') for c in df_focal_windows.columns):
        audio_cols = [c for c in df_focal_windows.columns if c.startswith('perch_')]
        print(f"[*] Mode Perch détecté. Dimension des features : {len(audio_cols)}")
    else:
        audio_cols = [c for c in df_focal_windows.columns if any(k in c for k in ['mfcc', 'delta', 'centroid', 'zcr', 'bp_ratio', 'rms', 'flatness'])]
        print(f"[*] Mode Tabulaire détecté. Dimension des features : {len(audio_cols)}")

    xgb_params = {
        'objective': 'binary:logistic', 'tree_method': 'hist', 'device': xgb_device, 
        'random_state': 42, 'base_score': 0.5,
        'max_depth': 5,
        'colsample_bytree': 0.6,
        'subsample': 0.8
    }
    lgb_params = {
        'n_estimators': 300, 'learning_rate': 0.05, 'random_state': 42, 'n_jobs': -1,
        'max_depth': 5, 
        'colsample_bytree': 0.6, 
        'subsample': 0.8,
        'subsample_freq': 1
    }

    print("[*] PHASE 1 : Nettoyage MIL (Instance-Level Disentanglement)...")
    X_train_audio_full = df_focal_windows[audio_cols].values.astype(np.float32)
    original_y_train = encoder.transform([[lbl for lbl in str(x).split(';') if lbl] for x in df_focal_windows['target_multi']])

    base_xgb = MultiOutputClassifier(XGBClassifier(**xgb_params, n_estimators=300, learning_rate=0.05))

    curr_X_train = X_train_audio_full
    curr_y_train = original_y_train
    
    for i in range(n_iter):
        base_xgb.fit(curr_X_train, curr_y_train)
        probas = extract_positive_probas(base_xgb.predict_proba(X_train_audio_full))
        
        selected_mask = np.zeros(len(X_train_audio_full), dtype=bool)
        new_y_pure = np.zeros((len(X_train_audio_full), original_y_train.shape[1]), dtype=np.float32)
        
        for _, group in df_focal_windows.groupby('file_id'):
            g_idx = group.index.values
            y_true_group = original_y_train[g_idx[0]]
            true_classes = np.where(y_true_group == 1)[0]

            if len(true_classes) == 0: continue

            for c in true_classes:
                p_c = probas[g_idx, c]
                sorted_args = np.argsort(p_c)[::-1]
                sorted_probs = p_c[sorted_args]
                diffs = np.abs(np.diff(sorted_probs))
                
                elbow_idx = np.argmax(diffs) + 1 if len(diffs) > 0 else 1
                max_allowed_chunks = max(4, len(g_idx) // 2) 
                keep_n = max(1, min(elbow_idx, max_allowed_chunks))
                
                selected_chunk_idx = g_idx[sorted_args[:keep_n]]
                
                selected_mask[selected_chunk_idx] = True
                new_y_pure[selected_chunk_idx, c] = 1
                    
        curr_X_train = X_train_audio_full[selected_mask]
        curr_y_train = new_y_pure[selected_mask]
        print(f"    -> Itération {i+1} : Extraction de {selected_mask.sum()} chunks PURES uniques.")

    print("[*] Entraînement final du duo sur les extraits purs...")
    base_xgb.fit(curr_X_train, curr_y_train)
    base_lgb = MultiOutputClassifier(LGBMClassifier(**lgb_params))
    base_lgb.fit(curr_X_train, curr_y_train)

    print("[*] PHASE 2 : Calibration Méta-Modèle (Random Forest)...")
    y_soundscape_multi = encoder.transform([[lbl for lbl in str(x).split(';') if lbl] for x in df_soundscapes_feat['target_multi']])

    X_val_xgb = extract_positive_probas(base_xgb.predict_proba(df_soundscapes_feat[audio_cols].values))
    X_val_lgb = extract_positive_probas(base_lgb.predict_proba(df_soundscapes_feat[audio_cols].values))
    X_all_base = np.hstack([X_val_xgb, X_val_lgb])

    unique_files = df_soundscapes_feat['filename'].unique()
    train_files, val_files = train_test_split(unique_files, test_size=0.2, random_state=42)

    train_mask = df_soundscapes_feat['filename'].isin(train_files)
    val_mask = df_soundscapes_feat['filename'].isin(val_files)
    
    meta_model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=50, max_depth=2, min_samples_leaf=10, max_features=None, random_state=42, n_jobs=-1)
    )
    meta_model.fit(X_all_base[train_mask], y_soundscape_multi[train_mask])

    val_prob = extract_positive_probas(meta_model.predict_proba(X_all_base[val_mask]))
    return {'base_xgb': base_xgb, 'base_lgb': base_lgb, 'meta': meta_model, 'val_prob': val_prob, 'val_true': y_soundscape_multi[val_mask]}
