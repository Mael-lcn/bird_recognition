import numpy as np
import torch
import json
import pandas as pd
from cuml.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler



class HierarchicalSVM:
    def __init__(self, config):
        self.config = config
        with open(config.taxonomy_path, 'r') as f:
            self.taxonomy = json.load(f)
        self.models = {}
        self.species_cols = []
        self.scaler = StandardScaler()

    def fit(self, X, Y_train):
        self.species_cols = Y_train.columns.tolist()
        
        # ALERTE DE SÉCURITÉ TAXONOMIE
        matched_leaves = [s for s in self._get_leaves(self.taxonomy) if s in self.species_cols]
        print(f"[*] Vérification Taxonomie : {len(matched_leaves)}/{len(self.species_cols)} espèces trouvées dans l'arbre json.")
        if len(matched_leaves) == 0:
            print("[!!!] ERREUR CRITIQUE : Aucune espèce du CSV ne correspond au JSON. L'AUC sera de 0.5. Vérifiez les IDs.")

        # Normalisation obligatoire pour SVM
        X_scaled = self.scaler.fit_transform(X.values.astype(np.float32))
        
        def train_node(node_name, node_dict):
            Y_node = pd.DataFrame(index=Y_train.index)
            valid_keys = []
            for k, v in node_dict.items():
                species = self._get_leaves(v)
                target = [s for s in species if s in self.species_cols]
                if target:
                    Y_node[k] = Y_train[target].max(axis=1)
                    valid_keys.append(k)
            
            if Y_node.shape[1] > 1:
                clf = MultiOutputClassifier(SVC(probability=True))
                clf.fit(X_scaled, Y_node.values.astype(np.float32)) # Utilise X_scaled !
                self.models[node_name] = {'clf': clf, 'keys': valid_keys}
            
            for k in valid_keys:
                if isinstance(node_dict[k], dict): train_node(k, node_dict[k])
                
        train_node('root', self.taxonomy)

    def _get_leaves(self, node):
        return [node] if isinstance(node, list) else [s for v in node.values() for s in self._get_leaves(v)]

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X.values.astype(np.float32)) # Normalise la validation
        final_log_probs = np.full((X_scaled.shape[0], len(self.species_cols)), -np.inf)
        
        def predict_node(node_name, node_dict, current_X, current_log_probs, indices):
            if node_name in self.models:
                clf_data = self.models[node_name]
                preds = np.array([p[:, 1] for p in clf_data['clf'].predict_proba(current_X)]).T
                preds_log = np.log(preds + 1e-9)
                for i, k in enumerate(clf_data['keys']):
                    c_val = node_dict[k]
                    new_log_probs = current_log_probs + preds_log[:, i]
                    if isinstance(c_val, dict): predict_node(k, c_val, current_X, new_log_probs, indices)
                    else:
                        for c in [s for s in c_val if s in self.species_cols]:
                            final_log_probs[indices, self.species_cols.index(c)] = new_log_probs
            else:
                for k, v in node_dict.items():
                    if isinstance(v, dict): predict_node(k, v, current_X, current_log_probs, indices)
                    else:
                        for c in [s for s in v if s in self.species_cols]:
                            final_log_probs[indices, self.species_cols.index(c)] = current_log_probs
                            
        predict_node('root', self.taxonomy, X_scaled, np.zeros(X_scaled.shape[0]), np.arange(X_scaled.shape[0]))
        return np.exp(final_log_probs)

def train_kaggle_pipeline(df_focal_windows, df_soundscapes_feat, encoder, args):
    audio_cols = [c for c in df_focal_windows.columns if 'mfcc_' in c]
    print(f"[*] Entraînement Hierarchical SVM avec {len(audio_cols)} features...")

    X_train = df_focal_windows[audio_cols]
    y_raw = encoder.transform([[lbl for lbl in str(x).split(';') if lbl] for x in df_focal_windows['target_multi']])
    Y_train = pd.DataFrame(y_raw, columns=encoder.classes_)

    h_svm = HierarchicalSVM(args)
    h_svm.fit(X_train, Y_train)

    y_soundscape_multi = encoder.transform([[lbl for lbl in str(x).split(';') if lbl] for x in df_soundscapes_feat['target_multi']])
    val_prob = h_svm.predict_proba(df_soundscapes_feat[audio_cols])

    val_end_sec = df_soundscapes_feat['end_sec'].values if 'end_sec' in df_soundscapes_feat.columns else np.zeros(len(df_soundscapes_feat))

    return {
        'model': h_svm, 'val_prob': val_prob, 'val_true': y_soundscape_multi,
        'val_filename': df_soundscapes_feat['filename'].values, 'val_end_sec': val_end_sec
    }
