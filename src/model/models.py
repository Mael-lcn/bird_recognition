import numpy as np
import torch
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score



def execute_mil_and_cv(df_windows, encoder, n_iter, top_n, n_splits):
    """
    Exécute l'entraînement complet basé sur l'apprentissage multi-instances (MIL) 
    couplé à un mécanisme de Late Fusion (Stacking) avec validation croisée.

    Le processus se déroule en trois phases :
    1. Purification des données (Expectation-Maximization) : Sélection itérative des 
       meilleures fenêtres audio par fichier pour éliminer le silence et le bruit.
    2. Validation Croisée et Stacking : Entraînement d'un modèle audio de base, puis 
       d'un méta-modèle combinant les prédictions audio aux coordonnées géographiques.
    3. Entraînement final : Création des modèles de production sur l'ensemble des données purifiées.

    Args:
        df_windows: Le DataFrame contenant toutes les fenêtres extraites, leurs 
            caractéristiques acoustiques (MFCC, flatness), spatiales et la cible.
        encoder: L'instance de l'encodeur de labels pour récupérer les classes.
        n_iter: Le nombre d'itérations pour la boucle d'Expectation-Maximization.
        top_n: Le nombre de meilleures fenêtres à retenir par fichier audio.
        n_splits: Le nombre de plis (folds) pour la validation croisée.

    Returns:
        Un tuple contenant :
            - Un dictionnaire avec les modèles finaux ("base" et "meta").
            - Le tableau des véritables étiquettes des fenêtres purifiées (y_clean).
            - Le tableau des prédictions fermes (Out-Of-Fold).
            - La matrice des probabilités continues (Out-Of-Fold).
    """
    # Détection automatique de l'accélération matérielle
    if torch.cuda.is_available():
        xgb_device = "cuda"
        print("\n[*] GPU NVIDIA détecté : Accélération CUDA activée pour XGBoost.")
    else:
        xgb_device = "cpu"
        print("\n[*] CPU détecté : Mode classique activé pour XGBoost.")

    # Conservation d'une copie globale indispensable pour évaluer toutes les fenêtres lors de l'EM
    df_full = df_windows.copy()

    # Isolation exclusive des caractéristiques acoustiques pures pour le modèle de base
    audio_cols = [c for c in df_full.columns if c.startswith('mfcc') or c == 'flatness']

    # Hyperparamètres spécifiques au modèle d'analyse du signal audio
    xgb_params_audio = {
        'objective': 'multi:softprob', 
        'n_estimators': 350,        
        'learning_rate': 0.05,      
        'max_depth': 6,             
        'tree_method': 'hist',
        'device': xgb_device,
        'subsample': 0.8,           
        'colsample_bytree': 0.7,    
        'min_child_weight': 5,      
        'gamma': 0.2,               
        'n_jobs': -1,
        'random_state': 42
    }

    # Hyperparamètres spécifiques au méta-modèle (fusion audio + géographie)
    # L'arbre est moins profond pour éviter le surapprentissage sur les probabilités
    xgb_params_meta = {
        'objective': 'multi:softprob', 
        'n_estimators': 100,        
        'learning_rate': 0.03,      
        'max_depth': 3,             
        'tree_method': 'hist',
        'device': xgb_device,
        'subsample': 0.85,          
        'colsample_bytree': 1.0,    
        'reg_lambda': 5.0,          
        'n_jobs': -1,
        'random_state': 42
    }

    print(f"\n[*] PHASE 1 : Purification EM guidée par Heuristique")

    # Initialisation : Sélection du "top N" basée sur l'énergie maximale (score heuristique)
    top_initial = df_full.groupby('file_id')['heuristic_score'].nlargest(top_n).index.get_level_values(1)
    df_train = df_full.loc[top_initial]

    # Boucle d'Expectation-Maximization
    for iteration in range(1, n_iter + 1):
        print(f"  > Itération {iteration} ({len(df_train)} fenêtres purifiées utilisées)...")
        
        X_audio_train = df_train[audio_cols]
        y_train = df_train['target'].values 

        # Phase de Maximisation : Entraînement du modèle sur les fenêtres supposées contenir l'oiseau
        model_audio = XGBClassifier(**xgb_params_audio)
        model_audio.fit(X_audio_train, y_train)

        # On arrête l'évaluation si on a atteint la dernière itération pour économiser du temps
        if iteration == n_iter:
            break 

        # Phase d'Expectation : On réévalue absolument toutes les fenêtres avec le modèle mis à jour
        X_audio_full = df_full[audio_cols]
        y_full = df_full['target'].values
        probas_full = model_audio.predict_proba(X_audio_full)

        # Extraction de la probabilité attribuée à la classe théorique du fichier
        df_full['score_oiseau'] = probas_full[np.arange(len(y_full)), y_full]

        # Mise à jour de la vérité terrain : on garde les fenêtres où le modèle est le plus confiant
        idx_to_keep = df_full.groupby('file_id')['score_oiseau'].nlargest(top_n).index.get_level_values(1)
        df_train = df_full.loc[idx_to_keep]


    print(f"\n[*] PHASE 2 : Méta-Modélisation (Late Fusion Audio + Geo)")

    # Préparation des données purifiées (sans le bruit/silence)
    X_clean_audio = df_train[audio_cols]
    X_geo = df_train[['geo_latitude', 'geo_longitude']]
    y_clean = df_train['target'].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y_clean))
    oof_proba = np.zeros((len(y_clean), len(encoder.classes_)), dtype=np.float32)

    # Validation croisée pour évaluer la capacité de généralisation réelle
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_clean_audio, y_clean)):
        x_aud_tr, y_tr = X_clean_audio.iloc[train_idx], y_clean[train_idx]
        x_aud_val, y_val = X_clean_audio.iloc[val_idx], y_clean[val_idx]

        # 1. Entraînement du modèle de base sur le signal audio uniquement
        base_model = XGBClassifier(**xgb_params_audio)
        base_model.fit(x_aud_tr, y_tr)

        # Extraction des caractéristiques latentes (probabilités de sortie)
        train_probas = base_model.predict_proba(x_aud_tr)
        val_probas = base_model.predict_proba(x_aud_val)

        # 2. Construction de la matrice pour le méta-modèle (Probabilités + Coordonnées)
        X_meta_train = np.hstack([train_probas, X_geo.iloc[train_idx].values])
        X_meta_val = np.hstack([val_probas, X_geo.iloc[val_idx].values])

        # Entraînement du méta-modèle de fusion
        meta_model = XGBClassifier(**xgb_params_meta)
        meta_model.fit(X_meta_train, y_tr)

        # Sauvegarde des prédictions strictes hors pli pour le calcul des métriques
        oof_proba[val_idx] = meta_model.predict_proba(X_meta_val)
        oof_preds[val_idx] = meta_model.predict(X_meta_val)

        print(f"  > Pli {fold+1} | Meta-Accuracy : {accuracy_score(y_val, oof_preds[val_idx]):.4f}")


    print("\n[*] Entraînement des modèles finaux pour l'inférence...")

    # Le pipeline d'inférence nécessite le modèle audio complet et le méta-modèle complet
    final_base_model = XGBClassifier(**xgb_params_audio).fit(X_clean_audio, y_clean)

    final_probas = final_base_model.predict_proba(X_clean_audio)
    X_meta_final = np.hstack([final_probas, X_geo.values])

    final_meta_model = XGBClassifier(**xgb_params_meta).fit(X_meta_final, y_clean)

    # Restitution des artefacts d'entraînement et de validation
    return {"base": final_base_model, "meta": final_meta_model}, y_clean, oof_preds, oof_proba
