import pandas as pd
import joblib
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import parse_arguments
from dataset import FocalAudioDataset, focal_collate_fn, build_soundscape_dataset, prepare_for_mil
from features import TorchFeatureExtractor, PerchFeatureExtractor
from models import train_kaggle_pipeline
from metrics import print_full_report, generate_class_analysis, optimize_f1_thresholds, export_worst_errors



def main():
    args = parse_arguments()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- PRÉPARATION DONNÉES FOCALES (Augmentation Conditionnelle) ---")
    df_focal_meta = pd.read_csv(args.focal_meta)
    df_snd_labels = pd.read_csv(args.soundscape_labels)

    # Calcul de l'union des labels
    all_birds = set()
    for labels in df_focal_meta['target_multi']:
        all_birds.update([l for l in str(labels).split(';') if l and l != 'nocall'])
    for labels in df_snd_labels['target_multi']:
        all_birds.update([l for l in str(labels).split(';') if l and l != 'nocall'])
    
    official_classes = sorted(list(all_birds))
    print(f"[*] Total des espèces uniques identifiées (Focal + Soundscapes) : {len(official_classes)}")

    dataset = FocalAudioDataset(
        df_focal_meta, 
        args.focal_audio, 
        args.window_sec, 
        args.stride_sec, 
        args.vad_threshold
    )
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=False, 
        num_workers=args.workers, collate_fn=focal_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    if args.feature_mode == "perch":
        ExtractorClass = PerchFeatureExtractor
        MAX_GPU_CHUNKS = 64
    else:
        ExtractorClass = TorchFeatureExtractor
        MAX_GPU_CHUNKS = 128

    extractor_clean = ExtractorClass(aug_mode="none")
    extractor_light = ExtractorClass(aug_mode="noise_light")
    extractor_heavy = ExtractorClass(aug_mode="noise_heavy")

    csv_temp_path = output_dir / f"focal_features_temp_{args.feature_mode}.csv"
    if csv_temp_path.exists(): csv_temp_path.unlink()

    first_batch = True
    total_chunks = 0
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    def extract_safe_microbatch(extractor, chunks):
        lst = []
        # On utilise MAX_GPU_CHUNKS dynamique
        for i in range(0, chunks.shape[0], MAX_GPU_CHUNKS):
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                lst.append(extractor.extract_features_batch(chunks[i:i+MAX_GPU_CHUNKS]))
        return np.vstack(lst)

    for batch_chunks, batch_meta in tqdm(dataloader, desc="Extraction GPU"):
        if batch_chunks.shape[0] == 0: continue

        ratings = np.array([m.get('rating', 3.0) for m in batch_meta])
        batch_results = []

        # Toujours extraire la version clean
        feats_c = extract_safe_microbatch(extractor_clean, batch_chunks)
        for i, meta in enumerate(batch_meta):
            row = meta.copy()
            for j, name in enumerate(extractor_clean.feat_names): row[name] = feats_c[i, j]
            batch_results.append(row)

        # Qualité Moyenne à Parfaite (>= 3.0) : On ajoute le Light Noise
        idx_light = np.where(ratings >= 3.0)[0]
        if len(idx_light) > 0:
            feats_l = extract_safe_microbatch(extractor_light, batch_chunks[idx_light])
            for idx_enum, i in enumerate(idx_light):
                row = batch_meta[i].copy()
                for j, name in enumerate(extractor_light.feat_names): row[name] = feats_l[idx_enum, j]
                batch_results.append(row)

        # Qualité Parfaite (>= 4.0) : On ajoute le Heavy Noise
        idx_heavy = np.where(ratings >= 4.0)[0]
        if len(idx_heavy) > 0:
            feats_h = extract_safe_microbatch(extractor_heavy, batch_chunks[idx_heavy])
            for idx_enum, i in enumerate(idx_heavy):
                row = batch_meta[i].copy()
                for j, name in enumerate(extractor_heavy.feat_names): row[name] = feats_h[idx_enum, j]
                batch_results.append(row)

        total_chunks += len(batch_results)
        df_batch = pd.DataFrame(batch_results)
        df_batch.to_csv(csv_temp_path, mode='a', header=first_batch, index=False)
        first_batch = False

    print(f"[+] Extraction terminée ! {total_chunks} chunks conditionnels générés.")

    df_focal_win = pd.read_csv(csv_temp_path, low_memory=False)
    for col in df_focal_win.columns:
        if col not in ['file_id', 'target_multi', 'end_sec', 'rating']:
            df_focal_win[col] = pd.to_numeric(df_focal_win[col], errors='coerce')
    df_focal_win = df_focal_win.dropna().reset_index(drop=True)

    df_focal_win, _, mlb = prepare_for_mil(df_focal_win, official_classes=official_classes)

    print("\n--- PRÉPARATION DONNÉES SOUNDSCAPES ---")
    df_snd_feat = build_soundscape_dataset(df_snd_labels, args.soundscape_audio, feature_mode=args.feature_mode)

    print("\n--- ENTRAÎNEMENT DU PIPELINE ---")
    results = train_kaggle_pipeline(df_focal_win, df_snd_feat, mlb, args.em_iter)

    # Calcul du ROC-AUC global
    print_full_report(results['val_true'], results['val_prob'], mlb)
    
    # Recherche immédiate des seuils optimaux
    optimal_thresholds = optimize_f1_thresholds(results['val_true'], results['val_prob'], mlb.classes_)

    # Génération de l'analyse en utilisant les seuils qu'on vient de trouver !
    generate_class_analysis(
        results['val_true'], 
        results['val_prob'], 
        mlb, 
        output_path=output_dir / "analyse_classes_opti.csv",
        thresholds=optimal_thresholds
    )

    # Génération du rapport des pires erreurs
    export_worst_errors(
        y_true=results['val_true'], 
        y_prob=results['val_prob'], 
        filenames=results['val_filename'], 
        end_secs=results['val_end_sec'], 
        encoder=mlb, 
        output_path=output_dir / "worst_predictions_report.csv",
        thresholds=optimal_thresholds,
        top_k=100
    )

    joblib.dump({
        'base_xgb': results['base_xgb'], 
        'base_lgb': results['base_lgb'], 
        'meta': results['meta'],
        'thresholds': optimal_thresholds
    }, output_dir / "model_dict.joblib")

    joblib.dump(mlb, output_dir / "label_encoder.joblib")
    print(f"\n[+] Pipeline terminé. Artefacts dans {output_dir}")

if __name__ == "__main__":
    main()
