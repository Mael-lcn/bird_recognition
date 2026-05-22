import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer

# Importations locales
from config import parse_arguments
from dataset import FocalAudioDataset, focal_collate_fn, build_soundscape_dataset
from features import TorchFeatureExtractor

from model import train_kaggle_pipeline
from metrics import print_full_report, generate_class_analysis, optimize_f1_thresholds, export_worst_errors



def main():
    args = parse_arguments()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- PRÉPARATION DONNÉES FOCALES ---")
    df_focal_meta = pd.read_csv(args.focal_meta)
    df_snd_labels = pd.read_csv(args.soundscape_labels)

    # 1. Union des labels
    all_birds = set()
    for labels in df_focal_meta['target_multi']:
        all_birds.update([l for l in str(labels).split(';') if l and l != 'nocall'])
    for labels in df_snd_labels['target_multi']:
        all_birds.update([l for l in str(labels).split(';') if l and l != 'nocall'])
    
    official_classes = sorted(list(all_birds))
    print(f"[*] Total des espèces uniques : {len(official_classes)}")

    extractor = TorchFeatureExtractor()

    # 3. Extraction des Features (Le pont manquant)
    print("\n--- EXTRACTION DES FEATURES ---")

    # A. Soundscapes
    df_snd_feat = build_soundscape_dataset(df_snd_labels, args.soundscape_audio)

    # B. Focales
    dataset = FocalAudioDataset(
        df_focal_meta, args.focal_audio, 
        args.window_sec, args.vad_threshold
    )
    loader = DataLoader(dataset, batch_size=64, num_workers=args.workers)

    focal_feats = []
    print("[*] Extraction Audio Focaux (First 5s)...")
    for chunk, meta in tqdm(loader):
        feats = extractor.extract_features_batch(chunk) 
        for i in range(len(meta['file_id'])):
            feat_dict = {extractor.feat_names[j]: feats[i, j] for j in range(len(extractor.feat_names))}
            for key in meta:
                feat_dict[key] = meta[key][i]
            focal_feats.append(feat_dict)

    df_focal_win = pd.DataFrame(focal_feats)

    # 4. Encodage
    mlb = MultiLabelBinarizer(classes=official_classes)
    mlb.fit([[c] for c in official_classes])

    print("\n--- ENTRAÎNEMENT DU PIPELINE ---")
    # Appel du Hierarchical SVM
    results = train_kaggle_pipeline(df_focal_win, df_snd_feat, mlb, args)

    # 5. Métriques et Sauvegarde
    print_full_report(results['val_true'], results['val_prob'], "TEST")
    
    optimal_thresholds = optimize_f1_thresholds(results['val_true'], results['val_prob'], mlb.classes_)

    generate_class_analysis(
        results['val_true'], 
        results['val_prob'], 
        mlb, 
        output_path=output_dir / "analyse_classes_opti.csv",
        thresholds=optimal_thresholds
    )

    export_worst_errors(
        y_true=results['val_true'], 
        y_prob=results['val_prob'], 
        filenames=results['val_filename'], 
        end_secs=results['val_end_sec'], 
        encoder=mlb, 
        output_path=output_dir / "worst_predictions_report.csv",
        thresholds=optimal_thresholds
    )

    joblib.dump({
        'model': results['model'], 
        'thresholds': optimal_thresholds
    }, output_dir / "model_dict.joblib")

    print(f"\n[*] Pipeline terminé. Modèle sauvegardé dans {output_dir}")

if __name__ == "__main__":
    main()
