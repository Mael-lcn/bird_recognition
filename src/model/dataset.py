import torch
import soundfile as sf
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from features import TorchFeatureExtractor, PerchFeatureExtractor, SAMPLE_RATE



class FocalAudioDataset(Dataset):
    def __init__(self, df_meta, audio_dir, window_sec, stride_sec, vad_threshold):
        self.df_meta = df_meta.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.vad_threshold = vad_threshold

    def __len__(self):
        return len(self.df_meta)

    def __getitem__(self, idx):
        row = self.df_meta.iloc[idx]
        file_path = self.audio_dir / row['filename']

        valid_chunks, metadata = [], []

        if not file_path.exists():
            return valid_chunks, metadata

        try:
            y_np, sr = sf.read(file_path)
            y = torch.tensor(y_np, dtype=torch.float32)

            if y.ndim > 1: y = y.mean(dim=1) 
            if sr != SAMPLE_RATE:
                y = torchaudio.functional.resample(y, sr, SAMPLE_RATE)

            window_samples = self.window_sec * SAMPLE_RATE
            stride_samples = self.stride_sec * SAMPLE_RATE
            duration_samples = y.shape[0]

            rating = float(row.get('rating', 3.0))

            for start in range(0, duration_samples, stride_samples):
                end = start + window_samples
                chunk = y[start:end]
                
                if chunk.shape[0] < window_samples:
                    chunk = torch.nn.functional.pad(chunk, (0, window_samples - chunk.shape[0]))

                rms = torch.sqrt(torch.mean(chunk**2))
                if rms >= self.vad_threshold:
                    valid_chunks.append(chunk)
                    metadata.append({
                        'file_id': idx,
                        'target_multi': row['target_multi'],
                        'rating': rating,
                        'end_sec': min((start + window_samples) / SAMPLE_RATE, duration_samples / SAMPLE_RATE)
                    })
        except Exception:
            pass 
            
        return valid_chunks, metadata

def focal_collate_fn(batch):
    all_chunks, all_meta = [], []
    for chunks, meta in batch:
        all_chunks.extend(chunks)
        all_meta.extend(meta)
    if not all_chunks:
        return torch.empty(0), []
    return torch.stack(all_chunks), all_meta


def build_soundscape_dataset(df_labels, audio_dir, feature_mode="tabular"):
    print(f"[*] Extraction Audio Soundscapes (GPU) - Mode: {feature_mode}...")
    all_feats = []

    # Choix dynamique de l'extracteur
    if feature_mode == "perch":
        extractor = PerchFeatureExtractor(aug_mode="none")
    else:
        extractor = TorchFeatureExtractor(aug_mode="none")
    all_feats = []
    extractor = TorchFeatureExtractor(aug_mode="none")

    for filename in tqdm(df_labels['filename'].unique()):
        path = Path(audio_dir) / filename
        if not path.exists(): continue

        try:
            y_np, sr = sf.read(path)
            y = torch.tensor(y_np, dtype=torch.float32)
            if y.ndim > 1: y = y.mean(dim=1)
            if sr != SAMPLE_RATE: y = torchaudio.functional.resample(y, sr, SAMPLE_RATE)
        except Exception: continue

        file_labels = df_labels[df_labels['filename'] == filename]
        window_samples = 5 * SAMPLE_RATE
        chunks, row_data = [], []

        for _, row in file_labels.iterrows():
            end_val = str(row['end'])
            if ':' in end_val:
                parts = [int(p) for p in end_val.split(':')]
                end_sec = parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else parts[0] * 60 + parts[1]
            else:
                end_sec = int(float(end_val))

            start_sample = (end_sec - 5) * SAMPLE_RATE
            end_sample = end_sec * SAMPLE_RATE

            if start_sample >= y.shape[0]: continue

            y_chunk = y[start_sample:end_sample]
            if y_chunk.shape[0] < window_samples:
                y_chunk = torch.nn.functional.pad(y_chunk, (0, window_samples - y_chunk.shape[0]))

            chunks.append(y_chunk)
            row_data.append(row)

        if chunks:
            chunks_tensor = torch.stack(chunks)
            feats_matrix = extractor.extract_features_batch(chunks_tensor)
            
            for i, r in enumerate(row_data):
                feat_dict = {extractor.feat_names[j]: feats_matrix[i, j] for j in range(len(extractor.feat_names))}
                feat_dict['filename'] = filename
                feat_dict['target_multi'] = r.get('target_multi', 'nocall')
                all_feats.append(feat_dict)

    return pd.DataFrame(all_feats)


def prepare_for_mil(df_windows, official_classes=None):
    print("[*] Encodage Multi-Label...")
    labels_list = [[lbl for lbl in str(x).split(';') if lbl] for x in df_windows['target_multi']]
    
    if official_classes is not None:
        mlb = MultiLabelBinarizer(classes=official_classes)
    else:
        mlb = MultiLabelBinarizer()
        
    y_multi = mlb.fit_transform(labels_list)
    return df_windows, y_multi, mlb
