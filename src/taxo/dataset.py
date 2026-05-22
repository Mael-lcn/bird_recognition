import torch
import soundfile as sf
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from features import SAMPLE_RATE, TorchFeatureExtractor



class FocalAudioDataset(Dataset):
    def __init__(self, df_meta, audio_dir, window_sec, vad_threshold):
        self.df_meta = df_meta.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.window_sec = window_sec
        self.vad_threshold = vad_threshold

    def __len__(self):
        return len(self.df_meta)

    def __getitem__(self, idx):
        row = self.df_meta.iloc[idx]
        file_path = self.audio_dir / row['filename']
        window_samples = self.window_sec * SAMPLE_RATE

        try:
            y_np, sr = sf.read(file_path)
            y = torch.tensor(y_np, dtype=torch.float32)
            if y.ndim > 1: y = y.mean(dim=1) 
            if sr != SAMPLE_RATE: y = torchaudio.functional.resample(y, sr, SAMPLE_RATE)

            chunk = y[:window_samples]
            if chunk.shape[0] < window_samples:
                chunk = torch.nn.functional.pad(chunk, (0, window_samples - chunk.shape[0]))

            metadata = {'file_id': idx, 'target_multi': row['target_multi'], 'filename': row['filename']}
            return chunk, metadata
        except Exception:
            return torch.zeros(window_samples), {'file_id': idx, 'target_multi': 'nocall', 'filename': row['filename']}

def build_soundscape_dataset(df_labels, audio_dir):
    extractor = TorchFeatureExtractor()
    audio_dir = Path(audio_dir)
    all_feats = []
    
    print("[*] Extraction Audio Soundscapes...")
    for idx, row in tqdm(df_labels.iterrows(), total=len(df_labels)):
        file_path = audio_dir / row['filename']
        if not file_path.exists(): continue
        try:
            y_np, sr = sf.read(file_path)
            y = torch.tensor(y_np, dtype=torch.float32)
            if y.ndim > 1: y = y.mean(dim=1)
            if sr != SAMPLE_RATE: y = torchaudio.functional.resample(y, sr, SAMPLE_RATE)
            
            # Gestion de end_sec (ou 5s par défaut)
            end_sec = int(row.get('end_sec', 5)) if pd.notna(row.get('end_sec')) else 5
            start_sample = max(0, (end_sec - 5) * SAMPLE_RATE)
            end_sample = end_sec * SAMPLE_RATE
            
            chunk = y[start_sample:end_sample]
            target_samples = 5 * SAMPLE_RATE
            if chunk.shape[0] < target_samples:
                chunk = torch.nn.functional.pad(chunk, (0, target_samples - chunk.shape[0]))
            
            chunk = chunk.unsqueeze(0) 
            feats = extractor.extract_features_batch(chunk)[0] 
            
            feat_dict = {extractor.feat_names[j]: feats[j] for j in range(len(extractor.feat_names))}
            for k, v in row.items(): feat_dict[k] = v
            all_feats.append(feat_dict)
        except Exception: pass
    
    return pd.DataFrame(all_feats)
