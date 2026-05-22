import torch
import soundfile as sf
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

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
            if sr != SAMPLE_RATE:
                y = torchaudio.functional.resample(y, sr, SAMPLE_RATE)

            chunk = y[:window_samples]
            if chunk.shape[0] < window_samples:
                chunk = torch.nn.functional.pad(chunk, (0, window_samples - chunk.shape[0]))

            metadata = {
                'file_id': idx,
                'target_multi': row['target_multi'],
                'filename': row['filename']
            }
            return chunk, metadata
        except Exception:
            return torch.zeros(window_samples), {'file_id': idx, 'target_multi': 'nocall', 'filename': row['filename']}

def build_soundscape_dataset(df_labels, audio_dir):
    """Extrait les features pour le dataset de validation/soundscapes."""
    extractor = TorchFeatureExtractor()
    audio_dir = Path(audio_dir)
    features_list = []

    print("[*] Extraction Audio Soundscapes...")
    for idx, row in tqdm(df_labels.iterrows(), total=len(df_labels)):
        file_path = audio_dir / row['filename']
        try:
            y_np, sr = sf.read(file_path)
            y = torch.tensor(y_np, dtype=torch.float32)
            if y.ndim > 1: y = y.mean(dim=1)
            if sr != SAMPLE_RATE:
                y = torchaudio.functional.resample(y, sr, SAMPLE_RATE)

            # Découpage autour du end_sec si présent, sinon les 5 premières secondes
            if 'end_sec' in row and pd.notna(row['end_sec']):
                end_sample = int(row['end_sec'] * SAMPLE_RATE)
                start_sample = max(0, end_sample - 5 * SAMPLE_RATE)
                chunk = y[start_sample:end_sample]
            else:
                chunk = y[:5 * SAMPLE_RATE]
                
            target_samples = 5 * SAMPLE_RATE
            if chunk.shape[0] < target_samples:
                chunk = torch.nn.functional.pad(chunk, (0, target_samples - chunk.shape[0]))
            
            # Passage dans l'extracteur
            chunk = chunk.unsqueeze(0) 
            feats = extractor.extract_features_batch(chunk)[0] 
            
            feat_dict = {extractor.feat_names[j]: feats[j] for j in range(len(extractor.feat_names))}
            for k, v in row.items():
                feat_dict[k] = v
            features_list.append(feat_dict)
        except Exception as e:
            pass 
    
    return pd.DataFrame(features_list)
