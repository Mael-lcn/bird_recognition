import torch
import soundfile as sf
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
from features import SAMPLE_RATE



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

            # On prend juste le début, pad si nécessaire
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
            # Retourner un chunk vide pour gérer les erreurs proprement
            return torch.zeros(window_samples), {'file_id': idx, 'target_multi': 'nocall'}
