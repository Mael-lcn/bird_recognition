import torch
import torchaudio
import torchaudio.functional as F

SAMPLE_RATE = 32000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class TorchFeatureExtractor:
    def __init__(self, sr=SAMPLE_RATE, device=DEVICE):
        self.sr = sr
        self.device = device
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sr, n_mfcc=20,
            melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128}
        ).to(self.device)
        self.feat_names = [f"mfcc_{i}_mean" for i in range(20)] + [f"mfcc_{i}_std" for i in range(20)]

    def extract_features_batch(self, y_chunks_tensor):
        with torch.no_grad():
            y_chunks_tensor = y_chunks_tensor.to(self.device)
            # Filtre Passe-haut rigoureux
            y_filtered = F.highpass_biquad(y_chunks_tensor, sample_rate=self.sr, cutoff_freq=1000.0)
            
            mfcc = self.mfcc_transform(y_filtered)
            mean = torch.mean(mfcc, dim=2)
            std = torch.std(mfcc, dim=2)
            
            return torch.cat([mean, std], dim=1).cpu().numpy()
