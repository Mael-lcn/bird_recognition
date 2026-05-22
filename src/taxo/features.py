import torch
import torchaudio



SAMPLE_RATE = 32000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TorchFeatureExtractor:
    def __init__(self, sr=SAMPLE_RATE, device=DEVICE, aug_mode="none"):
        self.sr = sr
        self.device = device
        self.aug_mode = aug_mode

        # 1. Filtre Passe-Haut à 1000 Hz
        self.highpass = torchaudio.transforms.HighPassFilter(
            sample_rate=self.sr, cutoff_freq=1000.0
        ).to(self.device)

        # 2. Transformation STFT
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=2048, hop_length=512, power=2
        ).to(self.device)

        # 3. MFCC avec 20 coefficients
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sr, n_mfcc=20,
            melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128}
        ).to(self.device)

        # Génération des noms de colonnes : 20 moyennes + 20 écarts-types = 40 features
        self.feat_names = [f"mfcc_{i}_mean" for i in range(20)] + [f"mfcc_{i}_std" for i in range(20)]

    def extract_features_batch(self, y_chunks_tensor):
        with torch.no_grad():
            y_chunks_tensor = y_chunks_tensor.to(self.device)

            # Application du filtre passe-haut
            y_filtered = self.highpass(y_chunks_tensor)

            # Calcul des MFCC sur le signal filtré
            mfcc = self.mfcc_transform(y_filtered) # (batch, 20, time)

            # Calcul des statistiques (moyenne et écart-type) sur l'axe temporel
            mean = torch.mean(mfcc, dim=2)   # (batch, 20)
            std = torch.std(mfcc, dim=2)     # (batch, 20)

            # Assemblage
            features = torch.cat([mean, std], dim=1)
            
            return features.cpu().numpy()
