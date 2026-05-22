import torch
import torchaudio
import gc
import torch
import tensorflow as tf

from perch_hoplite.zoo import model_configs



SAMPLE_RATE = 32000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TorchFeatureExtractor:
    def __init__(self, sr=SAMPLE_RATE, device=DEVICE, aug_mode="none"):
        self.sr = sr
        self.device = device
        self.aug_mode = aug_mode

        # Outils de transformation de base
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=2048, hop_length=512, power=2
        ).to(self.device)

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sr, n_mfcc=20,
            melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128, "f_min": 50, "f_max": 15000}
        ).to(self.device)

        self.centroid_transform = torchaudio.transforms.SpectralCentroid(
            sample_rate=self.sr, n_fft=2048, hop_length=512
        ).to(self.device)

        # Génération des noms de colonnes
        self.feat_names = []
        q_names = ['q05', 'q25', 'median', 'q75', 'q95']

        for feat_type in ['mfcc', 'delta', 'delta2']:
            for i in range(20):
                for stat in q_names:
                    self.feat_names.append(f"{feat_type}_{i}_{stat}")

        for feat_type in ['centroid', 'rms', 'flatness', 'band1_ratio', 'band2_ratio', 'band3_ratio', 'band4_ratio']:
            for stat in q_names:
                self.feat_names.append(f"{feat_type}_{stat}")

        self.feat_names.append('zcr_mean') # 336 Features au total

    def add_background_noise(self, y_tensor, noise_level):
        return (y_tensor * 0.8) + (torch.randn_like(y_tensor) * noise_level)

    def extract_features_batch(self, y_chunks_tensor):
        with torch.no_grad():
            y_chunks_tensor = y_chunks_tensor.to(self.device)
            if self.aug_mode == "noise_light": y_chunks_tensor = self.add_background_noise(y_chunks_tensor, 0.01)
            elif self.aug_mode == "noise_heavy": y_chunks_tensor = self.add_background_noise(y_chunks_tensor, 0.04)

            # Spectre de puissance de base
            spec = self.spec_transform(y_chunks_tensor)

            # Spectral Flatness (Séparation Bruit / Sifflement)
            geom_mean = torch.exp(torch.mean(torch.log(spec + 1e-8), dim=1, keepdim=True))
            arith_mean = torch.mean(spec, dim=1, keepdim=True) + 1e-8
            flatness = geom_mean / arith_mean # (batch, 1, time)

            # Bio-Bandes d'Énergie (Les 4 zones)
            # Résolution de fréquence : 32000 / 2048 = 15.625 Hz par bin
            total_energy = torch.sum(spec, dim=1, keepdim=True) + 1e-8
            b1 = torch.sum(spec[:, 3:128, :], dim=1, keepdim=True) / total_energy   # ~50Hz - 2000Hz (Vent, Hiboux)
            b2 = torch.sum(spec[:, 128:256, :], dim=1, keepdim=True) / total_energy # ~2000Hz - 4000Hz (Corbeaux)
            b3 = torch.sum(spec[:, 256:512, :], dim=1, keepdim=True) / total_energy # ~4000Hz - 8000Hz (Passereaux)
            b4 = torch.sum(spec[:, 512:960, :], dim=1, keepdim=True) / total_energy # ~8000Hz - 15000Hz (Insectes)

            # RMS Temporel
            rms = torch.sqrt(torch.mean(spec, dim=1, keepdim=True)) # (batch, 1, time)

            # Spectral Centroid
            centroid = self.centroid_transform(y_chunks_tensor)
            if centroid.ndim == 2: centroid = centroid.unsqueeze(1)

            # MFCC et Dérivées (Timbre)
            mfcc = self.mfcc_transform(y_chunks_tensor)
            delta = torchaudio.functional.compute_deltas(mfcc)
            delta2 = torchaudio.functional.compute_deltas(delta)

            # On assemble tout le long de l'axe des fréquences/features, en gardant la ligne temporelle intacte
            temporal_feats = torch.cat([mfcc, delta, delta2, centroid, rms, flatness, b1, b2, b3, b4], dim=1)

            # Calcul des quantiles robustes sur l'axe du temps
            q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float32).to(self.device)
            stats = torch.quantile(temporal_feats, q, dim=2) 
            stats = stats.permute(1, 2, 0).reshape(temporal_feats.shape[0], -1) # Aplatissement: (batch, 67 * 5)

            # Feature globale isolée
            zcr = (y_chunks_tensor[:, 1:] * y_chunks_tensor[:, :-1] < 0).float().mean(dim=1, keepdim=True)

            return torch.cat([stats, zcr], dim=1).cpu().numpy()



class PerchFeatureExtractor:
    def __init__(self, aug_mode="none"):
        self.aug_mode = aug_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("[*] Chargement du modèle Perch v2...")
        self.model = model_configs.load_model_by_name('perch_v2')

        # Le vecteur Perch fait toujours 1536 de long
        self.feat_names = [f"perch_{i}" for i in range(1536)]

    def add_background_noise(self, y_tensor, noise_level):
        return (y_tensor * 0.8) + (torch.randn_like(y_tensor) * noise_level)

    def extract_features_batch(self, y_chunks_tensor):
        # 1. Gestion de l'augmentation de données (comme avant)
        y_chunks_tensor = y_chunks_tensor.to(self.device)
        if self.aug_mode == "noise_light":
            y_chunks_tensor = self.add_background_noise(y_chunks_tensor, 0.01)
        elif self.aug_mode == "noise_heavy": 
            y_chunks_tensor = self.add_background_noise(y_chunks_tensor, 0.04)

        # Conversion GPU (Torch) -> CPU (Numpy)
        y_np = y_chunks_tensor.cpu().numpy()

        # Extraction via Perch (accepte les batchs directement)
        outputs = self.model.embed(y_np)

        # On renvoie uniquement l'embedding (batch_size, 1536)
        return outputs.embeddings[:, 0, :]

    def release_gpu(self):
        print("[*] Libération élégante de la mémoire GPU (TensorFlow/Torch)...")
        # 1. Supprime le modèle de la mémoire
        del self.model
        # 2. Force le nettoyage de Python
        gc.collect()
        # 3. Vide le cache Torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tf.keras.backend.clear_session()