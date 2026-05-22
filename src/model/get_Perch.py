!pip install git+https://github.com/google-research/perch-hoplite.git
!pip install tensorflow[and-cuda]~=2.20.0

from perch_hoplite.zoo import model_configs

# Input: 5 seconds of silence as mono 32 kHz waveform samples.
waveform = np.zeros(5 * 32000, dtype=np.float32)

# Automatically downloads the model from Kaggle.
# If no GPU, try 'perch_v2_cpu'
model = model_configs.load_model_by_name('perch_v2')

outputs = model.embed(waveform)
# do something with outputs.embeddings and outputs.logits['label']
