import librosa
import soundfile as sf
from pathlib import Path
import os 
import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
from scipy.signal import get_window
#from sklearn.preprocessing import LabelEncoder
import json

dataset_path = Path('../dataset')
train_path = os.path.join(dataset_path, 'train.csv')
data_train_path = os.path.join(dataset_path, 'train_audio')

df = pd.read_csv(train_path)

def show_float_ratings():

	#si le rating est un entier alors il n'y a pas d'autres animaux en fond.
	t = 0
	t1 = 0
	for i, row in df.iterrows():
		filename = row['filename']
		rating = row['rating']
		if not rating.is_integer():
			print(filename, rating)
			t1 += 1
		t += 1

	print(t, t1, t1/t * 100)
	#35549 4354 12.247883203465639 donc ~12% de fichiers avec ces animaux en fond

#show_float_ratings()

def show_labels():

	t = 0
	t1 = 0
	speciess = []
	for i, row in df.iterrows():
		species = row['primary_label']
		speciess.append(species)
		if species.isnumeric():
			t1 += 1
		print(species)
		t += 1

	speciess = np.array(speciess)
	print(np.unique(speciess).shape, np.unique(speciess))
	print(t, t1, t1/t * 100)
	#35549 750 2.1097639877352385 donc ~2% de fichiers avec autre chose que des oiseaux

#show_labels()

def freq_to_mel(freq):

    return 2595.0 * np.log10(1.0 + freq / 700.0)

def met_to_freq(mels):

    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=32000):

    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = met_to_freq(mels)
    
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

def get_filters(filter_points, FFT_size):

    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters

def dct(dct_filter_num, filter_len):

    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
    return basis

#https://www.kaggle.com/code/ilyamich/mfcc-implementation-and-tutorial
def stft_audio(sr, blocksize, overlap, file):

	path = os.path.join(data_train_path, file)
	data, samplerate = sf.read(path)
	original_duration = len(data) / samplerate

	#pas utile mais dans le doute
	if len(data.shape) > 1:
		data = np.mean(data, axis=1)

	#threshold = 0.01 * np.max(np.abs(data))
	#indices = np.where(np.abs(data) > threshold)[0]
	#if len(indices) > 0:
	#	data = data[indices[0]:indices[-1]]

	target_len = sr * 5
	if len(data) > target_len:
		start = np.random.randint(0, len(data) - target_len)
		data = data[start:start+target_len]
	else:
		data = np.pad(data, (0, max(0, target_len - len(data))))

	window = get_window("hann", blocksize, fftbins=True)

	frames = []
	step = blocksize - overlap
	for i in range(0, len(data) - blocksize, step):
		frames.append(data[i:i+blocksize] * window)

	data = np.array(frames)

	data_fft = np.empty((data.shape[0], int(1 + blocksize // 2)), dtype=np.complex64)
	for n in range(data.shape[0]):
		data_fft[n, :] = np.fft.fft(data[n])[:data_fft.shape[1]]

	data_power = np.square(np.abs(data_fft))

	min_freq = 0
	max_freq = sr / 2
	mel_filter_num = 64
	filter_points, mel_freqs = get_filter_points(min_freq, max_freq, mel_filter_num, blocksize, sample_rate=sr)
	filters = get_filters(filter_points, blocksize)

	enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
	filters *= enorm[:, np.newaxis]

	audio_filtered = np.dot(filters, data_power.T)
	audio_log = 10.0 * np.log10(np.clip(audio_filtered, a_min=1e-10, a_max=None))

	dct_filter_num = 20
	dct_filters = dct(dct_filter_num, mel_filter_num)
	cepstral_coefficents = np.dot(dct_filters, audio_log)

	mean = np.mean(cepstral_coefficents, axis=1)
	std = np.std(cepstral_coefficents, axis=1)

	return mean, std, original_duration

def create_label_map(filename):

	map = {}
	labels = np.unique(np.array(df['primary_label']))
	for i in range(len(labels)):
		map[labels[i]] = i

	with open(filename, 'w') as f:
		json.dump(map, f)

	#print(map.keys())
	return map

def get_map(filename):

	pass


def get_features(sample_rate, blocksize, overlap):

	#On met les features dans un csv direct, les features pour le moment c'est la mfcc avec 20 valeurs (faudra que j'étudie les paramètres) les coordonnées latitude/longitude (super important)
	#la collection et le rating, le coef et la collection je vais les passer 
	files = df['filename']

	map = create_label_map('map.json')
	#le = LabelEncoder()
	header = (
		[f'mean_coeff{i}' for i in range(1, 21)] +
		[f'std_coeff{i}' for i in range(1, 21)] +
		['latitude', 'longitude', 'rating', 'duration', 'label']
	)
	lines = []

	for file in files:
		

		mean, std, duration = stft_audio(sample_rate, blocksize, overlap, file)
		label = df.loc[df['filename'] == file, 'primary_label'].values[0]
		latitude = df.loc[df['filename'] == file, 'latitude'].values[0]
		longitude = df.loc[df['filename'] == file, 'longitude'].values[0]
		rating = df.loc[df['filename'] == file, 'rating'].values[0]
		
		line = list(mean) + list(std) + [latitude, longitude, rating, duration, label]
		lines.append(line)

	csv_features = pd.DataFrame(lines, columns=header)
	csv_features.to_csv("features_audio.csv", index=False)

	return csv_features

_ = get_features(32000, 16000, 8000)

#xgboost ptet
#Late fusion donc 2 modèles