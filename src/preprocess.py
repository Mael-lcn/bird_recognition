import librosa
import soundfile as sf
from pathlib import Path
import os
import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
from scipy.signal import get_window
from sklearn.model_selection import train_test_split
import json
import ast

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
def stft_audio(sr, blocksize, overlap, path, start_sec=None, end_sec=None):

	if start_sec is not None and end_sec is not None:
		info = sf.info(path)
		start_frame = int(start_sec * info.samplerate)
		stop_frame = int(end_sec * info.samplerate)
		data, samplerate = sf.read(path, start=start_frame, stop=stop_frame)
	else:
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
		data = data[:target_len]
	else:
		data = np.pad(data, (0, max(0, target_len - len(data))))

	window = get_window("hann", blocksize, fftbins=True)

	frames = []
	step = blocksize - overlap
	for i in range(0, len(data) - blocksize, step):
		frames.append(data[i:i+blocksize] * window)

	data = np.array(frames)

	cutoff_freq = 1000
	data_fft = np.empty((data.shape[0], int(1 + blocksize // 2)), dtype=np.complex64)
	for n in range(data.shape[0]):
		data_fft[n, :] = np.fft.fft(data[n])[:data_fft.shape[1]]

	freqs = np.fft.rfftfreq(blocksize, d=1/samplerate)
	mask = freqs >= cutoff_freq
	data_fft_filtered = data_fft * mask

	data_power = np.square(np.abs(data_fft_filtered))

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

	second_derivative = np.mean(np.diff(cepstral_coefficents, 2), axis=1)
	mean = np.mean(cepstral_coefficents, axis=1)
	std = np.std(cepstral_coefficents, axis=1)

	return mean, std, second_derivative, original_duration

def create_label_map(json_filename, taxonomy_filename):

	df = pd.read_csv(taxonomy_filename)
	labels = np.unique(np.array(df['primary_label']))
	map = {}

	for i in range(len(labels)):
		map[labels[i]] = i

	with open(json_filename, 'w') as f:
		json.dump(map, f)

	#print(map.keys())
	return map

def split_dataframe(df):

	train_df, test_df = train_test_split(
		df,
		test_size=0.2,
		random_state=42,
		stratify=df['primary_label']
	)

	return train_df, test_df

def get_features(df_input, sample_rate, blocksize, overlap, label_map):

	files = df_input['filename']

	species_list = list(label_map.keys())
	num_classes = len(label_map)

	header = (
		['filename'] + 
		[f'mean_coeff{i}' for i in range(1, 21)] +
		[f'std_coeff{i}' for i in range(1, 21)] +
		[f'diff2_coeff{i}' for i in range(1, 21)] +
		['latitude', 'longitude', 'rating', 'duration'] +
		species_list
	)

	lines = []

	for file in files:
		
		path = os.path.join(data_train_path, file)
		mean, std, second_derivative, duration = stft_audio(sample_rate, blocksize, overlap, path)
		row = df_input.loc[df_input['filename'] == file].iloc[0]

		label_vector = np.zeros(num_classes)

		p_label = row['primary_label']
		if p_label in label_map:
			label_vector[label_map[p_label]] = 1.0

		s_labels_raw = row['secondary_labels']
		if isinstance(s_labels_raw, str) and s_labels_raw != '[]':
			try:
				secondary_list = ast.literal_eval(s_labels_raw)
				for s_bird in secondary_list:
					if s_bird in label_map:
						label_vector[label_map[s_bird]] = 1.0
			except:
				pass

		meta = [row['latitude'], row['longitude'], row['rating'], duration]
		line = [file] + list(mean) + list(std) + list(second_derivative) + meta + list(label_vector)
		lines.append(line)

	csv_features = pd.DataFrame(lines, columns=header)

	return csv_features

def time_to_seconds(time_str):
    if time_str.startswith("'"):
        time_str = time_str[1:]
    
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def get_features_soundscape(labels_csv, sample_rate, blocksize, overlap, label_map):

	df_labels = pd.read_csv(labels_csv)
	data_soundscape_path = os.path.join(dataset_path, 'train_soundscapes')
	species_list = list(label_map.keys())
	num_classes = len(label_map)
    
	header = (
		['filename'] +
		[f'mean_coeff{i}' for i in range(1, 21)] +
		[f'std_coeff{i}' for i in range(1, 21)] +
		[f'diff2_coeff{i}' for i in range(1, 21)] +
		['latitude', 'longitude', 'rating', 'duration'] +
		species_list
	)

	lines = []

	for _, row in df_labels.iterrows():

		s_sec = time_to_seconds(row['start'])
		e_sec = time_to_seconds(row['end'])

		path = os.path.join(data_soundscape_path, row['filename'])
		mean, std, second_derivative, duration = stft_audio(
			sample_rate, blocksize, overlap, path,
			start_sec=s_sec, end_sec=e_sec
		)

		label_vector = np.zeros(num_classes)

		if isinstance(row['primary_label'], str):

			birds = row['primary_label'].split(';')
			for bird in birds:
				bird_code = bird.strip()
				if bird_code in label_map:
					label_vector[label_map[bird_code]] = 1.0

		latitude = np.nan
		longitude = np.nan
		rating = np.nan

		meta = [latitude, longitude, rating, duration]
		line = [row['filename']] + list(mean) + list(std) + list(second_derivative) + meta + list(label_vector)
		lines.append(line)

	csv_features = pd.DataFrame(lines, columns=header)
	csv_features.to_csv("features_soundscape_multilabel.csv", index=False)

	return csv_features

bird_map = create_label_map('map.json', '../dataset/taxonomy.csv')

train_features = get_features(df, 32000, 16000, 8000, bird_map)
train_features.to_csv("features_audio_train.csv", index=False)

#soundscape_features = get_features_soundscape('../dataset/train_soundscapes_labels.csv', 32000, 16000, 8000, bird_map)
#soundscape_features.to_csv("features_soundscape_multilabel.csv", index=False)




