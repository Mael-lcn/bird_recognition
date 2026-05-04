import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def prepare_data(filename):

	df = pd.read_csv(filename)
	idx_duration = df.columns.get_loc('duration')
	X = df.iloc[:, :idx_duration + 1]
	X = X.loc[:, ~X.columns.str.contains('diff2_coeff|latitude|longitude')]
	Y = df.iloc[:, idx_duration + 1:]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	return X_train, X_test, Y_train, Y_test


def prepare_mixed_data(audio_features_file, soundscape_features_file, test_size=0.2):

	df_audio = pd.read_csv(audio_features_file)
	df_sound = pd.read_csv(soundscape_features_file)

	df_combined = pd.concat([df_audio, df_sound], axis=0).reset_index(drop=True)

	idx_duration = df_combined.columns.get_loc('duration')
	X = df_combined.iloc[:, :idx_duration + 1]
	#X = X.loc[:, ~X.columns.str.contains('diff2_coeff|latitude|longitude')]
	X = X.loc[:, ~X.columns.str.contains('diff2_coeff')]
	Y = df_combined.iloc[:, idx_duration + 1:]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	return X_train, X_test, Y_train, Y_test

def predict_model(filename_audio, filename_soundscape):

	X_train, X_test, Y_train, Y_test = prepare_mixed_data(filename_audio, filename_soundscape, 0.2)
	total_elements = Y_train.shape[0] * Y_train.shape[1]
	positives = np.sum(Y_train.values)
	negatives = total_elements - positives
	pos_weight = max(1.0, negatives / positives)

	model_xgboost = xgb.XGBClassifier(
		objective='binary:logistic',
		#num_class=n_class,
		n_estimators=500,
		max_depth=8, #valeur par défaut de xgboost, j'ai pas trop touché pour éviter l'overfit, vu le nombre de valeurs qu'on a
		learning_rate=0.05,
		subsample=0.8,
		colsample_bytree=0.8,
		gamma=2,
		min_child_weight=5,
		max_delta_step=1,
		#reg_lambda=1,
		#reg_alpha=0.5,
		scale_pos_weight=pos_weight,
		tree_method='hist',
		random_state=42
	)

	model_xgboost.fit(X_train, Y_train)
	pred_probs = model_xgboost.predict_proba(X_test)
	cols_to_keep = Y_test.columns[Y_test.nunique() > 1]

	if len(cols_to_keep) > 0:
		y_test_filtered = Y_test[cols_to_keep]
		indices = [Y_test.columns.get_loc(c) for c in cols_to_keep]
		pred_probs_filtered = pred_probs[:, indices]

		roc_auc = roc_auc_score(y_test_filtered, pred_probs_filtered, average='macro', multi_class='ovr')
		print(f"ROC AUC Score (Macro) on {len(cols_to_keep)} species: {roc_auc:.4f}")
	else:
		print("ROC AUC Score: Not defined")

	pred_bin = (pred_probs > 0.3).astype(int)

	print("\nClassification Report:")
	print(classification_report(Y_test, pred_bin, target_names=Y_train.columns, zero_division=0))

predict_model('features_audio.csv', 'features_soundscape_multilabel.csv')
	
