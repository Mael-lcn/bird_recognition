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
from sklearn.multioutput import MultiOutputClassifier


def load_data(train, soundscape):

	df_train = pd.read_csv(train)
	df_soundscape = pd.read_csv(soundscape)

	files = np.unique(np.array(df_soundscape['filename']))
	np.random.shuffle(files)
	print(files.shape)
	train_soundscape, test_soundscape = np.array_split(files, 2)

	train_split = df_soundscape[df_soundscape['filename'].isin(train_soundscape)]
	test_split = df_soundscape[df_soundscape['filename'].isin(test_soundscape)]

	train = pd.concat([df_train, train_split])
	idx_duration = df_train.columns.get_loc('duration')

	X_train = train.iloc[:, :idx_duration + 1]
	Y_train = train.iloc[:, idx_duration + 1:]

	X_test = test_split.iloc[:, :idx_duration + 1]
	Y_test = test_split.iloc[:, idx_duration + 1:]

	X_train = X_train.loc[:, ~X_train.columns.str.contains('diff2_coeff|filename')]
	X_test = X_test.loc[:, ~X_test.columns.str.contains('diff2_coeff|filename')]

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	return X_train, X_test, Y_train, Y_test

def train_xgb_model(X_train, Y_train):

	base_model = xgb.XGBClassifier(
		objective='binary:logistic',
		n_estimators=500,
		max_depth=6,
		learning_rate=0.05,
		subsample=0.8,
		colsample_bytree=0.8,
		gamma=2,
		min_child_weight=5,
		max_delta_step=1,
		scale_pos_weight=10,
		tree_method='hist',
		random_state=42
	)

	model = MultiOutputClassifier(base_model)
	model.fit(X_train, Y_train)

	return model

def evaluate_model(model, X_test, Y_test):

	pred_probs = np.array([estimator.predict_proba(X_test)[:, 1] for estimator in model.estimators_]).T

	valid_classes = Y_test.sum(axis=0) > 0
	real_classes = valid_classes[valid_classes > 0].index.tolist()

	if valid_classes.sum() > 0:
		roc_auc = roc_auc_score(
			Y_test.loc[:, valid_classes],
			pred_probs[:, valid_classes],
			average='macro'
		)
		print(f"ROC AUC (macro) valid_classes = {len(real_classes)}: {roc_auc:.4f}")

	pred_bin = (pred_probs > 0.05).astype(int)

	print("\nClassification Report:")
	print(classification_report(Y_test, pred_bin, zero_division=0))

X_train, X_test, Y_train, Y_test = load_data(
	'features_audio_train.csv',
	'features_soundscape_multilabel.csv'
)

model = train_xgb_model(X_train, Y_train)

evaluate_model(model, X_test, Y_test)


