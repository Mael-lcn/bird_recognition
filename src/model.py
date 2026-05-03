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

def prepare_data(filename):

	df = pd.read_csv(filename)
	X = df.loc[:, df.columns != 'label']
	Y = df['label']
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
	#pb de continuité après le clean donc je dois reencoder
	le = LabelEncoder()
	Y_train = le.fit_transform(Y_train)
	Y_test = le.transform(Y_test)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	n_class = len(le.classes_)

	return X_train, X_test, Y_train, Y_test, n_class

def predict_model(filename):

	X_train, X_test, Y_train, Y_test, n_class = prepare_data(filename)
	sample_weights = compute_sample_weight(class_weight='balanced', y=Y_train)

	model_xgboost = xgb.XGBClassifier(
		objective='multi:softprob',
		num_class=n_class,
		n_estimators=300,
		max_depth=6,
		learning_rate=0.05,
		subsample=0.8,
		colsample_bytree=0.8,
		gamma=1,
		min_child_weight=2,
		reg_lambda=1,
		reg_alpha=0.5,
		tree_method='hist',
		random_state=42
	)

	model_xgboost.fit(X_train, Y_train, sample_weight=sample_weights)
	pred = model_xgboost.predict(X_test)

	print("Accuracy : ", accuracy_score(Y_test, pred))
	print(classification_report(Y_test, pred))

predict_model('features_audio.csv')
	
