import pandas as pd
import numpy as np

def remove_bad_ratings(filename):

	#le problème c'est que les ratings de 0 ça veut dire qu'il n'y a pas de rating, donc on peut pas faire un < 1.5 par exemple
	#on doit garder ceux à 0

	df = pd.read_csv(filename)
	df = df[(df['rating'] < 2) | (df['rating'] > 0)]
	df.to_csv(filename, index=False)

def remove_uneven_ratings(filename):

	#les ratings pas entiers c'est quand y'a d'autres espèces, si on les vire ça peut aider

	df = pd.read_csv(filename)
	df = df[df['rating'].apply(lambda x: float(x).is_integer() if pd.notnull(x) else False)]
	df.to_csv(filename, index=False)

remove_bad_ratings('features_audio.csv')
remove_uneven_ratings('features_audio.csv')