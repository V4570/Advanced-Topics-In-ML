from query_by_committee import query_by_cmt
from imblearn.combine import SMOTETomek
from sklearn.ensemble import AdaBoostClassifier
from pathlib import Path
import pandas as pd


def read_preprocessed(filepath):
	df = pd.read_csv(filepath)
	
	return df.drop('target', axis=1), df['target']


def main():
	datapath = Path("data")
	
	clf = AdaBoostClassifier()
	
	x, y = read_preprocessed(datapath / "processed.csv")
	sm = SMOTETomek(random_state=42, sampling_strategy='minority')
	x_resampled, y_resampled = sm.fit_resample(x, y)
	
	query_by_cmt(x_resampled, y_resampled, clf)


if __name__ == '__main__':
	main()
