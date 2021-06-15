from query_by_committee import query_by_cmt
from imblearn.combine import SMOTETomek
from sklearn.ensemble import AdaBoostClassifier
from pool_based import pool_based, plot_accuracy, build_conf_matrix
from pathlib import Path
from modAL.uncertainty import uncertainty_sampling, margin_sampling
import pandas as pd


def read_preprocessed(filepath):
	df = pd.read_csv(filepath)
	
	return df.drop('target', axis=1), df['target']


def main():
	datapath = Path(".")
	
	clf = AdaBoostClassifier()
	
	x, y = read_preprocessed(datapath / "processed.csv")
	print(x)
	sm = SMOTETomek(random_state=42, sampling_strategy='minority')
	x_resampled, y_resampled = sm.fit_resample(x, y)
	print(x_resampled)
	#query_by_cmt(x_resampled, y_resampled, clf)

	queries = 40
	for i in [uncertainty_sampling, margin_sampling]:
		print('-----------------------------------------------------------------')
		print('-----------------------------------------------------------------')
		query_strategy = i

		list, array_1, array_2 = pool_based(x_resampled, y_resampled, clf, queries, query_strategy)
		plot_accuracy(list)
		build_conf_matrix(array_1, array_2)


if __name__ == '__main__':
	main()
