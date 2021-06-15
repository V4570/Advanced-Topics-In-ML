from query_by_committee import qbc
from imblearn.combine import SMOTETomek
from sklearn.ensemble import AdaBoostClassifier
from pool_based import pool_based, plot_accuracy, build_conf_matrix
from pathlib import Path
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
import pandas as pd


def read_preprocessed(filepath):
	train_df = pd.read_csv(filepath / 'processed_train.csv')
	test_df = pd.read_csv(filepath / 'processed_test.csv')
	
	# x_train, x_test, y_train, y_test
	return train_df.drop('target', axis=1), test_df.drop('target', axis=1), train_df['target'], test_df['target']


def main():
	datapath = Path("data")
	
	clf = AdaBoostClassifier()
	
	x_train, x_test, y_train, y_test = read_preprocessed(datapath)
	
	sm = SMOTETomek(random_state=42, sampling_strategy='minority')
	x_train_resampled, y_train_resampled = sm.fit_resample(x_train, y_train)
	
	qbc(x_train_resampled, x_test, y_train_resampled, y_test, clf)

	# queries = 40
	# for i in [uncertainty_sampling, margin_sampling, entropy_sampling]:
	# 	print('-----------------------------------------------------------------')
	# 	print('-----------------------------------------------------------------')
	# 	query_strategy = i
	#
	# 	list, array_1, array_2 = pool_based(x_resampled, y_resampled, clf, queries, query_strategy)
	# 	plot_accuracy(list)
	# 	build_conf_matrix(array_1, array_2)


if __name__ == '__main__':
	main()
