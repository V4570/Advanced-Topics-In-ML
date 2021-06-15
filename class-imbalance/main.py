from data_preprocessing import preprocess_data, read_preprocessed
from easy_ensemble import easy_ensemble
from smote_tomek import smote_tomek
from no_sampling import no_sampling, only_majority_class
from cluster_based_sampling import cbs
from sklearn.ensemble import AdaBoostClassifier
from pathlib import Path

PREPROCESS = False


def main():
	datapath = Path("data")
	test_size = 0.2
	
	if PREPROCESS:
		x_train, x_test, y_train, y_test = preprocess_data(datapath / "aug_train.csv", test_size)
	else:
		x_train, x_test, y_train, y_test = read_preprocessed(datapath)
	
	clf = AdaBoostClassifier()
	
	only_majority_class(x_train, x_test, y_train, y_test, clf)
	no_sampling(x_train, x_test, y_train, y_test, clf)
	easy_ensemble(x_train, x_test, y_train, y_test, clf)
	smote_tomek(x_train, x_test, y_train, y_test, clf)
	cbs(x_train, x_test, y_train, y_test, clf)


if __name__ == '__main__':
	main()

