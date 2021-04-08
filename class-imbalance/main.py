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
	
	if PREPROCESS:
		x, y = preprocess_data(datapath / "aug_train.csv")
	else:
		x, y = read_preprocessed(datapath / "processed.csv")
	
	clf = AdaBoostClassifier()
	test_size = 0.2
	
	only_majority_class(x, y, test_size, clf)
	no_sampling(x, y, test_size, clf)
	easy_ensemble(x, y, test_size, clf)
	smote_tomek(x, y, test_size, clf)
	cbs(x, y, test_size, clf)


if __name__ == '__main__':
	main()

