from data_preprocessing import preprocess_data, read_preprocessed
from rakeld import rakeld_classifier
from classifier_chains import classifier_chains
from label_powerset import label_powerset
from sklearn.ensemble import AdaBoostClassifier
from pathlib import Path

PREPROCESS = False


def main():
	datapath = Path("data")
	test_size = 0.2
	
	if PREPROCESS:
		x_train, x_test, y_train, y_test = preprocess_data(datapath / "articles.csv", test_size)
	else:
		x_train, x_test, y_train, y_test = read_preprocessed(datapath)
	
	clf = AdaBoostClassifier()
	test_size = 0.3

	# RakelD: f1 = 67.95%, acc = 49.18%
	# RakelD: f1_sampled = 98.19%
	# rakeld_classifier(x_train, x_test, y_train, y_test, clf)
	# classifier_chains(x_train, x_test, y_train, y_test, clf)
	# label_powerset(x_train, x_test, y_train, y_test, clf)


if __name__ == '__main__':
	main()

