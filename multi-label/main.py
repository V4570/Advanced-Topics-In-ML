from data_preprocessing import preprocess_data, read_preprocessed
from rakeld import rakeld_classifier
from classifier_chains import classifier_chains
from label_powerset import label_powerset
from sklearn.ensemble import AdaBoostClassifier
from pathlib import Path

PREPROCESS = True


def main():
	datapath = Path("data")
	test_size = 0.2
	
	if PREPROCESS:
		x_train, x_test, y_train, y_test = preprocess_data(datapath / "articles.csv", test_size)
	else:
		x_train, x_test, y_train, y_test = read_preprocessed(datapath)
	
	clf = AdaBoostClassifier()
	test_size = 0.3

	# RakelD: f1 = 69.01%, acc = 49.60%
	# RakelD: f1_sampled = 97.21%
	rakeld_classifier(x_train, x_test, y_train, y_test, clf)
	#classifier_chains(x, y, test_size, clf)
	# Label Powerset: f1 = 49.73%, acc = 39.97%
	# Label Powerset: f1_sampled = 100.00% uhmmm(?)
	#label_powerset(x, y, test_size, clf)


if __name__ == '__main__':
	main()

