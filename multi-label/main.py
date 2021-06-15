from data_preprocessing import preprocess_data, read_preprocessed
from rakeld import rakeld_classifier
from classifier_chains import classifier_chains
from label_powerset import label_powerset
from sklearn.ensemble import AdaBoostClassifier
from pathlib import Path

PREPROCESS = False


def main():
	datapath = Path(".")
	
	if PREPROCESS:
		x, y = preprocess_data(datapath / "articles.csv")
	else:
		x, y = read_preprocessed(datapath / "processed.csv")
	
	clf = AdaBoostClassifier()
	test_size = 0.3

	# RakelD: f1 = 69.01%, acc = 49.60%
	# RakelD: f1_sampled = 97.21%
	#rakeld_classifier(x, y, test_size, clf)
	#classifier_chains(x, y, test_size, clf)
	# Label Powerset: f1 = 49.73%, acc = 39.97%
	# Label Powerset: f1_sampled = 100.00% uhmmm(?)
	#label_powerset(x, y, test_size, clf)


if __name__ == '__main__':
	main()

