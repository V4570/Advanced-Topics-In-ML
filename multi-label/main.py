from data_preprocessing import preprocess_data, read_preprocessed
from rakeld import rakeldClassifier
from classifier_chains import ClassifierChains
from sklearn.ensemble import AdaBoostClassifier
from pathlib import Path

PREPROCESS = False


def main():
	datapath = Path("data")
	
	if PREPROCESS:
		x, y = preprocess_data(datapath / "articles.csv")
	else:
		x, y = read_preprocessed(datapath / "processed.csv")
	
	clf = AdaBoostClassifier()
	test_size = 0.3
	
	# RakelD: f1 = 69.01%, acc = 49.60%
	# RakelD: f1_sampled = 97.21%
	rakeldClassifier(x, y, test_size, clf)
	ClassifierChains(x, y, test_size, clf)


if __name__ == '__main__':
	main()

