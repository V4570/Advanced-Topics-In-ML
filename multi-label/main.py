from data_preprocessing import preprocess_data, read_preprocessed
from rakeld import rakeldClassifier
from sklearn.ensemble import AdaBoostClassifier
from pathlib import Path

PREPROCESS = True

def main():
	datapath = Path("data")
	
	if PREPROCESS:
		x, y = preprocess_data(datapath / "articles.csv")
	else:
		x, y = read_preprocessed(datapath / "processed.csv")
	
	clf = AdaBoostClassifier()
	test_size = 0.3
	print("program is running")
	rakeldClassifier(x, y, test_size, clf)

if __name__ == '__main__':
	main()

