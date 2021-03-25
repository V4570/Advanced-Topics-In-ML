from data_preprocessing import preprocess_data, read_preprocessed
from pathlib import Path

PREPROCESS = False


def main():
	datapath = Path("data")
	
	if PREPROCESS:
		x, y = preprocess_data(datapath / "aug_train.csv")
	else:
		x, y = read_preprocessed(datapath / "processed.csv")
	
	print(x.shape, y.shape)


if __name__ == '__main__':
	main()

