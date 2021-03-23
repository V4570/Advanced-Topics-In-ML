from data_preprocessing import preprocess_data
from pathlib import Path


def main():
	datapath = Path("data")
	
	preprocess_data(datapath / "aug_train.csv")


if __name__ == '__main__':
	main()

