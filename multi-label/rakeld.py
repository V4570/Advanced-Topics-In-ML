from sklearn.naive_bayes import GaussianNB
from skmultilearn.ensemble import RakelD
from data_preprocessing import read_preprocessed

X, y = read_preprocessed("./data/processed.csv")

classifier = RakelD(
    base_classifier=GaussianNB(),
    base_classifier_require_dense=[True, True],
    labelset_size=4
)

classifier.fit(X_train, y_train)
prediction = classifier.predict(X_train, y_train)