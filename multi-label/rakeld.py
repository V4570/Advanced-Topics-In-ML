from sklearn.naive_bayes import GaussianNB
from skmultilearn.ensemble import RakelD
from data_preprocessing import read_preprocessed, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import average_precision_score, f1_score
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def rakeldClassifier(x, y, test_size, clf):
    classifier = RakelD(base_classifier=clf, labelset_size=2)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    classifier.fit(X_train, y_train)
    yhat = classifier.predict(X_test)
    acc = accuracy_score(y_test, yhat)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, yhat, average='micro')
    print('RakelD: f1 = %.2f%%, acc = %.2f%%' % (f1*100, acc*100))
    f1_s = f1_sampled(y_test.to_numpy(), yhat.toarray())
    print('RakelD: f1_sampled = %.2f%%' % (f1_s*100))

def f1_sampled(actual, pred):
    
    #converting the multi-label classification to a binary output
    mlb = MultiLabelBinarizer()
    actual = mlb.fit_transform(actual)
    pred = mlb.fit_transform(pred)

    #fitting the data for calculating the f1 score 
    f1 = f1_score(actual, pred, average = "samples")
    return f1
