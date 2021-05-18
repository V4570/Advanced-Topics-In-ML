from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split
from data_preprocessing import read_preprocessed, preprocess_data
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer


def ClassifierChains(x,y,test_size,clf):

    # defining classifier
    classifier = ClassifierChain(base_estimator=clf, order='random', cv='None', random_state=42)

    # splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(random_state=0)

    # fitting dataset
    classifier.fit(X_train, y_train)

    # predicting
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='micro')
    print('Classifier Chains: f1 = %.2f%%, acc = %.2f%%' % (f1 * 100, acc * 100))
    f1_s = f1_sampled(y_test.to_numpy(), y_pred.toarray())
    print('Classifier Chains: f1_sampled = %.2f%%' % (f1_s * 100))


def f1_sampled(actual, pred):
    # converting the multi-label classification to a binary output
    mlb = MultiLabelBinarizer()
    actual = mlb.fit_transform(actual)
    pred = mlb.fit_transform(pred)

    # fitting the data for calculating the f1 score
    f1 = f1_score(actual, pred, average="samples")
    return f1