from skmultilearn.ensemble import RakelD
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def rakeld_classifier(x_train, x_test, y_train, y_test, clf, standalone=False):
    classifier = RakelD(base_classifier=clf, labelset_size=2)

    classifier.fit(x_train, y_train)
    yhat = classifier.predict(x_test)
    
    if standalone:
        acc = accuracy_score(y_test, yhat)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, yhat, average='micro')
        # f1_s = f1_sampled(y_test.to_numpy(), yhat.toarray())
    
        print('RakelD: f1 = %.2f%%, acc = %.2f%%' % (f1 * 100, acc * 100))
        # print('RakelD: f1_sampled = %.2f%%' % (f1_s * 100))
    
    return yhat
