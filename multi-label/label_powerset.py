from skmultilearn.problem_transform import LabelPowerset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def label_powerset(x_train, x_test, y_train, y_test, clf, standalone=False):
	lb = LabelPowerset(classifier=clf)
	
	lb.fit(x_train, y_train)
	
	y_pred = lb.predict(x_test)
	
	if standalone:
		acc = accuracy_score(y_test, y_pred)
		prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='micro')
		
		print('Label powerset: f1 = %.2f%%, acc = %.2f%%' % (f1 * 100, acc * 100))
	
	return y_pred
