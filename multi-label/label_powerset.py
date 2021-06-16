from skmultilearn.problem_transform import LabelPowerset
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer


def label_powerset(x_train, x_test, y_train, y_test, clf, standalone=False):
	lb = LabelPowerset(classifier=clf)
	
	lb.fit(x_train, y_train)
	
	y_pred = lb.predict(x_test)
	
	if standalone:
		acc = accuracy_score(y_test, y_pred)
		prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='micro')
		# f1_s = f1_sampled(y_test.to_numpy(), y_pred.toarray())
		
		print('Label powerset: f1 = %.2f%%, acc = %.2f%%' % (f1 * 100, acc * 100))
		'''
		print('Label powerset: f1_sampled = %.2f%%' % (f1_s * 100))


def f1_sampled(actual, pred):
	# converting the multi-label classification to a binary output
	mlb = MultiLabelBinarizer()
	actual = mlb.fit_transform(actual)
	pred = mlb.fit_transform(pred)
	
	# fitting the data for calculating the f1 score
	f1 = f1_score(actual, pred, average="samples")
	return f1
'''