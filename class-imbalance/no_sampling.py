from sklearn.metrics import f1_score, precision_recall_curve, precision_recall_fscore_support, auc
from matplotlib import pyplot as plt
import numpy as np


def no_sampling(x_train, x_test, y_train, y_test, classifier):

	clf = classifier
	clf.fit(x_train, y_train)
	
	y_pred = clf.predict(x_test)
	
	# calculate scores
	lr_probs = classifier.predict_proba(x_test)
	lr_probs = lr_probs[:, 1]
	lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
	lr_f1, lr_auc = f1_score(y_test, y_pred, average='micro'), auc(lr_recall, lr_precision)
	print('No Sampling: f1 = %.2f%% auc = %.2f%%' % (lr_f1 * 100, lr_auc * 100))
	prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='micro')
	print('No Sampling: precision = %.2f%% recall = %.2f%%' % (prec * 100, rec * 100))
	no_skill = len(y_test[y_test == 0]) / len(y_test)
	
	# Precision - Recall Curve Plot
	plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
	plt.plot(lr_recall, lr_precision, marker='.', label='No Sampling')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend()
	plt.title("Precision-Recall Curve")
	plt.show()


def only_majority_class(x_train, x_test, y_train, y_test, classifier):
	'''
	classifier is dump variable, only for reasons of uniformity. 
	Function classifies data into the majority class.
	'''
	
	y_pred = np.zeros(y_test.shape[0])
	
	# calculate scores
	prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='micro')
	print('Only majority: f1 = %.2f%%' % (f1 * 100))