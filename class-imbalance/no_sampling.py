from sklearn.metrics import f1_score, precision_recall_curve, precision_recall_fscore_support, auc
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def no_sampling(x, y, test_size, classifier):
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
	
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