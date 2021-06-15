from sklearn.metrics import f1_score
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.metrics import precision_recall_curve, auc
from matplotlib import pyplot
from sklearn.metrics import precision_recall_fscore_support


def easy_ensemble(x_train, x_test, y_train, y_test, classifier):
	
	model = EasyEnsembleClassifier(
		base_estimator=classifier,
		n_estimators=10,
		sampling_strategy="majority",
		replacement=True,
		random_state=42, n_jobs=-1
	)
	model.fit(x_train, y_train)
	
	# Calculate scores
	lr_probs = model.predict_proba(x_test)
	lr_probs = lr_probs[:, 1]
	yhat = model.predict(x_test)
	lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
	lr_f1, lr_auc = f1_score(y_test, yhat, average='micro'), auc(lr_recall, lr_precision)
	print('EasyEnsemble: f1 = %.2f%% auc = %.2f%%' % (lr_f1*100, lr_auc*100))
	prec, rec, f1, _ = precision_recall_fscore_support(y_test, yhat, average='micro')
	print('EasyEnsemble: precision = %.2f%% recall = %.2f%%' % (prec*100, rec*100))
	no_skill = len(y_test[y_test==0]) / len(y_test)
	
	# Precision - Recall Curve Plot
	pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
	pyplot.plot(lr_recall, lr_precision, marker='.', label='EasyEnsemble')
	pyplot.xlabel('Recall')
	pyplot.ylabel('Precision')
	pyplot.legend()
	pyplot.title("Precision-Recall Curve")
	pyplot.show()