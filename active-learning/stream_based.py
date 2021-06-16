import numpy as np
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
from modAL.models import ActiveLearner
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, confusion_matrix
import seaborn as sn


def stream_based(X_train, x_test, y_train, y_test, clf, standalone=False):

	x_pool = deepcopy(X_train.to_numpy())
	y_pool = deepcopy(y_train.to_numpy())

	n_initial = 50
	initial_idx = np.random.choice(range(len(x_pool)), size=n_initial, replace=False)
	X_train, y_train = x_pool[initial_idx], y_pool[initial_idx]

	learner = ActiveLearner(
		estimator=clf,
		X_training=X_train, y_training=y_train
	)
	unqueried_score = learner.score(x_pool, y_pool)

	if standalone:
		print('Initial prediction accuracy: %f' % unqueried_score)

	from modAL.uncertainty import classifier_uncertainty

	performance_history = [unqueried_score]
	conf = 0.4
	# learning until the accuracy reaches a given threshold
	while learner.score(x_pool, y_pool) < 0.76:
		stream_idx = np.random.choice(range(len(x_pool)))
		
		if classifier_uncertainty(learner, x_pool[stream_idx].reshape(1, -1)) >= conf:
			learner.teach(x_pool[stream_idx].reshape(1, -1), y_pool[stream_idx].reshape(-1, ))
			new_score = learner.score(x_pool, y_pool)
			performance_history.append(new_score)
			if standalone:
				print('row no. %d queried, new accuracy: %f' % (stream_idx, new_score))
	
	y_pred = learner.predict(x_test)
	
	if standalone:
		fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
		
		ax.plot(performance_history)
		ax.scatter(range(len(performance_history)), performance_history, s=13)
		
		ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
		ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
		ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
		
		ax.set_ylim(bottom=0, top=1)
		ax.grid(True)
		
		ax.set_title('Incremental classification accuracy')
		ax.set_xlabel('Query iteration')
		ax.set_ylabel('Classification Accuracy')
		
		plt.show()
		
		acc = accuracy_score(y_test, y_pred)
		prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
		roc_auc = roc_auc_score(y_test, y_pred)
		
		print('Stream Based: f1 = %.2f%%, acc = %.2f%%' % (f1 * 100, acc * 100))
		print('Stream Based: prec = %.2f%%, rec = %.2f%%' % (prec * 100, rec * 100))
		print(roc_auc)
		conf_matrix = confusion_matrix(y_test, y_pred)
		labels = np.array([['TN', 'FP'], ['FN', 'TP']])
		sn.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues')
		plt.show()
	
	return y_pred


def read_preprocessed(filepath):
	import pandas as pd
	train_df = pd.read_csv(filepath / 'processed_train.csv')
	test_df = pd.read_csv(filepath / 'processed_test.csv')
	
	# x_train, x_test, y_train, y_test
	return train_df.drop('target', axis=1), test_df.drop('target', axis=1), train_df['target'], test_df['target']


if __name__ == '__main__':
	from pathlib import Path
	from sklearn.ensemble import AdaBoostClassifier
	
	clf = AdaBoostClassifier()
	
	datapath = Path('data')
	x_train, x_test, y_train, y_test = read_preprocessed(datapath)
	
	stream_based(x_train, x_test, y_train, y_test, clf, standalone=True)
