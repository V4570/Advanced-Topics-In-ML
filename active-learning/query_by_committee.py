from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, confusion_matrix
import numpy as np
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
from modAL.models import ActiveLearner, Committee
import seaborn as sn

RANDOM_STATE_SEED = 1
np.random.seed(RANDOM_STATE_SEED)


def qbc(x_train, x_test, y_train, y_test, clf, standalone=False):
	
	x_pool = deepcopy(x_train.to_numpy())
	y_pool = deepcopy(y_train.to_numpy())
	
	# committee members
	n_members = 5
	learner_list = list()
	
	# initializing the committee members (learners)
	for member_idx in range(n_members):
		# initial training data
		n_initial = 25
		train_idx = np.random.choice(range(x_pool.shape[0]), size=n_initial, replace=False)
		queryX_train = x_pool[train_idx]
		queryY_train = y_pool[train_idx]
		
		x_pool = np.delete(x_pool, train_idx, axis=0)
		y_pool = np.delete(y_pool, train_idx, axis=0)
		
		learner = ActiveLearner(
			estimator=clf,
			X_training=queryX_train, y_training=queryY_train
		)
		learner_list.append(learner)
	
	committee = Committee(learner_list=learner_list)
	
	unqueried_score = committee.score(x_train, y_train)
	
	performance_history = [unqueried_score]
	
	# actual query by committee process
	n_queries = 350
	for idx in range(n_queries):
		# gets the most valuable point in the data
		query_idx, query_instance = committee.query(x_pool)
		# retrains the learners based on the valuable point
		committee.teach(
			X=x_pool[query_idx].reshape(1, -1),
			y=y_pool[query_idx].reshape(1, )
		)
		
		performance_history.append(committee.score(x_train, y_train))
		
		# remove queried instance from pool
		x_pool = np.delete(x_pool, query_idx, axis=0)
		y_pool = np.delete(y_pool, query_idx)
	
	y_pred = committee.predict(x_test)
	
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
		
		print('Query By Committee: f1 = %.2f%%, acc = %.2f%%' % (f1 * 100, acc * 100))
		print('Query By Committee: prec = %.2f%%, rec = %.2f%%' % (prec * 100, rec * 100))
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
	
	qbc(x_train, x_test, y_train, y_test, clf, standalone=True)
