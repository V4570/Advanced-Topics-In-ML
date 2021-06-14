from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score, roc_auc_score
import numpy as np
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
from modAL.models import ActiveLearner, Committee

RANDOM_STATE_SEED = 1
np.random.seed(RANDOM_STATE_SEED)


def query_by_cmt(x, y, clf):
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
	
	x_pool = deepcopy(x_train.to_numpy())
	y_pool = deepcopy(y_train.to_numpy())
	
	n_members = 2
	learner_list = list()
	
	for member_idx in range(n_members):
		# initial training data
		n_initial = 100
		train_idx = np.random.choice(range(x_pool.shape[0]), size=n_initial, replace=False)
		queryX_train = x_pool[train_idx]
		queryY_train = y_pool[train_idx]
		
		x_pool = np.delete(x_pool, train_idx, axis=0)
		y_pool = np.delete(y_pool, train_idx)
		
		learner = ActiveLearner(
			estimator=clf,
			X_training=queryX_train, y_training=queryY_train
		)
		learner_list.append(learner)
	
	committee = Committee(learner_list=learner_list)
	
	unqueried_score = committee.score(x_train, y_train)
	
	performance_history = [unqueried_score]
	
	n_queries = 350
	for idx in range(n_queries):
		query_idx, query_instance = committee.query(x_pool)
		committee.teach(
			X=x_pool[query_idx].reshape(1, -1),
			y=y_pool[query_idx].reshape(1, )
		)
		performance_history.append(committee.score(x_train, y_train))
		# remove queried instance from pool
		x_pool = np.delete(x_pool, query_idx, axis=0)
		y_pool = np.delete(y_pool, query_idx)
	
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
	
	y_pred = committee.predict(x_test)
	
	acc = accuracy_score(y_test, y_pred)
	prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='micro')
	roc_auc = roc_auc_score(y_test, y_pred)
	
	print('Query By Committee: f1 = %.2f%%, acc = %.2f%%' % (f1 * 100, acc * 100))
	print('Query By Committee: prec = %.2f%%, rec = %.2f%%' % (prec * 100, rec * 100))
	print(roc_auc)