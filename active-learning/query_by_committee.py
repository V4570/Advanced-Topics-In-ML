import numpy as np
from copy import deepcopy
from modAL.models import ActiveLearner, Committee

RANDOM_STATE_SEED = 1
np.random.seed(RANDOM_STATE_SEED)


def query_by_cmt(x, y, clf):
	x_pool = deepcopy(x)
	y_pool = deepcopy(y)
	
	n_members = 2
	learner_list = list()
	
	for member_idx in range(n_members):
		# initial training data
		n_initial = 2
		train_idx = np.random.choice(range(x_pool.shape[0]), size=n_initial, replace=False)
		X_train = x_pool[train_idx]
		y_train = y_pool[train_idx]
		
		x_pool = np.delete(x_pool, train_idx, axis=0)
		y_pool = np.delete(y_pool, train_idx)
		
		learner = ActiveLearner(
			estimator=clf(),
			X_training=X_train, y_training=y_train
		)
		learner_list.append(learner)
	
	committee = Committee(learner_list=learner_list)
	
	unqueried_score = committee.score(x, y)
	
	performance_history = [unqueried_score]
	
	n_queries = 20
	for idx in range(n_queries):
		query_idx, query_instance = committee.query(x_pool)
		committee.teach(
			X=x_pool[query_idx].reshape(1, -1),
			y=y_pool[query_idx].reshape(1, )
		)
		performance_history.append(committee.score(x, y))
		# remove queried instance from pool
		x_pool = np.delete(x_pool, query_idx, axis=0)
		y_pool = np.delete(y_pool, query_idx)
