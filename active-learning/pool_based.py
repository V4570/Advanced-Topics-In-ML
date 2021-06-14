import numpy as np
import matplotlib.pyplot as plt
from modAL.models import ActiveLearner


def pool_based(x, y, clf, queries):

    X_raw = x
    y_raw = y

    n_labeled = X_raw.shape[0]
    idxs = np.random.randint(low=0, high=n_labeled, size=5000)

    X_train = X_raw[idxs]
    y_train = y_raw[idxs]

    X_pool = np.delete(X_raw, idxs, axis=0)
    y_pool = np.delete(y_raw, idxs, axis=0)

    n_queries = queries

    learner = ActiveLearner(estimator=clf, X_training=X_train, y_training=y_train)

    for idx in range(n_queries):

        query_idx, query_instance = learner.query(X_pool)

        X_learn = X_pool[query_idx].reshape(1, -1)
        y_learn = y_pool[query_idx].reshape(1, )

        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)

        acc = learner.score(X_raw, y_raw)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=idx + 1, acc=acc))


