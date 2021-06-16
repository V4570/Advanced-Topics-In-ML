import numpy as np
from modAL.models import ActiveLearner
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sn
from copy import deepcopy
# from sklearn.model_selection import train_test_split


def pool_based(x_train, x_test, y_train, y_test, clf, queries=100, standalone=True):

    # x = x.iloc[:, :].values
    # y = y.iloc[:].values

    # X_raw, X_test, y_raw, y_test = train_test_split(x, y, test_size=0.3)

    x_train = deepcopy(x_train.to_numpy())
    y_train = deepcopy(y_train.to_numpy())

    n_labeled = x_train.shape[0]

    idxs = np.random.randint(low=0, high=n_labeled, size=10)

    X_train = x_train[idxs]
    Y_train = y_train[idxs]

    X_pool = np.delete(x_train, idxs, axis=0)
    y_pool = np.delete(y_train, idxs, axis=0)

    n_queries = queries

    learner = ActiveLearner(estimator=clf, X_training=X_train, y_training=Y_train)

    unqueried_score = learner.score(x_train, y_train)

    performance_hist = [unqueried_score]

    for idx in range(n_queries):

        query_idx, query_instance = learner.query(X_pool)

        X_learn = X_pool[query_idx].reshape(1, -1)
        y_learn = y_pool[query_idx].reshape(1, )

        learner.teach(X=X_learn, y=y_learn)

        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)

        acc = learner.score(x_train, y_train)
        if standalone:
            print('Accuracy after query {n}: {acc:0.4f}'.format(n=idx + 1, acc=acc))

        performance_hist.append(acc)

        # predicting:

    y_pred = learner.predict(x_test)

    if standalone:

        fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

        ax.plot(performance_hist)

        ax.scatter(range(len(performance_hist)), performance_hist, s=13)

        ax.set_ylim(bottom=0, top=1)
        ax.grid(True)

        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=10))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))

        ax.set_title('Incremental Classification Accuracy')
        ax.set_xlabel('Query iteration')
        ax.set_ylabel('Classification Accuracy')

        plt.show()

        conf_matrix = confusion_matrix(y_test, y_pred)
        labels = np.array([['TN', 'FP'], ['FN', 'TP']])
        sn.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues')
        plt.show()

    return y_pred

'''return performance_hist, y_test, y_pred'''

