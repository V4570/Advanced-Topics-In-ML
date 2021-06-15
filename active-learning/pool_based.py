import numpy as np
from modAL.models import ActiveLearner
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sn
from imblearn.over_sampling import SMOTE


def pool_based(data, clf, queries, query_strategy):

    dataset = pd.read_csv(data)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X_raw, y_raw = SMOTE().fit_resample(X, y)

    n_labeled = X_raw.shape[0]

    idxs = np.random.randint(low=0, high=n_labeled, size=10)

    X_train = X_raw[idxs]
    y_train = y_raw[idxs]

    X_pool = np.delete(X_raw, idxs, axis=0)
    y_pool = np.delete(y_raw, idxs, axis=0)

    n_queries = queries

    learner = ActiveLearner(estimator=clf, X_training=X_train, y_training=y_train, query_strategy=query_strategy)

    unqueried_score = learner.score(X_raw, y_raw)

    performance_hist = [unqueried_score]

    for idx in range(n_queries):

        query_idx, query_instance = learner.query(X_pool)

        X_learn = X_pool[query_idx].reshape(1, -1)
        y_learn = y_pool[query_idx].reshape(1, )

        learner.teach(X=X_learn, y=y_learn)

        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)

        acc = learner.score(X_raw, y_raw)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=idx + 1, acc=acc))

        performance_hist.append(acc)

        # predicting:

    y_pred = learner.predict(X_raw)

    return performance_hist, y_raw, y_pred


def plot_accuracy(list):

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

    ax.plot(list)

    ax.scatter(range(len(list)), list, s=13)

    ax.set_ylim(bottom=0, top=1)
    ax.grid(True)

    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))


    ax.set_title('Incremental Classification Accuracy')
    ax.set_xlabel('Query iteration')
    ax.set_ylabel('Classification Accuracy')

    plt.show()


def build_conf_matrix(array_1, array_2):

    conf_matrix = confusion_matrix(array_1, array_2)
    labels = np.array([['TN', 'FP'], ['FN', 'TP']])
    sn.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.show()


