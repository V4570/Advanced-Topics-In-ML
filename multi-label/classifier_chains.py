from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from time import time



def classifier_chains(x_train, x_test, y_train, y_test, clf):

    # creating an ensemble of the base classifiers:

    classifier = KNeighborsClassifier(n_neighbors=3)
    base_class = DecisionTreeClassifier()
    ovr = OneVsRestClassifier(estimator=classifier)
    ovr_new = OneVsRestClassifier(estimator=base_class)
    print('fitting')
    ovr.fit(x_train, y_train)
    y_pred_ovr = ovr.predict(x_test)
    print(y_pred_ovr)
    acc_ovr = accuracy_score(y_test, y_pred_ovr)
    print(acc_ovr)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_ovr, average='micro')
    print('Classifier Chains: f1 = %.2f%%, acc = %.2f%%, precision = %.2f%%' % (f1 * 100, acc_ovr * 100, prec * 100))

    ovr_new.fit(x_train, y_train)
    y_pred_ovr_new = ovr_new.predict(x_test)
    print(y_pred_ovr_new)
    acc_ovr = accuracy_score(y_test, y_pred_ovr_new)
    print(acc_ovr)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_ovr_new, average='micro')
    print('Classifier Chains: f1 = %.2f%%, acc = %.2f%%, precision = %.2f%%' % (f1 * 100, acc_ovr * 100, prec * 100))
    chains = [ClassifierChain(base_estimator=clf, order='random', random_state=i) for i in range(5)]

    for chain in chains:
        print('loading')
        t1 = time()
        chain.fit(x_train, y_train)
        print('fit done in ', time() - t1)

    y_pred_chains = np.array([chain.predict(x_test) for chain in chains])
    chain_accuracy_scores = [accuracy_score(y_test, y_pred_chain) for y_pred_chain in y_pred_chains]
    y_pred_ensemble = y_pred_chains.mean(axis=0)
    print(y_pred_ensemble >= .5)
    print('chains done')

    ensemble_accuray_score = accuracy_score(y_test, y_pred_ensemble >= .5)

    print(ensemble_accuray_score)

    model_scores = chain_accuracy_scores
    model_scores.append(ensemble_accuray_score)
    model_names = ('Chain 1',
                   'Chain 2',
                   'Chain 3',
                   'Chain 4',
                   'Chain 5',
                   'Ensemble')
    x_pos = np.arange(len(model_names))

    print('graphing')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.grid(True)
    ax.set_title('Classifier Chain Ensemble Performance Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation='vertical')
    ax.set_ylabel('Accuracy Score')
    ax.set_ylim([min(model_scores) * .9, max(model_scores) * 1.1])
    colors = ['b'] * (len(chain_accuracy_scores) - 1) + ['r']
    ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
    plt.tight_layout()
    plt.show()

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_ensemble >= .5, average='micro')
    print('Classifier Chains: f1 = %.2f%%, acc = %.2f%%' % (f1 * 100, ensemble_accuray_score * 100))

'''
    f1_s = f1_sampled(y_test, y_pred_ensemble >= .5)
    print('Classifier Chains: f1_sampled = %.2f%%' % (f1_s * 100))



def f1_sampled(actual, pred):
    # converting the multi-label classification to a binary output
    mlb = MultiLabelBinarizer()
    actual = mlb.fit_transform(actual)
    pred = mlb.fit_transform(pred)

    # fitting the data for calculating the f1 score
    f1 = f1_score(actual, pred, average="samples")
    return f1
'''
