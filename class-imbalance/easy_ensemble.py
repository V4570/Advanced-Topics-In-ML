from data_preprocessing import read_preprocessed, preprocess_data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_curve, auc
from matplotlib import pyplot
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA

#X, y = preprocess_data("./data/aug_train.csv")
X, y = read_preprocessed("./data/processed.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pca = PCA(n_components=2, svd_solver='full')
X_train = pca.fit_transform(X_train, y_train)
X_test = pca.transform(X_test)

#Set machine learning model
adaboost = AdaBoostClassifier()
model = EasyEnsembleClassifier(base_estimator=adaboost, n_estimators=10, sampling_strategy="majority", replacement=True, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

#y_test = 1 - y_test

#Calculate scores
lr_probs = model.predict_proba(X_test)
lr_probs = lr_probs[:, 1]
yhat = model.predict(X_test)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat, average='micro'), auc(lr_recall, lr_precision)
print('EasyEnsemble: f1 = %.2f%% auc = %.2f%%' % (lr_f1*100, lr_auc*100))
prec, rec, f1, _ = precision_recall_fscore_support(y_test, yhat, average='micro')
print('EasyEnsemble: precision = %.2f%% recall = %.2f%%' % (prec*100, rec*100))
no_skill = len(y_test[y_test==0]) / len(y_test)

#Precision - Recall Curve Plot
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='EasyEnsemble')
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
pyplot.title("Precision-Recall Curve")
pyplot.show()