from utils import load_dataset
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from matplotlib import pyplot

X, y = make_classification(n_classes=2, class_sep=2, weights=[0.80, 0.20], n_samples=5000, random_state=10, n_features=100)
print('Original dataset shape %s' % Counter(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

model = EasyEnsembleClassifier(random_state=42, sampling_strategy="majority", replacement=True, n_jobs=-1, n_estimators=10)
model.fit(X_train, y_train)

lr_probs = model.predict_proba(X_test)
lr_probs = lr_probs[:, 1]
yhat = model.predict(X_test)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
print('EasyEnsemble: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
no_skill = len(y_test[y_test==0]) / len(y_test)

pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='EasyEnsemble')
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
pyplot.title("Precision-Recall Curve")
pyplot.show()