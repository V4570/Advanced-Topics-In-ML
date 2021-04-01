from sklearn.cluster import KMeans
from imblearn.over_sampling import KMeansSMOTE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import silhouette_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import multiprocessing

TESTING = False


def cbs(x, y):
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
	
	# split train set into train and validation for more unbiased data
	# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=0)
	
	pca = PCA(n_components=2)
	x_train = pca.fit_transform(x_train, y_train)
	x_test = pca.transform(x_test)
	
	if TESTING:
		elbow(x_train, y_train)
	
	kmeans_kwargs = {
		"n_clusters": 3,
		"init": "random",
		"n_init": 10,
		"max_iter": 500,
		"random_state": 0
	}
	
	kmeans = KMeans(**kmeans_kwargs)
	
	kmeans_smote = KMeansSMOTE(kmeans_estimator=kmeans, random_state=0,
	                           sampling_strategy='minority', cluster_balance_threshold=0.24)
	
	x_train, y_train = kmeans_smote.fit_resample(x_train, y_train)
	
	clf = AdaBoostClassifier()
	clf.fit(x_train, y_train)
	
	y_pred = clf.predict(x_test)
	
	acc = accuracy_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred, average='macro')
	precision = precision_score(y_test, y_pred, average='macro')
	f1 = f1_score(y_test, y_pred, average='macro')
	
	print('------Scores------')
	print("\tAccuracy: {:.3f}".format(acc))
	print("\tPrecision: {:.3f}".format(precision))
	print("\tRecall: {:.3f}".format(recall))
	print("\tF1: {:.3f}".format(f1))


def elbow(x, y):
	kmeans_kwargs = {
		"init": "random",
		"n_init": 10,
		"max_iter": 500,
		"random_state": 0
	}
	
	sse = []
	for k in range(1, 15):
		kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
		kmeans.fit(x, y)
		sse.append(kmeans.inertia_)
	
	plt.plot(range(1, 15), sse)
	plt.xticks(range(1, 11))
	plt.xlabel("Number of Clusters")
	plt.ylabel("SSE")
	plt.show()
