from sklearn.cluster import KMeans
from imblearn.over_sampling import KMeansSMOTE
from sklearn.metrics import f1_score, precision_recall_curve, precision_recall_fscore_support, auc, silhouette_score
import numpy as np
from matplotlib import pyplot as plt

TESTING = False
PLOTTING = False


def cbs(x_train, x_test, y_train, y_test, classifier):
	
	if TESTING:
		elbow(x_train, y_train)
		return()
	
	kmeans_kwargs = {
		"n_clusters": 2,
		"init": "random",
		"n_init": 10,
		"max_iter": 500,
		"random_state": 0
	}
	
	kmeans = KMeans(**kmeans_kwargs)
	
	kmeans_smote = KMeansSMOTE(kmeans_estimator=kmeans, random_state=0,
	                           sampling_strategy='minority', cluster_balance_threshold=0.25)
	
	before_sampling_0 = y_train.where(y_train == 0).count()
	before_sampling_1 = y_train.where(y_train == 1).count()
	
	x_train, y_train = kmeans_smote.fit_resample(x_train, y_train)
	
	if PLOTTING:
		after_sampling_0 = y_train.where(y_train == 0).count()
		after_sampling_1 = y_train.where(y_train == 1).count()
		
		labels = ['Before Sampling', 'After Over Sampling']
		class_0 = [before_sampling_0, after_sampling_0]
		class_1 = [before_sampling_1, after_sampling_1]
		
		x_axis = np.arange(len(labels))  # the label locations
		width = 0.35  # the width of the bars
		
		fig, ax = plt.subplots()
		rects1 = ax.bar(x_axis - width / 2, class_0, width, label='Class 0')
		rects2 = ax.bar(x_axis + width / 2, class_1, width, label='Class 1')
		
		# Add some text for labels, title and custom x-axis tick labels, etc.
		ax.set_ylabel('Counts')
		ax.set_title('Counts before and after sampling')
		ax.set_xticks(x_axis)
		ax.set_xticklabels(labels)
		ax.legend()
		
		ax.bar_label(rects1, padding=3)
		ax.bar_label(rects2, padding=3)
		
		fig.tight_layout()
		
		plt.show()
	
	clf = classifier
	clf.fit(x_train, y_train)
	
	y_pred = clf.predict(x_test)
	
	# calculate scores
	lr_probs = classifier.predict_proba(x_test)
	lr_probs = lr_probs[:, 1]
	lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
	lr_f1, lr_auc = f1_score(y_test, y_pred, average='micro'), auc(lr_recall, lr_precision)
	print('Cluster Based Sampling: f1 = %.2f%% auc = %.2f%%' % (lr_f1 * 100, lr_auc * 100))
	prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='micro')
	print('Cluster Based Sampling: precision = %.2f%% recall = %.2f%%' % (prec * 100, rec * 100))
	no_skill = len(y_test[y_test == 0]) / len(y_test)
	
	# Precision - Recall Curve Plot
	plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
	plt.plot(lr_recall, lr_precision, marker='.', label='Cluster Based Sampling')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend()
	plt.title("Precision-Recall Curve")
	plt.show()


def elbow(x, y):
	kmeans_kwargs = {
		"init": "random",
		"n_init": 10,
		"max_iter": 500,
		"random_state": 0
	}
	
	sse = []
	silhouette = []
	for k in range(1, 15):
		kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
		kmeans.fit(x)
		if k > 1:
			silhouette.append(silhouette_score(x, kmeans.labels_))
		sse.append(kmeans.inertia_)
	
	plt.plot(range(1, 15), sse)
	plt.xticks(range(1, 15))
	plt.xlabel("Number of Clusters")
	plt.ylabel("SSE")
	plt.show()
	
	plt.plot(range(2, 15), silhouette)
	plt.xticks(range(2, 15))
	plt.xlabel("Number of Clusters")
	plt.ylabel("Silhouette score")
	plt.show()
