from data_preprocessing import preprocess_data, read_preprocessed
from rakeld import rakeld_classifier
from classifier_chains import classifier_chains
from label_powerset import label_powerset
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from pathlib import Path

PREPROCESS = True


def main():
	datapath = Path("data")
	test_size = 0.2
	
	if PREPROCESS:
		x_train, x_test, y_train, y_test = preprocess_data(datapath / "articles.csv", test_size)
	else:
		x_train, x_test, y_train, y_test = read_preprocessed(datapath)
	
	ada_boost = AdaBoostClassifier()
	decision_tree = DecisionTreeClassifier()
	random_forest = RandomForestClassifier(
		random_state=0,
		n_estimators=350,
		criterion='gini',
		max_features='auto',
		max_depth=5
	)
	
	classifiers = [
		{'name': 'Adaboost', 'clf': ada_boost},
		{'name': 'Decision Tree', 'clf': decision_tree},
		{'name': 'Random Forest', 'clf': random_forest}
	]
	
	results = {}
	for clf_dict in classifiers:
		name = clf_dict['name']
		clf = clf_dict['clf']
		results[name] = []

		# RakelD: f1 = 67.95%, acc = 49.18%
		# RakelD: f1_sampled = 98.19%
		pred = rakeld_classifier(x_train, x_test, y_train, y_test, clf, standalone=False)
		results[name].append(calc_scores(y_test, pred, 'RaKEL'))
		
		pred = label_powerset(x_train, x_test, y_train, y_test, clf)
		results[name].append(calc_scores(y_test, pred, 'Label Powerset'))
		
		# pred = classifier_chains(x_train, x_test, y_train, y_test, clf, standalone=False)
		# results[name].append(calc_scores(y_test, pred, 'Classifier Chains'))
	
	pretty_print(results)


def pretty_print(results):
	for r in results:
		print(f'{"/":-<69}')
		print(f'Classifier: {r}')
		print(f'{"Type":<24} {"Accuracy":<10} {"Precision":<10} {"Recall":<10} {"F1":<8} {"ROC/AUC":<8}')
		for d in results[r]:
			print(
				(
					f'{d["method"]:<24} '
					f'{d["accuracy"]:<10.2f} '
					f'{d["precision"]:<10.2f} '
					f'{d["recall"]:<10.2f} '
					f'{d["f1"]:<8.2f} '
					f'{d["roc_auc"]:<8.2f}'
				)
			)
			print(d['cf_matrix'])
		print(f'\\{"":-<69}')


def calc_scores(y_true, y_predicted, method):
	acc = accuracy_score(y_true, y_predicted)
	pr, rec, f1, _ = precision_recall_fscore_support(y_true, y_predicted, average='macro')
	roc_auc = roc_auc_score(y_true, y_predicted)
	cf_matrix = confusion_matrix(y_true, y_predicted)
	
	res = {
		'method': method,
		'accuracy': acc,
		'precision': pr,
		'recall': rec,
		'f1': f1,
		'roc_auc': roc_auc,
		'cf_matrix': cf_matrix
	}
	return res


if __name__ == '__main__':
	main()

