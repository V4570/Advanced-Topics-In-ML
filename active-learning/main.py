from query_by_committee import qbc
from imblearn.combine import SMOTETomek
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from pool_based import pool_based
from stream_based import stream_based
from pathlib import Path
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
import pandas as pd


def read_preprocessed(filepath):
	train_df = pd.read_csv(filepath / 'processed_train.csv')
	test_df = pd.read_csv(filepath / 'processed_test.csv')
	
	# x_train, x_test, y_train, y_test
	return train_df.drop('target', axis=1), test_df.drop('target', axis=1), train_df['target'], test_df['target']


def main():
	datapath = Path("data")
	
	x_train, x_test, y_train, y_test = read_preprocessed(datapath)
	
	sm = SMOTETomek(random_state=42, sampling_strategy='minority')
	x_train_resampled, y_train_resampled = sm.fit_resample(x_train, y_train)
	
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
		
		pred = qbc(x_train_resampled, x_test, y_train_resampled, y_test, clf, standalone=False)
		results[name].append(calc_scores(y_test, pred, 'Query By Committee'))

		pred_pool = pool_based(x_train_resampled, x_test, y_train_resampled, y_test, clf, standalone=False)
		results[name].append(calc_scores(y_test, pred_pool, 'Pool Based'))
		
		pred_stream = stream_based(x_train_resampled, x_test, y_train_resampled, y_test, clf, standalone=False)
		results[name].append(calc_scores(y_test, pred_stream, 'Stream Based'))

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
