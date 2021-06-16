import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def get_numeric_categorical_cols(df: pd.DataFrame):
	"""
	Given a dataframe it returns the column names for the numeric and categorical columns.
	:param df
	:return
	"""
	
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	return df.select_dtypes(include=numerics).columns, df.select_dtypes(include=object).columns


def preprocess_data(filepath, test_size):
	df = pd.read_csv(filepath)

	print(df.isna().any())

	# drop columns which are irrelevant
	df.drop(['ID'], axis=1, inplace=True)

	# y data, DONE!
	y = df.iloc[:, 2:]
	df.drop(['Computer Science','Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance'], axis=1, inplace=True)

	x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=test_size, random_state=0)

	# TITLE feature to word_tokenizer!
	df_title_train = x_train['TITLE']
	df_title_test = x_test['TITLE']

	# df_title = df['TITLE']
	df.drop(['TITLE'], axis=1, inplace=True)
	
	
	# tfidf_vectoriser = HashingVectorizer(n_features=n_components, stop_words=nltk_stop_words)
	# tfidf_vectoriser = CountVectorizer(min_df=5, stop_words=nltk_stop_words,ngram_range=(2,2))
	tfidf_vectoriser = TfidfVectorizer(min_df=5, stop_words='english',ngram_range=(2,2))
	# description data, DONE!
	
	title_train = tfidf_vectoriser.fit_transform(df_title_train).todense()
	title_test = tfidf_vectoriser.transform(df_title_test).todense()

	n_components = 250
	pca = PCA(n_components=n_components)

	# create names for dataframe
	df_title_feature_names = []
	for i in range(n_components):
		df_title_feature_names.append("title_"+str(i))

	df_title_new_train = pd.DataFrame(pca.fit_transform(title_train), columns=df_title_feature_names)
	df_title_new_test = pd.DataFrame(pca.transform(title_test), columns=df_title_feature_names)
	print(f"Explained variance for n components [TITLE]: {pca.explained_variance_ratio_[:n_components].sum():.4f}")

	############ end title preprocessing ###########



	df_abstract_train = x_train['ABSTRACT']
	df_abstract_test = x_test['ABSTRACT']
	df.drop(['ABSTRACT'], axis=1, inplace=True)

	n_components = 3000

	tfidf_vectoriser = TfidfVectorizer(min_df=10, stop_words='english',ngram_range=(2,2))
	abstract_train = tfidf_vectoriser.fit_transform(df_abstract_train).todense().astype('float32')
	abstract_test = tfidf_vectoriser.transform(df_abstract_test).todense().astype('float32')

	pca = PCA(n_components=n_components)
	df_abstract_feature_names = []
	for i in range(n_components):
		df_abstract_feature_names.append("abstract_"+str(i))

	df_abstract_new_train = pd.DataFrame(pca.fit_transform(abstract_train), columns=df_abstract_feature_names)
	df_abstract_new_test = pd.DataFrame(pca.transform(abstract_test), columns=df_abstract_feature_names)
	print(f"Explained variance for n components [ABSTRACT]: {pca.explained_variance_ratio_[:n_components].sum():.4f}")

	x_train = pd.concat([df_title_new_train, df_abstract_new_train], axis=1).astype('float32')
	x_test = pd.concat([df_title_new_test, df_abstract_new_test], axis=1).astype('float32')

	pd.concat([x_train, y_train.reset_index(drop=True)], axis=1).to_csv('data/processed_train.csv')
	pd.concat([x_test, y_test.reset_index(drop=True)], axis=1).to_csv('data/processed_test.csv')

	return x_train, x_test, y_train, y_test


def get_labels():
	return (['Computer Science','Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance'])


def read_preprocessed(filepath):
	df_train = pd.read_csv(filepath/'processed_train.csv', dtype='float32')
	df_test = pd.read_csv(filepath/'processed_test.csv', dtype='float32')

	return df_train.drop(get_labels(), axis=1), df_test.drop(get_labels(), axis=1), df_train[get_labels()], df_test[get_labels()]

