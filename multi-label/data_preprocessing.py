import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import PCA

def get_numeric_categorical_cols(df: pd.DataFrame):
	"""
	Given a dataframe it returns the column names for the numeric and categorical columns.
	:param df
	:return
	"""
	
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	return df.select_dtypes(include=numerics).columns, df.select_dtypes(include=object).columns

def preprocess_data(filepath):
	df = pd.read_csv(filepath)

	print(df.isna().any())

	#drop columns which are irrelevant
	df.drop(['ID'], axis=1, inplace=True)

	#y data, DONE!
	y = df.iloc[:,2:]
	df.drop(['Computer Science','Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance'], axis=1, inplace=True)

	#TITLE feature to word_tokenizer!
	df_title = df['TITLE']
	df.drop(['TITLE'], axis=1, inplace=True)

	#nltk.download('stopwords')
	#nltk.download('punkt')	
	#nltk_stop_words = stopwords.words("english")
	
	n_components = 250
	#tfidf_vectoriser = HashingVectorizer(n_features=n_components, stop_words=nltk_stop_words)
	#tfidf_vectoriser = CountVectorizer(min_df=5, stop_words=nltk_stop_words,ngram_range=(2,2))
	tfidf_vectoriser = TfidfVectorizer(min_df=5, stop_words='english',ngram_range=(2,2))
	# #description data, DONE!
	title = tfidf_vectoriser.fit_transform(df_title).todense()
	pca = PCA(n_components=n_components)
	df_title_feature_names = []
	for i in range(n_components):
		df_title_feature_names.append("title_"+str(i))
	df_title_new = pd.DataFrame(pca.fit_transform(title), columns=df_title_feature_names)
	print(f"Explained variance for n components [TITLE]: {pca.explained_variance_ratio_[:n_components].sum():.4f}")

	df_abstract = df['ABSTRACT']
	df.drop(['ABSTRACT'], axis=1, inplace=True)
	n_components = 3000
	#tfidf_vectoriser = HashingVectorizer(n_features=n_components, stop_words=nltk_stop_words)
	tfidf_vectoriser = TfidfVectorizer(min_df=10, stop_words='english',ngram_range=(2,2))
	# #description data, DONE!
	abstract = tfidf_vectoriser.fit_transform(df_abstract).todense().astype('float32')
	pca = PCA(n_components=n_components)
	df_abstract_feature_names = []
	for i in range(n_components):
		df_abstract_feature_names.append("abstract_"+str(i))
	df_abstract_new = pd.DataFrame(pca.fit_transform(abstract), columns=df_abstract_feature_names)
	print(f"Explained variance for n components [ABSTRACT]: {pca.explained_variance_ratio_[:n_components].sum():.4f}")

	x = pd.concat([df_title_new, df_abstract_new], axis=1).astype('float32')
	#x = pd.concat([df_num, df_cats, df_description_new], axis=1).astype('float32')
	pd.concat([x, y], axis=1).to_csv('data/processed.csv')

	print(x.isna().sum())
	print(y.isna().sum())
	return x, y

def get_labels():
	return (['Computer Science','Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance'])

def read_preprocessed(filepath):
	df = pd.read_csv(filepath, dtype='float32')
	
	return df.drop(get_labels(), axis=1), df[get_labels()]
