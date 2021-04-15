import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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

	cols = df.columns
	for col in cols:
		#find columns which nan values are more than threshold (10%)
		nan_percentage = df[col].isnull().sum()/len(df.index)*100
		if(nan_percentage > 10):
			print("name that droped: " + col + ", nan percentage of column is : " + str(nan_percentage))
			df.drop(col, axis=1, inplace=True)


	#drop columns which are irrelevant (original_title it's almost the same with title).
	df.drop(['imdb_title_id', 'date_published', 'original_title'], axis=1, inplace=True)


	nums, cats = get_numeric_categorical_cols(df)
	print("total index[before dropna in categorical columns] " + str(len(df.index)))
	df.dropna(subset=cats, inplace=True)
	print("total index [after dropna in categorical columns]: " + str(len(df.index)))

	#y data, DONE!
	y = df['genre'].str.get_dummies(sep=",")
	df.drop(['genre'], axis=1, inplace=True)

	#fill with mean value where is nan in 'reviews_from_users' feature
	df['reviews_from_users'].fillna(df['reviews_from_users'].mean(), inplace=True)
	#print(df.isna().any())
	
	nums, cats = get_numeric_categorical_cols(df)
	df_cats = df[cats]
	df.drop(cats, axis=1, inplace=True)

	#numeric data, DONE!
	scaler = MinMaxScaler()
	numeric_matrix = scaler.fit_transform(df)

	#description feature to word_tokenizer!
	df_description = df_cats['description']
	df_cats.drop(['description'], axis=1, inplace=True)

	nltk.download('stopwords')
	nltk.download('punkt')	
	nltk_stop_words = stopwords.words("english")
	tfidf_vectoriser = TfidfVectorizer(min_df=10, stop_words=nltk_stop_words,ngram_range=(2,2))
	#description data, DONE!
	x_t = tfidf_vectoriser.fit_transform(df_description).todense()
	n_components = 1200
	pca = PCA(n_components=n_components)
	description_matrix = pca.fit_transform(x_t)
	print(f"Explained variance for n components: {pca.explained_variance_ratio_[:n_components].sum():.4f}")

	#categories to codes.
	df_cats = df_cats.apply(lambda c: c.astype('category').cat.codes)
	
	x = np.concatenate((numeric_matrix, df_cats.to_numpy(), description_matrix), axis=1)
	
	np.savetxt("./data/processed.csv", x, delimiter=",")
	return x, y

def read_preprocessed(filepath):
	df = pd.read_csv(filepath)
	
	return df.drop('target', axis=1), df['target']


df = preprocess_data('.\data\IMDb movies.csv')
