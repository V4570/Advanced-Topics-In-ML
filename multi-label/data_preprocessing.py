import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
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

	#df = df.loc[:10000]

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
	df_num = pd.DataFrame(scaler.fit_transform(df), columns=nums)

	#description feature to word_tokenizer!
	df_description = df_cats['description']
	df_cats.drop(['description'], axis=1, inplace=True)

	# nltk.download('stopwords')
	# nltk.download('punkt')	
	# nltk_stop_words = stopwords.words("english")
	
	# n_components = 25
	# tfidf_vectoriser = HashingVectorizer(n_features=n_components, stop_words=nltk_stop_words)
	# #tfidf_vectoriser = CountVectorizer(min_df=5, stop_words=nltk_stop_words,ngram_range=(2,2))
	# #tfidf_vectoriser = TfidfVectorizer(min_df=10, stop_words=nltk_stop_words,ngram_range=(2,2))
	# # #description data, DONE!
	# x_t = tfidf_vectoriser.fit_transform(df_description).todense()
	# pca = PCA(n_components=n_components)
	# df_description_feature_names = []
	# for i in range(n_components):
	# 	df_description_feature_names.append("description_"+str(i))
	# df_description_new = pd.DataFrame(pca.fit_transform(x_t), columns=df_description_feature_names)
	# print(f"Explained variance for n components: {pca.explained_variance_ratio_[:n_components].sum():.4f}")


	#categories to codes.
	df_cats = df_cats.apply(lambda c: c.astype('category').cat.codes)

	df_num.reset_index(drop=True, inplace=True)
	df_cats.reset_index(drop=True, inplace=True)
	#df_description_new.reset_index(drop=True, inplace=True)

	x = pd.concat([df_num, df_cats], axis=1).astype('float32')
	#x = pd.concat([df_num, df_cats, df_description_new], axis=1).astype('float32')
	pd.concat([x, y], axis=1).to_csv('data/processed.csv')

	print(x.isna().sum())
	print(y.isna().sum())
	return x, y

def get_labels():
	return ([' Action', ' Adventure', ' Animation', ' Biography', ' Comedy',
       ' Crime', ' Drama', ' Family', ' Fantasy', ' Film-Noir', ' History',
       ' Horror', ' Music', ' Musical', ' Mystery', ' Romance', ' Sci-Fi',
       ' Sport', ' Thriller', ' War', ' Western', 'Action', 'Adventure',
       'Animation', 'Biography', 'Comedy', 'Crime', 'Drama', 'Family',
       'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical',
       'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

def read_preprocessed(filepath):
	df = pd.read_csv(filepath, dtype='float32')
	
	return df.drop(get_labels(), axis=1), df[get_labels()]
