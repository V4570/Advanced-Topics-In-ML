import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


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
	
	# set the target column as our y
	y = df['target']
	
	# drop enrollee_id as it doesn't have any relevance
	df.drop(['enrollee_id', 'target'], axis=1, inplace=True)
	
	# get the names of numeric and categorical columns
	df_num_cols, df_cat_cols = get_numeric_categorical_cols(df)
	
	# normalize numeric columns
	scaler = MinMaxScaler()
	df_num = df[df_num_cols].copy()
	# create a dataframe from the normalized output and store it back to the variable
	df_num = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num_cols)
	
	df_cat = df[df_cat_cols].copy()
	# convert categorical to numeric
	df_cat = df_cat.apply(lambda c: c.astype('category').cat.codes)
	
	# because when converting to categorical NaN gets mapped to -1 replace it with nan
	# so that the imputer can work it's magic
	df_cat.replace({-1: np.nan}, inplace=True)
	# get the names of the columns that have at least one NaN value
	missing_cat_cols = df_cat.columns[df_cat.isna().any()].tolist()
	
	# get only the columns with null values
	missing_cat = df_cat[missing_cat_cols]
	
	df_cat.drop(missing_cat_cols, axis=1, inplace=True)
	
	# normalize the now numeric, categorical columns
	missing_cat = pd.DataFrame(scaler.fit_transform(missing_cat), columns=missing_cat_cols)
	missing_cat.fillna(method='backfill', inplace=True)
	
	# fill missing values
	# knn_imputer = KNNImputer(n_neighbors=3)
	# missing_cat_impute = knn_imputer.fit_transform(missing_cat)
	
	# missing_cat = pd.DataFrame(missing_cat_impute, columns=missing_cat_cols)
	# concatenate categorical values
	df_cat = pd.concat([df_cat, missing_cat], axis=1)
	
	# construct the x dataframe
	x = pd.concat([df_num, df_cat], axis=1)
	
	# save to dataframe to file
	pd.concat([x, y], axis=1).to_csv('data/processed.csv')
	
	return x, y


def read_preprocessed(filepath):
	df = pd.read_csv(filepath)
	
	return df.drop('target', axis=1), df['target']
