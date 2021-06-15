import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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
	
	# set the target column as our y
	y = df['target']
	
	# drop enrollee_id as it doesn't have any relevance
	df.drop(['enrollee_id', 'target'], axis=1, inplace=True)
	
	# get the names of numeric and categorical columns
	df_num_cols, df_cat_cols = get_numeric_categorical_cols(df)
	
	x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=test_size, random_state=0)
	
	scaler = MinMaxScaler()
	
	train_num = x_train[df_num_cols].copy()
	train_num = pd.DataFrame(scaler.fit_transform(train_num), columns=df_num_cols, index=train_num.index)
	
	test_num = x_test[df_num_cols].copy()
	test_num = pd.DataFrame(scaler.transform(test_num), columns=df_num_cols, index=test_num.index)
	
	train_cat, missing_train_cat, missing_train_cols = format_cat_cols(x_train[df_cat_cols].copy())
	test_cat, missing_test_cat, missing_test_cols = format_cat_cols(x_test[df_cat_cols].copy())
	
	missing_train_cat = pd.DataFrame(
		scaler.fit_transform(missing_train_cat),
		columns=missing_train_cols,
		index=missing_train_cat.index
	)
	missing_train_cat = missing_train_cat.ffill(axis=1).bfill(axis=1)
	
	missing_test_cat = pd.DataFrame(
		scaler.transform(missing_test_cat),
		columns=missing_test_cols,
		index=missing_test_cat.index
	)
	missing_test_cat = missing_test_cat.ffill(axis=1).bfill(axis=1)
	
	train_cat = pd.concat([train_cat, missing_train_cat], axis=1)
	test_cat = pd.concat([test_cat, missing_test_cat], axis=1)
	
	x_train = pd.concat([train_num, train_cat], axis=1)
	x_test = pd.concat([test_num, test_cat], axis=1)
	
	pd.concat([x_train, y_train], axis=1).to_csv('data/processed_train.csv')
	pd.concat([x_test, y_test], axis=1).to_csv('data/processed_test.csv')
	
	return x_train, x_test, y_train, y_test


def format_cat_cols(df):
	
	# convert categorical to numeric
	df = df.apply(lambda c: c.astype('category').cat.codes)
	
	# because when converting to categorical NaN gets mapped to -1 replace it with nan
	# so that the imputer can work it's magic
	df.replace({-1: np.nan}, inplace=True)
	
	# get the names of the columns that have at least one NaN value
	missing_cat_cols = df.columns[df.isna().any()].tolist()
	# get only the columns with null values
	
	missing_cat = pd.DataFrame(df[missing_cat_cols], columns=missing_cat_cols, index=df[missing_cat_cols].index)
	
	df.drop(missing_cat_cols, axis=1, inplace=True)

	return df, missing_cat, missing_cat_cols


def read_preprocessed(filepath):
	train_df = pd.read_csv(filepath/'processed_train.csv')
	test_df = pd.read_csv(filepath/'processed_test.csv')
	
	# x_train, x_test, y_train, y_test
	return train_df.drop('target', axis=1), test_df.drop('target', axis=1), train_df['target'], test_df['target']
