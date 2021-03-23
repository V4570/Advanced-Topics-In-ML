import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


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
	
	y = df['target']
	
	df.drop(['enrollee_id', 'target'], axis=1, inplace=True)
	# df.replace({'Male': 0, 'Female': 1, 'Other': 3}, inplace=True)
	
	df_num_cols, df_cat_cols = get_numeric_categorical_cols(df)
	
	df_num = df[df_num_cols].copy()
	
	df_cat = df[df_cat_cols].copy()
	missing_cat_cols = df_cat.columns[df_cat.isna().any()].tolist()
	
	missing_cat = df_cat[missing_cat_cols].apply(lambda c: c.astype('category').cat.codes)
	missing_cat.replace({-1: np.nan}, inplace=True)
	
	knn_imputer = KNNImputer()
	missing_cat_impute = knn_imputer.fit_transform(missing_cat)
	
	missing_cat = pd.DataFrame(missing_cat_impute, columns=missing_cat_cols)
	
	x = pd.concat(df_num, )
