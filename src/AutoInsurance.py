import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder, LabelEncoder
from fancyimpute import IterativeImputer
from collections import defaultdict
import argparse

from Train import *

TRAIN_PATH = '../auto-insurance-fall-2017/train_auto.csv'
TEST_PATH = '../auto-insurance-fall-2017/test_auto.csv'
MODEL_PATH = '../model/xgb_classifier.pkl'

def clean_dataframe(df, dollar_cols, train):

	# Removing '$' from the columns with dollar values
	df[dollar_cols] = df[dollar_cols].apply(lambda x : x.str.replace(',','').str.strip('$'))
	df[dollar_cols] = df[dollar_cols].astype('float')
	if train:
		# This is a data point with a claim and -3.0 as CAR_AGE so I'm going to just make it +3 years
		# df[df['CAR_AGE']<0]
		df.at[6940, 'CAR_AGE'] = 3.0
	return df

def create_features(df):
	def age_grouping(x):
		if x<40:
			return '20-40'
		if x<60:
			return '41-60'
		return '60+'

	def risk_grouping(x):
		if x<4:
			return 'Low Risk'
		if x<6:
			return 'Minor Risk'
		if x<8:
			return 'Border Line Risk'
		if x<10:
			return 'Medium Risk'
		return 'High Risk'

	# age binning
	df['AGE_GROUPED'] = df['AGE'].apply(lambda x : age_grouping(x))

	# MVR risk grouping
	df['MVR_GROUPED'] = df['MVR_PTS'].apply(lambda x: risk_grouping(x))
	
	# Group Jobs as professional/non-professional
	df['PROFESSIONAL_BIN'] = np.where(df['JOB'].str.lower().str.strip()
								  .isin(['professional', 'doctor', 'lawyer', 'manager']), 1, 0)
	
	# Group Education as highly educated/not
	df['HIGHLY_EDUCATED_BIN'] = np.where(df['EDUCATION'].str.lower().str.strip()
								  .isin(['phd', 'masters', 'bachelors']), 1, 0)
	
	new_cols = ['AGE_GROUPED', 'MVR_GROUPED', 'PROFESSIONAL_BIN', 'HIGHLY_EDUCATED_BIN']
	
	# apply binarization
	cols_to_binary = ['KIDSDRIV','HOMEKIDS', 'HOME_VAL', 'OLDCLAIM',]

	for col, col_bin in zip(cols_to_binary, [c + '_BIN' for c in cols_to_binary]):
		df[col_bin] = np.where(df[col]==0, 0, 1)
		
	
	new_cols += [c + '_BIN' for c in cols_to_binary]
	
	return df, new_cols

class MiceImputer():
	def __init__(self, df, categorical_cols):
		self.df = df
		self.categorical_cols = categorical_cols
		self.ordinal_enc_dict = defaultdict(OrdinalEncoder)
		self.mice_imputer = IterativeImputer()

	def encode_non_nulls(self,):
		for col_name in self.categorical_cols:
			# Select the non-null values in the column
			col = self.df[col_name]
			col_not_null = col[col.notnull()]
			reshaped_vals = col_not_null.values.reshape(-1, 1)
			
			# Encode the non-null values of the column
			encoded_vals = self.ordinal_enc_dict[col_name].fit_transform(reshaped_vals)
			self.df.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)
	
	def mice_impute(self,):
		# imputing the missing value with mice imputer
		self.encode_non_nulls()
		self.df.iloc[:, :] = np.round(self.mice_imputer.fit_transform(self.df))
		
		for col in self.categorical_cols:
			reshaped_col = self.df[col].values.reshape(-1, 1)
			self.df[col] = self.ordinal_enc_dict[col].inverse_transform(reshaped_col)

class FeatureTransformer():
	# this dictionary keeps all categorical encoders. Use it for inverse transformations later on
	def __init__(self,):
		self.encoder_dict = defaultdict(LabelEncoder)

	def transform_continuous_columns(self, df, cols):
		features = df[cols]
		scaler = MinMaxScaler().fit(features.values)
		features = scaler.transform(features.values)
		df[cols] = features
		return df

	def fit_categorical_transformer(self, df, cols):
	#     df_joined = learn_df.append(test_df, ignore_index=True)
		for col in cols:
			self.encoder_dict[col].fit(df[col])

	def transform_categorical_columns(self, df, cols):
		# Encoding the variable
		df[cols] = (df[cols].apply(lambda x: self.encoder_dict[x.name].transform(x)))
		return df

	def inverse_transform_categorical(self, df, cols):
		df[cols] = (df[cols].apply(lambda x: self.encoder_dict[x.name].inverse_transform(x)))
		return df

def prepare_data(df, train=False):
	target = 'TARGET_FLAG'
	dollar_cols = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM']
	continuous_cols = ['AGE', 'KIDSDRIV', 'HOMEKIDS', 'YOJ', 'TRAVTIME', 'TIF', 'CLM_FREQ', 'MVR_PTS', 'CAR_AGE']
	continuous_cols.extend(dollar_cols)
	categorical_cols = ['PARENT1', 'MSTATUS', 'SEX', 'EDUCATION', 'JOB', 'CAR_USE', 'CAR_TYPE', 'RED_CAR', 'REVOKED', 'URBANICITY']

	# 1. Reading and cleaning data
	df = clean_dataframe(df, dollar_cols, True)

	# 2. Using MICE Imputation to fill in missing values
	mice_imputer = MiceImputer(df[continuous_cols+categorical_cols], categorical_cols)
	mice_imputer.mice_impute()
	df[continuous_cols+categorical_cols] = mice_imputer.df[continuous_cols+categorical_cols]

	# 3. Create New Features
	df, new_features = create_features(df)
	categorical_cols += new_features

	# 4. Transform Features
	transformer = FeatureTransformer()
	transformer.fit_categorical_transformer(df, categorical_cols)
	df = transformer.transform_categorical_columns(df, categorical_cols)
	df[dollar_cols] = df[dollar_cols].apply(lambda x: np.log10(x+1))

	return df, categorical_cols, continuous_cols, target

def main(train):

	# if --train flag is set then we train models on data, compare performance and then predict on test data as well
	if train:
		df_train = pd.read_csv(TRAIN_PATH)
		df_train, categorical_cols, continuous_cols, target = prepare_data(df_train, train=True)
		X = pd.DataFrame(data=df_train, columns = categorical_cols+continuous_cols,)
		y = df_train[target]

		train_xgb = ClassifierModels(X, y)

		# 1. Compare different algorithms and print performance metrics : AUC, F1 and Accuracy
		train_xgb.compare_models()

		# 2. Choose xgboost as the best model and fit on data with best parameters
		train_xgb.fit_train()

	df_test = pd.read_csv(TEST_PATH)
	df_test, categorical_cols, continuous_cols, target = prepare_data(df_test)
	X_test = pd.DataFrame(data=df_test, columns = categorical_cols + continuous_cols,)
	# All the values of TARGET_FLAG are Nans
	y_test = pd.DataFrame(data=df_test, columns = ['INDEX', target],)

	model_xgb = pickle.load(open(MODEL_PATH, 'rb'))
	preds_y = model_xgb.predict(X_test)
	# Assign predictions to the right index and write to .csv file
	y_test[target] = preds_y
	y_test.to_csv('../predicted_TARGET_FLAG.csv', index=False)

	print('\nPredictions saved to file predicted_TARGET_FLAG.csv\n')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', action='store_true')
	args = parser.parse_args()
	train = args.train

	main(train)

