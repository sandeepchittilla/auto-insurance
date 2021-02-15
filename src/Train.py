import pickle

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, plot_roc_curve, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


MODEL_PATH = '../model/xgb_classifier.pkl'
# Performed Stratified CV Grid Search to find best parameters
BEST_PARAMS = {'colsample_bytree': 0.8,
 'gamma': 5,
 'learning_rate': 0.1,
 'max_depth': 5,
 'min_child_weight': 10,
 'n_estimators': 600}

class ClassifierModels():
	def __init__(self, X, y):
		self.X = X
		self.y = y
		self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.20, random_state=42)

	def compare_models(self,):

		fprs = []
		tprs = []
		roc_aucs = []
		f1scores = []
		accuracies = []
		classification_reports = []
		# I am comparing 3 algorithms here with their AUC, F1scores
		models = [LogisticRegression(), RandomForestClassifier(), xgb.XGBClassifier()]

		def display_performance(models, roc_aucs, f1scores, accuracies):
			for model_, auc_, f1_, acc_ in zip(models, roc_aucs, f1scores, accuracies):
				print(f'\n********* {model_} **********')
				print(f'\tROC AUC : {auc_}\n\tF1 : {f1_}\n\tAccuracy : {acc_}')

		for model in models:
			print(f'Fitting {model} model ...')
			model.fit(self.X_train, self.y_train)
			# predict probabilities
			probs = model.predict_proba(self.X_val)
			# predict classes
			preds_target = model.predict(self.X_val)
			# compute False Positive Rate, True Positive Rate and Thresholds
			fpr, tpr, threshold = roc_curve(self.y_val, probs[:,1])
			# build classification report with Precision, Recall, F1 scores
			classification_reports.append(classification_report(self.y_val, preds_target))
			precision, recall, pr_threshold = precision_recall_curve(self.y_val, probs[:,1])
			fprs.append(fpr)
			tprs.append(tpr)
			f1scores.append(f1_score(self.y_val, preds_target))
			accuracies.append(accuracy_score(self.y_val, preds_target))
			roc_aucs.append(auc(fpr,tpr))

		display_performance(models, roc_aucs, f1scores, accuracies)

	def fit_train(self,):
		model_xgb = xgb.XGBClassifier(params=BEST_PARAMS, silent=True)
		model_xgb.fit(self.X, self.y)

		pickle.dump(model_xgb, open(MODEL_PATH, 'wb'))
		print('\nModel saved to ../models/')
