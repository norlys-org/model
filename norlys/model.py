from norlys.data_utils import get_training_data
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import logging
import joblib

def train_0m_classifier(log=True):
	"""
	Train the '0m' classifier, i.e. the model predicting the label for the present. 
	"""

	logging.info('Gathering training data...')
	X_train, X_test, y_train, y_test = get_training_data('label')

	logging.info('Training classifier...')
	clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
	clf.fit(X_train, y_train)
	
	logging.info('Running tests to compute accuracy score...')
	y_pred = clf.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	logging.info(f'Trained ExtraTreesClassifier 0m, accurary: {accuracy * 100}%')

	logging.info('Saving model to `0m-model.joblib`')
	joblib.dump(clf, '0m-model.joblib')

def load_0m_classifier():
	logging.info('Loading 0m-model.joblib...')
	return joblib.load('0m-model.joblib')
