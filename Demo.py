from BBF import BFClassifier, BBFClassifier, BFRegressor, BBFRegressor
import joblib
from sklearn.linear_model import RidgeClassifier, Ridge
import numpy as np

data = joblib.load('data/test_data.joblib')
X, y = np.r_[data['training_data'], data['eval_data']], np.r_[data['training_label'], data['eval_label']]
test_X, test_y = data['testing_data'], data['testing_label']
init_clf = RidgeClassifier()
init_clf.fit(X, y)
print('Baseline: ', init_clf.score(test_X, test_y))
clf = BFClassifier(max_iterations=100, n_nodes_H=100, learning_rate=0.2, initLearner=init_clf)
clf.fit(X, y)
print('BFClassifier: ', clf.score(test_X, test_y))
clf = BBFClassifier(max_iterations=100, n_nodes_H=100, learning_rate=0.2, n_estimators=20, initLearner=init_clf)
clf.fit(X, y)
print('BBFClassifier: ', clf.score(test_X, test_y))
init_reg = Ridge()
init_reg.fit(X, y)
print('Baseline: ', init_reg.score(test_X, test_y))
reg = BFRegressor(max_iterations=100, n_nodes_H=100, learning_rate=0.05, initLearner=init_reg)
reg.fit(X, y, eval_data=(test_X, test_y))
print('BFRegressor: ', reg.score(test_X, test_y))
reg = BBFRegressor(max_iterations=100, n_nodes_H=100, learning_rate=0.05, n_estimators=20, initLearner=init_reg)
reg.fit(X, y)
print('BBFRegressor: ', reg.score(test_X, test_y))
