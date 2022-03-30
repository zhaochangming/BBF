# BBF
This page shows some Python demos.
The related data can be downloaded at [data](https://github.com/zhaochangming/BBF/tree/main/data).
## BBF for Classification

```python
from BBF import BFClassifier, BBFClassifier
import joblib
from sklearn.linear_model import RidgeClassifier
import numpy as np

data = joblib.load('../../data/test_data.joblib')
X, y = np.r_[data['training_data'], data['eval_data']], np.r_[data['training_label'], data['eval_label']]
test_X, test_y = data['testing_data'], data['testing_label']
init_clf = RidgeClassifier()
init_clf.fit(X, y)
print('ACC of baseline: %.3f' % init_clf.score(test_X, test_y))
clf = BFClassifier(max_iterations=100, n_nodes_H=100, learning_rate=0.2, initLearner=init_clf)
clf.fit(X, y)
print('ACC of BFClassifier: %.3f' % clf.score(test_X, test_y))
clf = BBFClassifier(max_iterations=100, n_nodes_H=100, learning_rate=0.2, n_estimators=20, initLearner=init_clf)
clf.fit(X, y)
print('ACC of BBFClassifier: %.3f' % clf.score(test_X, test_y))
```

    ACC of baseline: 0.842
    ACC of BFClassifier: 0.834
    ACC of BBFClassifier: 0.854


## BBF for Regression


```python
from BBF import BFRegressor, BBFRegressor
import joblib
from sklearn.linear_model import Ridge
import numpy as np

data = joblib.load('data/test_data.joblib')
X, y = np.r_[data['training_data'], data['eval_data']], np.r_[data['training_label'], data['eval_label']]
test_X, test_y = data['testing_data'], data['testing_label']
init_reg = Ridge()
init_reg.fit(X, y)
print('R^2 of baseline: %.3f' % init_reg.score(test_X, test_y))
reg = BFRegressor(max_iterations=100, n_nodes_H=100, learning_rate=0.05, initLearner=init_reg)
reg.fit(X, y, eval_data=(test_X, test_y))
print('R^2 of BFRegressor: %.3f' % reg.score(test_X, test_y))
reg = BBFRegressor(max_iterations=100, n_nodes_H=100, learning_rate=0.05, n_estimators=20, initLearner=init_reg)
reg.fit(X, y)
print('R^2 of BBFRegressor: %.3f' % reg.score(test_X, test_y)) 
```

    R^2 of baseline: 0.443
    R^2 of BFRegressor: 0.479
    R^2 of BBFRegressor: 0.518

