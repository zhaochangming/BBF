import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy.special import softmax, expit
from copy import deepcopy
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import column_or_1d
from sklearn.utils.estimator_checks import check_estimator
import joblib

MAX_RESPONSE = 4
MACHINE_EPSILON = np.finfo(np.float64).eps


class NNLinear:
    def __init__(self, W, bias):
        self.W = W
        self.bias = bias

    def decision_function(self, X):
        return X @ self.W + self.bias.reshape((1, -1))

    def predict(self, X):
        return X @ self.W + self.bias.reshape((1, -1))


class myEncoder:
    def __init__(self):

        self.labelEncoder = preprocessing.LabelBinarizer(pos_label=1, neg_label=0)

    def fit(self, y):
        self.labelEncoder.fit(y)
        self.n_classes_ = len(self.labelEncoder.classes_)

    def transform(self, y):
        if self.n_classes_ == 2:
            y_ = self.labelEncoder.transform(y)
            return np.c_[1 - y_, y_]
        else:
            return self.labelEncoder.transform(y)


class node_generator:
    def __init__(self, active_function='relu', n_nodes_H=10):
        self.n_nodes_H = n_nodes_H
        self.active_function = active_function

    @staticmethod
    def _sigmoid(data):
        return 1.0 / (1 + np.exp(-data))

    @staticmethod
    def _linear(data):
        return data

    @staticmethod
    def _tanh(data):
        return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))

    @staticmethod
    def _relu(data):
        return np.maximum(data, 0)

    @staticmethod
    def _generator(fea_dim, node_size):
        W = 2 * random.random(size=(fea_dim, node_size)) - 1
        b = 2 * random.random() - 1
        return W, b

    def _generate_h(self, X, sample_weight=None):

        self.nonlinear_ = {
            'linear': self._linear,
            'sigmoid': self._sigmoid,
            'tanh': self._tanh,
            'relu': self._relu
        }[self.active_function]
        fea_dim = X.shape[1]
        W_, b_ = self._generator(fea_dim, self.n_nodes_H)
        data_ = np.dot(X, W_) + b_
        scaler_ = StandardScaler()
        scaler_.fit(data_, sample_weight=sample_weight)
        try:
            self.W_.append(W_)
            self.b_.append(b_)
            self.scaler_.append(scaler_)
        except:
            self.W_ = []
            self.b_ = []
            self.scaler_ = []
            self.W_.append(W_)
            self.b_.append(b_)
            self.scaler_.append(scaler_)
        return self.nonlinear_(scaler_.transform(data_))

    def _transform_iter(self, X, i=None):
        return self.nonlinear_(self.scaler_[i].transform(X.dot(self.W_[i]) + self.b_[i]))


class BF(BaseEstimator, node_generator):
    def __init__(self, active_function='relu', n_nodes_H=10):
        node_generator.__init__(self, active_function, n_nodes_H)

    def _output2prob(self, output):
        if self.n_classes_ == 1:
            return expit(output)
        else:
            return softmax(output, axis=1)

    def save_model(self, file):
        """
        Parameters
        ----------
        file: str
            Controls the filename.
        """
        check_is_fitted(self, ['estimators_'])
        joblib.dump(self, filename=file)

    @staticmethod
    def _MixUp(X, y):
        l = np.random.beta(0.75, 0.75)
        l = max(l, 1 - l)
        random_index = np.random.choice(range(len(X)), len(X), replace=False)
        X_mix = l * X + (1 - l) * X[random_index]
        z_mix = l * y + (1 - l) * y[random_index]
        return X_mix, z_mix

    @staticmethod
    def _down_sample(index, weight, bs):
        if weight.sum() == 0:
            return random.choice(index, bs, replace=True)
        else:
            return random.choice(index, bs, replace=True, p=weight / weight.sum())


class BFClassifier(ClassifierMixin, BF):
    """
        TrBF classifier. Construct a TrBF model to fine-tune the initial model.

        Parameters
        ----------
        max_iterations: int, default=10
                Controls the number of boosting iterations.
        active_function: {str, ('relu', 'tanh', 'sigmoid' or 'linear')}, default='relu'
                        Controls the active function of enhancement nodes.
        n_nodes_H: int, default=100
                    Controls the number of enhancement nodes.
        reg_alpha: float, default=0.001
                    Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates. Larger values specify stronger regularization.
        verbose: bool, default=False
                Controls wether to show the boosting process.
        boosting_model: str, default='ridge'
                Controls the base learner used in boosting.
        batch_size: int, default=256
                Controls the batch size.
        learning_rate: float, default=0.05
                Controls the learning rate.
        noise_scale: float, default=1.0
                Controls the noise level.
        initLearner: obj, default=None
                Controls the initial model.
        random_state: int, default=0
                        Controls the randomness of the estimator.
    """

    def __init__(self, max_iterations=10, active_function='relu', n_nodes_H=100, reg_alpha=0.001, verbose=False, boosting_model='ridge',
                 batch_size=256, learning_rate=0.05, noise_scale=1.0, initLearner=None, random_state=0):
        BF.__init__(self, active_function, n_nodes_H)
        self.reg_alpha = reg_alpha
        self.noise_scale = noise_scale
        self.initLearner = initLearner
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.boosting_model = boosting_model
        self.verbose = verbose
        self.random_state = random_state

    def _predict_score(self, node_model, X):
        new_scores = [e.predict(X) for e in node_model]
        new_scores = np.asarray(new_scores).T
        new_scores = np.clip(new_scores, a_min=-MAX_RESPONSE, a_max=MAX_RESPONSE)
        if self.n_classes_ > 1:
            new_scores -= new_scores.mean(keepdims=True, axis=1)
            new_scores *= (self.n_classes_ - 1) / self.n_classes_
            new_scores = normalize(new_scores)
        return new_scores * self.learning_rate

    @staticmethod
    def _weight_and_response(y, prob):
        z = np.where(y, 1. / (prob + MACHINE_EPSILON), -1. / (1. - prob + MACHINE_EPSILON))
        z = np.clip(z, a_min=-MAX_RESPONSE, a_max=MAX_RESPONSE)
        sample_weight = (y - prob) / z
        sample_weight = np.maximum(sample_weight, 2. * MACHINE_EPSILON)
        return sample_weight, z

    def _get_init_output(self, X):
        if self.initLearner is not None:
            initOutput = self.initLearner.decision_function(X)
            if len(initOutput.shape) == 1:
                initOutput = np.c_[-initOutput, initOutput]
            initOutput -= initOutput.mean(keepdims=True, axis=1)
            initOutput = normalize(initOutput)
        else:
            initOutput = np.zeros((len(X), self.n_classes_))
        return initOutput

    def _get_balance_index(self, y, weight=None, bs=None):
        if weight is None:
            weight = np.ones(len(y))
        # balance index
        balance_index = []
        index_positive = np.where(y == 1)[0]
        index_negative = np.where(y == 0)[0]
        if len(index_positive) > 0:
            balance_index.extend(self._down_sample(index_positive, weight[index_positive], bs))
        if len(index_negative) > 0:
            balance_index.extend(self._down_sample(index_negative, weight[index_negative], bs))
        return balance_index

    def _decision_function(self, X, iter=None):
        if iter is None:
            iter = self.max_iterations
        output = self._get_init_output(X)
        for i in range(iter):
            output += self._predict_score(self.estimators_[i], self._transform_iter(X, i))
        return output

    def predict(self, X, iter=None):
        """
            Predict class labels for samples in X.

            Parameters
            ----------
            X : array_like or sparse matrix, shape (n_samples, n_features)
                Samples.

            iter: int
                Total number of iterations used in the prediction.

            Returns
            -------
            C : array, shape [n_samples]
                Predicted class label per sample.
        """
        check_is_fitted(self, ['estimators_'])
        X = check_array(X)
        y_pred = self._decision_function(X, iter)
        scores = y_pred.reshape(len(X), -1)
        scores = scores.ravel() if scores.shape[1] == 1 else scores
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def predict_proba(self, X, iter=None):
        """
            Predict class probabilities for samples in X.

            Parameters
            ----------
            X : array_like or sparse matrix, shape (n_samples, n_features)
                Samples.

            iter: int
                Total number of iterations used in the prediction.

            Returns
            -------
            p : array, shape [n_samples, n_classes]
                Predicted class probability per sample.
        """
        check_is_fitted(self, ['estimators_'])
        X = check_array(X)
        y_pred = self._decision_function(X, iter)
        scores = y_pred.reshape(len(X), -1)
        scores = scores.ravel() if scores.shape[1] == 1 else scores
        if self.n_classes_ == 1:
            prob = expit(scores)
        else:
            prob = softmax(scores, axis=1)
        return prob

    @property
    def classes_(self):
        """
        Classes labels
        """
        return self.labelEncoder_.labelEncoder.classes_

    def fit(self, X=None, y=None, eval_data=None):
        """
            Build a TrBF model.

            Parameters
            ----------
            X : {ndarray, sparse matrix} of shape (n_samples, n_features) or dict
                Training data.

            y : ndarray of shape (n_samples,)
                Target values.

            eval_data : tuple (X_test, y_test)
                tuple to use for watching the boosting process.

            Returns
            -------
            self : object
                Instance of the estimator.
        """
        np.random.seed(self.random_state)
        self.labelEncoder_ = myEncoder()
        self.labelEncoder_.fit(y)
        Y = self.labelEncoder_.transform(y)
        if not self.labelEncoder_.labelEncoder.y_type_.startswith('multilabel'):
            _ = column_or_1d(y, warn=True)
        else:
            raise ValueError(
                "%s doesn't support multi-label classification" % (
                    self.__class__.__name__))
        X, Y = check_X_y(X, Y, dtype=[np.float64, np.float32], multi_output=True, y_numeric=True)
        self.n_classes_ = len(self.classes_)
        model = Ridge(alpha=self.reg_alpha)
        output_total = self._get_init_output(X)
        prob = self._output2prob(output_total)
        if self.verbose:
            if Y is not None:
                print('Init Loss: {}, ACC: {}'.format(log_loss(Y, prob), accuracy_score(Y.argmax(axis=1), prob.argmax(axis=1))))
        H = self._generate_h(X)
        if eval_data is not None:
            eval_X, eval_y = eval_data
            eval_output = self._get_init_output(eval_X)
            if self.verbose:
                print('Init Test ACC: {}'.format(accuracy_score(eval_y, eval_output.argmax(axis=1))))
        else:
            eval_X, eval_y, eval_output = None, None, None
        self.estimators_ = []
        for i in range(self.max_iterations):
            new_estimators_ = []
            for j in range(self.n_classes_):
                model_copy = deepcopy(model)
                w, z = self._weight_and_response(Y[:, j], prob[:, j])
                batch_index = self._get_balance_index(Y[:, j], weight=w, bs=int(self.batch_size / 2.0))
                X_batch, z_batch = H[batch_index], z[batch_index]
                model_copy.fit(X_batch, z_batch)
                new_estimators_.append(model_copy)

            self.estimators_.append(new_estimators_)
            new_scores = self._predict_score(new_estimators_, H)
            output_total += new_scores
            prob = self._output2prob(output_total)
            if self.verbose:
                print('Iteration: {} Loss: {}, ACC: {}'.format(i + 1, log_loss(Y, prob),
                                                               accuracy_score(Y.argmax(axis=1), prob.argmax(axis=1))))
            if eval_data is not None:
                eval_H = self._transform_iter(eval_X, i)
                eval_output += self._predict_score(new_estimators_, eval_H)
                if self.verbose:
                    print('Iteration: {} Test ACC: {}'.format(i + 1, accuracy_score(eval_y, eval_output.argmax(axis=1))))
            if i < self.max_iterations - 1:
                H = self._generate_h(X)
        return self


class BFRegressor(RegressorMixin, BF):
    """
        TrBF regressor. Construct a TrBF model to fine-tune the initial model.

        Parameters
        ----------
        max_iterations: int, default=10
                Controls the number of boosting iterations.
        active_function: {str, ('relu', 'tanh', 'sigmoid' or 'linear')}, default='relu'
                        Controls the active function of enhancement nodes.
        n_nodes_H: int, default=100
                    Controls the number of enhancement nodes.
        reg_alpha: float, default=0.001
                    Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates. Larger values specify stronger regularization.
        verbose: bool, default=False
                Controls wether to show the boosting process.
        boosting_model: str, default='ridge'
                Controls the base learner used in boosting.
        batch_size: int, default=256
                Controls the batch size.
        learning_rate: float, default=0.05
                Controls the learning rate.
        noise_scale: float, default=1.0
                Controls the noise level.
        initLearner: obj, default=None
                Controls the initial model.
        random_state: int, default=0
                        Controls the randomness of the estimator.
    """

    def __init__(self, max_iterations=10, active_function='relu', n_nodes_H=100, reg_alpha=0.001, verbose=False, boosting_model='ridge',
                 batch_size=256, learning_rate=0.05, noise_scale=1.0, initLearner=None, random_state=0):
        BF.__init__(self, active_function, n_nodes_H)
        self.reg_alpha = reg_alpha
        self.noise_scale = noise_scale
        self.initLearner = initLearner
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.boosting_model = boosting_model
        self.verbose = verbose
        self.random_state = random_state

    def _predict_score(self, node_model, X):
        new_scores = node_model.predict(X)
        new_scores = np.clip(new_scores, a_min=-MAX_RESPONSE, a_max=MAX_RESPONSE)
        return new_scores * self.learning_rate

    @staticmethod
    def _weight_and_response(y, output):
        z = y - output
        sample_weight = np.ones(z.shape)
        z = np.clip(z, a_min=-MAX_RESPONSE, a_max=MAX_RESPONSE)
        return sample_weight, z

    def _get_init_output(self, X):
        if self.initLearner is not None:
            initOutput = self.initLearner.predict(X)
            initOutput = initOutput.reshape(len(X))
        else:
            initOutput = np.zeros(len(X))
        return initOutput

    def _decision_function(self, X, iter=None):
        if iter is None:
            iter = self.max_iterations
        output = self._get_init_output(X)
        for i in range(iter):
            output += self._predict_score(self.estimators_[i], self._transform_iter(X, i))
        return output

    def predict(self, X, iter=None):
        """
            Return the predicted value for each sample.

            Parameters
            ----------
            X : array_like or sparse matrix, shape (n_samples, n_features)
                Samples.

            iter: int
                Total number of iterations used in the prediction.
            Returns
            -------
            C : array, shape (n_samples,)
                Returns predicted values.
        """
        check_is_fitted(self, ['estimators_'])
        X = check_array(X)
        y_pred = self._decision_function(X, iter)
        y_pred = y_pred.reshape(len(X))
        return y_pred

    def fit(self, X=None, y=None, eval_data=None):
        """
            Build a TrBF model.

            Parameters
            ----------
            X : {ndarray, sparse matrix} of shape (n_samples, n_features) or dict
                Training data.

            y : ndarray of shape (n_samples,)
                Target values.

            eval_data : tuple (X_test, y_test)
                tuple to use for watching the boosting process.

            Returns
            -------
            self : object
                Instance of the estimator.
        """
        np.random.seed(self.random_state)
        y = column_or_1d(y, warn=True)
        X, y = check_X_y(X, y, dtype=[np.float64, np.float32], multi_output=True, y_numeric=True)
        model = Ridge(alpha=self.reg_alpha)
        output = self._get_init_output(X)
        if self.verbose:
            print('Init MSE Loss: '.format(mean_squared_error(y, output)))
        H = self._generate_h(X)
        if eval_data is not None:
            eval_X, eval_y = eval_data
            eval_output = self._get_init_output(eval_X)
            if self.verbose:
                print('Init Test MSE Loss: {}'.format(mean_squared_error(eval_y, eval_output)))
        else:
            eval_X, eval_y, eval_output = None, None, None
        self.estimators_ = []
        for i in range(self.max_iterations):
            new_estimators_ = deepcopy(model)
            w, z = self._weight_and_response(y, output)
            batch_index = random.choice(len(w), self.batch_size, replace=True, p=w / w.sum())
            X_batch, z_batch = H[batch_index], z[batch_index]
            new_estimators_.fit(X_batch, z_batch)
            self.estimators_.append(new_estimators_)
            new_scores = self._predict_score(new_estimators_, H)
            output += new_scores
            if self.verbose:
                print('Iteration: {} MSE Loss: {}'.format(i + 1, mean_squared_error(y, output)))
            if eval_data is not None:
                eval_output += self._predict_score(new_estimators_, self._transform_iter(eval_X, i))
                if self.verbose:
                    print('Iteration: {} Test MSE Loss: {}'.format(i + 1, mean_squared_error(eval_y, eval_output)))
            if i < self.max_iterations - 1:
                H = self._generate_h(X)
        return self


if __name__ == "__main__":
    check_estimator(BFClassifier())
    check_estimator(BFRegressor())
