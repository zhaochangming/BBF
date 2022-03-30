from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from .BF import BFClassifier, BFRegressor
from sklearn.ensemble._base import _set_random_states
from sklearn.base import clone


class BBFClassifier(BaggingClassifier):
    """
        Construct a BoostForestClassifier, referred to ``sklearn.ensemble.BaggingClassifier``.

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

        initLearner: obj, default=None
                Controls the initial model.

        n_estimators : int, default=10
            The number of base estimators in the ensemble.

        max_samples : int or float, default=1.0
            The number of samples to draw from X to train each base estimator (with
            replacement by default, see `bootstrap` for more details).

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

        max_features : int or float, default=1.0
            The number of features to draw from X to train each base estimator (
            without replacement by default, see `bootstrap_features` for more
            details).

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

        bootstrap : bool, default=True
            Whether samples are drawn with replacement. If False, sampling
            without replacement is performed.

        bootstrap_features : bool, default=False
            Whether features are drawn with replacement.

        oob_score : bool, default=False
            Whether to use out-of-bag samples to estimate
            the generalization error.

        warm_start : bool, default=False
            When set to True, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit
            a whole new ensemble.

        n_jobs : int, default=None
            The number of jobs to run in parallel for both :meth:`fit` and
            :meth:`predict`. ``None`` means 1 unless in a
            :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors.

        random_state : int or RandomState, default=None
            Controls the random resampling of the original dataset
            (sample wise and feature wise).
            If the base estimator accepts a `random_state` attribute, a different
            seed is generated for each instance in the ensemble.
            Pass an int for reproducible output across multiple function calls.

        verbose : int, default=0
        Controls the verbosity when fitting and predicting.
    """

    def __init__(self, max_iterations=10, active_function='relu', n_nodes_H=100, reg_alpha=0.001, boosting_model='ridge',
                 batch_size=256, learning_rate=0.05, initLearner=None, **kwargs):
        BaggingClassifier.__init__(self,
                                   base_estimator=BFClassifier(max_iterations=max_iterations, active_function=active_function,
                                                               n_nodes_H=n_nodes_H, reg_alpha=reg_alpha, boosting_model=boosting_model,
                                                               batch_size=batch_size, learning_rate=learning_rate,
                                                               initLearner=None),
                                   **kwargs)
        self.initLearner = initLearner

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator_)
        estimator.set_params(**{p: getattr(self, p)
                                for p in self.estimator_params})

        if random_state is not None:
            _set_random_states(estimator, random_state)
        estimator.set_params(**{'initLearner': self.initLearner})
        if append:
            self.estimators_.append(estimator)

        return estimator


class BBFRegressor(BaggingRegressor):
    """
        Construct a BoostForestClassifier, referred to ``sklearn.ensemble.BaggingRegressor``.

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

        initLearner: obj, default=None
                Controls the initial model.

        n_estimators : int, default=10
            The number of base estimators in the ensemble.

        max_samples : int or float, default=1.0
            The number of samples to draw from X to train each base estimator (with
            replacement by default, see `bootstrap` for more details).

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

        max_features : int or float, default=1.0
            The number of features to draw from X to train each base estimator (
            without replacement by default, see `bootstrap_features` for more
            details).

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

        bootstrap : bool, default=True
            Whether samples are drawn with replacement. If False, sampling
            without replacement is performed.

        bootstrap_features : bool, default=False
            Whether features are drawn with replacement.

        oob_score : bool, default=False
            Whether to use out-of-bag samples to estimate
            the generalization error.

        warm_start : bool, default=False
            When set to True, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit
            a whole new ensemble.

        n_jobs : int, default=None
            The number of jobs to run in parallel for both :meth:`fit` and
            :meth:`predict`. ``None`` means 1 unless in a
            :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors.

        random_state : int or RandomState, default=None
            Controls the random resampling of the original dataset
            (sample wise and feature wise).
            If the base estimator accepts a `random_state` attribute, a different
            seed is generated for each instance in the ensemble.

        verbose : int, default=0
            Controls the verbosity when fitting and predicting.
    """

    def __init__(self, max_iterations=10, active_function='relu', n_nodes_H=100, reg_alpha=0.001, boosting_model='ridge',
                 batch_size=256, learning_rate=0.05, initLearner=None, **kwargs):
        BaggingRegressor.__init__(self,
                                  base_estimator=BFRegressor(max_iterations=max_iterations, active_function=active_function,
                                                             n_nodes_H=n_nodes_H, reg_alpha=reg_alpha, boosting_model=boosting_model,
                                                             batch_size=batch_size, learning_rate=learning_rate, initLearner=None),
                                  **kwargs)
        self.initLearner = initLearner

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator_)
        estimator.set_params(**{p: getattr(self, p)
                                for p in self.estimator_params})

        if random_state is not None:
            _set_random_states(estimator, random_state)
        estimator.set_params(**{'initLearner': self.initLearner})
        if append:
            self.estimators_.append(estimator)

        return estimator
