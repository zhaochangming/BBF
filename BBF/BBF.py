from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from .BF import BFClassifier, BFRegressor


class BBFClassifier(BaggingClassifier):
    """
        Construct a BoostForestClassifier, referred to ``sklearn.ensemble.BaggingClassifier``.

        Parameters
        ----------


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
    def __init__(self, , **kwargs):
        BaggingClassifier.__init__(self,
                                   base_estimator=BFClassifier(), **kwargs)


class BBFRegressor(BaggingRegressor):
    """
        Construct a BoostForestClassifier, referred to ``sklearn.ensemble.BaggingRegressor``.

        Parameters
        ----------


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
    def __init__(self, , **kwargs):
        BaggingRegressor.__init__(self,
                                  base_estimator=BFRegressor(), **kwargs)
