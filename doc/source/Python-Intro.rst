Python-package Introduction
===========================

This document gives a basic walk-through of BBF Python-package.

**List of other helpful links**

-  `Python API <Python-API.html>`__

-  `Parameters Tuning <Parameters-Tuning.html>`__

-  `Python Examples <Demo.html>`__

Install
-------

The preferred way to install BBF is via pip from `Pypi <https://pypi.org/project/BBF>`__:

::

    pip install BBF


To verify your installation, try to ``import BBF`` in Python:

::

    import BBF

Data Interface
--------------

The BBF Python module can load data from:

-  NumPy 2D array(s)
    .. code:: python

        import numpy as np
        data = np.random.rand(500, 10)
        label = np.random.randint(2, size=500)



Setting Parameters
------------------

BBF can use a dictionary to set parameters.
For instance:

   .. code:: python

       param = {'max_iterations': 200, 'active_function': 'relu', 'n_nodes_H': 100, 'reg_alpha': 0.001, 'random_state': 0}


Training
--------

Training a model requires a parameter dictionary and data set:

.. code:: python


    estimator = BBF.BFClassifier(**param).fit(data, label)

After training, the model can be saved:

.. code:: python

    estimator.save_model('model.joblib')

A saved model can be loaded:

.. code:: python

    import joblib
    estimator = joblib.load('model.joblib')


Predicting
----------

A model that has been trained or loaded can perform predictions on datasets:

.. code:: python

    # 7 entities, each contains 10 features
    data = np.random.rand(7, 10)
    ypred = estimator.predict(data)
