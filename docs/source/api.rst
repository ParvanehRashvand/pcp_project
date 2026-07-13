Estimators API
==============

EEG preprocessing transformers
------------------------------

StateSelector
~~~~~~~~~~~~~~

.. autoclass:: pcp_project.estimators.StateSelector
   :members:

BandPassFilter
~~~~~~~~~~~~~~

.. autoclass:: pcp_project.estimators.BandPassFilter
   :members:

NotchFilter
~~~~~~~~~~~

.. autoclass:: pcp_project.estimators.NotchFilter
   :members:

BatchCovariances
~~~~~~~~~~~~~~~~

.. autoclass:: pcp_project.estimators.BatchCovariances
   :members:

MeanProbabilityAggregator
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pcp_project.estimators.MeanProbabilityAggregator
   :members:

SlidingWindow
~~~~~~~~~~~~~

.. autoclass:: pcp_project.estimators.SlidingWindow
   :members:


Covariance helper functions
---------------------------

.. autofunction:: pcp_project.estimators.batch_empirical_covariance

.. autofunction:: pcp_project.estimators.batch_ledoit_wolf_shrinkage

.. autofunction:: pcp_project.estimators.batch_ledoit_wolf

.. autofunction:: pcp_project.estimators.batch_oas