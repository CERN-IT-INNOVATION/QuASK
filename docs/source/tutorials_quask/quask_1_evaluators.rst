Criteria to evaluate a quantum kernel
=====================================

One of the main features of *quask* is the opportunity to evaluate a
quantum kernel according to the various criteria proposed in the
literature. These criteria are especially important in the context of
seeking a quantum advantage, a model that outperforms existing classical
choices.

All the criteria are available as classes that inherit the abstract
class ``KernelEvaluator``. This object has only one abstract method,
``evaluate``, which takes four arguments:

1. The ``Kernel`` object.
2. The set of training data ``X``, which might be used by some criteria.
3. The set of labels ``y``, which might be used by some criteria in
   conjunction with the training data.
4. The kernel Gram matrix ``K``, which is entirely optional and can be
   built from ``kappa`` and ``X``.

The argument ``K`` is provided in case such an object has been
previously calculated and is kept for the purpose of speeding up the
computation.

Depending uniquely on the structure of the kernel
-------------------------------------------------

These first criteria measure the expressibility of the given ansatz,
thus do not need any information about the data used.

Haar evaluator evaluator
~~~~~~~~~~~~~~~~~~~~~~~~

A criteria inspired by the definition of expressiblity given in [sim19].
A discretized, approximated version of this metric is given, and
compares the histogram of inner products between Haar random vectors,
with the inner product of vectors generated with the given kernel kappa.
Note that, for :math:`n \to \infty`, the Haar random histogram
concentrates around zero.

.. code:: ipython3

    from quask.evaluator import HaarEvaluator
    n = 100 # number of bins discretizing the histogram
    m = 10000 #number of randomly sampled data for creating the ansatz's ensemble of states
    h_eval = HaarEvaluator(n, m)

Lie Rank evaluator evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A criteria inspired by the work in [lar21]. The rank of the Lie algebra
associated with the ansatz is computed, truncated to a maximum value
:math:`T`. In this case, the criteria can both be associated with the
expressibility (higher rank leads to higher expressibility) or with the
efficiency of simulation on a classical device (higher rank leads to
harder to simulate unitaries).

.. code:: ipython3

    from quask.evaluator import LieRankEvaluator
    t = 10_000 # threshold on the rank of the Lie algebra
    lr_eval = LieRankEvaluator(t)

Covering numbers evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~

A criteria inspired by the work in [du22]. The expressibility is upper
bounded by a quantity exponential in the number of trainable gates. In
our context, it is quite a loose bound, but the original article allows
to consider a more precise bounds in some cases.

.. code:: ipython3

    from quask.evaluator import CoveringNumberEvaluator
    cn_eval = CoveringNumberEvaluator()

Depending on the kernel and on the training features, but not on the labels
---------------------------------------------------------------------------

These criteria depends on the kernel itself and the training data.

Geometric difference evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A criteria inspired by the work in [hua21]. The geometric difference has
been extensively studied in the `Projected quantum kernels
tutorial <%22../tutorial_quantum/quantum_2_projected%22>`__. It is used
as follows:

.. code:: ipython3

    from quask.evaluator import GeometricDifferenceEvaluator
    
    Kc1 = ... # first classical kernel
    Kc2 = ... # second classical kernel
    # ...
    Kc100 = ... # last classical kernel
    lam = 0.0001 # regularization 
    
    gd_eval = GeometricDifferenceEvaluator([Kc1, Kc2, ..., Kc100], lam)

Depending on the kernel, the training features and training labels
------------------------------------------------------------------

Kernel alignment evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~

A criteria inspired by the work in [cri01].

.. code:: ipython3

    from quask.evaluator import KernelAlignmentEvaluator
    ka_eval = KernelAlignmentEvaluator()

Centered Kernel alignment evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A criteria inspired by the work in [cor12].

.. code:: ipython3

    from quask.evaluator import CenteredKernelAlignmentEvaluator
    cka_eval = CenteredKernelAlignmentEvaluator()

Ridge generalization evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from quask.evaluator import RidgeGeneralizationEvaluator
    rg_eval = RidgeGeneralizationEvaluator()

‘S’ model complexity evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A criteria inspired by the work in [hua21]. The ‘S’ model complexity has
been extensively studied in the `Projected quantum kernels
tutorial <%22../tutorial_quantum/quantum3_spectralbias.html%22>`__. It
is used as follows:

.. code:: ipython3

    from quask.evaluator import EssModelComplexityEvaluator
    smc_eval = EssModelComplexityEvaluator()

Spectral bias evaluator
~~~~~~~~~~~~~~~~~~~~~~~

A criteria inspired by the work in [can21].

.. code:: ipython3

    from quask.evaluator import SpectralBiasEvaluator
    sb_eval = SpectralBiasEvaluator(10)

Add your own criteria
---------------------

References
----------

[sim19] Sim, Sukin, Peter D. Johnson, and Alán Aspuru‐Guzik.
“Expressibility and entangling capability of parameterized quantum
circuits for hybrid quantum‐classical algorithms.” Advanced Quantum
Technologies 2.12 (2019): 1900070.

[lar21] Larocca, Martin, et al. “Diagnosing barren plateaus with tools
from quantum optimal control.” Quantum 6 (2022): 824.

[du22] Du, Yuxuan, et al. “Efficient measure for the expressivity of
variational quantum algorithms.” Physical Review Letters 128.8 (2022):
080506.

[cri01] Cristianini, Nello, et al. “On kernel-target alignment.”
Advances in neural information processing systems 14 (2001).

[cor12] Cortes, Corinna, Mehryar Mohri, and Afshin Rostamizadeh.
“Algorithms for learning kernels based on centered alignment.” The
Journal of Machine Learning Research 13.1 (2012): 795-828.

[can21] Canatar, Abdulkadir, Blake Bordelon, and Cengiz Pehlevan.
“Spectral bias and task-model alignment explain generalization in kernel
regression and infinitely wide neural networks.” Nature communications
12.1 (2021): 2914.

[hua21] Huang, HY., Broughton, M., Mohseni, M. et al. Power of data in
quantum machine learning. Nat Commun 12, 2631 (2021).
https://doi.org/10.1038/s41467-021-22539-9

.. note::

   Author's note.

