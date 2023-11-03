Expressibility in quantum kernels
=================================

In this tutorial, we start to charatetize the properties of a quantum
kernel. The *expressibility* is definitively the most important. *quask*
has some built in facilities to evaluate a quantum kernel from this
perspective.

Expressibility of a parameterized quantum circuit
-------------------------------------------------

The concept of *expressibility* for :math:`U(\mathbf{x})` can be
understood as its capacity to distribute classical data throughout the
Hilbert space of the quantum system. To formally quantify this
expressibility, we rely on the norm of the super-operator :math:`A`,
given by :math:`||A||=\mathrm{Trace}[AA^\dagger]`. This can be defined
as:

.. math::


   A = \int_\text{Haar} (\ketbra{\phi}{\phi})^{\otimes t} d\phi - \int_\Theta (U(\mathbf{\theta}) \ketbra{0}{0} U^\dagger (\mathbf{\theta}))^{\otimes t} d\mathbf{\theta},

where :math:`t` is an integer with :math:`t \ge 2` [sim19].

The super-operator :math:`A` quantifies the extent to which the ensemble
of states, obtained by initiating the system in :math:`\ket{0}` and
evolving it with :math:`U(\theta)` (where :math:`\theta` is randomly
chosen), deviates from the Haar-random ensemble of states. The
Haar-random ensemble represents a uniform distribution of quantum states
throughout the Hilbert space. When this deviation is small, we consider
the unitary :math:`U` to be expressible.

As comparing a distribution to the true Haar-random distribution of
states can be challenging, the concept of
:math:`\varepsilon`-approximate state :math:`t`-design is employed. This
ensemble of states closely approximates the Haar random ensemble of
states up to the :math:`t`-th statistical moment, hence the parameter
:math:`t`. For :math:`t = 1`, only the first statistical moment (the
average) is considered, which is generally less informative. For
:math:`t=2`, both the average and standard deviation are taken into
account, making it a suitable choice for most use cases. The higher the
value of :math:`t`, the more precise the quantification of the
deviation, but the more computationally expensive the calculation
becomes.

Understanding the expressibility of the parameterized quantum circuit
:math:`U` provides insights into the potential performance of the kernel
that utilizes :math:`U`. Further details on these aspects are covered in
the tutorial `Spectral Bias <xxx>`__.

The ``KernelEvaluator`` object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In *quask*, you can assess a kernel object based on specific criteria
using the ``KernelEvaluator``. Each sub-class of KernelEvaluator
contains a single method, ``evaluate``, which provides a cost associated
with the given object. The kernel’s quality with respect to a particular
criterion is indicated by a lower cost. Additionally, the cost may be
influenced by a set of data, as is the case with accuracy.

The ``HaarEvaluator`` object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``HaarEvaluator`` class inherits ``KernelEvaluator`` and is used to
calculate :math:`||A||`. It can be used as follows,

.. code:: ipython3

    from quask.core import Ansatz, Kernel, KernelFactory, KernelType
    from quask.evaluator import HaarEvaluator
    
    N_FEATURES = 2
    N_OPERATIONS = 3
    N_QUBITS = 2
    ansatz = Ansatz(n_features=N_FEATURES, n_qubits=N_QUBITS, n_operations=N_OPERATIONS)
    ansatz.initialize_to_identity()
    ansatz.change_operation(0, new_feature=0, new_wires=[0, 1], new_generator="ZZ", new_bandwidth=1.0)
    ansatz.change_operation(1, new_feature=1, new_wires=[0, 1], new_generator="XX", new_bandwidth=1.0)
    ansatz.change_operation(2, new_feature=2, new_wires=[0, 1], new_generator="IX", new_bandwidth=0.123)
    kernel = KernelFactory.create_kernel(ansatz, "Z" * N_QUBITS, KernelType.FIDELITY)
    
    he = HaarEvaluator(n_bins=40, n_samples=10000)
    cost = he.evaluate(kernel=kernel, K=None, X=None, y=None)
    print(f"Cost (norm of A): {cost:3.5f}")


.. parsed-literal::

    Cost (norm of A): 0.16714


Usually, the kernel evaluator needs only the ``kernel`` argument. If the
criteria depend on the data, ``X`` and ``y`` parameters must be
provided, which correspond to the dataset features and dataset labels,
respectively. If the kernel Gram matrix must be calculated, it will be
done using ``kernel`` and ``X``; however, if it has been pre-calculated
previously, then it should be passed to the ``K`` parameter to avoid
losing time to re-calculate it.

Dynamical Lie algebra of a parameterized quantum circuit
--------------------------------------------------------

The parameterized unitary transformation :math:`U(\cdot)`, which
operates on :math:`n` qubits, can be expressed as a sum of Pauli
generators:

.. math:: U(\mathbf{x}) = \exp\left(-i \sum_{j = 0}^{4^n-1} f_j(\mathbf{x}) \sigma_j \right)

Here, :math:`f_j(\mathbf{x})` belongs to :math:`\mathbb{R}`, and
:math:`\sigma_j` represents the tensor product of :math:`n` Pauli
matrices. For instance:

-  :math:`\sigma_0 = \mathrm{Id} \otimes \mathrm{Id} \otimes ... \otimes \mathrm{Id}`,
-  :math:`\sigma_1 = \mathrm{Id} \otimes \mathrm{Id} \otimes ... \otimes X`,
-  :math:`\sigma_{4^n-1} = Z \otimes Z \otimes ... \otimes Z`.

Each unitary transformation can depend on all :math:`4^n` generators or
only on a subset of them. In the context of a quantum circuit, :math:`U`
is represented by the product of elementary one-qubit and two-qubit
gates. To determine the generator of this transformation, we employ the
tools of the Dynamical Lie Algebra, a vector space with a bilinear
operation denoted as the commutator
(:math:`[\sigma, \eta] = \sigma\eta - \eta\sigma`). This vector space is
spanned by the generators of :math:`U`, which include the generators of
both single one- and two-qubit gates and the generators derived through
repeated application of the commutator until a fixed point is reached.

The number of generators is the rank of the Dynamical Lie Algebra.

The ``LieRankEvaluator``
~~~~~~~~~~~~~~~~~~~~~~~~

The rank of the Dynamical Lie Algebra can be determined using the
``LieRankEvaluator``. This class inherits from ``KernelEvaluator`` and
is employed in a manner similar to the previous examples. It’s important
to note that from a computational perspective, exact calculation is
feasible only for small values of :math:`n` or specific cases. In the
worst-case scenario, calculating an exponential number of commutations
is required to cover all the :math:`4^n` potential Pauli strings. To
address this, the search can be truncated once a predefined threshold
:math:`T` is reached.

.. code:: ipython3

    from quask.evaluator import LieRankEvaluator
    lre = LieRankEvaluator(T=500)
    cost = lre.evaluate(kernel=kernel, K=None, X=None, y=None)
    print(f"Cost (-1 * rank of DLA): {cost:3.5f}")


.. parsed-literal::

    Cost (-1 * rank of DLA): -8.00000


Use of the DLA in quantum kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The application of the Dynamical Lie Algebra (DLA) in quantum machine
learning has significantly contributed to the theoretical development of
various tools. It has been demonstrated that the rank of the DLA serves
as a proxy for expressibility. Essentially, the more generators a
unitary transformation :math:`U` possesses, the greater the capacity to
map quantum states across the Hilbert space of the quantum system
[lar21].

However, it’s worth noting that this measure lacks some precision. For
instance, it doesn’t account for the density of the distribution of
quantum states, a consideration addressed by the norm of the
super-operator :math:`A`. Moreover, when we introduce a bandwidth
parameter :math:`\beta` to restrict the rotational angles, we
effectively limit the region in which states can be mapped. In such
cases, we may encounter a unitary transformation that, despite having an
exponential number of generators, exhibits only mild expressibility.

The rank of the DLA also sheds light on another intriguing aspect. Some
relatively simple quantum circuits can be efficiently simulated on
classical computers, rendering the use of quantum hardware redundant.
This is particularly evident for circuits consisting solely of
single-qubit gates. [som06] has established that unitary transformations
with a polynomial number of generators can be efficiently simulated in
polynomial time on classical hardware. While the reverse is not
universally proven, having a multitude of generators offers favorable
evidence that can be used to speculate that the chosen quantum circuit
is challenging to simulate classically.

References
----------

[sim19] Sim, Sukin, Peter D. Johnson, and Alán Aspuru‐Guzik.
“Expressibility and entangling capability of parameterized quantum
circuits for hybrid quantum‐classical algorithms.” Advanced Quantum
Technologies 2.12 (2019): 1900070.

[lar21] Larocca, Martin, et al. “Diagnosing barren plateaus with tools
from quantum optimal control.” Quantum 6 (2022): 824.

[som06] Somma, Rolando, et al. “Efficient solvability of Hamiltonians
and limits on the power of some quantum computational models.” Physical
review letters 97.19 (2006): 190501.

.. code:: ipython3

    .. note::
    
       Author's note.
