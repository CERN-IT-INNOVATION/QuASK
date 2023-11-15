Projected quantum kernels
=========================

To understand Projected Quantum Kernels we should understand the limitations
of "traditional" quantum kernels. These limitations have very deep implications
in the understanding of kernel methods also on a classical perspective.

Expressibility and curse of dimensionality in kernel methods
------------------------------------------------------------

When approaching a ML problem we could ask if it makes sense at all to use
QML techniques, such as quantum kernel methods. We understood in the last years 
[kbs21],[Hua21] that having a large Hilbert space where we can compute 
classically intractable inner products does not guarantee an advantage. But, why?

When dealing with kernel methods, whether classical or quantum, we must
exercise caution when working in high-dimensional (or even
infinite-dimensional) Hilbert spaces. This is due to the fact that in
high dimensions, the problem of generalization becomes hard, *i.e.* the 
trained kernel is prone to overfitting.
In turn, an exponential (in the number of features/qubits) number of datapoints 
are needed to learn the target function we aim to estimate.
These phenomena are explored in the `upcoming tutorial <xxx>`__.

For instance, in the classical context, the Gaussian kernel maps any
:math:`\mathbf{x} \in \mathbb{R}^d` to a multi-dimensional Gaussian
distribution with an average of :math:`\mathbf{x}` and a covariance
matrix of :math:`\sigma I`. When :math:`\sigma` is small, data points
are mapped to different regions of this infinite-dimensional Hilbert
space, and :math:`\kappa(\mathbf{x}, \mathbf{x}') \approx 0` for all
:math:`\mathbf{x} \neq \mathbf{x}'`. This is known as the phenonenon of
*curse of dimensionality*, or *orthogonality catastrophe*. To avoid this, a larger
:math:`\sigma` is chosen to ensure that most data points relevant to our
task have some nontrivial overlap.

As the Hilbert space for quantum systems grows exponentially with the
number of qubits :math:`n`, similar challenges can arise when using
quantum kernels. This situation occurs with expressible
:math:`U(\cdot)`, which allows access to various regions within the
Hilbert space. In such cases, similar to classical kernels, techniques
must be employed to control expressibility and, consequently, the
model’s complexity.

Projection as expressibility control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The authors of [Hua21], who initially addressed the challenge of the
exponential dimensionality of Hilbert space in the context of quantum
kernels, have introduced the concept of *projected quantum kernels* to
mitigate this issue. Then, in [kbs21] they proved as this projected kernel
must intertwine with a correct inductive bias to obtain some positive results.

The concept is straightforward: first, the unitary transformation
:math:`U` maps classical data into the Hilbert space of the quantum
system. Subsequently, a projection maps these elements back to a
lower-dimensional Hilbert space. The overall transformation, thanks to
the contribution of :math:`U`, remains beyond the capabilities of
classical kernels.

For a single data point encoded in the quantum system, denoted as
:math:`\rho_x = U(x) \rho_0 U(x)`, projected quantum kernels can be
implemented in two different ways: - We can implement the feature map
:math:`\phi(x) = \mathrm{\tilde{Tr}}[\rho_x]`, with
:math:`\mathrm{\tilde{Tr}}` representing partial trace. - Alternatively,
we can implement the feature map
:math:`\phi(x) = \{ \mathrm{Tr}[\rho_x O^{(j)}] \}_{j=1}^k`, where the
observable :math:`O^{(j)}` is employed for the projections.

Finally, the kernel :math:`\kappa(x, x')` is explicitly constructed as
the inner product between :math:`\phi(x)` and :math:`\phi(x')`.

Implementation of projected quantum kernel in *quask*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We first create the parameterized quantum circuit :math:`U` as in the
previous tutorials.

.. code:: ipython3

    from quask.core import Ansatz, Kernel, KernelFactory, KernelType
    
    N_FEATURES = 2
    N_OPERATIONS = 3
    N_QUBITS = 2
    ansatz = Ansatz(n_features=N_FEATURES, n_qubits=N_QUBITS, n_operations=N_OPERATIONS)
    ansatz.initialize_to_identity()
    ansatz.change_operation(0, new_feature=0, new_wires=[0, 1], new_generator="ZZ", new_bandwidth=1.0)
    ansatz.change_operation(1, new_feature=1, new_wires=[0, 1], new_generator="XX", new_bandwidth=1.0)
    ansatz.change_operation(2, new_feature=2, new_wires=[0, 1], new_generator="IX", new_bandwidth=0.123)

Now, by employing the SWAP test over a subset of the :math:`n` qubits,
only a small and constant number of qubits are measured while the rest
remain unmeasured. This calculation is equivalent to performing the
inner product between partial traces of two quantum-encoded data points
can be achieved.

In the following example, the measurement is performed only on the first
of two qubits.

.. code:: ipython3

    kernel = KernelFactory.create_kernel(ansatz, "ZI", KernelType.SWAP_TEST)

We can also obtain the kernel by projecting onto a single observable
described by a Pauli string.

.. code:: ipython3

    kernel = KernelFactory.create_kernel(ansatz, "XY", KernelType.OBSERVABLE)

Multiple observable can be tested if we compose together kernel
functions made of different observables. Due to the properties of
positive semidefinite functions, the sum and product and tensor of
positive semidefinite operators is again positive semidefinite.

Learning of quantum processes
-----------------------------

The projected quantum kernel finds application in the realm of learning
a quantum process, described by a function:

.. math:: f(x) = \mathrm{Tr}[U^\dagger(x) \rho_0 U(x) O]

Here, :math:`U` represents a parameterized quantum circuit,
:math:`\rho_0` is the initial state, and :math:`O` stands for the
observable. This family of functions carries significant theoretical
importance, as it has facilitated the formal demonstration of quantum
advantages. It also holds practical significance, as certain use cases
in physics and chemistry can be conceptualized as quantum processes.

We are given a dataset, denoted as
:math:`\{ (x^{(j)}, y^{(j)}) \}_{j=1}^m`. Additionally, we assume that
each label in this dataset is noise-free, meaning that
:math:`y^{(j)} = f(x^{(j)})`.

S-value
~~~~~~~

We can train a kernel machine on a dataset using a kernel
:math:`\kappa`. The resulting model takes the form
:math:`h(x) = w^\top \phi(x)`. This representation is a kernel machine
in its primal form, and the corresponding kernel Gram matrix is defined
as :math:`K = [\kappa(x^{(i)}, x^{(j)})]_{i,j=1}^m`. Assuming that the
kernel Gram matrix is normalized, i.e., :math:`\mathrm{Tr}[K]=m`, we can
define the *s-value*, a quantity that depends on the process :math:`f`,
the input data, and the kernel Gram matrix $K:

.. math:: s_K = \sum_{i,j=1}^m (K_{i,j}^{-1}) \, f(x^{(i)}) \, f(x^{(j)})

This value quantifies how well the kernel function captures the behavior
of the quantum process. The kernel is indeed able to capture the
relationship within the data if:

.. math:: \kappa(x^{(i)}, x^{(j)}) \approx f(x^{(i)}) \, f(x^{(j)})

It’s important to note that :math:`s_K = \lVert w \rVert`, making it a
measure of the model’s complexity. Higher values of :math:`s_K` suggest
that the kernel machine :math:`h` becomes a more complex function, which
can lead to overfitting and poor generalization performance.

Geometric difference
~~~~~~~~~~~~~~~~~~~~

While the quantity :math:`s_K` compare a kernel and the target function,
the geometric difference quantifies the divergence between two kernels.

Assume for the two kernel matrices :math:`K_1, K_2` that their trace is
equal to :math:`m`. This is a valid assumption for quantum kernels, as
the inner product between unitary vectors (or corresponding density
matrices) is one, which then has to be multiplied for the :math:`m`
elements. For classical kernels, the Gram matrix needs to be normalized.

The geometric difference is defined by

.. math:: g(K_1, K_2) = \sqrt{\lVert \sqrt{K_2} K_1^{-1} \sqrt{K_2} \rVert_{\infty}},

where :math:`\lVert \cdot \rVert_\infty` is the spectral norm, i.e. the
largest singular value.

One should use the geometric difference to compare the quantum kernel
:math:`K_Q` with several classical kernels
:math:`K_{C_1}, K_{C_2}, ...`. Then, :math:`\min g(K_C, K_Q)` has to be
calculated: \* if this difference is small,
:math:`g(K_C, K_Q) \ll \sqrt{m}`, then one of the classical kernels, the
one with the smallest geometric difference, is guaranteed to provide
similar performances; \* if the difference is high,
:math:`g(K_C, K_Q) \approx \sqrt{m}`, the quantum kernel might
outperform all the classical kernels tested.

Geometry Test
~~~~~~~~~~~~~

The geometry test, introduced by [Hua21], serves as a means to assess
whether a particular dataset holds the potential for a quantum advantage
or if such an advantage is unlikely. The test operates as follows:

-  When :math:`g(K_C, K_Q) \ll \sqrt{m}`, a classical kernel exhibits
   behavior similar to the quantum kernel, rendering the use of the
   quantum kernel redundant.

-  When :math:`g(K_C, K_Q) \approx \sqrt{m}`, the quantum kernel
   significantly deviates from all tested classical kernels. The outcome
   depends on the complexity of classical kernel machines:

   -  If the complexity of any classical kernel machine is low
      (:math:`s_{K_C} \ll m`), classical kernels perform well, and the
      quantum kernel’s divergence from classical :math:`K_C`, doesn’t
      yield superior performance.
   -  When the complexity of all classical kernel machines is high
      (:math:`s_{K_C} \approx m`), classical models struggle to learn
      the function :math:`f`. In this scenario:

      -  If the quantum model’s complexity is low
         (:math:`s_{K_Q} \ll m`), the quantum kernel successfully solves
         the task while the classical models do not.
      -  If the quantum model’s complexity is high
         (:math:`s_{K_Q} \approx m`), even the quantum model struggles
         to solve the problem.



.. code:: ipython3

    from quask.evaluator import EssEvaluator, GeometricDifferenceEvaluator, GeometryTestEvaluator


.. parsed-literal::

    
    KeyboardInterrupt
    


References & acknowledgements
-----------------------------

[Hua21] Huang, HY., Broughton, M., Mohseni, M. et al."Power of data in
quantum machine learning." Nat Commun 12, 2631 (2021).
https://doi.org/10.1038/s41467-021-22539-9

[kbs21] Jonas M. Kübler, Simon Buchholz, Bernhard Schölkopf. "The 
Inductive Bias of Quantum Kernels." arXiv:2106.03747 (2021).
https://doi.org/10.48550/arXiv.2106.03747 

