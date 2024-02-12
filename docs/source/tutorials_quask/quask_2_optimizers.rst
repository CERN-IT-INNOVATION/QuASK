How to optimize a quantum kernel
================================

.. code:: ipython3

    import sys
    import os
    # to import quask, move from docs/source/notebooks to src
    sys.path.append('../../../src')
    import quask

One of the main challenges regarding quantum kernels is the need to
choose the appropriate ansatz for each different problem. When
little-to-no information about a certain task is known, we can use
techniques to attempt to construct a suitable quantum kernel following
an optimization problem. Two main approaches are possible:

1. Choosing an ansatz where some of the parameters are the usual
   features, while others are freely tunable parameters that are
   optimized according to some cost function via a stochastic gradient
   descent-based algorithm.
2. Avoid making any choice and let an optimization algorithm pick the
   entire quantum circuit.

In this tutorial, we will demonstrate how to easily implement the first
technique. Furthermore, we will also discuss how to leverage the *quask*
built-in features to implement the second technique. Finally, we will
show how it is possible to efficiently achieve the best-performing
linear combination of these kernels when dealing with multiple different
kernels.

Optimization of quantum kernels in *quask*
------------------------------------------

The package ``quask.optimizer`` allows defining an optimization
procedure for quantum kernels. The main interface is the
``BaseKernelOptimizer`` class, which requires: \* a kernel function,
which will serve as the initial point of the optimization routine; \* a
kernel evaluator, which will be the cost function guiding the
optimization; \* possibly some input data, if needed by the kernel
evaluator.

Then, the ``optimizer`` method starts the optimization and will return a
new instance of ``quask.core.Kernel`` to be used. You will use
``BaseKernelOptimizer`` directly if only to create a new optimization
method, which can be done as follows:

.. code:: ipython3

    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.svm import SVC
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import copy
    
    from quask.core import Ansatz, Kernel, KernelFactory, KernelType
    from quask.core_implementation import QiskitKernel
    from quask.optimizer.base_kernel_optimizer import BaseKernelOptimizer
    from quask.evaluator import CenteredKernelAlignmentEvaluator
    
    def create_qiskit_noiseless(ansatz: Ansatz, measurement: str, type: KernelType):
        return QiskitKernel(ansatz, measurement, type, n_shots=None)
    
    KernelFactory.add_implementation('qiskit_noiseless', create_qiskit_noiseless)
    KernelFactory.set_current_implementation('qiskit_noiseless')

.. code:: ipython3

    class RandomOptimizer(BaseKernelOptimizer):
    
        def __init__(self, initial_kernel, X, y, ke):
            super().__init__(initial_kernel, X, y, ke)
    
        def optimize(self):
            kernel = copy.deepcopy(self.initial_kernel)
            cost = self.ke.evaluate(kernel, None, self.X, self.y)
            N_TENTATIVES = 10
            for i in range(N_TENTATIVES):
                new_kernel = copy.deepcopy(kernel)
                i_operation = np.random.randint(new_kernel.ansatz.n_operations)
                i_feature = np.random.randint(new_kernel.ansatz.n_features)
                i_wires = np.random.choice(range(new_kernel.ansatz.n_qubits), 2, replace=False).tolist()
                i_gen = np.random.choice(['I', 'Z', 'X', 'Y'], 2, replace=True)
                i_gen = "".join(i_gen.tolist())
                i_bandwidth = np.random.rand()
                new_kernel.ansatz.change_operation(i_operation, i_feature, i_wires, i_gen, i_bandwidth)
                new_cost = self.ke.evaluate(new_kernel, None, self.X, self.y)
                print("Cost of the new solution:", new_cost)
                if cost > new_cost:
                    kernel = new_kernel
                    cost = new_cost
            return kernel

Let’s unpack the content of this function. The initialization is
identical to the one in ``BaseKernelOptimizer``, as we don’t really need
any new parameters (we may have added ``N_TENTATIVES``, but for the sake
of simplicity, we can keep it as is).

The ``optimize`` function effectively starts from the initial kernel and
proceeds iteratively N_TENTATIVES times by changing a single operation
within the quantum circuit with a completely random operation. When the
result improves, indicating a lower cost for the kernel evaluator, the
new solution is accepted. Clearly, this is a rather inefficient way to
optimize the quantum kernel, and more sophisticated techniques are shown
below.

We can test this approach in a simple context.

.. code:: ipython3

    N_FEATURES = 4
    N_OPERATIONS = 5
    N_QUBITS = 4
    ansatz = Ansatz(n_features=N_FEATURES, n_qubits=N_QUBITS, n_operations=N_OPERATIONS)
    ansatz.initialize_to_identity()
    kernel = KernelFactory.create_kernel(ansatz, "Z" * N_QUBITS, KernelType.FIDELITY)
    
    N_ELEMENTS_PER_CLASS = 20
    iris = load_iris()
    X = np.row_stack([iris.data[0:N_ELEMENTS_PER_CLASS], iris.data[50:50+N_ELEMENTS_PER_CLASS]])
    y = np.array([0] * N_ELEMENTS_PER_CLASS + [1] * N_ELEMENTS_PER_CLASS)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=5454)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    ce = CenteredKernelAlignmentEvaluator()

.. code:: ipython3

    print("The initial cost is:", ce.evaluate(kernel, None, X_train, y_train))
    optimizer = RandomOptimizer(kernel, X_train, y_train, ce)
    optimized_kernel = optimizer.optimize()
    print("The final cost is:", ce.evaluate(optimized_kernel, None, X_train, y_train))


.. parsed-literal::

    The initial cost is: -0.42299755542166695
    Cost of the new solution: -0.28109278237003543
    Cost of the new solution: -0.0961605171721972
    Cost of the new solution: -0.27011666753748764
    Cost of the new solution: -0.03337677439165289
    Cost of the new solution: -0.2771358170950197
    Cost of the new solution: -0.15948562200769945
    Cost of the new solution: -0.07431221779515468
    Cost of the new solution: -0.2751971110940814
    Cost of the new solution: -0.27668879041145483
    Cost of the new solution: -0.03265594563162022
    The final cost is: -0.42299755542166695


The result of the optimization can be used exactly like any other kernel
object.

.. code:: ipython3

    model = SVC(kernel='precomputed')
    K_train = optimized_kernel.build_kernel(X_train, X_train)
    model.fit(K_train, y_train)
    K_test = optimized_kernel.build_kernel(X_test, X_train)
    y_pred = model.predict(K_test)
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    print("Accuracy:", accuracy)


.. parsed-literal::

    Accuracy: 0.3


Combinatorial optimization of a quantum kernel
----------------------------------------------

A plethora of techniques has been implemented in *quask* and can be used
in different contexts according to the available computational resources
and the volume of data to be analyzed. The functioning of these
algorithms is detailed in [inc23].

Bayesian optimizer
~~~~~~~~~~~~~~~~~~

Bayesian optimization is the simplest and usually the most effective
technique to use. It is known to work best for low-dimensional problems
where the function to optimize is a black box costly to evaluate, which
is often the case in our context (although the optimization might not be
so low-dimensional). This approach is based on the library
`scikit-optimize <https://scikit-optimize.github.io/stable/>`__, which
needs to be installed separately from *quask* via the command
``pip install scikit-optimize``.

Note that a KeyError ’’ might occur at this point if you have not
configured a default backend for *quask*.

.. code:: ipython3

    from quask.optimizer.bayesian_optimizer import BayesianOptimizer

.. code:: ipython3

    print("The initial cost is:", ce.evaluate(kernel, None, X_train, y_train))
    optimizer = BayesianOptimizer(kernel, X_train, y_train, ce)
    optimized_kernel = optimizer.optimize(n_epochs=2, n_points=1, n_jobs=1)
    print("The final cost is:", ce.evaluate(optimized_kernel, None, X_train, y_train))


.. parsed-literal::

    The initial cost is: -0.42299755542166695
    Epoch of training i=0
    Epoch of training i=1
    The final cost is: -0.22919895370100746


Meta-heuristic optimizer
~~~~~~~~~~~~~~~~~~~~~~~~

At the moment we only support Particle Swarm but we plan to support
other techniques, such as evolutionary (genetic) algorithms. This
approach is based on the library
`opytimizer <https://opytimizer.readthedocs.io/en/latest/>`__, which
needs to be installed separately from *quask* via the command
``pip install opytimizer``.

Due to the extremely high computational cost, you can only use this
technique for the smallest circuits (<2 operations, <2 qubits); in
general is better to rely on the other techniques.

.. code:: ipython3

    from quask.optimizer.metaheuristic_optimizer import MetaheuristicOptimizer

Greedy optimizer
~~~~~~~~~~~~~~~~

The greedy optimization tries any possible value for the first
operation, chooses the best one, and proceeds with the following
operations in a sequential fashion. Despite its simplicity, it is quite
an expensive technique. This approach is based on the library
`mushroom-rl <https://mushroomrl.readthedocs.io/en/latest/>`__, which
needs to be installed separately from *quask* via the command
``pip install mushroom_rl``.

.. code:: ipython3

    from quask.optimizer.greedy_optimizer import GreedyOptimizer

Reinforcement learning optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimizes the quantum kernel by setting up a reinforcement learning
environment and using SARSA Lambda algorithm. This approach is based on
the library
`mushroom-rl <https://mushroomrl.readthedocs.io/en/latest/>`__, which
needs to be installed separately from *quask* via the command
``pip install mushroom_rl``.

.. code:: ipython3

    from quask.optimizer.reinforcement_learning_optimizer import ReinforcementLearningOptimizer

References
----------

[llo20] Lloyd, S., Schuld, M., Ijaz, A., Izaac, J., & Killoran, N.
(2020). Quantum embeddings for machine learning. arXiv preprint
arXiv:2001.03622.

[inc23] Incudini, M., Lizzio Bosco, D., Martini, F., Grossi, M., Serra,
G., and Di Pierro, A., “Automatic and effective discovery of quantum
kernels”, arXiv e-prints, 2022. doi:10.48550/arXiv.2209.11144.

.. note::

    Author's note
