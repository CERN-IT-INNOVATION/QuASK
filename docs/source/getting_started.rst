Getting started
===============

.. code:: ipython3

    import sys
    import os
    # to import quask, move from docs/source/notebooks to src
    sys.path.append('../../../src')

Are you curious to know if *quask* is the right library for your
project? Here is a quick and straightforward guide on how to get started
using this tool. We will see:

1. how to quickly install the framework;
2. what are the main components of *quask* and how to use them;
3. how to solve a toy classification problem using quantum kernels via
   the *quask* application programming interface.

.. warning::

    This first tutorial illustrates the functionalities of the framework. It presumes a pre-existing knowledge of kernel methods and quantum computing. For a more beginner's level introduction, take a look at the `Intro to classical kernels <tutorials_classical/index.html>`__ page. 

Fast installation
-----------------

The easiest way to use *quask* is by installing it in your Python3
environment (version >= 3.10) via the *pip* packet manager,

.. code:: sh

    python3 -m pip install quask==2.0.0-alpha1

You also need any quantum SDK installed on your system. For example, we
can install Qiskit. For more information about the installation process,
you can see `the Installation section <installation.html>`__.

.. code:: sh

    python3 -m pip install qiskit qiskit_ibm_runtime
    python3 -m pip install qiskit_ibm_runtime --upgrade
    python3 -m pip install qiskit-aer

You can check if the installation has been successful by running the
command:

.. code:: ipython3

    import quask
    print(quask.__version__)

In this way, *quask* has been started as a standalone application,
meaning it can be used via the command line without the need for coding.
This modality is explored in depth later.

Hello world, quantum kernels!
-----------------------------

.. code:: ipython3

    from quask.core import Ansatz, Kernel, KernelFactory, KernelType
    import numpy as np

Here, we can see the main objects of *quask*. The class ``Ansatz``
represents the function that maps classical data to the Hilbert space of
the quantum system using a parametric quantum circuit. This class is
parameterized by the number of qubits in the underlying quantum circuit,
which often corresponds to the number of features (although it’s not a
strict rule), the number of gates applied to the quantum circuit, and
the number of features that the classical data point has.

The class ``Kernel`` represents a kernel object, which is essentially an
ansatz along with additional information on how to effectively implement
the quantum circuit for the entire procedure. The kernel object must be
executed using one of the available backends (Qiskit, Pennylane, Qibo,
etc.). To achieve this, the kernel class has been designed as an
abstract object, meaning it cannot be used directly. Instead, we can use
one of its subclasses, with each subclass interfacing with a particular
backend. We can instantiate the concrete (non-abstract) kernel objects
using the ``KernelFactory`` class.

Since there are several ways to design a quantum kernel using a single
ansatz, the ``KernelType`` class is an enumeration whose values indicate
the kind of kernel to be implemented.

.. code:: ipython3

    # Number of features in the data point to be mapped in the Hilbert space of the quantum system
    N_FEATURES = 1
    
    # Number of gates applied to the quantum circuit
    N_OPERATIONS = 1
    
    # Number of qubits of the quantum circuit
    N_QUBITS = 2
    
    # Ansatz object, representing the feature map
    ansatz = Ansatz(n_features=N_FEATURES, n_qubits=N_QUBITS, n_operations=N_OPERATIONS)

The ansatz class is not immediately usable when instantiated. It needs
to be initialized so that all its operations correspond to valid gates,
in this case, corresponding to the identity.

.. code:: ipython3

    ansatz.initialize_to_identity()

Each operation acts on two qubits and is defined as

.. math:: U(\beta \theta) = \exp(-i \beta \frac{\theta}{2} \sigma_1 \sigma_2),

where the generators :math:`\sigma_1` and :math:`\sigma_2` correspond to
the Pauli matrices :math:`X, Y, Z`, and :math:`\mathrm{Id}`. When one of
these generators is the identity, the gate effectively applies
nontrivially to a single qubit.

All the gates are parameterized by a single real-valued parameter,
:math:`\theta`, which can optionally be rescaled by a global scaling
parameter :math:`0 < \beta < 1`. We can characterize each parametric
gate by the following:

-  The feature that parameterizes the rotation, with
   :math:`0 \le f \le N\_FEATURES - 1`, or the constant feature
   :math:`1`. The constant features allow us to construct
   non-parameterized gates.
-  A pair of generators, represented by a 2-character string.
-  The qubits on which the operation acts, denoted as
   :math:`(q_1, q_2)`, where :math:`0 \le q_i < N\_QUBITS`, and
   :math:`q_1 \neq q_2`. For ‘single-qubit gates’ with the identity as
   one or both generators, the qubit on which the identity is applied
   has a negligible effect on the transformation.
-  The scaling parameter :math:`\beta`.

.. code:: ipython3

    ansatz.change_operation(0, new_feature=0, new_wires=[0, 1], new_generator="XX", new_bandwidth=0.9)

The ansatz serves as the feature map for our quantum kernel. To
calculate kernel values, however, we have the opportunity to specify the
method of calculation. This can be done using the fidelity test or by
computing the expectation value of some observable. Additionally, we
need to specify the backend to be used.

Currently, we support Qiskit, Pennylane, and Braket. More detailed
information is available at the `Backends in
quask <tutorials_quask/quask_0_backends.html>`__ tutorial. Here, we
suppose to use Qiskit as a backend, which has to be installed
separately. To create the commonly used fidelity kernel, we provide the
ansatz, the basis on which we will perform the measurement (typically
the computational basis), and the type of kernel.

.. code:: ipython3

    from quask.core_implementation import QiskitKernel
    kernel = QiskitKernel(ansatz, "Z" * N_QUBITS, KernelType.FIDELITY)

To test if the kernel object function correctly we can call the kernel
function on a pair of data point.

.. code:: ipython3

    x1 = np.array([0.001])
    x2 = np.array([0.999])
    similarity = kernel.kappa(x1, x2)
    print("The kernel value between x1 and x2 is", similarity)


.. parsed-literal::

    The kernel value between x1 and x2 is 0.4033203125


We can decouple the actual backend used from the high-level APIs. The
decoupling is managed by the ``KernelFactory`` class. to the
``create_kernel`` method. By default, ``KernelFactory`` creates objects
that rely on the noiseless, infinite-shot simulation of Pennylane as a
backend. To use the ``KernelFactory`` class, you first have to set up
with backend are you using.

.. code:: ipython3

    def create_qiskit_noiseless(ansatz: Ansatz, measurement: str, type: KernelType):
        return QiskitKernel(ansatz, measurement, type, n_shots=None)
    
    KernelFactory.add_implementation('qiskit_noiseless', create_qiskit_noiseless)
    KernelFactory.set_current_implementation('qiskit_noiseless')
    kernel = KernelFactory.create_kernel(ansatz, "Z" * N_QUBITS, KernelType.FIDELITY) # QiskitKernel

Solve the iris dataset classification using *quask*
---------------------------------------------------

We demonstrate how to integrate *quask* into a machine learning pipeline
based on the library `scikit-learn <https://scikit-learn.org/stable>`__.
This package allows us to effortlessly set up a toy classification
problem that can be solved using kernel machines with quantum kernels.

.. code:: ipython3

    from sklearn.datasets import load_iris
    from sklearn.svm import SVC
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np

First, we load the dataset. It can be retrieved directly from the Python
package of scikit-learn.

It contains 150 samples associated with the three different subspecies
of the Iris flower, with 50 samples for each subspecies. To simplify the
task, we classify only the first two classes and select 10 samples for
each class.

Each sample has 4 real featues.

.. code:: ipython3

    N_ELEMENTS_PER_CLASS = 20
    iris = load_iris()
    X = np.row_stack([iris.data[0:N_ELEMENTS_PER_CLASS], iris.data[50:50+N_ELEMENTS_PER_CLASS]])
    y = np.array([0] * N_ELEMENTS_PER_CLASS + [1] * N_ELEMENTS_PER_CLASS)

We preprocess our data and divide the dataset in training and testing
set.

.. code:: ipython3

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=5454)

We then define the machine learning model to solve the classification
task. Among the possibilities, we choose the Support Vector Machine. In
order to use the quantum kernel, we specify we will give the kernel
machine the kernel Gram matrix instead of the original features, by
using the precomputed option.

.. code:: ipython3

    # Instantiate a machine learning model
    model = SVC(kernel='precomputed')

We then calculate the kernel Gram matrices and train the model.

.. code:: ipython3

    # Create a quantum kernel
    ansatz = Ansatz(n_features=4, n_qubits=4, n_operations=4)
    ansatz.initialize_to_identity()
    ansatz.change_operation(0, new_feature=0, new_wires=[0, 1], new_generator="XX", new_bandwidth=0.9)
    ansatz.change_operation(1, new_feature=1, new_wires=[1, 2], new_generator="XX", new_bandwidth=0.9)
    ansatz.change_operation(2, new_feature=2, new_wires=[2, 3], new_generator="XX", new_bandwidth=0.9)
    ansatz.change_operation(3, new_feature=3, new_wires=[3, 0], new_generator="XX", new_bandwidth=0.9)
    kernel = KernelFactory.create_kernel(ansatz, "ZZZZ", KernelType.FIDELITY)
    
    # Fit the model to the training data
    K_train = kernel.build_kernel(X_train, X_train)
    model.fit(K_train, y_train)




.. raw:: html

    <style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SVC(kernel=&#x27;precomputed&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">SVC</label><div class="sk-toggleable__content"><pre>SVC(kernel=&#x27;precomputed&#x27;)</pre></div></div></div></div></div>



We then use the model to predict the label of elements in the testing
set. Again, we need to create the kernel Gram matrix of the elements in
the testing set.

.. code:: ipython3

    # Predict the labels for the test data
    K_test = kernel.build_kernel(X_test, X_train)
    y_pred = model.predict(K_test)

Finally, we can calculate the accuracy with respect to the testing set.

.. code:: ipython3

    # Calculate the accuracy
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    print("Accuracy:", accuracy)


.. parsed-literal::

    Accuracy: 0.4


Among the features of *quask* is the ability to evaluate the kernel
according to criteria known in the literature. We demonstrate one
possible method for evaluating our quantum kernel with respect to the
Centered Kernel Target Alignment. The lower the cost, the better the
kernel is suited for the task.

.. code:: ipython3

    from quask.evaluator import CenteredKernelAlignmentEvaluator
    ce = CenteredKernelAlignmentEvaluator()
    cost = ce.evaluate(None, K_train, X_train, y_train)
    print("The cost according to the Centered-KTA is:", cost)


.. parsed-literal::

    The cost according to the Centered-KTA is: -0.0441181503241057

