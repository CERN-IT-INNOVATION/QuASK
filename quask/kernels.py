from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel

import jax
import jax.numpy as jnp
import optax
import pennylane as qml
import pennylane.numpy as np


class KernelRegister:

    def __init__(self):
        self.kernel_functions = []
        self.kernel_names = []
        self.parameters = []
        self.current = 0

    def register(self, fn, name, params):
        self.kernel_functions.append(fn)
        self.kernel_names.append(name)
        self.parameters.append(params)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= len(self.kernel_functions):
            raise StopIteration
        ret = (self.kernel_functions[self.current], self.kernel_names[self.current], self.parameters[self.current])
        self.current += 1
        return ret


def pennylane_quantum_kernel(feature_map, X_1, X_2=None):
    """
    Create a Quantum Kernel given the template written in Pennylane framework
    :param feature_map: Pennylane template for the quantum feature map
    :param X_1: First dataset
    :param X_2: Second dataset
    :return: Gram matrix
    """
    if X_2 == None: X_2 = X_1  # Training Gram matrix
    assert X_1.shape[1] == X_2.shape[1], "The training and testing data must have the same dimensionality"
    N = X_1.shape[1]

    # create device using JAX
    device = qml.device("default.qubit.jax", wires=N)

    # create projector (measures probability of having all "00...0")
    projector = np.zeros((2 ** N, 2 ** N))
    projector[0, 0] = 1

    # define the circuit for the quantum kernel ("overlap test" circuit)
    @jax.jit
    @qml.qnode(device)
    def kernel(x1, x2):
        feature_map(x1, wires=range(N))
        qml.adjoint(feature_map)(x2, wires=range(N))
        return qml.expval(qml.Hermitian(projector, wires=range(N)))

    # build the gram matrix
    gram = np.zeros(shape=(X_1.shape[0], X_2.shape[0]))
    for i in range(X_1.shape[0]):
        for j in range(X_2.shape[0]):
            gram[i][j] = kernel(X_1[i], X_2[j])

    return gram


def pennylane_projected_feature_map(feature_map, X):
    """
    Create a Quantum Kernel given the template written in Pennylane framework
    :param feature_map: Pennylane template for the quantum feature map
    :param X: First dataset
    :return: projected quantum feature map X
    """
    N = X.shape[1]

    # create device using JAX
    device = qml.device("default.qubit.jax", wires=N)

    # define the circuit for the quantum kernel ("overlap test" circuit)
    @jax.jit
    @qml.qnode(device)
    def proj_feature_map(x):
        feature_map(x, wires=range(N))
        return [qml.expval(qml.PauliX(i)) for i in range(N)] \
               + [qml.expval(qml.PauliY(i)) for i in range(N)] \
               + [qml.expval(qml.PauliZ(i)) for i in range(N)]

    # build the gram matrix
    X_proj = [proj_feature_map(x) for x in X]

    return X_proj


def zz_quantum_kernel(X_1, X_2=None, params=None):
    """
    Create the kernel matrix using the Quantum ZZ Feature Map (with full entanglement scheme)
    using the structure described in https://qiskit.org/documentation/stubs/qiskit.circuit.library.ZZFeatureMap.html.
    To create the training Gram matrix pass X_1 = training set and X_2 = None
    To create the testing Gram matrix pass X_1 = testing set and X_2 = training set
    :param X_1: First dataset
    :param X_2: Second dataset
    :param params: ignored
    :return: Gram matrix
    """

    # define ZZ Feature Map with full entanglement scheme
    def ZZFullEntFeatureMap(x, wires):
        N = len(wires)
        for i in range(N):
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[i], wires=i)
        for i in range(N):
            for j in range(i+1, N):
                qml.CRZ(2 * (np.pi - x[i]) * (np.pi - x[j]), wires=[i, j])

    return pennylane_quantum_kernel(ZZFullEntFeatureMap, X_1, X_2)


def projected_zz_quantum_kernel(X_1, X_2=None, params=[0.01]):
    """

    To create the training Gram matrix pass X_1 = training set and X_2 = None
    To create the testing Gram matrix pass X_1 = testing set and X_2 = training set
    :param X_1: First dataset
    :param X_2: Second dataset
    :param params: list of floats, params[0] is the gamma parameter
    :return: Gram matrix
    """

    # define ZZ Feature Map with full entanglement scheme
    def ZZFullEntFeatureMap(x, wires):
        N = len(wires)
        for i in range(N):
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[i], wires=i)
        for i in range(N):
            for j in range(i + 1, N):
                qml.CRZ(2 * (np.pi - x[i]) * (np.pi - x[j]), wires=[i, j])

    X_1_proj = pennylane_projected_feature_map(ZZFullEntFeatureMap, X_1)
    X_2_proj = pennylane_projected_feature_map(ZZFullEntFeatureMap, X_2)

    # build the gram matrix
    gamma = params[0]

    gram = np.zeros(shape=(X_1.shape[0], X_2.shape[0]))
    for i in range(X_1_proj.shape[0]):
        for j in range(X_2_proj.shape[0]):
            gram[i][j] = np.exp(-gamma * ((X_1_proj[i] - X_2_proj[j])**2).sum())

    return gram

# create the global register
the_kernel_register = KernelRegister()

# register linear kernel
linear_kernel_wrapper = lambda X1, X2, params: linear_kernel(X1, X2)
the_kernel_register.register(linear_kernel_wrapper, 'linear_kernel', [])

# register gaussian kernel
rbf_kernel_wrapper = lambda X1, X2, params: rbf_kernel(X1, X2, gamma=float(params[0]))
the_kernel_register.register(rbf_kernel_wrapper, 'rbf_kernel', ['gamma'])

# register polynomial kernel
poly_kernel_wrapper = lambda X1, X2, params: polynomial_kernel(X1, X2, degree=int(params[0]))
the_kernel_register.register(poly_kernel_wrapper, 'poly_kernel', ['degree'])

# register custom quantum kernels
the_kernel_register.register(zz_quantum_kernel, 'zz_quantum_kernel', [])
the_kernel_register.register(projected_zz_quantum_kernel, 'projected_zz_quantum_kernel', ['gamma'])

# TODO add registration of more quantum kernels
