from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel
from .template_pennylane import ZZFullEntFeatureMap, pennylane_quantum_kernel, pennylane_projected_feature_map
from .template_qiskit import encoding, qiskit_quantum_kernel, zz_norm_feature_map
from qiskit_machine_learning.kernels import QuantumKernel
import numpy as np

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


def zz_quantum_kernel(X_1, X_2=None, params=None, pennylane=True):
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

    if pennylane:
        return pennylane_quantum_kernel(ZZFullEntFeatureMap, X_1, X_2)
    else:
        return qiskit_quantum_kernel(zz_norm_feature_map, X_1, X_2)


def projected_zz_quantum_kernel(X_1, X_2=None, params=[0.01]):
    """
    To create the training Gram matrix pass X_1 = training set and X_2 = None
    To create the testing Gram matrix pass X_1 = testing set and X_2 = training set
    :param X_1: First dataset
    :param X_2: Second dataset
    :param params: list of floats, params[0] is the gamma parameter
    :return: Gram matrix
    """
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
