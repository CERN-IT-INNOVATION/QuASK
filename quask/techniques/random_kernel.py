import copy

import pennylane as qml
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


class RandomKernel:

    def __init__(self, X_train, y_train, X_validation, y_validation, n_layers, seed=12345):
        self.X_train = X_train
        self.y_train = y_train
        self.training_gram = None
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.validation_gram = None
        self.n_qubits = X_train.shape[1]
        self.n_layers = n_layers
        self.state = jnp.array(np.random.normal(size=((n_layers - 1) * self.n_qubits,)))
        self.random_kernel = self.create_pennylane_function()
        self.seed = seed

    def create_pennylane_function(self):
        # define function to compile
        def trainable_kernel_wrapper(x1, x2, weights, bandwidth):
            device = qml.device("default.qubit.jax", wires=self.n_qubits)

            # create projector (measures probability of having all "00...0")
            projector = np.zeros((2 ** self.n_qubits, 2 ** self.n_qubits))
            projector[0, 0] = 1

            # define the circuit for the quantum kernel ("overlap test" circuit)
            @qml.qnode(device, interface='jax')
            def random_kernel():
                qml.RandomLayers(bandwidth * jnp.concatenate([x1, weights]).reshape((self.n_layers, self.n_qubits)),
                                 wires=range(self.n_qubits),
                                 seed=self.seed)
                qml.adjoint(qml.RandomLayers)(
                                 bandwidth * jnp.concatenate([x2, weights]).reshape((self.n_layers, self.n_qubits)),
                                 wires=range(self.n_qubits),
                                 seed=self.seed)
                return qml.expval(qml.Hermitian(projector, wires=range(self.n_qubits)))

            return random_kernel()

        return jax.jit(trainable_kernel_wrapper)

    def estimate_mse(self, weights=None, X_test=None, y_test=None):
        X_test = self.X_validation if X_test is None else X_test
        y_test = self.y_validation if y_test is None else y_test
        training_gram = self.get_kernel_values(self.X_train, weights=weights)
        testing_gram = self.get_kernel_values(X_test, self.X_train, weights=weights)
        return self.estimate_mse_svr(training_gram, self.y_train, testing_gram, y_test)

    def estimate_mse_svr(self, gram_train, y_train, gram_test, y_test):
        svr = SVR()
        svr.fit(gram_train, y_train.ravel())
        y_pred = svr.predict(gram_test)
        return mean_squared_error(y_test.ravel(), y_pred.ravel())

    def get_kernel_values(self, X1, X2=None, weights=None, bandwidth=None):
        weights = self.state if weights is None else weights
        bandwidth = 1.0 if bandwidth is None else bandwidth
        if X2 is None:
            m = self.X_train.shape[0]
            kernel_gram = np.eye(m)
            for i in range(m):
                for j in range(i + 1, m):
                    value = self.random_kernel(X1[i], X1[j], weights, bandwidth)
                    kernel_gram[i][j] = value
                    kernel_gram[j][i] = value
                    print(".", end="")
        else:
            kernel_gram = np.zeros(shape=(len(X1), len(X2)))
            for i in range(len(X1)):
                for j in range(len(X2)):
                    kernel_gram[i][j] = self.random_kernel(X1[i], X2[j], weights, bandwidth)
                    print(".", end="")
        return kernel_gram
