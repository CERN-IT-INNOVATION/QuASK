"""
Code base for https://arxiv.org/abs/2209.11144. This module optimize a quantum circuit with a
greedy optimization algorithm.
"""

from __future__ import division

import numpy as np
import pennylane as qml
from .combinatorial_kernel import CombinatorialFeatureMap
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import jax


class CombinatorialKernelGreedySearch:
    """
    Generate a kernel function by perturbing a random circuit according to a greedy local search algorithm
    """

    def __init__(self, solution, n_qubits, n_layers, n_operations, X_train, y_train, X_validation, y_validation):
        """
        Initialization

        Args:
            solution: initial matrix representation of the circuit
            n_qubits: number of qubits
            n_layers: number of layers
            n_operations: number of candidate operation
            X_train: training feature data matrix
            y_train: training label data array
            X_validation: validation feature data matrix
            y_validation: validation label data array
        """
        self.solution = solution
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_gates = n_qubits * 2 * n_layers
        self.n_operations = n_operations
        self.energy_calculation_performed = 0
        self.energy_calculation_discarded = 0
        self.X_train = X_train
        self.y_train = y_train
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.combinatorial_kernel = self.create_pennylane_function()

    def search(self):
        """
        Search the best value for each cell, sequentially

        Returns:
            None
        """
        for gate in range(self.n_gates):
            self.search_gate(gate)

    def search_gate(self, gate):
        """
        Search the best value of the given gate

        Args:
            gate: number of operation which call

        Returns:
            None
        """
        best_i, best_j, best_energy = self.solution[gate][0], self.solution[gate][1], self.energy()
        for i in range(16):
            for j in range(self.n_operations):
                self.solution[gate][0] = i
                self.solution[gate][1] = j
                this_energy = self.energy()
                if this_energy < best_energy:
                    best_energy = this_energy
                    best_i = i
                    best_j = j
        self.solution[gate][0] = best_i
        self.solution[gate][1] = best_j
        print(f"\nBest {gate=} {best_i=} {best_j=} | {best_energy}")
        print(self.solution.ravel(), "\n")

    def energy(self, solution=None):
        """
        Return the energy of the best solution after optimization

        Args:
            solution: optional matrix representation of a candidate solution

        Return:
            energy or 100000 if no good solution has been found
        """
        solution = self.solution if solution is None else solution
        self.energy_calculation_performed += 1
        # print(solution.ravel())
        # first use "concentration around mean" criteria
        estimated_variance, _ = self.estimate_variance_of_kernel(solution=solution)
        # print(f"Estimated variance: {estimated_variance:0.3f}", end="")
        if estimated_variance < 0.1:
            self.energy_calculation_discarded += 1
            # print("")
            return 100000.0
        else:
            # then estimate accuracy
            mse = self.estimate_mse(solution=solution)
            # print(f"\tMSE: {mse:0.3f}")
            return mse

    def create_pennylane_function(self):
        """
        Create pennylane kernel function

        Returns:
            kernel function
        """

        def combinatorial_kernel_wrapper(x1, x2, solution, bandwidth):
            device = qml.device("default.qubit.jax", wires=self.n_qubits)

            # create projector (measures probability of having all "00...0")
            projector = np.zeros((2 ** self.n_qubits, 2 ** self.n_qubits))
            projector[0, 0] = 1

            # define the circuit for the quantum kernel ("overlap test" circuit)
            @qml.qnode(device, interface='jax')
            def combinatorial_kernel():
                CombinatorialFeatureMap(x1, self.n_qubits, self.n_layers, solution, bandwidth)
                qml.adjoint(CombinatorialFeatureMap)(x2, self.n_qubits, self.n_layers, solution, bandwidth)
                return qml.expval(qml.Hermitian(projector, wires=range(self.n_qubits)))

            return combinatorial_kernel()

        return jax.jit(combinatorial_kernel_wrapper)

    def estimate_variance_of_kernel(self, solution=None, n_sample_variance=5):
        """
        Estimate the variance of the kernel function

        Args:
            solution: optional matrix representation of a candidate solution
            n_sample_variance: number of pairs of value to be selected

        Returns:
            (v, l); l = list of randomly selected kernel values, v = variance of l
        """
        solution = self.solution if solution is None else solution
        kernel_values = []
        for i in range(n_sample_variance):
            indexes = np.random.choice(len(self.X_train), 2)
            x1, x2 = self.X_train[indexes[0]], self.X_train[indexes[1]]
            inner_product = self.combinatorial_kernel(x1, x2, solution, 1.0)
            kernel_values.append(inner_product)
        return np.var(kernel_values), kernel_values

    def estimate_mse(self, solution=None, X_test=None, y_test=None):
        """
        Estimate the MSE of the current solution.

        Args:
            solution: matrix representation of the kernel
            X_test: testing feature data matrix
            y_test: testing label data array

        Returns:
            MSE
        """
        X_test = self.X_validation if X_test is None else X_test
        y_test = self.y_validation if y_test is None else y_test
        training_gram = self.get_kernel_values(self.X_train, solution=solution)
        validation_gram = self.get_kernel_values(X_test, self.X_train, solution=solution)
        svr = SVR()
        svr.fit(training_gram, self.y_train.ravel())
        y_pred = svr.predict(validation_gram)
        return mean_squared_error(y_test.ravel(), y_pred.ravel())

    def get_kernel_values(self, X1, X2=None, solution=None, bandwidth=None):
        """
        Calculate kernel gram matrix

        Args:
            X1: feature data of the first batch
            X2: feature data of the second batch
            solution: matrix representation of the circuit
            bandwidth: optional constant limiting the rotational angles of the transformations

        Returns:
            kernel gram matrix
        """
        solution = self.solution if solution is None else solution
        bandwidth = 1.0 if bandwidth is None else bandwidth
        if X2 is None:
            m = self.X_train.shape[0]
            kernel_gram = np.eye(m)
            for i in range(m):
                for j in range(i + 1, m):
                    value = self.combinatorial_kernel(X1[i], X1[j], solution, bandwidth)
                    kernel_gram[i][j] = value
                    kernel_gram[j][i] = value
                    print(".", end="")
        else:
            kernel_gram = np.zeros(shape=(len(X1), len(X2)))
            for i in range(len(X1)):
                for j in range(len(X2)):
                    kernel_gram[i][j] = self.combinatorial_kernel(X1[i], X2[j], solution, bandwidth)
                    print(".", end="")
        return kernel_gram
