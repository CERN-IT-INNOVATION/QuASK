import copy
import datetime
import operator

import numpy as np
import pygad
import pennylane as qml
from quask.combinatorial_kernel import create_random_combinatorial_kernel
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import jax
import jax.numpy as jnp


class CombinatorialKernelGenetic:

    def __init__(self, X_train, y_train, X_validation, y_validation, ck, n_qubits, n_layers, num_iterations=1000, num_populations=50, num_breeding_parents=10):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ck = ck
        self.X_train = X_train
        self.y_train = y_train
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.energy_calculation_performed = 0
        self.energy_calculation_discarded = 0
        self.state = None
        self.num_iterations = num_iterations
        self.num_population = num_populations
        self.num_breeding_parents = num_breeding_parents
        self.initial_population = None
        self.current_population = None

    def generate_initial_population(self):
        self.initial_population = []
        for _ in range(self.num_population):
            item = create_random_combinatorial_kernel(self.n_qubits, self.n_layers, self.n_qubits)
            self.state = item
            energy = self.energy()
            self.initial_population.append((item, energy))

    def run_genetic_optimization(self):
        self.current_population = copy.deepcopy(self.initial_population)

        for epoch in range(self.num_iterations):
            # sort operator by increasing cost
            self.current_population.sort(key=operator.itemgetter(1))
            # get 10 best items (we should delete
            self.current_population = self.current_population[:self.num_breeding_parents]
            # generate all other items
            for i in range(self.num_breeding_parents, self.num_population):
                print(f"Epoch {epoch} Iteration {i} best solution energy={self.current_population[0][1]}")
                # crossover (1 point)
                one, two = np.random.choice(self.num_breeding_parents, 2, replace=False)
                split = np.random.randint(2 * self.n_layers * self.n_qubits)
                first_part_item = self.current_population[one][0][:split]
                second_part_item = self.current_population[two][0][split:]
                new_item = np.concatenate([first_part_item, second_part_item])
                assert new_item.shape == (2 * self.n_layers * self.n_qubits, 2), f"Shape is {new_item.shape} instead of {(2 * self.n_layers * self.n_qubits, 2)}"
                # mutate
                for gene in range(2 * self.n_layers * self.n_qubits):
                    if np.random.uniform() < 0.10:
                        new_item[gene][0] = np.random.randint(16)
                        new_item[gene][1] = np.random.randint(self.n_qubits)
                # save
                self.state = new_item
                new_energy = self.energy()
                self.current_population.append((new_item, new_energy))

    def energy(self):
        self.energy_calculation_performed += 1
        estimated_variance, _ = self.estimate_variance_of_kernel()
        print(f"Estimated variance: {estimated_variance:0.3f}", end="")
        if estimated_variance < 0.1:
            self.energy_calculation_discarded += 1
            print("")
            return 100000
        else:
            # then estimate accuracy
            mse = self.estimate_mse()
            print(f"\tMSE: {mse:0.3f}")
            return mse

    def estimate_variance_of_kernel(self, n_sample_variance=5):
        kernel_values = []
        solution = self.state.reshape((self.n_qubits * self.n_layers * 2, 2))
        for i in range(n_sample_variance):
            indexes = np.random.choice(len(self.X_train), 2)
            x1, x2 = self.X_train[indexes[0]], self.X_train[indexes[1]]
            inner_product = self.ck(x1, x2, solution, 1.0)
            kernel_values.append(inner_product)
        return np.var(kernel_values), kernel_values

    def estimate_mse(self, solution=None, X_test=None, y_test=None):
        X_test = self.X_validation if X_test is None else X_test
        y_test = self.y_validation if y_test is None else y_test
        training_gram = self.get_kernel_values(self.X_train, solution=solution)
        validation_gram = self.get_kernel_values(X_test, self.X_train, solution=solution)
        svr = SVR()
        svr.fit(training_gram, self.y_train.ravel())
        y_pred = svr.predict(validation_gram)
        return mean_squared_error(y_test.ravel(), y_pred.ravel())

    def get_kernel_values(self, X1, X2=None, solution=None, bandwidth=None):
        solution = self.state if solution is None else solution
        bandwidth = 1.0 if bandwidth is None else bandwidth
        if X2 is None:
            m = self.X_train.shape[0]
            kernel_gram = np.eye(m)
            for i in range(m):
                for j in range(i + 1, m):
                    value = self.ck(X1[i], X1[j], solution, bandwidth)
                    kernel_gram[i][j] = value
                    kernel_gram[j][i] = value
                    print(".", end="")
        else:
            kernel_gram = np.zeros(shape=(len(X1), len(X2)))
            for i in range(len(X1)):
                for j in range(len(X2)):
                    kernel_gram[i][j] = self.ck(X1[i], X2[j], solution, bandwidth)
                    print(".", end="")
        return kernel_gram

