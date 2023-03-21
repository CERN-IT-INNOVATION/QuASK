from abc import ABC, abstractmethod
import numpy as np
import copy
from ..core import Operation, Ansatz, Kernel, KernelFactory
from ..evaluator import KernelEvaluator


class BaseKernelOptimizer(ABC):
    """
    Abstract class implementing a procedure to optimize the kernel
    """

    def __init__(self, initial_kernel: Kernel, X: np.ndarray, y: np.ndarray, ke: KernelEvaluator):
        """
        Initialization
        :param initial_kernel: initial kernel object
        :param X: datapoints
        :param y: labels
        :param ke: kernel evaluator object
        """
        self.initial_kernel = initial_kernel
        self.X = X
        self.y = y
        self.ke = ke

    @abstractmethod
    def optimize(self):
        """
        Run the optimization
        :return: optimized kernel object
        """
        pass

    # def configuration_to_ansatz(self, ansatz_configuration):
    #     n_operations = self.initial_kernel.ansatz.n_operations
    #     assert len(ansatz_configuration) == 5 * n_operations, \
    #         f"ansatz_configuration vector is not of the correct length (expected: {5 * n_operations}, actual: {len(ansatz_configuration)})"
    #     ansatz = copy.deepcopy(self.initial_kernel.ansatz)
    #     for i in range(len(ansatz_configuration) // 5):
    #         wires = ansatz_configuration[5*i+1:5*i+3]
    #         if wires[1] >= wires[0]:
    #             wires[1] += 1
    #         ansatz.change_generators(i, Operation.OPERATIONS[ansatz_configuration[i * 5]])
    #         ansatz.change_wires(i, wires)
    #         ansatz.change_feature(i, ansatz_configuration[5*i+3])
    #         ansatz.change_bandwidth(i, ansatz_configuration[5*i+4])
    #     return ansatz

    # def configuration_to_measurement(self, measurement_configuration):
    #     n_qubits = self.initial_kernel.ansatz.n_qubits
    #     assert len(measurement_configuration) == n_qubits, \
    #         f"measurement_configuration vector is not of the correct length (expected: {n_qubits}, actual: {len(measurement_configuration)})"
    #     assert 0 <= min(measurement_configuration) <= max(measurement_configuration) <= 3, "measurement_configuration has invalid element"
    #     return "".join([['X', 'Y', 'Z', 'I'][i] for i in measurement_configuration])

    # def configuration_to_kernel(self, configuration):
    #     n_operations = self.initial_kernel.ansatz.n_operations
    #     n_qubits = self.initial_kernel.ansatz.n_qubits
    #     assert len(configuration) == 5 * n_operations + n_qubits, \
    #         f"Configuration vector is not of the correct length (expected: {5 * n_operations + n_qubits}, actual {len(configuration)})"
    #     ansatz = self.configuration_to_ansatz(configuration[:5 * n_operations])
    #     measurement = self.configuration_to_measurement(configuration[-n_qubits:])
    #     kernel = KernelFactory.create_kernel(ansatz, measurement, self.initial_kernel.type)
    #     return kernel

    # def cost(self, configuration):
    #     kernel = self.configuration_to_kernel(configuration)
    #     the_cost = self.ke.evaluate(kernel, None, self.X, self.y)
    #     return the_cost
