import numpy as np
from scipy.spatial.distance import cdist
from abc import ABC, abstractmethod
from . import Ansatz, KernelType, KernelFactory


class Kernel(ABC):
    """
    Abstract class representing a kernel object
    """

    PAULIS = ['I', 'X', 'Y', 'Z']

    def __init__(self, ansatz: Ansatz, measurement: str, type: KernelType):
        """
        Initialization
        :param ansatz: Ansatz object representing the unitary transformation
        :param measurement: Pauli string representing the measurement
        :param type: type of kernel, fidelity or observable
        """
        assert ansatz.n_qubits == len(measurement), "Measurement qubits and number of ansatz qubits do not match"
        assert len(set(measurement).difference(Kernel.PAULIS)) == 0, "Unknown Pauli in measurement"
        self.ansatz = ansatz
        self.measurement = measurement
        self.type = type
        self.last_probabilities = None

    def get_last_probabilities(self):
        """
        Get the last kernel value calculated
        :return: last probability array
        """
        return np.array(self.last_probabilities)

    @abstractmethod
    def kappa(self, x1, x2) -> float:
        """
        Calculate the kernel given two datapoints
        :param x1: first data point
        :param x2: second data point
        :return: Kernel similarity between the two data points
        """
        pass

    @abstractmethod
    def phi(self, x) -> float:
        """
        Calculate the feature map of a data point
        :param x: data point
        :return: feature map of the datapoint as numpy array
        """
        pass

    def get_allowed_operations(self):
        """
        Get the list of allowed operations
        :return:  list of generators allowed (the information is saved in the ansatz)
        """
        return self.ansatz.get_allowed_operations()

    def build_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Build a kernel
        :param X1: a single datapoint or a list of datapoints
        :param X2: a single datapoint or a list of datapoints
        :return: a single or a matrix of kernel inner products
        """
        # if you gave me only one sample
        if len(X1.shape) == 1 and len(X2.shape) == 1:
            if self.type == KernelType.FIDELITY:
                return self.kappa(X1, X2)
            else:
                return self.phi(X1) * self.phi(X2)
        # if you gave me multiple samples
        assert self.ansatz.n_features == X1.shape[1], "Number of features and X1.shape[1] do not match"
        assert self.ansatz.n_features == X2.shape[1], "Number of features and X2.shape[1] do not match"
        if self.type == KernelType.FIDELITY:
            return cdist(X1, X2, metric=self.kappa)
        else:
            n = X1.shape[0]
            m = X2.shape[0]
            Phi1 = np.array([self.phi(x) for x in X1]).reshape((n, 1))
            Phi2 = np.array([self.phi(x) for x in X2]).reshape((m, 1))
            return Phi1.dot(Phi2.T)

    def to_numpy(self):
        """
        Serialize the kernel object as a numpy array
        :return: numpy array
        """
        ansatz_numpy = self.ansatz.to_numpy()
        measurement_numpy = np.array([Kernel.PAULIS.index(p) for p in self.measurement])
        type_numpy = np.array([self.type.value])
        return np.concatenate([ansatz_numpy, measurement_numpy, type_numpy], dtype=object).ravel()

    @staticmethod
    def from_numpy(array, n_features, n_qubits, n_operations, allow_midcircuit_measurement, shift_second_wire=False):
        """
        Deserialize the object from a numpy array
        :param array: numpy array
        :param n_features: number of feature that can be used to parametrize the operation
        :param n_qubits: number of qubits of the circuit
        :param n_operations: number of operations
        :param allow_midcircuit_measurement: True if mid-circuit measurement are allowed
        :return: Kernel object (created using default instance in KernelFactory)
        """
        assert len(array) == 5 * n_operations + n_qubits + 1, f"Size of the array is {len(array)} instead of {5 * n_operations + n_qubits + 1}"
        ansatz_numpy = array[:n_operations*5]
        measurement_numpy = array[n_operations*5:-1]
        type_numpy = array[-1]
        ansatz = Ansatz.from_numpy(ansatz_numpy, n_features, n_qubits, n_operations, allow_midcircuit_measurement, shift_second_wire)
        measurement = "".join(Kernel.PAULIS[np.rint(i).astype(int)] for i in measurement_numpy)
        the_type = KernelType.convert(type_numpy)
        kernel = KernelFactory.create_kernel(ansatz, measurement, the_type)
        return kernel

    def __str__(self):
        return str(self.ansatz) + " -> " + self.measurement

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return Kernel.from_numpy(self.to_numpy(), self.ansatz.n_features, self.ansatz.n_qubits, self.ansatz.n_operations, self.ansatz.allow_midcircuit_measurement)
