import numpy as np
from typing import List
from . import Operation


class Ansatz:
    """
    Class representing the Ansatz as list of Operations
    """

    def __init__(self, n_features: int, n_qubits: int, n_operations: int, allow_midcircuit_measurement=False):
        """
        Initialization
        :param n_features: number of feature that can be used to parametrize the operation
        :param n_qubits: number of qubits of the circuit
        :param n_operations: number of operations
        :param allow_midcircuit_measurement: True if mid-circuit measurement are allowed
        """
        assert n_qubits >= 2, "This ansatz is specified for >= 2 qubits"
        assert n_features > 0, "Cannot have zero or negative number of features"
        assert n_operations > 0, "Cannot have zero or negative number or operations"
        self.n_features: int = n_features
        self.n_qubits: int = n_qubits
        self.n_operations: int = n_operations
        self.operation_list: List[Operation] = [None] * n_operations
        self.allow_midcircuit_measurement: bool = allow_midcircuit_measurement

    def change_operation(self, operation_index: int, new_feature: int, new_wires: List[int], new_generator: str, new_bandwidth: float):
        """
        Overwrite the operation at the given index with a whole new set of data
        :param operation_index: index of the operation
        :param new_feature: feature parameterizing the operation
        :param new_wires: wires on which the operation is applied
        :param new_generator: generator of the operation
        :param new_bandwidth: bandwidth of the operation
        :return: None
        """
        self.change_feature(operation_index, new_feature)
        self.change_wires(operation_index, new_wires)
        self.change_generators(operation_index, new_generator)
        self.change_bandwidth(operation_index, new_bandwidth)

    def change_bandwidth(self, operation_index: int, new_bandwidth: float):
        """
        Overwrite the operation at the given index with a new bandwidth
        :param operation_index: index of the operation
        :param new_bandwidth: bandwidth of the operation
        :return: None
        """
        assert 0 <= operation_index < self.n_operations, "Operation index out of bounds"
        self.operation_list[operation_index].bandwidth = new_bandwidth

    def change_generators(self, operation_index: int, new_generator: str):
        """
        Overwrite the operation at the given index with a new generator
        :param operation_index: index of the operation
        :param new_generator: generator of the operation
        :return: None
        """
        assert 0 <= operation_index < self.n_operations, "Operation index out of bounds"
        assert new_generator in Operation.OPERATIONS, f"Unknown generator {new_generator}"
        if not self.allow_midcircuit_measurement:
            assert new_generator not in Operation.MEASUREMENT_OPERATIONS, "Mid-circuit measurement not allowed"
        self.operation_list[operation_index].generator = new_generator

    def change_feature(self, operation_index: int, new_feature: int):
        """
        Overwrite the operation at the given index with a new feature
        :param operation_index: index of the operation
        :param new_feature: feature parameterizing the operation
        :return: None
        """
        assert 0 <= operation_index < self.n_operations, "Operation index out of bounds"
        assert 0 <= new_feature <= self.n_features, f"Feature index out of bounds ({new_feature=})"
        self.operation_list[operation_index].feature = new_feature

    def change_wires(self, operation_index: int, new_wires: List[int]):
        """
        Overwrite the operation at the given index with a new pair of wires
        :param operation_index: index of the operation
        :param new_wires: wires on which the operation is applied
        :return: None
        """
        assert 0 <= operation_index < self.n_operations, "Operation index out of bounds"
        assert len(new_wires) == 2, "The location is a list of two integers, not less no more"
        assert 0 <= new_wires[0] < self.n_qubits, f"First wire index out of bounds ({new_wires=})"
        assert 0 <= new_wires[1] < self.n_qubits, f"Second wire index out of bounds ({new_wires=})"
        assert new_wires[0] != new_wires[1], f"Cannot specify the same wire twice ({new_wires=})"
        self.operation_list[operation_index].wires = new_wires

    def get_allowed_operations(self):
        """
        Get the list of allowed operation for the ansatz, either only the PAULI_GENERATORS or any operation including measurements
        :return: list of allowed operations
        """
        if self.allow_midcircuit_measurement:
            return Operation.OPERATIONS
        else:
            return Operation.PAULI_GENERATORS

    def initialize_to_identity(self):
        """
        Initialize the ansatz to the identity circuit
        :return: None
        """
        self.operation_list = [None] * self.n_operations
        for i in range(self.n_operations):
            self.operation_list[i] = Operation("II", [0, 1], self.n_features, 1.0)

    def initialize_to_random_circuit(self):
        """
        Initialize the ansatz to a random circuit
        :return: None
        """
        for i in range(self.n_operations):
            generator = np.random.choice(self.get_allowed_operations())
            wires = np.random.choice(list(range(self.n_qubits)), 2, replace=False)
            feature = np.random.choice(list(range(self.n_features + 1)))
            bandwidth = np.random.uniform(0.0, 1.0)
            self.operation_list[i] = Operation(generator, wires, feature, bandwidth)

    def initialize_to_known_ansatz(self, ansatz):
        """
        Initialize the ansatz form an already given element
        :param ansatz: Given ansatz
        :return: None
        """
        self.initialize_to_identity()
        for i in range(self.n_operations):
            op: Operation = ansatz.operation_list[i]
            self.change_operation(i, op.feature, op.wires, op.generator, op.bandwidth)

    def to_numpy(self):
        """
        Serialize the ansatz to a numpy array
        :return: numpy array
        """
        return np.array([op.to_numpy() for op in self.operation_list]).ravel()

    @staticmethod
    def from_numpy(array, n_features, n_qubits, n_operations, allow_midcircuit_measurement, shift_second_wire=False):
        """
        Deserialize the ansatz from a numpy array
        :param array: numpy array
        :param n_features: number of feature that can be used to parametrize the operation
        :param n_qubits: number of qubits of the circuit
        :param n_operations: number of operations
        :param allow_midcircuit_measurement: True if mid-circuit measurement are allowed
        :return: Ansatz deserialized
        """
        ans = Ansatz(n_features, n_qubits, n_operations, allow_midcircuit_measurement)
        ans.initialize_to_identity()
        for i in range(n_operations):
            # feature -> wires -> generator -> bandwidth
            generator = np.rint(array[i * 5]).astype(int)
            wires = [np.rint(array[i * 5 + 1]).astype(int), np.rint(array[i * 5 + 2]).astype(int)]
            feature = np.rint(array[i * 5 + 3]).astype(int)
            bandwidth = np.round(array[i * 5 + 4], decimals=4)
            if shift_second_wire and wires[1] >= wires[0]:
                wires[1] += 1
            ans.change_operation(i, feature, wires, Operation.OPERATIONS[generator], bandwidth)
        return ans

    def __str__(self):
        return str(self.operation_list)

    def __repr__(self):
        return self.__str__()