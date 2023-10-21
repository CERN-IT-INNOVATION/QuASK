import numpy as np
from typing import List
import itertools


class Operation:
    """
    Class representing a 2-qubit rotational quantum gates $exp(-i \theta \sigma_1 \otimes \sigma_2)$
    """

    PAULI_GENERATORS = list(a + b for a, b in itertools.product(["I", "X", "Y", "Z"], repeat=2))
    MEASUREMENT_OPERATIONS = ["IM", "MI"]
    OPERATIONS = PAULI_GENERATORS + MEASUREMENT_OPERATIONS

    def __init__(self, generator: str, wires: List[int], feature: int, bandwidth: float):
        """
        Operation initializer
        :param generator: one of the elements of Operation.OPERATIONS
        :param wires: pair of integers
        :param feature: index of the feature parameterizing the element (can be -1 for constant feature '1')
        :param bandwidth: bandwidth parameter in range [0,1]
        """
        self.generator: str = generator
        self.wires: List[int] = wires
        self.feature: int = feature
        self.bandwidth: float = bandwidth

    def to_numpy(self):
        """
        Serialize the Operation object to a numpy array format
        :return: numpy array representing the operation
        """
        return np.array([Operation.OPERATIONS.index(self.generator), self.wires[0], self.wires[1], self.feature, self.bandwidth])

    @staticmethod
    def from_numpy(array):
        """
        Deserialize the operation object given its numpy array description
        :param array: numpy array
        :return: Operation object
        """
        op = Operation(None, None, None, None)
        op.generator = Operation.OPERATIONS[int(array[0])]
        op.wires = [int(array[1]), int(array[2])]
        op.feature = int(array[3])
        op.bandwidth = float(array[4])
        return op

    def __str__(self):
        return f"-i {self.bandwidth:0.2f} * x[{self.feature}] {self.generator}^({self.wires[0]},{self.wires[1]})"

    def __repr__(self):
        return self.__str__()