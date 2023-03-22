import numpy as np
import pennylane as qml
from ..core import Ansatz, Kernel, KernelType


def AnsatzTemplate(ansatz: Ansatz, params: np.ndarray, wires: np.ndarray):
    for operation in ansatz.operation_list:
        if "M" not in operation.generator:
            feature = np.pi if operation.feature == ansatz.n_features else params[operation.feature]
            qml.PauliRot(operation.bandwidth * feature, operation.generator, wires=wires[operation.wires])
        elif operation.generator[0] == "M":
            qml.measure(wires[operation.wires[0]])
        else:
            qml.measure(wires[operation.wires[1]])


def ChangeBasis(measurement: str):
    for i, pauli in enumerate(measurement):
        if pauli == 'X':
            qml.Hadamard(wires=[i])
        elif pauli == 'Y':
            qml.S(wires=[i])
            qml.Hadamard(wires=[i])


class PennylaneKernel(Kernel):

    def __init__(self, ansatz: Ansatz, measurement: str, type: KernelType):
        super().__init__(ansatz, measurement, type)

        dev = qml.device("default.qubit", wires=self.ansatz.n_qubits, shots=None)
        wires = np.array(list(range(self.ansatz.n_qubits)))
        measurement_wires = np.array([i for i in wires if measurement[i] != 'I'])
        if len(measurement_wires) == 0:
            measurement_wires = range(self.ansatz.n_qubits)

        @qml.qnode(dev)
        def fidelity_kernel(x1, x2):
            AnsatzTemplate(self.ansatz, x1, wires=wires)
            qml.adjoint(AnsatzTemplate)(self.ansatz, x2, wires=wires)
            ChangeBasis(self.measurement)
            return qml.probs(wires=measurement_wires)

        self.fidelity_kernel = fidelity_kernel

        @qml.qnode(dev)
        def observable_phi(x):
            AnsatzTemplate(self.ansatz, x, wires=wires)
            ChangeBasis(self.measurement)
            return qml.probs(wires=measurement_wires)

        self.observable_phi = observable_phi

    def kappa(self, x1, x2) -> float:
        if self.type == KernelType.OBSERVABLE:
            return self.phi(x1) * self.phi(x2)

        elif self.type == KernelType.FIDELITY:
            probabilities = self.fidelity_kernel(x1, x2)
            self.last_probabilities = probabilities
            return probabilities[0]

    def phi(self, x) -> float:
        if self.type == KernelType.OBSERVABLE:
            probabilities = self.observable_phi(x)
            self.last_probabilities = probabilities
            parity = lambda i: 1 if bin(i).count('1') % 2 == 0 else -1
            probabilities = np.array([parity(i) * probabilities[i] for i in range(len(probabilities))])
            sum_probabilities = np.sum(probabilities)
            # print(f"{sum_probabilities=} {probabilities=}")
            return np.sum(probabilities)

        elif self.type == KernelType.FIDELITY:
            raise ValueError("phi not available for fidelity kernels")
