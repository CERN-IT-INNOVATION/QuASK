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

    def create_device(self, n_qubits): 
        return qml.device(self.device_name, wires=n_qubits, shots=self.n_shots)
    
    def __init__(self, ansatz: Ansatz, measurement: str, type: KernelType, device_name: str = "default.qubit", n_shots: int = None):
        """
        Initialization.

        :param ansatz: Ansatz object representing the unitary transformation
        :param measurement: Pauli string representing the measurement
        :param type: type of kernel, fidelity or observable
        :param device_name: name of the device, 'default.qubit' for noiseless simulation
        :param n_shots: number of shots when sampling the solution, None to have infinity
        """
        
        super().__init__(ansatz, measurement, type)
        self.device_name = device_name
        self.n_shots = n_shots

        dev = self.create_device(self.ansatz.n_qubits)
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

        dev_swap = self.create_device(1+2*self.ansatz.n_qubits)
        n = self.ansatz.n_qubits

        @qml.qnode(dev_swap)
        def swap_kernel(x1, x2):
            qml.Hadamard(wires=[0])
            AnsatzTemplate(self.ansatz, x1, wires=1+wires)
            AnsatzTemplate(self.ansatz, x2, wires=1+n+wires)
            for j in measurement_wires:
                qml.CSWAP(wires=[0, 1+j, 1+n+j])
            qml.Hadamard(wires=[0])
            return qml.probs(wires=[0])

        self.swap_kernel = swap_kernel

    def kappa(self, x1, x2) -> float:
        if self.type == KernelType.OBSERVABLE:
            return self.phi(x1) * self.phi(x2)

        elif self.type == KernelType.FIDELITY:
            probabilities = self.fidelity_kernel(x1, x2)
            self.last_probabilities = probabilities
            return probabilities[0]
        
        elif self.type == KernelType.SWAP_TEST:
            probabilities = self.swap_kernel(x1, x2)
            self.last_probabilities = probabilities
            return np.max([2 * probabilities[0] - 1, 0.0])

    def phi(self, x) -> float:
        if self.type == KernelType.OBSERVABLE:
            probabilities = self.observable_phi(x)
            self.last_probabilities = probabilities
            parity = lambda i: 1 if bin(i).count('1') % 2 == 0 else -1
            probabilities = np.array([parity(i) * probabilities[i] for i in range(len(probabilities))])
            # sum_probabilities = np.sum(probabilities)
            # print(f"{sum_probabilities=} {probabilities=}")
            return np.sum(probabilities)

        elif self.type in [KernelType.FIDELITY, KernelType.SWAP_TEST]:
            raise ValueError("phi not available for fidelity kernels")

        else:
            raise ValueError("Unknown type, possible erroneous loading from a numpy array")
        
