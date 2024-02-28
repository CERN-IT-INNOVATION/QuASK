import numpy as np
from ..core import Ansatz, Kernel, KernelType

from qiskit import Aer, BasicAer, QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Sampler as IbmSampler
from qiskit_ibm_runtime import Estimator as IbmEstimator
from qiskit_ibm_runtime import Options
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_aer.primitives import Estimator as AerEstimator

class QiskitKernel(Kernel):

    def get_estimator(self):
        if self.platform == "Aer":
            return AerEstimator(
                backend_options={"method": "statevector"}, 
                run_options={"shots": self.n_shots})
        else:
            options = Options()
            options.optimization_level = self.optimization_level
            options.resilience_level = self.resilience_level
            return IbmEstimator(backend=self.backend, options=self.options)
    
    def get_sampler(self):
        if self.platform == "Aer":
            return AerSampler(
                backend_options={"method": "statevector"}, 
                run_options={"shots": self.n_shots})
        else:
            options = Options()
            options.optimization_level = self.optimization_level
            options.resilience_level = self.resilience_level
            return IbmSampler(backend=self.backend, options=self.options)

    def get_qiskit_ansatz(self):
        n_params = self.ansatz.n_features + 1
        params = ParameterVector('p', n_params)
        qc = QuantumCircuit(self.ansatz.n_qubits)
        qc.rx(0.0*np.prod(params), 0) # fake instruction to include all parameters in the quantum circuit
        for operation in self.ansatz.operation_list:
            operator = SparsePauliOp(operation.generator)
            rotation = operation.bandwidth*params[operation.feature]
            evo = PauliEvolutionGate(operator, time=rotation)
            qc.append(evo, operation.wires)
        return qc

    def __init__(self, ansatz: Ansatz, measurement: str, type: KernelType, 
                 platform="Aer", backend="qasm_simulator", n_shots=2048,
                 optimization_level=2, resilience_level=2):
        """
        Initialization.

        :param ansatz: Ansatz object representing the unitary transformation
        :param measurement: Pauli string representing the measurement
        :param type: type of kernel, fidelity or observable
        :param device_name: name of the device, 'default.qubit' for noiseless simulation
        :param n_shots: number of shots when sampling the solution, None to have infinity
        """
        super().__init__(ansatz, measurement, type)
        assert platform in ["Aer", "QiskitRuntimeService"]
        self.platform = platform
        self.backend_name = backend
        self.n_shots = n_shots
        self.optimization_level = optimization_level
        self.resilience_level = resilience_level
        

    def kappa(self, x1, x2) -> float:
        assert len(x1) == self.ansatz.n_features
        assert len(x2) == self.ansatz.n_features

        if self.type == KernelType.OBSERVABLE:
            return self.phi(x1) * self.phi(x2)

        elif self.type == KernelType.FIDELITY:
            qc = QuantumCircuit(self.ansatz.n_qubits, self.ansatz.n_qubits)
            qc.append(self.get_qiskit_ansatz().bind_parameters(x1.tolist() + [1.0]), range(self.ansatz.n_qubits))
            qc.append(self.get_qiskit_ansatz().bind_parameters(x2.tolist() + [1.0]).inverse(), range(self.ansatz.n_qubits))
            qc.measure(range(self.ansatz.n_qubits), range(self.ansatz.n_qubits))
            job = self.get_sampler().run(qc)
            probabilities = job.result().quasi_dists[0]
            return probabilities.get(0, 0.0)
        
        elif self.type == KernelType.SWAP_TEST:
            qc = QuantumCircuit(1+2*self.ansatz.n_qubits, 1)
            qc.h(0)
            qc.append(self.get_qiskit_ansatz().bind_parameters(x1.tolist() + [1.0]), range(1, 1+self.ansatz.n_qubits))
            qc.append(self.get_qiskit_ansatz().bind_parameters(x2.tolist() + [1.0]), range(self.ansatz.n_qubits))
            for i in range(self.ansatz.n_qubits):
                qc.cswap(0, 1+i, 1+self.ansatz.n_qubits+i)
            qc.h(0)
            qc.measure(0, 0)
            job = self.get_sampler().run(qc)
            probabilities = job.result().quasi_dists[0]
            return probabilities.get(0, 0.0)


    def phi(self, x) -> float:
        if self.type == KernelType.OBSERVABLE:
            
            assert len(x) == self.ansatz.n_features
            complete_features = x.tolist() + [1.0]
            circuit = self.get_qiskit_ansatz().bind_parameters(complete_features)
            observable = SparsePauliOp(self.measurement)
            job = self.get_estimator().run(circuit, observable)
            exp_val = job.result().values[0]
            return exp_val

        elif self.type in [KernelType.FIDELITY, KernelType.SWAP_TEST]:
            raise ValueError("phi not available for fidelity kernels")

        else:
            raise ValueError("Unknown type, possible erroneous loading from a numpy array")
        
