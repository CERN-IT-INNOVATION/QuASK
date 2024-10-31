import numpy as np
from ..core import Ansatz, Kernel, KernelType

# from qiskit import Aer, BasicAer, QuantumCircuit
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_ibm_runtime import QiskitRuntimeService, IBMBackend
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_ibm_runtime import SamplerV2 as IBMSampler
from qiskit_ibm_runtime import EstimatorV2 as IBMEstimator
from qiskit_ibm_runtime import RuntimeJobV2
# from qiskit_ibm_runtime import Options
from qiskit_ibm_runtime.options import SamplerOptions as sop
from qiskit_ibm_runtime.options import EstimatorOptions as eop
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.result import QuasiDistribution
from qiskit.primitives import BackendSamplerV2 as BackendSampler
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
# from qiskit_aer.primitives import Sampler as AerSampler
# from qiskit_aer.primitives import Estimator as AerEstimator

class QiskitKernel(Kernel):

    def __init__(self, ansatz: Ansatz, measurement: str, type: KernelType, 
                 platform: str="finite_shots", backend: IBMBackend=None, n_shots: int=2048,
                 options: dict=None, optimization_level: int=2, layout: list=None):
        """
        Initialization.

        :param ansatz: Ansatz object representing the unitary transformation
        :param measurement: Pauli string representing the measurement
        :param type: type of kernel, fidelity or observable
        :param platform: name of the device, 'finite_shots' or 'infty_shots' for noiseless simulation
        :param backend: simulator or hardware backend
        :param n_shots: number of shots when sampling the solution, None to have infinity
        :param options: options of the sampler or estimator
        :param optimization_level: optimization level in case of transpilation
        :param layout: qubit layout of the physical circuit in case of transpilation
        """
        super().__init__(ansatz, measurement, type)
        assert platform in ["infty_shots", "finite_shots", "ibm_quantum"]
        self.platform = platform
        self.backend = backend
        self.n_shots = n_shots
        self.options = options
        self.optimization_level = optimization_level
        self.layout = layout

    def get_backend(self, channel: str, ibm_token: str, group_instance: str, device: str=None):
        service = QiskitRuntimeService(channel=channel, token=ibm_token, instance=group_instance)
        if device == None:
            self.backend = service.least_busy(operational=True, simulator=False)
        else:
            self.backend = service.backend(device)
        print(f"{self.backend.name} selected")
        return self

    def get_sampler_options(self):
        sampler_options = sop(
            default_shots = self.n_shots,
            dynamical_decoupling = {
                "enable": bool(self.options["dynamical_decoupling"]["sequence_type"]),
                "sequence_type": self.options["dynamical_decoupling"]["sequence_type"]
            },
            twirling = {
                "enable_gates": self.options["twirling"]["enable_gates"],
                "enable_measure": self.options["twirling"]["enable_measure"],
                "num_randomizations": "auto",
                "shots_per_randomization": "auto"
            }
        )
        return sampler_options
    
    def get_estimator_options(self):
        estimator_options = eop(
            default_shots = self.n_shots,
            dynamical_decoupling = {
                "enable": bool(self.options["dynamical_decoupling"]["sequence_type"]),
                "sequence_type": self.options["dynamical_decoupling"]["sequence_type"]
            },
            twirling = {
                "enable_gates": self.options["twirling"]["enable_gates"],
                "enable_measure": self.options["twirling"]["enable_measure"],
                "num_randomizations": "auto",
                "shots_per_randomization": "auto"
            }
        )
        return estimator_options
    
    def get_sampler(self):
        if self.platform == "infty_shots":
            return Statevector
        
        elif self.platform == "finite_shots":
            return StatevectorSampler(
                default_shots=self.n_shots
                )
        elif self.platform == "ibm_quantum":
            options = self.get_sampler_options()
            return IBMSampler(backend=self.backend, options=options)
        
    def get_estimator(self):
        if self.platform == "infty_shots":
            return StatevectorEstimator
        elif self.platform == "ibm_quantum":
            options = self.get_estimator_options()
            return IBMEstimator(backend=self.backend, options=options)

    def get_running_method(self, qc: QuantumCircuit):
        sampler = self.get_sampler()
        if self.platform == "infty_shots":
            res = Statevector.from_instruction(qc).data[0].real
        elif self.platform == "finite_shots":
            qc.measure_all()
            counts = (
                    sampler.run([qc]).result()[0].data.meas.get_int_counts()
            )
            dist = QuasiDistribution(
                {meas: count / self.n_shots for meas, count in counts.items()}, shots=self.n_shots
            )
            res = dist.get(0, 0.0)
        elif self.platform == "ibm_quantum":
            qc.measure_all()
            logical_circuit = qc
            pm = generate_preset_pass_manager(optimization_level=self.optimization_level, backend=self.backend, initial_layout=self.layout)
            physical_circuit = pm.run(logical_circuit)
            job = sampler.run([physical_circuit])
            print(f"Job sent to hardware. Job ID: {job.job_id()}")
            res = job

        return res
    
    def get_job_results(self, job: RuntimeJobV2):
        counts = job.result()[0].data.meas.get_int_counts()
        dist = QuasiDistribution(
            {meas: count / self.n_shots for meas, count in counts.items()}, shots=self.n_shots
        )
        res = dist.get(0, 0.0)
        return res

    def get_qiskit_ansatz(self):
        n_params = self.ansatz.n_features
        params = ParameterVector('p', n_params)
        qc = QuantumCircuit(self.ansatz.n_qubits)
        for operation in self.ansatz.operation_list:
            operator = SparsePauliOp(operation.generator)
            rotation = operation.bandwidth*params[operation.feature]/2
            evo = PauliEvolutionGate(operator, time=rotation)
            qc.append(evo, operation.wires)
        return qc

    def kappa(self, x1: np.ndarray, x2: np.ndarray) -> float:
        assert len(x1) == self.ansatz.n_features
        assert len(x2) == self.ansatz.n_features

        if self.type == KernelType.OBSERVABLE:
            return self.phi(x1) * self.phi(x2)

        elif self.type == KernelType.FIDELITY:
            qc = QuantumCircuit(self.ansatz.n_qubits, self.ansatz.n_qubits)
            qc.append(self.get_qiskit_ansatz().assign_parameters(x1.tolist()), range(self.ansatz.n_qubits))
            qc.append(self.get_qiskit_ansatz().assign_parameters(x2.tolist()).inverse(), range(self.ansatz.n_qubits))
            probabilities = self.get_running_method(qc)
            return probabilities
        
        elif self.type == KernelType.SWAP_TEST:
            qc = QuantumCircuit(1+2*self.ansatz.n_qubits, 1)
            qc.h(0)
            qc.append(self.get_qiskit_ansatz().assign_parameters(x1.tolist() + [1.0]), range(1, 1+self.ansatz.n_qubits))
            qc.append(self.get_qiskit_ansatz().assign_parameters(x2.tolist() + [1.0]), range(self.ansatz.n_qubits))
            for i in range(self.ansatz.n_qubits):
                qc.cswap(0, 1+i, 1+self.ansatz.n_qubits+i)
            qc.h(0)
            qc.measure(0, 0)
            job = self.get_sampler().run(qc)
            probabilities = job.result().quasi_dists[0]
            return probabilities.get(0, 0.0)
        
    def phi(self, x: np.ndarray) -> float:
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