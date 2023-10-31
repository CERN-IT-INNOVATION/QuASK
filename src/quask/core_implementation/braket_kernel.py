import numpy as np
import pennylane as qml
from ..core import Ansatz, Kernel, KernelType
from .pennylane_kernel import PennylaneKernel


class BraketKernel(PennylaneKernel):

    def create_device(self, n_qubits): 
        return qml.device(
            "braket.aws.qubit", 
            device_arn=self.device_name, 
            s3_destination_folder=(self.s3_bucket, self.s3_prefix), 
            wires=n_qubits
        )
    
    def __init__(self, ansatz: Ansatz, measurement: str, type: KernelType, 
                 device_name: str, s3_bucket: str, s3_prefix: str, n_shots: int = None):
        """
        Initialization.

        :param ansatz: Ansatz object representing the unitary transformation
        :param measurement: Pauli string representing the measurement
        :param type: type of kernel, fidelity or observable
        :param device_name: name of the device, 'default.qubit' for noiseless simulation
        :param n_shots: number of shots when sampling the solution, None to have infinity
        """
        
        super().__init__(ansatz, measurement, type, device_name, n_shots)
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix

