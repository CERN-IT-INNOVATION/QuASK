import sys
sys.path.append("src/")
import pytest
import quask
import numpy as np
from quask.core import Ansatz, Kernel, KernelFactory, KernelType
from quask.core_implementation import PennylaneKernel

def check_kernel_value(kernel: Kernel, x1: float, x2: float, expected: float):
    similarity = kernel.kappa(x1, x2)
    print(similarity, expected)
    assert np.isclose(similarity, expected), f"Kernel value is {similarity:0.3f} while {expected:0.3f} was expected"

def check_kernel_rx_value(kernel: Kernel, x1: float, x2: float):
    def rx(theta): 
        return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]])
    ket_zero = np.array([[1], [0]])
    ket_phi = np.linalg.inv(rx(x2)) @ rx(x1) @ ket_zero
    expected_similarity = (np.abs(ket_phi[0])**2).real
    check_kernel_value(kernel, np.array([x1]), np.array([x2]), expected_similarity)

def test_rx_kernel_fidelity():

    ansatz = Ansatz(n_features=1, n_qubits=2, n_operations=1, allow_midcircuit_measurement=False)
    ansatz.initialize_to_identity()
    ansatz.change_operation(0, new_feature=0, new_wires=[0, 1], new_generator="XI", new_bandwidth=1.0)
    kernel = PennylaneKernel(ansatz, "ZZ", KernelType.FIDELITY, device_name="default.qubit", n_shots=None)

    check_kernel_value(kernel, np.array([0.33]), np.array([0.33]), 1.0)

    check_kernel_rx_value(kernel, 0.00,     0.00)
    check_kernel_rx_value(kernel, 0.33,     0.33)
    check_kernel_rx_value(kernel, np.pi/2,  np.pi/2)
    check_kernel_rx_value(kernel, np.pi,    np.pi)
    check_kernel_rx_value(kernel, 0,        np.pi)
    check_kernel_rx_value(kernel, 0.33,     np.pi)
    check_kernel_rx_value(kernel, np.pi/2,  np.pi)
    check_kernel_rx_value(kernel, 0,        0.55)
    check_kernel_rx_value(kernel, 0.33,     0.55)
    check_kernel_rx_value(kernel, np.pi/2,  0.55)

def test_rx_kernel_fidelity():

    ansatz = Ansatz(n_features=1, n_qubits=2, n_operations=1, allow_midcircuit_measurement=False)
    ansatz.initialize_to_identity()
    ansatz.change_operation(0, new_feature=0, new_wires=[0, 1], new_generator="XI", new_bandwidth=1.0)
    kernel = PennylaneKernel(ansatz, "ZZ", KernelType.SWAP_TEST, device_name="default.qubit", n_shots=None)

    check_kernel_value(kernel, np.array([0.33]), np.array([0.33]), 1.0)

    check_kernel_rx_value(kernel, 0.00,     0.00)
    check_kernel_rx_value(kernel, 0.33,     0.33)
    check_kernel_rx_value(kernel, np.pi/2,  np.pi/2)
    check_kernel_rx_value(kernel, np.pi,    np.pi)
    check_kernel_rx_value(kernel, 0,        np.pi)
    check_kernel_rx_value(kernel, 0.33,     np.pi)
    check_kernel_rx_value(kernel, np.pi/2,  np.pi)
    check_kernel_rx_value(kernel, 0,        0.55)
    check_kernel_rx_value(kernel, 0.33,     0.55)
    check_kernel_rx_value(kernel, np.pi/2,  0.55)

