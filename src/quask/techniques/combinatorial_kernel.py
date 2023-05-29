"""
Code base for https://arxiv.org/abs/2209.11144. This module allow to convert a parametric quantum
circuit into a matrix representation. Furthermore, it allows to generate a parameteric kernel function
parameterized by such circuit matrix representation.
"""

import pennylane as qml
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm
import jax


sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_id = np.eye(2)
pauli_vector = jnp.array([sigma_id, sigma_x, sigma_y, sigma_z])


def create_operation(n_qubits, n_layers, index, pauli, angle):
    """
    Append to the Pennylane circuit a quantum transformation having the specified configuration according to the
    scheme in https://arxiv.org/abs/2209.11144

    Args:
        n_qubits: number of qubits of the circuit
        n_layers: number of layers of the circuit
        index: number of operation within the circuit (determine if it is a single or two-qubits operation and its position)
        pauli: determines the generators
        angle: feature corresponding to the angle of the transformation

    Returns:
        None
    """
    assert 0 <= index < n_qubits * 2 * n_layers
    # assert 0 <= pauli < 16
    index_within_layer = index % n_layers
    if index_within_layer < n_qubits:  # single qubit operation
        unitary = expm(-1j * angle * pauli_vector[pauli % 4])
        qml.QubitUnitary(unitary, wires=index_within_layer % n_qubits)
    else:  # two qubits operation
        unitary = expm(-1j * angle * jnp.kron(pauli_vector[pauli % 4], pauli_vector[pauli // 4]))
        qml.QubitUnitary(unitary, wires=(index_within_layer % n_qubits, (index_within_layer + 1) % n_qubits))


def create_identity_combinatorial_kernel(n_qubits, n_layers):
    """
    Create a matrix representation for the identity circuit according to the
    scheme in https://arxiv.org/abs/2209.11144

    Args:
        n_qubits: number of qubits of the circuit
        n_layers: number of layers of the circuit

    Returns:
        numpy matrix
    """
    n_gates = n_qubits * 2 * n_layers
    return np.zeros(shape=(n_gates, 2))


def create_random_combinatorial_kernel(n_qubits, n_layers, n_operations):
    """
    Create a matrix representation for a randomly generated circuit according to the
    scheme in https://arxiv.org/abs/2209.11144

    Args:
        n_qubits: number of qubits of the circuit
        n_layers: number of layers of the circuit

    Returns:
        numpy matrix
    """
    n_gates = n_qubits * 2 * n_layers
    return np.stack((
        np.random.randint(0, 16, size=(n_gates,)),
        np.random.randint(0, n_operations, size=(n_gates,))
    )).reshape((n_gates, 2))


def CombinatorialFeatureMap(x, n_qubits, n_layers, solution, bandwidth):
    """
    Generate a PennyLane feature map starting from the matrix representation of the circuit

    Args:
        x: feature data
        n_qubits: number of qubits of the circuit
        n_layers: number of layers of the circuit
        solution: matrix representation of the circuit
        bandwidth: optional constant limiting the rotational angles of the transformations

    Returns:
        numpy matrix
    """

    n_gates = n_qubits * 2 * n_layers
    assert solution.shape == (n_gates, 2), f"Shape is {solution.shape} instead of {(n_gates, 2)} {solution=}"
    for index in range(n_gates):
        pauli = solution[index][0]
        operation_idx = solution[index][1]
        angle = bandwidth * x[operation_idx]
        create_operation(n_qubits, n_layers, index, pauli, angle)


def CombinatorialKernel(n_qubits, n_layers):
    """
    Generate a kernel function starting from a given CombinatorialFeatureMap object.

    Args:
        n_qubits: number of qubits of the circuit
        n_layers: number of layers of the circuit

    Returns:
        kernel function having the parameters x1, x2, the_solution, the_bandwidth and returning a scalar value
    """

    def combinatorial_kernel_wrapper(x1, x2, the_solution, the_bandwidth):
        """
        Kernel function corresponding to the given embedding, simulated with Jax

        Args:
            x1: feature data of the first point
            x2: feature data of the first point
            the_solution: matrix representation of the circuit
            the_bandwidth: optional constant limiting the rotational angles of the transformations

        Returns:
            kernel value (scalar)
        """
        device = qml.device("default.qubit.jax", wires=n_qubits)
        the_solution = the_solution.reshape((2*n_qubits*n_layers, 2))

        # create projector (measures probability of having all "00...0")
        projector = np.zeros((2 ** n_qubits, 2 ** n_qubits))
        projector[0, 0] = 1

        # define the circuit for the quantum kernel ("overlap test" circuit)
        @qml.qnode(device, interface='jax')
        def combinatorial_kernel():
            CombinatorialFeatureMap(x1, n_qubits, n_layers, the_solution, the_bandwidth)
            qml.adjoint(CombinatorialFeatureMap)(x2, n_qubits, n_layers, the_solution, the_bandwidth)
            return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

        return combinatorial_kernel()

    return jax.jit(combinatorial_kernel_wrapper)
