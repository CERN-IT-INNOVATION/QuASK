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
    n_gates = n_qubits * 2 * n_layers
    return np.zeros(shape=(n_gates, 2))


def create_random_combinatorial_kernel(n_qubits, n_layers, n_operations):
    n_gates = n_qubits * 2 * n_layers
    return np.stack((
        np.random.randint(0, 16, size=(n_gates,)),
        np.random.randint(0, n_operations, size=(n_gates,))
    )).reshape((n_gates, 2))


def CombinatorialFeatureMap(x, n_qubits, n_layers, solution, bandwidth):

    n_gates = n_qubits * 2 * n_layers
    assert solution.shape == (n_gates, 2), f"Shape is {solution.shape} instead of {(n_gates, 2)} {solution=}"
    for index in range(n_gates):
        pauli = solution[index][0]
        operation_idx = solution[index][1]
        angle = bandwidth * x[operation_idx]
        create_operation(n_qubits, n_layers, index, pauli, angle)


def CombinatorialKernel(n_qubits, n_layers):

    def combinatorial_kernel_wrapper(x1, x2, the_solution, the_bandwidth):
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


