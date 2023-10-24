import pennylane as qml
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

sigma_x = jnp.array([[0, 1], [1, 0]])
sigma_y = jnp.array([[0, -1j], [1j, 0]])
sigma_z = jnp.array([[1, 0], [0, -1]])
sigma_id = jnp.eye(2)
pauli_vector = jnp.array([sigma_id, sigma_x, sigma_y, sigma_z])


def compile_pqc(matrix, parameters):
    """
    Convert a 6xn matrix parameterized by the features given into a quantum circuit in Pennylane
    :param matrix:
    :param parameters:
    :return: None
    """
    n_rows, n_gates = matrix.shape
    assert n_rows == 6, "There must be 6 rows (gen1, gen2, q1, q2, feature, bandwidth)"
    for i in range(n_gates):
        (gen1, gen2, q1, q2, feature, bandwidth) = matrix[:, i]
        generator = jnp.kron(pauli_vector[gen1], pauli_vector[gen2])
        unitary = expm(-1j * parameters[feature] * bandwidth * generator)
        qml.QubitUnitary(unitary, wires=[int(q1), int(q2)])


def pqc_integral(n, matrix, n_features, n_samples, key=None, dev=None):
    """
    Calculate the operator A = ∫_w |Ψ(w)⟩⟨Ψ(w)| dw
    :param n: number of qubits
    :param matrix: circuit matrix
    :param n_features: number of features
    :param n_samples: number of samples = precision
    :param key: random key
    :return: integral value
    """
    if dev is None:
        dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(params):
        compile_pqc(matrix, params)
        return qml.state()

    randunit_density = jnp.zeros((2**n, 2**n))

    for _ in range(n_samples):
        key1, key = jax.random.split(key)
        this_features = jax.random.normal(key, shape=(n_features,)) * jnp.pi
        A = circuit(this_features).reshape((-1, 1))
        randunit_density += jnp.kron(A, A.conj().T)

    return randunit_density / n_samples


def pqc_histogram(n, matrix, n_features, n_samples, n_bins, key=None, dev=None):
    """

    :param n:
    :param matrix:
    :param n_features:
    :param n_samples:
    :param n_bins:
    :param key:
    :param dev:
    :return:
    """
    if dev is None:
        dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(theta1, theta2):
        compile_pqc(matrix, theta1)
        qml.adjoint(compile_pqc)(matrix, theta2)
        return qml.probs(wires=range(n))

    histogram = [0] * n_bins

    for _ in range(n_samples):
        key1, key2, key = jax.random.split(key, num=3)
        theta_1 = jax.random.normal(key1, shape=(n_features,)) * jnp.pi
        theta_2 = jax.random.normal(key2, shape=(n_features,)) * jnp.pi
        fidelity = circuit(theta_1, theta_2).data[0]
        index = int(fidelity * n_bins)
        histogram[jnp.minimum(index, n_bins - 1)] += 1

    return jnp.array(histogram) / n_samples

