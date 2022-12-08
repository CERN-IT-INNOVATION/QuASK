"""
Module dedicated to define Templates for Pennylane quantum circuits.
See https://pennylane.readthedocs.io/en/stable/introduction/templates.html for details.
"""


import jax
import pennylane as qml
import numpy as np


def rx_embedding(x, wires):
    """
    Encode the data with one rotation on sigma_x per qubit per feature

    Args:
        x: feature vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)

    Returns:
        None
    """
    qml.AngleEmbedding(x, wires=wires, rotation="X")


def ry_embedding(x, wires):
    """
    Encode the data with one rotation on sigma_y per qubit per feature

    Args:
        x: feature vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)

    Returns:
        None
    """
    qml.AngleEmbedding(x, wires=wires, rotation="Y")


def rz_embedding(x, wires):
    """
    Encode the data with one hadamard then one rotation on sigma_y per qubit per feature

    Args:
        x: feature vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)

    Returns:
        None
    """
    qml.Hadamard(wires=wires)
    qml.AngleEmbedding(x, wires=wires, rotation="Z")


def zz_fullentanglement_embedding(x, wires):
    """
    Encode the data with the ZZ Feature Map (https://qiskit.org/documentation/stubs/qiskit.circuit.library.ZZFeatureMap.html)

    Args:
        x: feature vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)

    Returns:
        None
    """
    N = len(wires)
    for i in range(N):
        qml.Hadamard(wires=i)
        qml.RZ(2 * x[i], wires=i)
    for i in range(N):
        for j in range(i + 1, N):
            qml.CRZ(2 * (np.pi - x[i]) * (np.pi - x[j]), wires=[i, j])


def hardware_efficient_ansatz(theta, wires):
    """
    Hardware efficient ansatz

    Args:
        theta: parameter vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)

    Returns:
        None
    """
    N = len(wires)
    assert len(theta) == 2 * N
    for i in range(N):
        qml.RX(theta[2 * i], wires=wires[i])
        qml.RY(theta[2 * i + 1], wires=wires[i])
    for i in range(N - 1):
        qml.CZ(wires=[wires[i], wires[i + 1]])


def tfim_ansatz(theta, wires):
    """
    Transverse Field Ising Model
    Figure 6a (left) in https://arxiv.org/pdf/2105.14377.pdf

    Args:
        theta: parameter vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)

    Returns:
        None
    """
    N = len(wires)
    assert len(theta) == 2
    for i in range(N // 2):
        qml.MultiRZ(theta[0], wires=[wires[2 * i], wires[2 * i + 1]])
    for i in range(N // 2 - 1):
        qml.MultiRZ(theta[0], wires=[wires[2 * i + 1], wires[2 * i + 2]])
    for i in range(N):
        qml.RX(theta[1], wires=wires[i])


def ltfim_ansatz(theta, wires):
    """
    Transverse Field Ising Model with additional sigma_z rotations.
    Figure 6a (right) in https://arxiv.org/pdf/2105.14377.pdf

    Args:
        theta: parameter vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)

    Returns:
        None
    """
    N = len(wires)
    assert len(theta) == 3
    tfim_ansatz(theta[:2], wires)
    for i in range(N):
        qml.RZ(theta[2], wires=wires[i])


def zz_rx_ansatz(theta, wires):
    """
    ZZX Model
    Figure 7a in https://arxiv.org/pdf/2109.11676.pdf

    Args:
        theta: parameter vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)

    Returns:
        None
    """
    N = len(wires)
    assert len(theta) == 2
    for i in range(N // 2):
        qml.MultiRZ(theta[0], wires=[wires[2 * i], wires[2 * i + 1]])
    for i in range(N // 2 - 1):
        qml.MultiRZ(theta[0], wires=[wires[2 * i + 1], wires[2 * i + 2]])
    for i in range(N):
        qml.RX(theta[1], wires=wires[i])


def random_qnn_encoding(x, wires, trotter_number=10):
    """
    This function creates and appends a quantum neural network to the selected
    encoding. It follows formula S(116) in the Supplementary.

    Args:
        x: feature vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)
        trotter_number: number of repetitions (int)

    Returns:
        None
    """
    assert len(x) == len(wires)
    # embedding
    ry_embedding(x, wires)
    # random rotations
    for _ in range(trotter_number):
        for i in range(len(wires) - 1):
            angle = np.random.normal()
            qml.RXX(angle, wires=[wires[i], wires[i + 1]])
            qml.RYY(angle, wires=[wires[i], wires[i + 1]])
            qml.RZZ(angle, wires=[wires[i], wires[i + 1]])


def projected_xyz_embedding(embedding, X):
    """
    Create a Quantum Kernel given the template written in Pennylane framework

    Args:
        embedding: Pennylane template for the quantum feature map
        X: feature data (matrix)

    Returns:
        projected quantum feature map X
    """
    N = X.shape[1]

    # create device using JAX
    device = qml.device("default.qubit.jax", wires=N)

    # define the circuit for the quantum kernel ("overlap test" circuit)
    @jax.jit
    @qml.qnode(device)
    def proj_feature_map(x):
        embedding(x, wires=range(N))
        return (
            [qml.expval(qml.PauliX(i)) for i in range(N)]
            + [qml.expval(qml.PauliY(i)) for i in range(N)]
            + [qml.expval(qml.PauliZ(i)) for i in range(N)]
        )

    # build the gram matrix
    X_proj = [proj_feature_map(x) for x in X]

    return X_proj


def pennylane_quantum_kernel(feature_map, X_1, X_2=None):
    """
    Create a Quantum Kernel given the template written in Pennylane framework

    Args:
        feature_map: Pennylane template for the quantum feature map
        X_1: First dataset
        X_2: Second dataset

    Returns:
        Gram matrix
    """
    if X_2 is None:
        X_2 = X_1  # Training Gram matrix
    assert (
        X_1.shape[1] == X_2.shape[1]
    ), "The training and testing data must have the same dimensionality"
    N = X_1.shape[1]

    # create device using JAX
    device = qml.device("default.qubit.jax", wires=N)

    # create projector (measures probability of having all "00...0")
    projector = np.zeros((2**N, 2**N))
    projector[0, 0] = 1

    # define the circuit for the quantum kernel ("overlap test" circuit)
    @jax.jit
    @qml.qnode(device)
    def kernel(x1, x2):
        feature_map(x1, wires=range(N))
        qml.adjoint(feature_map)(x2, wires=range(N))
        return qml.expval(qml.Hermitian(projector, wires=range(N)))

    # build the gram matrix
    gram = np.zeros(shape=(X_1.shape[0], X_2.shape[0]))
    for i in range(X_1.shape[0]):
        for j in range(i, X_2.shape[0]):
            gram[i][j] = kernel(X_1[i], X_2[j])
            gram[j][i] = gram[i][j]

    return gram


def pennylane_projected_quantum_kernel(feature_map, X_1, X_2=None, params=[1.0]):
    """
    Create a Quantum Kernel given the template written in Pennylane framework.

    Args:
        feature_map: Pennylane template for the quantum feature map
        X_1: First dataset
        X_2: Second dataset
        params: List of one single parameter representing the constant in the exponentiation

    Returns:
        Gram matrix
    """
    if X_2 is None:
        X_2 = X_1  # Training Gram matrix
    assert (
        X_1.shape[1] == X_2.shape[1]
    ), "The training and testing data must have the same dimensionality"

    X_1_proj = projected_xyz_embedding(feature_map, X_1)
    X_2_proj = projected_xyz_embedding(feature_map, X_2)

    # build the gram matrix
    gamma = params[0]

    gram = np.zeros(shape=(X_1.shape[0], X_2.shape[0]))
    for i in range(X_1_proj.shape[0]):
        for j in range(X_2_proj.shape[0]):
            value = np.exp(-gamma * ((X_1_proj[i] - X_2_proj[j]) ** 2).sum())
            gram[i][j] = value

    return gram


def pennylane_linear_projected_kernel():
    """
    Function from Francesco's msc thesis - please fill with the description
    """
    pass
