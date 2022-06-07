import jax
import jax.numpy as jnp
import pennylane as qml
import pennylane.numpy as np


def ZZFullEntFeatureMap(x, wires):
    N = len(wires)
    for i in range(N):
        qml.Hadamard(wires=i)
        qml.RZ(2 * x[i], wires=i)
    for i in range(N):
        for j in range(i + 1, N):
            qml.CRZ(2 * (np.pi - x[i]) * (np.pi - x[j]), wires=[i, j])


def pennylane_quantum_kernel(feature_map, X_1, X_2=None):
    """
    Create a Quantum Kernel given the template written in Pennylane framework
    :param feature_map: Pennylane template for the quantum feature map
    :param X_1: First dataset
    :param X_2: Second dataset
    :return: Gram matrix
    """
    if X_2 == None: X_2 = X_1  # Training Gram matrix
    assert X_1.shape[1] == X_2.shape[1], "The training and testing data must have the same dimensionality"
    N = X_1.shape[1]

    # create device using JAX
    device = qml.device("default.qubit.jax", wires=N)

    # create projector (measures probability of having all "00...0")
    projector = np.zeros((2 ** N, 2 ** N))
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
        for j in range(X_2.shape[0]):
            gram[i][j] = kernel(X_1[i], X_2[j])

    return gram


def pennylane_projected_feature_map(feature_map, X):
    """
    Create a Quantum Kernel given the template written in Pennylane framework
    :param feature_map: Pennylane template for the quantum feature map
    :param X: First dataset
    :return: projected quantum feature map X
    """
    N = X.shape[1]

    # create device using JAX
    device = qml.device("default.qubit.jax", wires=N)

    # define the circuit for the quantum kernel ("overlap test" circuit)
    @jax.jit
    @qml.qnode(device)
    def proj_feature_map(x):
        feature_map(x, wires=range(N))
        return [qml.expval(qml.PauliX(i)) for i in range(N)] \
               + [qml.expval(qml.PauliY(i)) for i in range(N)] \
               + [qml.expval(qml.PauliZ(i)) for i in range(N)]

    # build the gram matrix
    X_proj = [proj_feature_map(x) for x in X]

    return X_proj
