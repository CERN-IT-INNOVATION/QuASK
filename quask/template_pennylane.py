"""
Module dedicated to define Templates for Pennylane quantum circuits.
See https://pennylane.readthedocs.io/en/stable/introduction/templates.html for details.
"""


import jax
import jax.numpy as jnp
import pennylane as qml
import numpy as np
import optax
from .metrics import (
    calculate_kernel_target_alignment,
    calculate_generalization_accuracy,
    calculate_geometric_difference,
    calculate_model_complexity,
)


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


class PennylaneTrainableKernel:
    """
    Create a trainable kernel using Pennylane framework.
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        embedding,
        var_form,
        layers,
        optimizer,
        metric,
        seed,
        keep_intermediate=True,
    ):
        """
        Init method.

        Args:
            X_train: training set feature vector
            y_train: training set label vector
            X_test: testing set feature vector
            y_test: testing set label vector
            embedding: one of the following list: "rx", "ry", "rz", "zz"
            var_form: one of the following list: "hardware_efficient", "tfim", "ltfim", "zz_rx"
            layers: number of ansatz repetition
            optimizer: one of the following list: "adam", "grid"
            metric: one of the following list: "kernel-target-alignment", "accuracy", "geometric-difference", "model-complexity"
            seed: random seed (int)
            keep_intermediate: True if you want to keep the intermediate results of the optimization (bool)

        Returns:
            None
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        assert embedding in ["rx", "ry", "rz", "zz"]
        self.embedding = embedding
        assert var_form in ["hardware_efficient", "tfim", "ltfim", "zz_rx"]
        self.var_form = var_form
        assert 1 <= layers < 1000
        self.layers = layers
        assert optimizer in ["adam", "grid"]
        self.optimizer = optimizer
        assert metric in [
            "kernel-target-alignment",
            "accuracy",
            "geometric-difference",
            "model-complexity",
        ]
        self.metric = metric
        self.seed = seed
        self.circuit = None
        self.params = None
        self.create_circuit()
        self.intermediate_params = []
        self.intermediate_grams = []
        self.keep_intermediate = keep_intermediate

    @staticmethod
    def jnp_to_np(value):
        """
        Convert jax numpy value to numpy

        Args:
            value: jax value

        Returns:
            numpy value
        """
        try:
            value_numpy = np.array(value.primal)
            return value_numpy
        except:
            pass
        try:
            value_numpy = np.array(value.primal.aval)
            return value_numpy
        except:
            pass
        try:
            value_numpy = np.array(value)
            return value_numpy
        except:
            raise ValueError(f"Cannot convert to numpy value {value}")

    def get_embedding(self):
        """
        Convert the embedding into its function pointer

        Returns:
            None
        """
        if self.embedding == "rx":
            return rx_embedding
        elif self.embedding == "ry":
            return ry_embedding
        elif self.embedding == "rz":
            return rz_embedding
        elif self.embedding == "zz":
            return zz_fullentanglement_embedding
        else:
            raise ValueError(f"Unknown embedding {self.embedding}")

    def get_var_form(self, n_qubits):
        """
        Convert the variational form into its function pointer

        Args:
            n_qubits: Number of qubits of the variational form

        Returns:
            (fn, n) tuple of function and integer, the former representing the ansatz and the latter the number of parameters
        """
        if self.var_form == "hardware_efficient":
            return hardware_efficient_ansatz, 2 * n_qubits
        elif self.var_form == "tfim":
            return tfim_ansatz, 2
        elif self.var_form == "ltfim":
            return ltfim_ansatz, 3
        elif self.var_form == "zz_rx":
            return zz_rx_ansatz, 2
        else:
            raise ValueError(f"Unknown var_form {self.var_form}")

    def create_circuit(self):
        """
        Creates the quantum circuit to be simulated with jax.

        Returns:
            None
        """
        N = self.X_train.shape[1]
        device = qml.device("default.qubit.jax", wires=N)
        embedding_fn = self.get_embedding()
        var_form_fn, params_per_layer = self.get_var_form(N)

        @jax.jit
        @qml.qnode(device, interface="jax")
        def circuit(x, theta):
            embedding_fn(x, wires=range(N))
            for i in range(self.layers):
                var_form_fn(
                    theta[i * params_per_layer : (i + 1) * params_per_layer],
                    wires=range(N),
                )
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(N)]

        self.circuit = circuit
        self.params = jax.random.normal(
            jax.random.PRNGKey(self.seed), shape=(self.layers * params_per_layer,)
        )

    def get_gram_matrix(self, X_1, X_2, theta):
        """
        Get the gram matrix given the actual parameters.

        Args:
            X_1: first set (testing)
            X_2: second set (training)
            theta: parameters

        Returns:
            Gram matrix
        """
        X_proj_1 = jnp.array([self.circuit(x, theta) for x in X_1])
        X_proj_2 = jnp.array([self.circuit(x, theta) for x in X_2])
        gamma = 1.0

        gram = np.zeros(shape=(X_1.shape[0], X_2.shape[0]))
        for i in range(X_proj_1.shape[0]):
            for j in range(X_proj_2.shape[0]):
                value = jnp.exp(-gamma * ((X_proj_1[i] - X_proj_2[j]) ** 2).sum())
                gram[i][j] = PennylaneTrainableKernel.jnp_to_np(value)
        return gram

    def get_loss(self, theta):
        """
        Get loss according to the wanted metric.

        Args:
            theta: parameter vector

        Returns:
            loss (float)
        """
        theta_numpy = PennylaneTrainableKernel.jnp_to_np(theta)
        training_gram = self.get_gram_matrix(self.X_train, self.X_train, theta)
        if self.keep_intermediate:
            self.intermediate_params.append(theta_numpy)
            self.intermediate_grams.append(training_gram)
        if self.metric == "kernel-target-alignment":
            return 1 / calculate_kernel_target_alignment(training_gram, self.y_train)
        elif self.metric == "accuracy":
            return 1 / calculate_generalization_accuracy(
                training_gram, self.y_train, training_gram, self.y_train
            )
        elif self.metric == "geometric-difference":
            comparison_gram = np.outer(self.X_train, self.X_train)
            return 1 / calculate_geometric_difference(training_gram, comparison_gram)
        elif self.metric == "model-complexity":
            return 1 / calculate_model_complexity(training_gram, self.y_train)
        else:
            raise ValueError(f"Unknown metric {self.metric} for loss function")

    def get_optimizer(self):
        """
        Convert the optimizer from string to object

        Returns:
            optimizer object
        """
        if self.optimizer == "adam":
            return optax.adam(learning_rate=0.1)
        elif self.optimizer == "grid":
            return "grid"
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")

    def optimize_circuit(self):
        """
        Run optimization of the circuit

        Returns:
            None
        """
        optimizer = self.get_optimizer()
        if optimizer == "grid":
            raise ValueError("Not implemented yet")
        else:
            opt_state = optimizer.init(self.params)
            epochs = 2
            for epoch in range(epochs):
                cost, grad_circuit = jax.value_and_grad(
                    lambda theta: self.get_loss(theta)
                )(self.params)
                updates, opt_state = optimizer.update(grad_circuit, opt_state)
                self.params = optax.apply_updates(self.params, updates)
                print(".", end="", flush=True)

    def get_optimized_gram_matrices(self):
        """
        Get optimized gram matrices

        Returns:
            (tr,te) tuple of training and testing gram matrices
        """
        training_gram = self.get_gram_matrix(self.X_train, self.X_train, self.params)
        testing_gram = self.get_gram_matrix(self.X_test, self.X_train, self.params)
        return training_gram, testing_gram
