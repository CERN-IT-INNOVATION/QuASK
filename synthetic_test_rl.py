import copy

import cirq
import numpy as np
np.random.seed(12345)
np.set_printoptions(precision=5, suppress=True)
np.set_printoptions(suppress=True)
from darqk.core import Ansatz, Kernel, KernelFactory, KernelType
from darqk.optimizer import ReinforcementLearningOptimizer, GreedyOptimizer, MetaheuristicOptimizer
from darqk.evaluator import RidgeGeneralizationEvaluator, CenteredKernelAlignmentEvaluator, KernelAlignmentEvaluator


N_FEATURES = 1  # allows constant feature too
N_OPERATIONS = 1
N_QUBITS = 2
N_SAMPLES = 4
ansatz = Ansatz(n_features=N_FEATURES, n_qubits=N_QUBITS, n_operations=N_OPERATIONS)
ansatz.initialize_to_identity()
ansatz.change_operation(0, new_feature=0, new_wires=[0, 1], new_generator="XI", new_bandwidth=0.5)
# ansatz.change_operation(1, new_feature=0, new_wires=[0, 1], new_generator="XZ", new_bandwidth=0.5)
real_kernel = KernelFactory.create_kernel(ansatz, "ZZ", KernelType.OBSERVABLE)


def quantum_process(x):
    return real_kernel.phi(x)


X = np.random.uniform(-np.pi, np.pi, size=(N_SAMPLES, N_FEATURES))
y = np.array([quantum_process(x) for x in X])
print(y)
ke = KernelAlignmentEvaluator()
print("R", ke.evaluate(real_kernel, None, X, y))

init_kernel = copy.copy(real_kernel)
init_kernel.ansatz.initialize_to_identity()
init_kernel.measurement = "Z" * len(init_kernel.measurement)
# rl_opt = ReinforcementLearningOptimizer(init_kernel, X, y, ke)
# rl_opt_kernel = rl_opt.optimize(initial_episodes=3, n_episodes=100, n_steps_per_fit=1, final_episodes=3)
# greedy_opt = GreedyOptimizer(copy.copy(init_kernel), X, y, ke)
# gr_opt_kernel = greedy_opt.optimize(verbose=True)
mh_opt = MetaheuristicOptimizer(init_kernel, X, y, ke)
mh_opt_kernel = mh_opt.optimize(1)
print("----------------------------------")
print("REAL", ke.evaluate(real_kernel, None, X, y))
# print("MH", ke.evaluate(mh_opt_kernel, None, X, y))
# print("RL", ke.evaluate(rl_opt_kernel, None, X, y))
# print("GR", ke.evaluate(gr_opt_kernel, None, X, y))


import sympy
from sympy.physics.quantum import TensorProduct

def H():
    return (1 / sympy.sqrt(2)) * sympy.Matrix([[1, 1], [1, -1]])

def RY(sym):
    return sympy.Matrix([[sympy.cos(sym/2), -1*sympy.sin(sym/2)], [sympy.sin(sym/2), sympy.cos(sym/2)]])

def CRY(sym):
    return sympy.Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, sympy.cos(sym / 2), -sympy.sin(sym / 2)],
        [0, 0, sympy.sin(sym / 2), sympy.cos(sym / 2)]
    ])


phi, theta = sympy.symbols('phi theta')
ket_0 = sympy.Matrix([[1],[0],[0],[0]])
ket_2 = TensorProduct(RY(phi), sympy.eye(2)).dot(ket_0)
ket_3 = CRY(theta).dot(ket_2)
ket_4 = TensorProduct(RY(-phi), sympy.eye(2)).dot(ket_2)



import pennylane as qml

dev = qml.device("default.qubit", wires=2, shots=None)

phi, theta = sympy.symbols('phi theta')

@qml.qnode(dev)
def linear_qnn():
    qml.RY(2 * phi, wires=[0])
    qml.CRY(2 * theta, wires=[0, 1])
    qml.RY(- 2 * phi, wires=[0])
    return qml.state()


def get_angle(phi, theta):
    res = linear_qnn(phi, theta)
    a0 = res[0] + res[2]
    a1 = res[1] + res[3]
    b0 = res[0] + res[1]
    b1 = res[2] + res[3]
    print([a0, a1, b0, b1])
    print(np.arcsin([a0, a1, b0, b1]))
    print(np.arccos([a0, a1, b0, b1]))


@qml.qnode(dev)
def guerreschi_qnn(phi, theta):
    qml.RY(2 * phi, wires=[0])
    qml.CRY(2 * theta, wires=[0, 1])
    qml.CY(wires=[1, 2])
    qml.CRY(-2 * theta, wires=[0, 1])
    _ = qml.measure(1)
    return qml.probs(wires=[2])


X = np.array([i/10 for i in range(10)])
y = np.array([guerreschi_qnn(x, 0.2) for x in X])
import matplotlib.pyplot as plt
plt.plot(X, y)
