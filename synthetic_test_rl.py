import copy

import cirq
import numpy as np
np.random.seed(12345)
np.set_printoptions(precision=5, suppress=True)
np.set_printoptions(suppress=True)
from darqk.core import Ansatz, Kernel, KernelFactory, KernelType
from darqk.optimizer import ReinforcementLearningOptimizer, GreedyOptimizer, MetaheuristicOptimizer, BayesianOptimizer
from darqk.evaluator import RidgeGeneralizationEvaluator, CenteredKernelAlignmentEvaluator, KernelAlignmentEvaluator


N_FEATURES = 1  # allows constant feature too
N_OPERATIONS = 1
N_QUBITS = 2
N_SAMPLES = 4
ansatz = Ansatz(n_features=N_FEATURES, n_qubits=N_QUBITS, n_operations=N_OPERATIONS)
ansatz.initialize_to_identity()
ansatz.change_operation(0, new_feature=0, new_wires=[0, 1], new_generator="XI", new_bandwidth=0.5)
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
mh_opt = BayesianOptimizer(init_kernel, X, y, ke)
mh_opt_kernel = mh_opt.optimize(n_epochs=1, n_points=1, n_jobs=1)
print("----------------------------------")
print("REAL", ke.evaluate(real_kernel, None, X, y))
# print("MH", ke.evaluate(mh_opt_kernel, None, X, y))
# print("RL", ke.evaluate(rl_opt_kernel, None, X, y))
# print("GR", ke.evaluate(gr_opt_kernel, None, X, y))
