import numpy as np
np.random.seed(12345)
np.set_printoptions(precision=5, suppress=True)
np.set_printoptions(suppress=True)
from darqk.kernel_evaluator import *
from darqk.pennylane_kernel import AnsatzTemplate, PennylaneKernel
from darqk import Ansatz, KernelType
from darqk.kernel_optimizer import SklearnBayesianOptimizer, ReinforcementLearningOptimizer



N_FEATURES = 1
N_QUBITS = 5
N_SAMPLES = 30

ansatz = Ansatz(n_features=N_FEATURES, n_qubits=N_QUBITS, n_operations=1)
ansatz.initialize_to_identity()
ansatz.change_operation(0, new_feature=0, new_wires=[3, 4], new_generator="XX", new_bandwidth=1.0)
real_kernel = PennylaneKernel(ansatz, "Z" * N_QUBITS, KernelType.FIDELITY)
magic_key = np.random.uniform(0, 1.0, size=(N_FEATURES,))


def quantum_process(x):
    return real_kernel.kappa(x, magic_key)


X = np.random.uniform(0, np.pi, size=(N_SAMPLES, N_FEATURES))
y = np.array([quantum_process(x) for x in X])
print(f"{X=}")
print(f"{y=}")
ke = MixKernelEvaluator()
rl_opt = ReinforcementLearningOptimizer(real_kernel, X, y, ke)
opt_kernel = rl_opt.optimize(n_steps=10000)
print("----------------------------------")
print("R", ke.evaluate(real_kernel, None, X, y))
print("O", ke.evaluate(opt_kernel, None, X, y))
