import numpy as np
np.random.seed(12345)
np.set_printoptions(precision=5, suppress=True)
np.set_printoptions(suppress=True)
from darqk.kernel_evaluator import *
from darqk.pennylane_kernel import AnsatzTemplate, PennylaneKernel
from darqk.ansatz import Ansatz
from darqk.kernel import KernelType
from darqk.kernel_optimizer import SklearnBayesianOptimizer


class MixKernelEvaluator(KernelEvaluation):

    def __init__(self):
        self.lie_evaluator = LieRankKernelEvaluation(16)
        self.kta_evaluator = KTAKernelEvaluation()
        self.ckta_evaluator = CenteredKTAKernelEvaluation()
        self.sbias_evaluator = SpectralBiasKernelEvaluation(1)
        self.queries = []

    def evaluate(self, kernel: Kernel, K: np.ndarray, X: np.ndarray, y: np.ndarray):
        if K is None:
            K = kernel.build_kernel(X, X)
        lie_cost = self.lie_evaluator.evaluate(kernel, K, X, y)
        kta_cost = self.kta_evaluator.evaluate(kernel, K, X, y)
        ckta_cost = self.ckta_evaluator.evaluate(kernel, K, X, y)
        sbias_cost, powers, w, a = self.sbias_evaluator.evaluate(kernel, K, X, y)
        # the_log = {'lie': len(lie_cost), 'kta': np.array(kta_cost), 'ckta': np.array(ckta_cost), 'sbias': np.array(powers), 'kernel': copy.deepcopy(kernel)}
        np.set_printoptions(precision=5, suppress=True)
        np.set_printoptions(suppress=True)
        divergence = 1 / np.linalg.norm((np.outer(y, y)) - (K))
        cdivergence = 1 / np.linalg.norm(CenteredKTAKernelEvaluation.center_kernel(np.outer(y, y)) - CenteredKTAKernelEvaluation.center_kernel(K))
        print(f"{divergence=: 0.3f} {cdivergence=: 0.3f} {kernel=}")
        return np.log(cdivergence)  # ckta must be maximized


N_FEATURES = 1
N_QUBITS = 10
N_SAMPLES = 64
ansatz = Ansatz(n_features=N_FEATURES, n_qubits=N_QUBITS, n_operations=1)
ansatz.initialize_to_identity()
ansatz.change_operation(0, new_feature=0, new_wires=[0, 1], new_generator="XX", new_bandwidth=1.0)
kernel = PennylaneKernel(ansatz, "Z" * N_QUBITS, KernelType.FIDELITY)
magic_key = np.random.uniform(0, 1.0, size=(N_FEATURES,))


def quantum_process(x):
    return kernel.kappa(x, magic_key)


X = np.random.uniform(0, np.pi, size=(N_SAMPLES, N_FEATURES))
y = np.array([quantum_process(x) for x in X])
print(f"{X=}")
print(f"{y=}")
ke = MixKernelEvaluator()
bo_opt = SklearnBayesianOptimizer(kernel, X, y, ke)
min_cost, min_solution = bo_opt.optimize(n_epochs=100, n_points=5)
k = bo_opt.configuration_to_kernel(min_solution)
print("----------------------------------")
print("W", ke.evaluate(kernel, None, X, y))
print("A", ke.evaluate(k, None, X, y))
for i in range(X.shape[0]):
    print(kernel.kappa(X[i], magic_key), k.kappa(X[i], magic_key))
