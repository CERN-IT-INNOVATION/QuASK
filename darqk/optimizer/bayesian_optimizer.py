import numpy as np
from joblib import Parallel, delayed
from skopt import Optimizer
from skopt.space import Real, Categorical

from ..core import Operation, Ansatz, Kernel, KernelFactory
from ..evaluator import KernelEvaluator
from . import BaseKernelOptimizer


class SklearnBayesianOptimizer(BaseKernelOptimizer):

    def __init__(self, initial_kernel: Kernel, X: np.ndarray, y: np.ndarray, ke: KernelEvaluator):
        super().__init__(initial_kernel, X, y, ke)
        self.optimizer = None

    def get_sklearn_dimensions(self):
        n_features = self.initial_kernel.ansatz.n_features
        n_operations = self.initial_kernel.ansatz.n_operations
        n_qubits = self.initial_kernel.ansatz.n_qubits
        allowed_generators = self.initial_kernel.get_allowed_operations()
        ansatz_dimension = [
            # generator
            Categorical(list(range(len(allowed_generators)))),
            # wires
            Categorical(list(range(n_qubits))),
            Categorical(list(range(n_qubits - 1))),
            # features
            Categorical(list(range(n_features))),
            # bandwidth
            Real(0.0, 1.0),
        ] * n_operations
        measurement_dimensions = [Categorical([0, 1, 2, 3])] * n_qubits
        return ansatz_dimension + measurement_dimensions

    def optimize(self, n_epochs=20, n_points=4, n_jobs=4):
        assert False, "Be careful at the return value, it must be a kernel"

        self.optimizer = Optimizer(
            dimensions=self.get_sklearn_dimensions(),
            random_state=1,
            base_estimator='gp',
            acq_func="PI",
            acq_optimizer="sampling",
            acq_func_kwargs={"xi": 10000.0, "kappa": 10000.0}
        )

        for i in range(n_epochs):
            x = self.optimizer.ask(n_points=n_points)  # x is a list of n_points points
            y = Parallel(n_jobs=n_jobs)(delayed(lambda cfg: self.cost(cfg))(v) for v in x)  # evaluate points in parallel
            self.optimizer.tell(x, y)
            print(f"Epoch of training {i=}")

        min_index = np.argmin(self.optimizer.yi)
        min_cost = self.optimizer.yi[min_index]
        min_solution = self.optimizer.Xi[min_index]

        # TODO ERROR must return the new kernel, not this
        return min_cost, min_solution
