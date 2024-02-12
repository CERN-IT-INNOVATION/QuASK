import copy
import numpy as np
from enum import Enum

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import GridSpace
from opytimizer.utils.callback import Callback
from ..core import Kernel
from ..evaluator import KernelEvaluator
from .base_kernel_optimizer import BaseKernelOptimizer


class CustomCallback(Callback):
    """A CustomCallback can be created by override its parent `Callback` class
    and by implementing the desired logic in its available methods.
    """

    def __init__(self):
        """Initialization method for the customized callback."""

        # You only need to override its parent class
        super(CustomCallback).__init__()

    def on_task_begin(self, opt_model):
        """Called at the beginning of an task."""
        print("Task begin")

    def on_task_end(self, opt_model):
        """Called at the end of an task."""
        print("Task end")

    def on_iteration_begin(self, iteration, opt_model):
        """Called at the beginning of an iteration."""
        print(f"Iteration {iteration} begin")

    def on_iteration_end(self, iteration, opt_model):
        """Called at the end of an iteration."""
        print(f"Iteration {iteration} end")

    def on_evaluate_before(self, *evaluate_args):
        """Called before the `evaluate` method."""
        print(f"Evaluate before {evaluate_args}")

    def on_evaluate_after(self, *evaluate_args):
        """Called after the `evaluate` method."""
        print(f"Evaluate after {evaluate_args}")

    def on_update_before(self, *update_args):
        """Called before the `update` method."""
        print(f"Update before {update_args}")

    def on_update_after(self, *update_args):
        """Called after the `update` method."""
        print(f"Update after {update_args}")


class MetaheuristicType(Enum):

    # evolutionary
    FOREST_OPTIMIZATION = 1
    GENETIC_ALGORITHM = 2
    # population
    EMPEROR_PENGUIN_OPTIMIZER = 3
    # swarm
    PARTICLE_SWARM_OPTIMIZATION = 0


class MetaheuristicOptimizer(BaseKernelOptimizer):

    def __init__(self, initial_kernel: Kernel, X: np.ndarray, y: np.ndarray, ke: KernelEvaluator):
        super().__init__(initial_kernel, X, y, ke)

        def cost(array):
            array = array.ravel()
            the_array = np.concatenate([array, np.array([initial_kernel.type])])
            # create kernel
            the_kernel = Kernel.from_numpy(the_array,
                                           initial_kernel.ansatz.n_features,
                                           initial_kernel.ansatz.n_qubits,
                                           initial_kernel.ansatz.n_operations,
                                           initial_kernel.ansatz.allow_midcircuit_measurement,
                                           shift_second_wire=True)
            the_cost = ke.evaluate(the_kernel, None, X, y)
            print(f"MetaheuristicOptimizer.cost -> {the_cost: 5.5f} -> {array}")
            return the_cost

        self.space = self.get_opytimize_space()
        self.optimizer = PSO()
        self.cost = cost
        self.function = Function(cost)
        self.opt = Opytimizer(self.space, self.optimizer, self.function, save_agents=True)
        self.history = None
        self.best_solution = None
        self.best_cost = None

    def optimize(self, n_iterations=1000, verbose=False):
        self.opt.start(n_iterations=n_iterations, callbacks=[CustomCallback()] if verbose else [])
        self.history = self.opt.history
        data_at_convergence = self.history.get_convergence("best_agent")
        self.best_solution = data_at_convergence[0].ravel()
        self.best_cost = data_at_convergence[1].ravel()
        the_array = np.concatenate([self.best_solution, np.array([self.initial_kernel.type])])
        return Kernel.from_numpy(the_array,
                                 self.initial_kernel.ansatz.n_features,
                                 self.initial_kernel.ansatz.n_qubits,
                                 self.initial_kernel.ansatz.n_operations,
                                 self.initial_kernel.ansatz.allow_midcircuit_measurement,
                                 shift_second_wire=True)

    def get_opytimize_space(self):
        n_features = self.initial_kernel.ansatz.n_features
        n_operations = self.initial_kernel.ansatz.n_operations
        n_qubits = self.initial_kernel.ansatz.n_qubits
        allowed_generators = self.initial_kernel.get_allowed_operations()

        n_variables = 5 * n_operations + n_qubits
        step = [1, 1, 1, 1, 0.2] * n_operations + [1] * n_qubits
        lower_bound = [0, 0, 0, 0, 0.2] * n_operations + [0] * n_qubits
        upper_bound = [len(allowed_generators) - 1, n_qubits - 1, n_qubits - 2, n_features, 1.0] * n_operations + [3] * n_qubits
        return GridSpace(n_variables, step, lower_bound, upper_bound)
