from abc import ABC, abstractmethod
import numpy as np
from joblib import Parallel, delayed
import copy
from typing import Tuple, Dict
from skopt import Optimizer
from skopt.space import Real, Categorical
from saasbo.saasbo import run_saasbo
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Box, Discrete
from mushroom_rl.utils.viewer import Viewer
from mushroom_rl.core import Core
from mushroom_rl.algorithms.value import TrueOnlineSARSALambda, SARSALambda
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.dataset import compute_J
from sklearn.ensemble import ExtraTreesRegressor
from mushroom_rl.algorithms.value import FQI
from .core import Ansatz, Kernel, Operation, KernelFactory
from .evaluator import KernelEvaluator


class BaseKernelOptimizer(ABC):

    def __init__(self, initial_kernel: Kernel, X: np.ndarray, y: np.ndarray, ke: KernelEvaluator):
        self.initial_kernel = initial_kernel
        self.X = X
        self.y = y
        self.ke = ke

    def configuration_to_ansatz(self, ansatz_configuration):
        n_operations = self.initial_kernel.ansatz.n_operations
        assert len(ansatz_configuration) == 5 * n_operations, \
            f"ansatz_configuration vector is not of the correct length (expected: {5 * n_operations}, actual: {len(ansatz_configuration)})"
        ansatz = copy.deepcopy(self.initial_kernel.ansatz)
        for i in range(len(ansatz_configuration) // 5):
            wires = ansatz_configuration[5*i+1:5*i+3]
            if wires[1] >= wires[0]:
                wires[1] += 1
            ansatz.change_generators(i, Operation.OPERATIONS[ansatz_configuration[i * 5]])
            ansatz.change_wires(i, wires)
            ansatz.change_feature(i, ansatz_configuration[5*i+3])
            ansatz.change_bandwidth(i, ansatz_configuration[5*i+4])
        return ansatz

    def configuration_to_measurement(self, measurement_configuration):
        n_qubits = self.initial_kernel.ansatz.n_qubits
        assert len(measurement_configuration) == n_qubits, \
            f"measurement_configuration vector is not of the correct length (expected: {n_qubits}, actual: {len(measurement_configuration)})"
        assert 0 <= min(measurement_configuration) <= max(measurement_configuration) <= 3, "measurement_configuration has invalid element"
        return "".join([['X', 'Y', 'Z', 'I'][i] for i in measurement_configuration])

    def configuration_to_kernel(self, configuration):
        n_operations = self.initial_kernel.ansatz.n_operations
        n_qubits = self.initial_kernel.ansatz.n_qubits
        assert len(configuration) == 5 * n_operations + n_qubits, \
            f"Configuration vector is not of the correct length (expected: {5 * n_operations + n_qubits}, actual {len(configuration)})"
        ansatz = self.configuration_to_ansatz(configuration[:5 * n_operations])
        measurement = self.configuration_to_measurement(configuration[-n_qubits:])
        kernel = KernelFactory.create_kernel(ansatz, measurement, self.initial_kernel.type)
        return kernel

    def cost(self, configuration):
        kernel = self.configuration_to_kernel(configuration)
        the_cost = self.ke.evaluate(kernel, None, self.X, self.y)
        return the_cost

    @abstractmethod
    def optimize(self):
        pass


class BaseBandwidthKernelOptimizer(ABC):

    def __init__(self, initial_kernel: Kernel, X: np.ndarray, y: np.ndarray, ke: KernelEvaluator):
        self.initial_kernel = initial_kernel
        self.X = X
        self.y = y
        self.ke = ke


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
        return min_cost, min_solution


class ReinforcementLearningOptimizer(BaseKernelOptimizer):

    class KernelEnv(Environment):

        def __init__(self, initial_kernel: Kernel, X: np.ndarray, y: np.ndarray, ke: KernelEvaluator):
            # Save important environment information
            self.initial_kernel = initial_kernel
            self.n_operations = self.initial_kernel.ansatz.n_operations
            self.n_features = self.initial_kernel.ansatz.n_features
            self.n_qubits = self.initial_kernel.ansatz.n_qubits
            self.allow_midcircuit_measurement = self.initial_kernel.ansatz.allow_midcircuit_measurement
            self.X = X
            self.y = y
            self.ke = ke

            # Create the action space.
            action_space = Discrete(
                len(self.initial_kernel.get_allowed_operations())
                * self.initial_kernel.ansatz.n_qubits
                * (self.initial_kernel.ansatz.n_qubits - 1)
                * self.initial_kernel.ansatz.n_features
            )

            # Create the observation space.
            observation_space = Discrete(
                len(self.initial_kernel.get_allowed_operations())
                * self.initial_kernel.ansatz.n_qubits
                * (self.initial_kernel.ansatz.n_qubits - 1)
                * self.initial_kernel.ansatz.n_features
                * self.initial_kernel.ansatz.n_operations
            )

            # Create the MDPInfo structure, needed by the environment interface
            mdp_info = MDPInfo(observation_space, action_space, gamma=0.99, horizon=max(2, self.initial_kernel.ansatz.n_operations // 2))
            super().__init__(mdp_info)

            # Create a state class variable to store the current state
            self._state = np.concatenate([np.array([0]), initial_kernel.to_numpy()]).ravel()

            # Create the viewer
            self._viewer = None

        def render(self):
            pass

        def reset(self, state=None):
            if state is None:
                self.initial_kernel.ansatz.initialize_to_identity()
                self._state = np.concatenate([np.array([0]), self.initial_kernel.to_numpy()]).ravel()
            else:
                self._state = state
            return self._state

        def step(self, action):
            # unpack action
            action = action[0]

            # convert the action in feature, wires, generator, bandwidth
            feature, wires, generator, bandwidth = None, [None, None], None, 1.0
            generator = action % len(self.initial_kernel.get_allowed_operations())
            generator = self.initial_kernel.get_allowed_operations()[generator]
            action //= len(self.initial_kernel.get_allowed_operations())
            wires[0] = action % self.initial_kernel.ansatz.n_qubits
            action //= self.initial_kernel.ansatz.n_qubits
            wires[1] = action % (self.initial_kernel.ansatz.n_qubits - 1)
            if wires[1] == wires[0]: wires[1] += 1
            action //= self.initial_kernel.ansatz.n_qubits - 1
            feature = action % self.initial_kernel.ansatz.n_features
            action //= self.initial_kernel.ansatz.n_features
            assert action == 0

            # Create kernel from state
            kernel = Kernel.from_numpy(self._state[1:], self.n_features, self.n_qubits, self.n_operations, self.allow_midcircuit_measurement)
            n_operations = int(self._state[0])

            # Update kernel
            kernel.ansatz.change_operation(n_operations, feature, wires, generator, bandwidth)  # update the operation
            n_operations += 1

            # Update state
            self._state = np.concatenate([np.array([n_operations]), kernel.to_numpy()]).ravel()

            # Compute the reward as distance penalty from goal
            reward = self.ke.evaluate(kernel, None, self.X, self.y)

            # Set the absorbing flag if goal is reached
            absorbing = n_operations == self.n_operations

            # Return all the information + empty dictionary (used to pass additional information)
            return self._state, reward, absorbing, {}

    def __init__(self, initial_kernel: Kernel, X: np.ndarray, y: np.ndarray, ke: KernelEvaluator):
        self.initial_kernel = copy.deepcopy(initial_kernel)
        self.X = X
        self.y = y
        self.ke = ke
        self.mdp = Environment.make('KernelEnv', initial_kernel=self.initial_kernel, X=X, y=y, ke=ke)

    def optimize(self, initial_episodes=3, n_steps=100, n_steps_per_fit=1, final_episodes=3):
        # Policy
        epsilon = Parameter(value=1.)
        pi = EpsGreedy(epsilon=epsilon)
        learning_rate = Parameter(.001)

        # Agent
        agent = SARSALambda(self.mdp.info, pi,
                            learning_rate=learning_rate,
                            lambda_coeff=.9)

        # Reinforcement learning experiment
        core = Core(agent, self.mdp)

        # Visualize initial policy for 3 episodes
        dataset = core.evaluate(n_episodes=initial_episodes, render=True)

        # Print the average objective value before learning
        J = np.mean(compute_J(dataset, self.mdp.info.gamma))
        print(f'Objective function before learning: {J}')

        # Train
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, render=True)

        # Visualize results for 3 episodes
        dataset = core.evaluate(n_episodes=final_episodes, render=True)

        # Print the average objective value after learning
        J = np.mean(compute_J(dataset, self.mdp.info.gamma))
        print(f'Objective function after learning: {J}')

        kernel = Kernel.from_numpy(self.mdp._state[1:], self.mdp.n_features, self.mdp.n_qubits, self.mdp.n_operations, self.mdp.allow_midcircuit_measurement)
        return kernel


ReinforcementLearningOptimizer.KernelEnv.register()
