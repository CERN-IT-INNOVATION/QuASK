import numpy as np
import copy
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Discrete
from mushroom_rl.core import Core
from mushroom_rl.algorithms.value import SARSALambda
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.dataset import compute_J
from ..core import Operation, Ansatz, Kernel, KernelFactory
from ..evaluator import KernelEvaluator
from . import BaseKernelOptimizer


class KernelMushroomEnv(Environment):
    """
    Implementation of a Mushroom-RL Environment for our problem
    """

    def __init__(self, initial_kernel: Kernel, X: np.ndarray, y: np.ndarray, ke: KernelEvaluator):
        """
        Initialization
        :param initial_kernel: initial kernel object
        :param X: datapoints
        :param y: labels
        :param ke: kernel evaluator object
        """
        self.initial_kernel = initial_kernel
        self.n_operations = self.initial_kernel.ansatz.n_operations
        self.n_features = self.initial_kernel.ansatz.n_features
        self.n_qubits = self.initial_kernel.ansatz.n_qubits
        self.allow_midcircuit_measurement = self.initial_kernel.ansatz.allow_midcircuit_measurement
        self.X = X
        self.y = y
        self.ke = ke
        self.last_reward = None

        # Create the action space.
        action_space = Discrete(
            len(self.initial_kernel.get_allowed_operations())
            * self.n_qubits
            * (self.n_qubits - 1)
            * (self.n_features + 1)
        )

        # Create the observation space.
        observation_space = Discrete(
            len(self.initial_kernel.get_allowed_operations())
            * self.n_qubits
            * (self.n_qubits - 1)
            * (self.n_features + 1)
            * self.n_operations
        )

        # Create the MDPInfo structure, needed by the environment interface
        mdp_info = MDPInfo(observation_space, action_space, gamma=0.99, horizon=100)
        super().__init__(mdp_info)

        # Create a state class variable to store the current state
        self._state = self.serialize_state(0, initial_kernel)

        # Create the viewer
        self._viewer = None

    def serialize_state(self, n_operation, kernel):
        """
        Pack the state of the optimization technique
        :param n_operation: number of operations currently performed
        :param kernel: kernel object
        :return: serialized state
        """
        state = np.concatenate([np.array([n_operation], dtype=int), kernel.to_numpy()], dtype=object).ravel()
        return state.astype(int)

    def deserialize_state(self, array):
        """
        Deserialized a previously packed state variable
        :param array: serialized state
        :return: tuple n_operations, kernel object
        """
        kernel = Kernel.from_numpy(array[1:], self.n_features, self.n_qubits, self.n_operations, self.allow_midcircuit_measurement)
        n_operations = int(array[0])
        return n_operations, kernel

    def render(self):
        """
        Rendering function - we don't need that
        :return: None
        """
        n_op, kernel = self.deserialize_state(self._state)
        print(f"{self.last_reward=:2.4f} {n_op=:2d} {kernel=}")

    def reset(self, state=None):
        """
        Reset the state
        :param state: optional state
        :return: self._state variable
        """
        if state is None:
            self.initial_kernel.ansatz.initialize_to_identity()
            self._state = self.serialize_state(0, self.initial_kernel)
        else:
            self._state = state
        return self._state

    def unpack_action(self, action):
        """
        Unpack an action to a operation
        :param action: integer representing the action
        :return: dictionary of the operation
        """
        generator_index = int(action % len(self.initial_kernel.get_allowed_operations()))
        action = action // len(self.initial_kernel.get_allowed_operations())

        wires_0 = int(action % self.n_qubits)
        action = action // self.n_qubits

        wires_1 = int(action % (self.n_qubits - 1))
        if wires_1 >= wires_0:
            wires_1 += 1
        action = action // (self.n_qubits - 1)

        feature = int(action % (self.n_features + 1)) - 1 # allows -1 as feature index for constant feature
        action = action // (self.n_features + 1)
        assert action == 0

        return {'generator': self.initial_kernel.get_allowed_operations()[generator_index],
                'wires': [wires_0, wires_1],
                'feature': feature,
                'bandwidth': 1.0}

    def step(self, action):

        the_action = self.unpack_action(action[0])

        # Create kernel from state
        n_operations, kernel = self.deserialize_state(self._state)

        # Update kernel
        kernel.ansatz.change_operation(n_operations, the_action['feature'], the_action['wires'], the_action['generator'], the_action['bandwidth'])
        n_operations += 1

        # Update state
        self._state = self.serialize_state(n_operations, kernel)

        # Compute the reward as distance penalty from goal
        reward = -1 * self.ke.evaluate(kernel, None, self.X, self.y)
        self.last_reward = reward

        # Set the absorbing flag if goal is reached
        absorbing = self.n_operations == n_operations

        # Return all the information + empty dictionary (used to pass additional information)
        return self._state, reward, absorbing, {}


class ReinforcementLearningOptimizer(BaseKernelOptimizer):
    """
    Reinforcement learning based technique for optimize a kernel function
    """

    def __init__(self, initial_kernel: Kernel, X: np.ndarray, y: np.ndarray, ke: KernelEvaluator):
        """
        Initialization
        :param initial_kernel: initial kernel object
        :param X: datapoints
        :param y: labels
        :param ke: kernel evaluator object
        """
        self.initial_kernel = copy.deepcopy(initial_kernel)
        self.X = X
        self.y = y
        self.ke = ke
        self.mdp = Environment.make('KernelMushroomEnv', initial_kernel=self.initial_kernel, X=X, y=y, ke=ke)
        self.agent = None
        self.core = None

    def optimize(self, initial_episodes=3, n_episodes=100, n_steps_per_fit=1, final_episodes=3):
        """
        Optimization routine
        :param initial_episodes:
        :param n_steps:
        :param n_steps_per_fit:
        :param final_episodes:
        :return:
        """

        # Policy
        epsilon = Parameter(value=1.)
        pi = EpsGreedy(epsilon=epsilon)
        learning_rate = Parameter(.1)

        # Agent
        self.agent = SARSALambda(self.mdp.info, pi,
                            learning_rate=learning_rate,
                            lambda_coeff=.9)

        # Reinforcement learning experiment
        self.core = Core(self.agent, self.mdp)

        # Visualize initial policy for 3 episodes
        dataset = self.core.evaluate(n_episodes=initial_episodes, render=True)
        print(f"{dataset=}")

        # Print the average objective value before learning
        J = np.mean(compute_J(dataset, self.mdp.info.gamma))
        print(f'Objective function before learning: {J}')

        # Train
        self.core.learn(n_episodes=n_episodes, n_steps_per_fit=n_steps_per_fit, render=True)

        # Visualize results for 3 episodes
        dataset = self.core.evaluate(n_episodes=final_episodes, render=True)

        # Print the average objective value after learning
        J = np.mean(compute_J(dataset, self.mdp.info.gamma))
        print(f'Objective function after learning: {J}')

        kernel = Kernel.from_numpy(self.mdp._state[1:], self.mdp.n_features, self.mdp.n_qubits, self.mdp.n_operations, self.mdp.allow_midcircuit_measurement)
        return kernel


KernelMushroomEnv.register()
