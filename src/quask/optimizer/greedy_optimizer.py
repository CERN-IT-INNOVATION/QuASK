import copy

import numpy as np
from mushroom_rl.core import Environment

from ..core import Kernel
from ..evaluator import KernelEvaluator
from .base_kernel_optimizer import BaseKernelOptimizer
from .wide_kernel_environment import WideKernelEnvironment


class GreedyOptimizer(BaseKernelOptimizer):

    def __init__(self, initial_kernel: Kernel, X: np.ndarray, y: np.ndarray, ke: KernelEvaluator):
        super().__init__(initial_kernel, X, y, ke)
        self.mdp: WideKernelEnvironment = Environment.make('WideKernelEnvironment', initial_kernel=self.initial_kernel, X=X, y=y, ke=ke)
        self.rewards_history = []
        self.actions_history = []

    def optimize(self, verbose=False):

        self.mdp.reset()
        state = copy.deepcopy(self.mdp._state)

        terminated = False
        n_actions = self.mdp._mdp_info.action_space.size[0]
        rewards = np.zeros(shape=(n_actions,))

        while not terminated:
            # list all actions at the first depth
            for action in range(n_actions):
                self.mdp.reset(state)
                new_state, reward, absorbed, _ = self.mdp.step((action,))
                rewards[action] = reward
                _, kernel = self.mdp.deserialize_state(new_state)
                if absorbed:
                    terminated = True
                print(f"{action=:4d} {reward=:0.6f} {kernel=}")
            # apply chosen action
            chosen_action = np.argmax(rewards)
            self.mdp.reset(state)
            state, _, _, _ = self.mdp.step((chosen_action,))
            if verbose:
                print(f"Chosen action: {chosen_action}")
                print(f"{self.mdp.deserialize_state(state)=}")
            # additional information
            self.rewards_history.append(rewards)
            self.actions_history.append(chosen_action)

        _, kernel = self.mdp.deserialize_state(state)
        return kernel
