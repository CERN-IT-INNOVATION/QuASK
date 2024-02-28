from abc import ABC, abstractmethod
import numpy as np
import copy
from ..core import Operation, Ansatz, Kernel, KernelFactory
from ..evaluator import KernelEvaluator


class BaseKernelOptimizer(ABC):
    """
    Abstract class implementing a procedure to optimize the kernel
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
        self.X = X
        self.y = y
        self.ke = ke

    @abstractmethod
    def optimize(self):
        """
        Run the optimization
        :return: optimized kernel object
        """
        pass
