import numpy as np
from ..core import Kernel
from . import KernelEvaluator


class KernelAlignmentEvaluator(KernelEvaluator):
    """
    Kernel compatibility measure based on the kernel-target alignment
    See: Cristianini, Nello, et al. "On kernel-target alignment." Advances in neural information processing systems 14 (2001).
    """

    def evaluate(self, kernel: Kernel, K: np.ndarray, X: np.ndarray, y: np.ndarray):
        """
        Evaluate the current kernel and return the corresponding cost. Lower cost values corresponds to better solutions
        :param kernel: kernel object
        :param K: optional kernel matrix \kappa(X, X)
        :param X: datapoints
        :param y: labels
        :return: cost of the kernel, the lower the better
        """
        if K is None:
            K = kernel.build_kernel(X, X)
        the_cost = -1 * np.abs(KernelAlignmentEvaluator.kta(K, y))
        # assert not np.isnan(the_cost), f"{kernel=} {K=} {y=}"
        return the_cost if not np.isnan(the_cost) else 1000

    @staticmethod
    def kta(K, y):
        """
        Calculates the kernel target alignment
        :param K: kernel matrix
        :param y: label vector
        :return: kernel target alignment
        """
        Y = np.outer(y, y)
        return np.sum(K * Y) / (np.linalg.norm(K) * np.linalg.norm(Y))
