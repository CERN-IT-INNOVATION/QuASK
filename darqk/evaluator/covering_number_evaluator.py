import numpy as np
from ..core import Kernel
from . import KernelEvaluator


class CoveringNumberEvaluator(KernelEvaluator):
    """
    Expressibility measure based on the covering number associated with the hypothesis class related to the current ansatz.
    See: Du, Yuxuan, et al. "Efficient measure for the expressivity of variational quantum algorithms." Physical Review Letters
    128.8 (2022): 080506.
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
        operations = kernel.ansatz.operation_list
        trainable_operations = [op for op in operations if op.feature >= 0]
        return 2 ** trainable_operations
