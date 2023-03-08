import numpy as np
from sklearn.kernel_ridge import KernelRidge
from ..core import Kernel
from . import KernelEvaluator


class RidgeGeneralizationEvaluator(KernelEvaluator):
    """
    Evaluates the generalization error of the given kernel
    """

    def __init__(self):
        """
        Initialization
        """
        pass

    def evaluate(self, kernel: Kernel, K: np.ndarray, X: np.ndarray, y: np.ndarray):
        """
        Evaluate the current kernel and return the corresponding cost. Lower cost values corresponds to better solutions
        :param kernel: kernel object
        :param K: optional kernel matrix \kappa(X, X)
        :param X: datapoints
        :param y: labels
        :return: cost of the kernel, the lower the better
        """
        krr = KernelRidge(kernel=lambda X1, X2: kernel.build_kernel(X1, X2))
        krr.fit(X[:len(y) // 2], y[:len(y) // 2])
        y_pred = np.array(krr.predict(X[len(y) // 2:]))
        mse = np.linalg.norm(y_pred - y[len(y) // 2:])
        return mse
