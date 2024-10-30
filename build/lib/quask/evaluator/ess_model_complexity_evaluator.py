import numpy as np
from scipy.linalg import sqrtm
from ..core import Kernel
from . import KernelEvaluator


def EssModelComplexityEvaluator(KernelEvaluator):
    """
    Calculate the model complexity s(K). 
    See Equation F1 in "The power of data in quantum machine learning" (https://arxiv.org/abs/2011.01938)
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

        return calculate_model_complexity(K, y)

    def calculate_model_complexity(k, y, normalization_lambda=0.001):
        """
        Calculate the model complexity s(K), which is equation F1 in
        "The power of data in quantum machine learning" (https://arxiv.org/abs/2011.01938).

        :param k: Kernel gram matrix
        :param y: Labels
        :param normalization_lambda: Normalization factor
        :return: model complexity of the given kernel
        """
        n = k.shape[0]
        k_inv = np.linalg.inv(k + normalization_lambda * np.eye(n))
        k_body = k_inv @ k @ k_inv
        model_complexity = y.T @ k_body @ y
        return model_complexity

    def calculate_model_complexity_training(k, y, normalization_lambda=0.001):
        """
        Subprocedure of the function 'calculate_model_complexity_generalized'.

        :param k: Kernel gram matrix
        :param y: Labels
        :param normalization_lambda: Normalization factor
        :return: model complexity of the given kernel
        """
        n = k.shape[0]
        k_inv = np.linalg.inv(k + normalization_lambda * np.eye(n))
        k_mid = k_inv @ k_inv # without k in the middle
        model_complexity = (normalization_lambda**2) * (y.T @ k_mid @ y)
        return model_complexity

    def calculate_model_complexity_generalized(k, y, normalization_lambda=0.001):
        """
        Calculate the model complexity s(K), which is equation M1 in
        "The power of data in quantum machine learning" (https://arxiv.org/abs/2011.01938).

        :param k: Kernel gram matrix
        :param y: Labels
        :param normalization_lambda: Normalization factor
        :return: model complexity of the given kernel
        """
        n = k.shape[0]
        a = np.sqrt(calculate_model_complexity_training(k, y, normalization_lambda) / n)
        b = np.sqrt(calculate_model_complexity(k, y, normalization_lambda) / n)
        return a + b
