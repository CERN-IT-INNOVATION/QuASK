import numpy as np
from scipy.linalg import sqrtm
from ..core import Kernel
from . import KernelEvaluator


class GeometricDifferenceEvaluator(KernelEvaluator):
    """
    Calculate the geometric difference g(K_1 || K_2), and characterize 
    the separation between classical and quantum kernels.    
    See Equation F9 in "The power of data in quantum machine learning" (https://arxiv.org/abs/2011.01938)
    """

    def __init__(self, list_classical_kernel_matrices, lam):
        """
        Initialization. 

        :param list_classical_kernel_matrices: List of kernel matrices obtained with classical kernels
        :param lam: normalization constant lambda
        """
        super().__init__()
        self.list_classical_kernel_matrices = list_classical_kernel_matrices
        self.lam = lam


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

        geometric_differences = [GeometricDifferenceEvaluator.g(K, Kc, self.lam) 
                                 for Kc in self.list_classical_kernel_matrices]

        return -1 * np.min(geometric_differences)

    @staticmethod
    def g(k_1, k_2, lam):
        """
        Method to calculate the geometric difference

        :param k_1: first matrix (quantum usually)
        :param k_2: second matrix (classical usually)
        :param lam: normalization lambda
        :return: value of g(K_1, K_2)
        """
        n = k_2.shape[0]
        assert k_2.shape == (n, n)
        assert k_1.shape == (n, n)
        # √K1
        k_1_sqrt = np.real(sqrtm(k_1))
        # √K2
        k_2_sqrt = np.real(sqrtm(k_2))
        # √(K2 + lambda I)^-2
        kc_inv = np.linalg.inv(k_2 + lam * np.eye(n))
        kc_inv = kc_inv @ kc_inv
        # Equation F9
        f9_body = k_1_sqrt.dot(k_2_sqrt.dot(kc_inv.dot(k_2_sqrt.dot(k_1_sqrt))))
        f9 = np.sqrt(np.linalg.norm(f9_body, np.inf))
        return f9
