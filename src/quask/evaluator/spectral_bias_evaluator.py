import numpy as np
from scipy.linalg import eigh
from ..core import Kernel
from . import KernelEvaluator


class SpectralBiasKernelEvaluator(KernelEvaluator):
    """
    Kernel compatibility measure based on the spectral bias framework.
    See: Canatar, Abdulkadir, Blake Bordelon, and Cengiz Pehlevan. "Spectral bias and task-model alignment explain generalization
    in kernel regression and infinitely wide neural networks." Nature communications 12.1 (2021): 2914.
    """

    def __init__(self, n_eigenvalues_cut):
        """
        Initialization
        :param n_eigenvalues_cut: number of eigenvalues contributing to the cumulative power
        """
        self.n_eigenvalues_cut = n_eigenvalues_cut

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
        Lambda, Phi = SpectralBiasKernelEvaluator.decompose_kernel(K)
        w, a = SpectralBiasKernelEvaluator.calculate_weights(Lambda, Phi, y)
        C, powers = SpectralBiasKernelEvaluator.cumulative_power_distribution(w, Lambda, self.n_eigenvalues_cut)
        self.last_result = (Lambda, Phi, w, a, C, powers)
        return C


    @staticmethod
    def decompose_kernel(K, eigenvalue_descending_order=True, eigenvalue_removal_threshold=1e-12):
        """
        Decompose the kernel matrix K in its eigenvalues Λ and eigenvectors Φ
        :param K: kernel matrix, real and symmetric
        :param eigenvalue_descending_order: if True, the biggest eigenvalue is the first one
        :return: Lambda vector (n elements) and Phi matrix (N*N matrix)
        """
        Lambda, Phi = eigh(K)

        # set the desired order for the eigenvalues
        if eigenvalue_descending_order:
            Lambda = Lambda[::-1]
            Phi = Phi[:, ::-1]

        # kernel matrix is positive definite, any (small) negative eigenvalue is effectively a numerical error
        Lambda[Lambda < 0] = 0

        # remove the smallest positive eigenvalues, as they are useless
        Lambda[Lambda < eigenvalue_removal_threshold] = 0

        return Lambda, Phi

    @staticmethod
    def calculate_weights(Lambda, Phi, labels):
        """
        Calculates the weights of a predictor given the labels and the kernel eigendecomposition,
        as shown in (Canatar et al 2021, inline formula below equation 18).
        :param Lambda: vectors of m nonnegative eigenvalues 'eta'
        :param Phi: vectors of m nonnegative eigenvectors 'phi'
        :param labels: vector of m labels corresponding to 'm' ground truth labels or predictor outputs
        :return: vector w of RKHS weights, vector a of out-of-RKHS weights
        """
        # get the number of training elements
        m = Lambda.shape[0]

        # invert nonzero eigenvalues
        inv_eigenvalues = np.reciprocal(Lambda, where=Lambda > 0)

        # weight vectors are calculated by inverting formula: y = \sum_k=1^M w_k \sqrt{lambda_k} \phi_k(x)
        the_w = (1 / m) * np.diag(inv_eigenvalues ** 0.5) @ Phi.T @ labels
        the_w[Lambda == 0] = 0

        # weight vector for the components out-of-RKHS
        the_a = (1 / m) * Phi.T @ labels
        the_a[Lambda > 0] = 0
        return the_w, the_a

    @staticmethod
    def cumulative_power_distribution(w, Lambda, n_eigenvalues):
        """

        :param w: vector of weights
        :param Lambda: vector of eigenvalues
        :param n_eigenvalues: number of eigenvalues contributing to the cumulative power
        :return:
        """
        powers = np.diag(Lambda) @ (w ** 2)
        return np.sum(powers[:n_eigenvalues]) / np.sum(powers), powers

    def __str__(self):
        (Lambda, Phi, w, a, C, powers) = self.last_result
        return f"""{Lambda=} {Phi=} {w=} {a=} {C=} {powers=}"""
