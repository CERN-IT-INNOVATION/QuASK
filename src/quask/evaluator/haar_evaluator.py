import numpy as np
from ..core import Kernel
from . import KernelEvaluator


class HaarEvaluator(KernelEvaluator):
    """
    Expressibility measure based on the comparison between the distribution of states obtained with an Haar random circuit and
    the one obtained with the current ansatz.
    See: Sim, Sukin, Peter D. Johnson, and Alán Aspuru‐Guzik. "Expressibility and entangling capability of parameterized quantum
    circuits for hybrid quantum‐classical algorithms." Advanced Quantum Technologies 2.12 (2019): 1900070.
    """

    def __init__(self, n_bins: int, n_samples: int):
        """
        Initialization
        :param n_bins: number of discretization buckets
        :param n_samples: number of samples approximating the distribution of values
        """
        self.n_bins = n_bins
        self.n_samples = n_samples

    def evaluate(self, kernel: Kernel, K: np.ndarray, X: np.ndarray, y: np.ndarray):
        """
        Evaluate the current kernel and return the corresponding cost. Lower cost values corresponds to better solutions
        :param kernel: kernel object
        :param K: optional kernel matrix \kappa(X, X)
        :param X: datapoints
        :param y: labels
        :return: cost of the kernel, the lower the better
        """
        haar_histogram = HaarEvaluator.haar_histogram(kernel, self.n_bins)
        ansatz_histogram = HaarEvaluator.ansatz_histogram(kernel, self.n_bins, self.n_samples)
        self.last_result = (haar_histogram, ansatz_histogram)
        return np.linalg.norm(haar_histogram - ansatz_histogram)

    @staticmethod
    def ansatz_histogram(kernel, n_bins, n_samples):
        """
        Create a histogram of the fidelities of the ansatz
        :param kernel: kernel object
        :param n_bins: number of discretization buckets
        :param n_samples: number of samples approximating the distribution of values
        :return: histogram of the given ansatz
        """
        histogram = [0] * n_bins

        for _ in range(n_samples):
            theta_1 = np.random.normal(size=(kernel.ansatz.n_features,)) * np.pi
            theta_2 = np.random.normal(size=(kernel.ansatz.n_features,)) * np.pi
            fidelity = kernel.kappa(theta_1, theta_2)
            index = int(fidelity * n_bins)
            histogram[np.minimum(index, n_bins - 1)] += 1

        return np.array(histogram) / n_samples

    @staticmethod
    def haar_histogram(kernel, n_bins):
        """
        Create a histogram of the Haar random fidelities
        :param n_bins: number of bins
        :return: histogram
        """
        N = 2 ** kernel.ansatz.n_qubits

        def prob(low, high):
            return (1-low) ** (N - 1) - (1 - high) ** (N - 1)

        histogram = np.array([prob(i / n_bins, (i+1) / n_bins) for i in range(n_bins)])
        return histogram

    def __str__(self):
        return "A = " + self.last_result[0] + " - " + self.last_result[1]
