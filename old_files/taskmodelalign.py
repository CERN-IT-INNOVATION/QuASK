from scipy.linalg import eigh
import numpy as np


def decompose_kernel(K, eigenvalue_descending_order=True, eigenvalue_removal_threshold=1e-12):
    """
    Decompose the kernel matrix in its eigenvalues Λ and eigenvectors Φ
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

    # remove smallest positive eigenvalues, as they are useless
    Lambda[Lambda < eigenvalue_removal_threshold] = 0

    return Lambda, Phi


def calculate_weight_coefficient(kernel_eigenvalues, kernel_eigenvectors, labels):
    """
    Calculates the weights of a predictor given the labels and the kernel eigendecomposition,
    as shown in (Canatar et al 2021, inline formula below equation 18).
    :param kernel_eigenvalues: vectors of m nonnegative eigenvalues 'eta'
    :param kernel_eigenvectors: vectors of m nonnegative eigenvectors 'phi'
    :param labels: vector of m labels corresponding to 'm' ground truth labels or predictor outputs
    :return: vector of m weights
    """
    # get the number of training elements
    m = kernel_eigenvalues.shape[0]

    # invert nonzero eigenvalues
    inv_eigenvalues = np.reciprocal(kernel_eigenvalues, where=kernel_eigenvalues > 0)

    # weight vectors are calculated by inverting formula: y = \sum_k=1^M w_k \sqrt{eta_k} \phi_k(x)
    the_w = (1 / m) * np.diag(inv_eigenvalues ** 0.5) @ kernel_eigenvectors.T @ labels
    return the_w


def cumulative_power_distribution(weight_coefficients, kernel_eigenvalues, rho):
    power = weight_coefficients * (kernel_eigenvalues ** 2)
    return np.sum(power[:rho]) / np.sum(power)


def f(x):
    return np.inner(x, np.array([0.5, 0.6, 0.7]))


F = 3
M = 20
X = np.random.normal(size=(M, F))
Y = f(X)

K = np.inner(X, X)
Lambda, Phi = decompose_kernel(K)
print(1, np.allclose(Phi @ np.diag(Lambda) @ Phi.T, K))
print(2, np.allclose(Phi @ np.diag(Lambda ** 0.5) @ np.diag(Lambda ** 0.5) @ Phi.T, K))
Psi = Phi @ np.diag(Lambda ** 0.5)
print(3, np.allclose(Psi @ Psi.T, K))
print(4, np.allclose(Psi @ np.diag([1] * M) @ Psi.T, K))
n = np.count_nonzero(Lambda)
print(5, np.allclose(Psi @ np.diag([1] * n + [0]*(M-n)) @ Psi.T, K))

InvLambda = np.reciprocal(Lambda, where=Lambda>0)

w = (1/M) * np.diag(InvLambda ** 0.5) @ Phi.T @ Y
w1 = (1/M) * np.diag(InvLambda) @ Psi.T @ Y
print(6, np.allclose(w, w1))