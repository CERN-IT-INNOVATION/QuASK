"""
Module dedicated to study the Gram matrices generated with the quantum kernels.
"""


import numpy as np
from sklearn.svm import SVC
from scipy.linalg import sqrtm
import numpy.linalg as la


def calculate_frobenius_inner_product(A, B, normalized=False):
    """
    Calculate the Frobenius inner product of two matrices.

    Args:
        A: first matrix (numpy ndarray)
        B: second matrix (numpy nodarray)
        normalized: if True the result is divided by its norm

    Returns:
        Frobenius inner product (float)
    """
    norm = np.sqrt(np.sum(A * A) * np.sum(B * B)) if normalized else 1
    return np.sum(A * B) / norm


def calculate_kernel_polarity(gram_matrix, label_vector):
    """
    Calculate the kernel polarity.

    Args:
        gram_matrix: data points (square) Gram matrix (numpy ndarray)
        label_vector: label vector (numpy ndarray)

    Return:
        Kernel polarity.
    """
    Y = np.outer(label_vector, label_vector)
    return calculate_frobenius_inner_product(gram_matrix, Y, normalized=False)


def calculate_kernel_target_alignment(gram_matrix, label_vector):
    """
    Calculate the kernel-target alignment.

    Args:
        gram_matrix: data points (square) Gram matrix (numpy ndarray)
        label_vector: label vector (numpy ndarray)

    Return:
        Kernel-target alignment (float).
    """
    Y = np.outer(label_vector, label_vector)
    return calculate_frobenius_inner_product(gram_matrix, Y, normalized=True)


def calculate_generalization_accuracy(
    training_gram, training_labels, testing_gram, testing_labels
):
    """
    Calculate accuracy wrt a precomputed kernel, a training and testing set

    Args:
        training_gram: Gram matrix of the training set, must have shape (N,N)
        training_labels: Labels of the training set, must have shape (N,)
        testing_gram: Gram matrix of the testing set, must have shape (M,N)
        testing_labels: Labels of the training set, must have shape (M,)

    Returns:
        generalization accuracy (float)
    """
    svm = SVC(kernel="precomputed")
    svm.fit(training_gram, training_labels)
    y_predict = svm.predict(testing_gram)
    correct = np.sum(testing_labels == y_predict)
    accuracy = correct / len(testing_labels)
    return accuracy


def calculate_approximate_dimension(k):
    """
    Calculate the approximate dimension (d), which is equation S111 in the Supplementary
    of the "The power of data in quantum machine learning"
    (https://www.nature.com/articles/s41467-021-22539-9).

    Args:
        k: Kernel gram matrix.

    Returns:
        approximate dimension of the given kernel (float).
    """
    u, t, udagger = la.svd(k, full_matrices=True)

    N = len(t)

    d = 0
    for i in range(N):
        d += 1 / (N - i) * sum(t[i:])
    return d


def calculate_geometric_difference(k_1, k_2, normalization_lambda=0.001):
    """
    Calculate the geometric difference g(K_1 || K_2), which is equation F9 in
    "The power of data in quantum machine learning" (https://arxiv.org/abs/2011.01938)
    and characterize the separation between classical and quantum kernels.

    Args:
        k_1: Quantum kernel Gram matrix
        k_2: Classical kernel Gram matrix
        normalization_lambda: normalization factor, must be close to zero

    Returns:
        geometric difference between the two kernel functions (float).
    """
    n = k_2.shape[0]
    assert k_2.shape == (n, n)
    assert k_1.shape == (n, n)
    # √K1
    k_1_sqrt = np.real(sqrtm(k_1))
    # √K2
    k_2_sqrt = np.real(sqrtm(k_2))
    # √(K2 + lambda I)^-2
    kc_inv = la.inv(k_2 + normalization_lambda * np.eye(n))
    kc_inv = kc_inv @ kc_inv
    # Equation F9
    f9_body = k_1_sqrt.dot(k_2_sqrt.dot(kc_inv.dot(k_2_sqrt.dot(k_1_sqrt))))
    f9 = np.sqrt(la.norm(f9_body, np.inf))
    return f9


def calculate_model_complexity(k, y, normalization_lambda=0.001):
    """
    Calculate the model complexity s(K), which is equation F1 in
    "The power of data in quantum machine learning" (https://arxiv.org/abs/2011.01938).

    Args:
        k: Kernel gram matrix
        y: Labels
        normalization_lambda: Normalization factor

    Returns:
        model complexity of the given kernel
    """
    n = k.shape[0]
    k_inv = la.inv(k + normalization_lambda * np.eye(n))
    k_body = k_inv.dot(k.dot(k_inv))
    model_complexity = 0
    for i in range(n):
        for j in range(n):
            model_complexity += k_body[i][j] * y[i] * y[j]
    return model_complexity
