import jax
import jax.numpy as jnp
from jax.scipy.linalg import qr


def haar_unitary(N, key=None):
    """
    Return a Haar distributed random unitary from U(N)
    :param N: number of amplitudes
    :param key: jax random key
    :return: jax numpy matrix
    """
    key1, key2 = jax.random.split(key)
    Z = jax.random.normal(key1, shape=(N, N)) + 1.0j * jax.random.normal(key2, shape=(N, N))
    [Q, R] = qr(Z)
    D = jnp.diag(jnp.diagonal(R) / jnp.abs(jnp.diagonal(R)))
    return jnp.dot(Q, D)


def haar_integral(N, n_samples, key=None):
    """
    Generate the operator (matrix) representing the integral
    ∫_Haar |Ψ⟩⟨Ψ| dΨ
    :param N: number of amplitudes
    :param n_samples: number of samples
    :param key: jax random key
    :return: jax matrix representing the operator
    """
    randunit_density = jnp.zeros((N, N))
    zero_state = jnp.zeros(N, dtype=complex)
    zero_state = zero_state.at[0].set(1)

    for _ in range(n_samples):
        key1, key = jax.random.split(key)
        A = jnp.matmul(zero_state, haar_unitary(N, key1)).reshape(-1, 1)
        randunit_density += jnp.kron(A, A.conj().T)

    return randunit_density / n_samples


def haar_histogram(N, n_bins):
    """
    Create a histogram of the Haar random fidelities
    :param N: number of amplitudes
    :param n_bins: number of bins
    :return: histogram
    """
    def prob(low, high):
        return (1-low) ** (N - 1) - (1 - high) ** (N - 1)

    histogram = jnp.array([prob(i / n_bins, (i+1) / n_bins) for i in range(n_bins)])
    return histogram

