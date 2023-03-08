import jax.numpy as jnp


def kl_divergence(hist_pqc, hist_haar, epsilon=0.001):
    """
    Calculate the KL Divergence between two probability distributions (histograms).
    Be careful! KL Divergence is fragile, and asymmetric
    :param hist_pqc:
    :param hist_haar:
    :return:
    """
    n_bins = len(hist_pqc)
    hist_pqc += epsilon
    hist_haar += epsilon
    the_sum = 0
    for j in range(n_bins):
        t = hist_pqc[j] * jnp.log(hist_pqc[j]/hist_haar[j])
        print(hist_pqc[j], hist_haar[j], "=", t)
        the_sum += t
    return the_sum


def hellinger_distance(hist_pqc, hist_haar):
    """
    More stable than KL Divergence, also is symmetric
    :param hist_pqc:
    :param hist_haar:
    :return:
    """
    return (1 / jnp.sqrt(2)) * jnp.linalg.norm(jnp.sqrt(hist_pqc) - jnp.sqrt(hist_haar))