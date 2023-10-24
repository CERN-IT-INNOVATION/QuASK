import haar
import pqc
import metrics

import jax
import jax.numpy as jnp

the_key = jax.random.PRNGKey(12345)
id_circuit = jnp.array([[0], [0], [0], [1], [0], [0]])
x_circuit = jnp.array([[1], [0], [0], [1], [0], [1]])

n = 2
n_bins = 8
h_hist = haar.haar_histogram(2**n, n_bins)
p_hist = pqc.pqc_histogram(n, x_circuit, 1, 10, n_bins, the_key)

print(metrics.hellinger_distance(p_hist, h_hist))