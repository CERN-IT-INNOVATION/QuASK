Spectral bias in quantum kernels
================================

Up to this point, we have introduced the concept of a quantum kernel and
discussed its potential applications (`Introduction to quantum kernels
0 <quantum_0_intro.html>`__). We have defined what the expressibility of
a quantum kernel is (`Introduction to quantum kernels
1 <quantum_1_expressibility.html>`__) and how to limit expressibility
through projected quantum kernels (`Introduction to quantum kernels
2 <quantum_2_projected.html>`__).

In this tutorial, we delve deeper into understanding the characteristics
of a quantum kernel and how to determine if and when this tool is
suitable for solving specific tasks. Our investigation is rooted in the
properties of the kernel’s spectrum, which is derived from Mercer’s
theorem and has been further examined in [Can21].

As per Mercer’s theorem, the integral equation:

.. math:: T_\kappa[\phi_j](x) = \int \kappa(x, x') \phi_j(x') p(x') dx = \lambda_j \phi_j(x)

decomposes the kernel function :math:`\kappa` into an infinite sequence
of orthogonal eigenfunctions :math:`\{ \phi_j \}_{j=0}^\infty` and real,
non-negative eigenvalues :math:`\{ \lambda_j \}_{j=0}^\infty`. These are
assumed to be ordered in descending order of eigenvalues. The
decomposition can be expressed as:

.. math:: \kappa(x, x') = \sum_{j=0}^\infty \lambda_j \phi_j(x) \phi_j(x').

This decomposition offers valuable insights into the feasibility of
using quantum kernels and their effectiveness for a spcific task. c
task.

Distribution of eigenvalues
---------------------------

Flat distributions of eigenvalues lead to classifiers with poor generalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider a kernel function
:math:`\kappa(x, x') = \langle \zeta(x), \zeta(x') \rangle_\mathcal{H}`,
with feature map :math:`\zeta : \mathbb{R}^d \to \mathcal{H}`.

Assume that :math:`\kappa(x, x) = 1`, with :math:`\lambda_0` being its
largest eigenvalue, as per Mercer’s decomposition. Let’s assume a
dataset :math:`\{ (x^j, y^j) \}_{j=1}^m`, where the labels are defined
as :math:`y^j = f(x^j)` with :math:`f` as the target function.
Additionally, consider a kernel ridge regressor, denoted as
:math:`\tilde{f}`, trained on the given dataset, possibly with
regularization.

For any :math:`\epsilon \ge 0`, the following inequality holds:

.. math:: \lVert f - \tilde{f} \rVert_2 \ge \sqrt{1 - \frac{\lambda_0 m^2}{\epsilon}} \lVert f \rVert_2

with a probability of at least :math:`1 - \epsilon - \lambda_0 m^4`.

It’s important to note that the sum of eigenvalues, under our
assumptions, equals 1 [dohmatob]. In the worst-case scenario, with the
smallest possible :math:`\lambda_0`, we would have
:math:`\lambda_0 = \lambda_1 = ... = 1/\text{dim}\mathcal{H}`. In the
case of infinite-dimensional Hilbert spaces, such as the Gaussian
kernel, :math:`\lambda` can even approach 0 as
:math:`\text{dim}\mathcal{H}=\infty`. In very large Hilbert spaces, like
those of quantum kernels for a moderately large number of qubits,
:math:`1/\text{dim}\mathcal{H} = 1/2^n \approx 0`.

Under these unfavorable conditions, the theorem above implies that for
sufficiently small :math:`\lambda_0`, the kernel machine
:math:`\tilde{f}` fails to learn any possible function:

.. math:: \lVert f - \tilde{f} \rVert_2 \ge \lVert f \rVert_2 \text{ as } \lambda_0 \to 0

More detailed insights into these results are provided in [Kub21,
Appendix D].

This theorem underscores that a kernel, whether classical or quantum,
with a flat distribution of eigenvalues, cannot generalize any function.
However, techniques can be employed to restore a favorable distribution
of eigenvalues.

Additionally, in [Hua21], as demonstrated in the previous tutorial: - A
large singular value of the kernel Gram matrix (indicating a low-rank
kernel Gram matrix) possesses favorable properties, and the associated
kernel exhibits a decaying spectrum. - A large singular value of the
kernel Gram matrix (indicating a high-rank kernel Gram matrix) has
unfavorable properties, and the associated kernel possesses a flat
spectrum.

The projected quantum kernel is one of the possible techniques to
address this issue.

The use of bandwidths can induce a non-flat distribution of eigenvalues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As evident, the expressibility of a kernel has a direct impact on its
spectrum.

For classical kernels :math:`\kappa` with infinite-dimensional Hilbert
spaces, such as the Gaussian kernel, the issue is managed by introducing
a variance parameter :math:`\sigma`: - With small :math:`\sigma` values,
each data point covers a limited region of the RKHS, resulting in very
small inner products between different data points, a flat spectrum for
:math:`\kappa`, and high-rank kernel Gram matrices. - Larger
:math:`\sigma` values cause each data point to cover a broader region of
the RKHS, leading to larger inner products between different data
points, a decaying spectrum for :math:`\kappa`, and low-rank kernel Gram
matrices.

.. code:: ipython3

    # example with gaussian kernel

In the case of quantum kernels, we have observed that projections can
reduce expressibility. However, a simple technique, akin to that used in
Gaussian kernels, can be employed either in conjunction with or as an
alternative to projection to control expressibility. This technique
leverages the fact that any unitary :math:`U(\cdot)` essentially
performs rotations, with each parameter serving as a rotational angle.
By constraining these angles within the range
:math:`[0, \beta] \subset [0, 2\pi]`, we limit the capacity of
:math:`\kappa` to disperse vectors across the Hilbert space of the
quantum system. This approach requires a bandwidth parameter
:math:`\beta`. [Can22] has formally demonstrated that this technique
restores a favorable spectrum.

In *quask*, each operation on every ansatz includes a mandatory
bandwidth parameter, ensuring native support for this capability.

.. code:: ipython3

    # example with quantum kernel and varying bandwidth

Task-model alignment
--------------------

Mercer’s decomposition of the kernel function provides valuable insights
into its general capabilities. However, this tool also allows us to
quantify how well a specific kernel can adapt to a particular task.

Consider the kernel function :math:`\kappa` and its eigendecomposition
:math:`\{\phi_p\}, \{\lambda_p\}`. Now, for a specific task :math:`f`
and a kernel regression model :math:`\tilde{f}` that employs our quantum
kernel, we can express:

-  The machine learning model in the form of
   :math:`\tilde{f}(x) = \sum_{j=0}^\infty \tilde{w}_j \sqrt{\lambda_j} \phi_j(x)`
-  The target function can be eigendecomposed using an orthonormal set
   of functions, yielding
   :math:`f(x) = \sum_{j=0}^\infty w_j \sqrt{\lambda_j} \phi_j(x)`.

Components with small :math:`\lambda_p` make a limited contribution to
the kernel function, and consequently, they contribute less to the
kernel machine. This implies that if the target function has a
significant contribution from :math:`\phi_j` that is not reflected in
the kernel, that particular component will be challenging to learn.

Based on these observations, we can define a measure of how well a
specific kernel function aligns with a particular task, referred to as
the *task model alignment*. It is defined as follows:

.. math:: C(k) = \frac{\sum_{j = 0}^{k-1} \lambda_j w_j^2}{\sum_{j = 0}^\infty \lambda_j w_j^2}

This metric represents the fraction of *power* in the top :math:`k`
components of the target function. If the target function concentrates
most of its power in the initial kernel components, then the kernel
machine will generalize effectively.

.. code:: ipython3

    # todo evaluate task model alignment

Exponential concentration of kernel values
------------------------------------------

A final, challenging aspect of poorly designed kernels affected by an
excess of expressibility must be examined. In both classical and quantum
kernels, we have observed that one consequence of expressibility is that
most inner products tend to diminish as the dimension of the Hilbert
space increases.

In classical kernels, we can observe this with random projection, as
follows:

.. code:: ipython3

    # todo

Similarly, for Gaussian kernels with small :math:`\sigma`, we have:

.. code:: ipython3

    # todo

This behavior also extends to quantum kernels. Notably, when
:math:`\lambda_0 = \lambda_1 = ... = O(2^{-n})`, [Tha22] has
demonstrated that :math:`\kappa(x, x') \in O(2^{-n})` for almost all
:math:`x, x'`. In the realm of quantum computing, where the kernel
values are estimated rather than precisely calculated, this has
significant implications for the algorithm’s scalability.

Specifically, if we require an accuracy of :math:`\epsilon = 1/2^n` to
distinguish zero and nonzero kernel values, we would need
:math:`O(1/\epsilon^2) = O(2^{2n})` shots to estimate the value
correctly. This phenomenon is referred to as the *exponential
concentration of kernel values*.

References and acknowledgments
------------------------------

[Can21] Canatar, A., Bordelon, B., & Pehlevan, C. (2021). Spectral bias
and task-model alignment explain generalization in kernel regression and
infinitely wide neural networks. Nature communications, 12(1), 2914.

[Can22] Canatar, A., Peters, E., Pehlevan, C., Wild, S. M., & Shaydulin,
R. (2022). Bandwidth enables generalization in quantum kernel models.
arXiv preprint arXiv:2206.06686.

[Kub21] Kübler, J., Buchholz, S., & Schölkopf, B. (2021). The inductive
bias of quantum kernels. Advances in Neural Information Processing
Systems, 34, 12661-12673.

[dohmatob]
https://mathoverflow.net/questions/391248/analytic-formula-for-the-eigenvalues-of-kernel-integral-operator-induced-by-lapl

[Hua21] Huang, HY., Broughton, M., Mohseni, M. et al. Power of data in
quantum machine learning. Nat Commun 12, 2631 (2021).
https://doi.org/10.1038/s41467-021-22539-9

[Tha22] Thanasilp, S., Wang, S., Cerezo, M., & Holmes, Z. (2022).
Exponential concentration and untrainability in quantum kernel methods.
arXiv preprint arXiv:2208.11060.

