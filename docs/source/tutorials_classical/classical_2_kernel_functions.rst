Kernel machines
===============

In the `previous tutorial <classical_1_linear_to_kernel.html>`__, we
introduced linear models for regression. These models are effective
under certain assumptions, primarily that the target function to be
learned is linear. In this tutorial, we’ll explore how to adapt Ridge
regression models to capture more complex relationships within the data
while maintaining efficient trainin.

.. note ::

    The ['Data Mining Book'](https://dataminingbook.info/book_html/) provides excellent material that can be consulted online for free.

Feature map and kernel functions
--------------------------------

Let :math:`\mathbf{x}` belong to the input space
:math:`\mathcal{X} = \mathbb{R}^d`. A *feature map*, denoted as
:math:`\phi: \mathbb{R}^d \to \mathcal{H}`, is a transformation of input
attributes into a Hilbert space :math:`\mathcal{H}`. This mapping allows
us to represent the original features in a richer way, allowing us to
solve the learning problem more effectively.

Consider a scenario where our target function is given by
:math:`f(\mathbf{x}) = v_0 \mathbf{x}_1^2 + v_1 \mathbf{x}_1 \mathbf{x}_2 + v_2 \mathbf{x}^2`.
This target function exhibits quadratic relationships within the
features, making it unsuitable for learning using a simple linear
regressor in the form
:math:`\tilde{f}(\mathbf{x}) = w_1 \mathbf{x}_1 + w_2 \mathbf{x}_2`. To
address this issue, we can define a feature map:

.. math:: \phi(\mathbf{x}) = \left[\begin{array}{c} \mathbf{x}_1 & \mathbf{x}_2 & \mathbf{x}_1 \mathbf{x}_2 & \mathbf{x}_1^2 &  \mathbf{x}_2^2 \end{array}\right]^\top

and then use a linear regressor that operates directly on the
transformed vector,
:math:`\tilde{f} = \sum_{j=1}^5 w_j \phi(\mathbf{x})_j`. In situations
where the dual form of the linear (Ridge) regressor is used, we can
directly replace the Euclidean inner product
:math:`\langle \cdot, \cdot \rangle` with the *kernel function*:

.. math:: \kappa(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle_\mathcal{H}

Calculating the kernel can be more convenient than calculating the explicit representation of the feature map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A major advantage of using a kernel function is the efficiency it offers
in calculating complex relationships without explicitly representing
:math:`\phi(\mathbf{x})`. This is particularly evident in cases like the
polynomial kernel, designed to capture polynomial relationships of
arbitrary degrees within the data:

.. math:: \kappa(\mathbf{x}, \mathbf{x}') = (\langle \mathbf{x}, \mathbf{x}' \rangle + b)^c

Its feature map has :math:`k = \sum_{j = 1}^c \binom{c}{j}` components,
:math:`\phi : \mathbb{R}^d \to \mathbb{R}^k`, with :math:`k \gg d`. In
many cases, the Hilbert space of the feature map has a much higher
dimensionality than the original input space.

A more striking example is the Gaussian or RBF kernel:

.. math:: \kappa(\mathbf{x}, \mathbf{x}') = \exp(-c \lVert \mathbf{x} - \mathbf{x}' \rVert_2^2)

Its calculation is straightforward, but the underlying feature map
transforms :math:`\mathbf{x} \in \mathbb{R}^d` into a Gaussian function
with a mean value in :math:`\mathbf{x}` itself,
:math:`\phi(\mathbf{x}) \in L_2(\mathbb{R}^d)`. In this case,
:math:`\mathcal{H} = L_2(\mathbb{R}^d)` is infinite-dimensional, and
constructing an explicit representation is unfeasible. A naive attempt
would require value discretization and truncation to finite intervals.

Positive semidefiniteness of kernel functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most important fact about a kernel function is that it represents an
inner product in some Hilbert space. To determine if a bilinear form is
a valid kernel, we only need to ensure that it behaves like an inner
product, which means it has to be positive semidefinite. This latter
property also implies that the function is symmetric.

If we can prove that :math:`\kappa` is a positive semidefinite bilinear
form, then there exists a (non-unique) Hilbert space :math:`\mathcal{H}`
and a feature map :math:`\phi` satisfying
:math:`\kappa(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}`.
This is true even if we don’t know their exact definitions or how to
compute them explicitly. Furthermore, positive semidefiniteness implies
that, given data :math:`\mathbf{x}^1, ..., \mathbf{x}^m`, the kernel
Gram matrix :math:`K_{i,j} = \kappa(\mathbf{x}^i, \mathbf{x}^j)` is
positive semidefinite.

Conversely, it is trivially true that if we define a kernel explicitly
from the feature map, positive semidefiniteness holds by construction.

Reproducing Kernel Hilbert Space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\kappa` be a kernel function. We have infinitely many feature
maps and Hilbert spaces corresponding to the given kernel. However,
there is a unique *reproducing kernel feature map*
:math:`\Phi_x : \mathcal{X} \to \mathbb{R}`, defined as
:math:`\Phi_x = \kappa(\cdot, \mathbf{x})`. We can use such a feature
map to define the following pre-Hilbert vector space:

.. math::  \mathcal{V} = \mathrm{span}\{ \Phi_x \mid x \in \mathcal{X} \} = \left\{f(\cdot) = \sum_{i=1}^n \alpha_i \kappa(\cdot, x^i) \mid n \in \mathbb{N}, x^i \in \mathcal{X} \right\}. 

We can prove that the following function is an inner product on
:math:`\mathcal{V}`, which means it is symmetric, bilinear, and positive
semidefinite:

.. math::

    \langle f, g \rangle 
   = \left\langle \sum_{i} \alpha_i \kappa(\cdot, x^i),  \sum_{j} \beta_j \kappa(\cdot, x^j) \right\rangle
   = \sum_{i, j} \alpha_i \beta_j \kappa(x^i, x^j).

We can also prove that :math:`\kappa` has the reproducing property,
which means that the following equation holds:

.. math:: \langle f, \kappa(\cdot, x^j)\rangle = f(x^j).

We can define the vector space
:math:`\mathcal{H} = \overline{\mathcal{V}}`, which is complete and,
thus, a Hilbert space. This latter one is denoted as the Reproducing
Kernel Hilbert Space (RKHS) of :math:`\kappa`.

Kernel Ridge regression
-----------------------

We have defined the Ridge regressor in the previous tutorial,

.. math:: \tilde{f}(\mathbf{x}) = \sum_{j=1}^m \alpha_j \langle \mathbf{x}, \mathbf{x}^j \rangle

where :math:`{\alpha} = (G + \lambda I)^{-1} \mathbf{y}` is the solution
of the optimization problem expressed via the Lagrangian multipliers,
and :math:`G = X^\top X` Gram matrix. To define a *kernel* Ridge
regressor we just have to substitute the inner product with the kernel
function, and the Gram matrix :math:`G` with the *kernel* Gram matrix
:math:`K_{i,j} = \kappa(x^i, x^j)`,

.. math:: \tilde{f}(\mathbf{x}) = \sum_{j=1}^m \alpha_j \kappa(\mathbf{x}, \mathbf{x}^j).

We can easily test the model on a synthetic dataset:

.. code:: ipython3

    import numpy as np
    m, d = 100, 2
    unknown_f = lambda x: 3.2 * x[0] * x[1] + 5.2 * x[0]**3
    
    X = np.random.random(size=(m, d))
    y = np.apply_along_axis(unknown_f, 1, X) + 0.33 * np.random.random(size=(m,))

To test the approach, we need to define a kernel. Given the problem’s
structure, we choose a polynomial kernel of degree three. As this is a
Ridge regression, the parameter ``alpha`` must be set, which corresponds
to the strength of the regularization term.

.. code:: ipython3

    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)
    
    kernel_regressor = KernelRidge(alpha=0.1, kernel='polynomial', degree=3)
    kernel_regressor.fit(X_train, y_train)
    y_pred = kernel_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")


.. parsed-literal::

    Mean Squared Error: 0.03978691374195227


The kernel has been specified, in this example, in the arguments
``kernel`` and ``degree``. Look at the documentation to see all the
possibilities offere built-in in the *scikit-learn* package.

Underfitting
~~~~~~~~~~~~

When we select a kernel that is not sophisticated enough to capture the
inherent relationships within the dataset, we end up with a model that
cannot effectively learn the target function. This phenomenon is known
as *underfitting*, and it occurs when both the training set and testing
set errors are high.

In our example, this may occur if we use a linear kernel when the actual
function is cubic:

.. code:: ipython3

    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1234)
    
    kernel_regressor = KernelRidge(alpha=0.1, kernel='polynomial', degree=1)
    kernel_regressor.fit(X_train, y_train)
    y_pred_train = kernel_regressor.predict(X_train)
    y_pred_test = kernel_regressor.predict(X_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    print(f"Mean Squared Error training: {mse_train}")
    print(f"Mean Squared Error testing: {mse_test}")


.. parsed-literal::

    Mean Squared Error training: 0.3511775695752495
    Mean Squared Error testing: 0.5276843667598432


Overfitting
~~~~~~~~~~~

When we select a kernel that is too sophisticated, the model
interpolates both the data and the noise within the dataset. This
results in a model whose underlying function is extremely complicated
and distant from the true target. This phenomenon is known as
*overfitting*, and it occurs when the training set error is low, and the
testing set error is high.

In this case, setting a large regularization constant can mitigate the
problem. A large regularization constant favors ‘simple’ solutions over
complicated ones, even if they better interpolate the data.

In our example, this may occur if we use a degree-50 kernel when the
actual function is ubic:

.. code:: ipython3

    # disable warning about singular matrices
    import warnings
    from scipy.linalg import LinAlgWarning
    warnings.filterwarnings(action='ignore', category=LinAlgWarning, module='sklearn')

.. code:: ipython3

    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)
    kernel_regressor = KernelRidge(alpha=0.000001, kernel='polynomial', degree=50)
    kernel_regressor.fit(X_train, y_train)
    y_pred_train = kernel_regressor.predict(X_train)
    y_pred_test = kernel_regressor.predict(X_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    print(f"Mean Squared Error training: {mse_train}")
    print(f"Mean Squared Error testing: {mse_test}")


.. parsed-literal::

    Mean Squared Error training: 3.117245929034068e-05
    Mean Squared Error testing: 24185905.35320369

