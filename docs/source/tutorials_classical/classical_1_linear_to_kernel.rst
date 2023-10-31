Linear and Ridge regression
===========================

Our first tutorial delves into the motivation behind the application of
kernel methods. We begin by demonstrating regression analysis using the
simplest model, the linear regressor. Next, we explore how Ridge
regularization can be applied to mitigate numerical errors during model
training. Finally, we delve into solving these learning problems using
the Lagrangian multiplier methods, highlighting how this method gives
rise to the primal and dual forms of the Ridge regressor.

Linear Regression
-----------------

Linear regression is the simplest machine learning model used for
learning a target function, denoted as :math:`f`. It requires a dataset
of samples, :math:`\{ (\mathbf{x}^j, y^j) \}_{j=1}^m`. Each feature
vector is represented as :math:`\mathbf{x}^j \in \mathbb{R}^d`, and has
been sampled i.i.d. from some unknown probability distribution
:math:`p(\mathbf{x})`. The labels are real numbers, defined as
:math:`y^j = f(\mathbf{x}^j) + \varepsilon^i`, with each
:math:`\varepsilon^i` representing random Gaussian noise with zero mean
and fixed variance.

Let
:math:`X = \left[ \begin{array}{c} (\mathbf{x}^1)^\top \\ \vdots \\ (\mathbf{x}^m)^\top \end{array}\right] \in \mathbb{R}^{m \times d}`,
which is the design matrix, and
:math:`\mathbf{y} = \left[ \begin{array}{c} y^1 \\ \vdots \\ y^m \end{array}\right] \in \mathbb{R}^{m \times 1}`,
representing the regressand.

We can generate randomly a synthetic dataset for our demo:

.. code:: ipython3

    import numpy as np
    
    # set the dimensionality of the problem: number of samples of the dataset, dimensionality of the feature vector
    m, d = 100, 2
    
    # create a target function f to learn, according to our assumptions it should be a linear function
    unknown_w = np.random.random(size=(d,))
    unknown_f = lambda x: unknown_w.dot(x)
    
    # generate the synthetic dataset: first the features...
    X = np.random.random(size=(m, d))
    
    # generate the synthetic dataset: ... and then the noisy labels
    noiseless_y = np.apply_along_axis(unknown_f, 1, X)
    noise = 0.1 * np.random.random(size=(m,))
    y = noiseless_y + noise
    
    print(f"Dataset dimensionality    : design matrix {X.shape=} | regressand {y.shape=}")
    print(f"Example of feature vector : {X[0]}")
    print(f"Example of label          : {y[0]}\n")


.. parsed-literal::

    Dataset dimensionality    : design matrix X.shape=(100, 2) | regressand y.shape=(100,)
    Example of feature vector : [0.57980451 0.18366092]
    Example of label          : 0.18893031545613453
    


We are going to build a function:

.. math:: \tilde{f}(\mathbf{x}) = \langle \mathbf{x}, \mathbf{w}\rangle + b

such that :math:`\tilde{f}` is as close as possible to :math:`f`. Note
that we will omit the bias :math:`b` [foot1]. The ideal scenario would
be to find the :math:`\tilde{f}`, or the vector of weights
:math:`\tilde{\mathbf{w}}`, that minimizes the *expected risk*:

.. math:: \tilde{f}(\mathbf{x}) = \arg\min_h \mathbb{E}_{\mathbf{x} \sim p(\mathbf{x})}\left[(h(\mathbf{x}) - f(y))^2\right]

However, we have no access to :math:`f` other than the given dataset, so
we can only minimize the *empirical risk*:

.. math:: \tilde{f}(\mathbf{x}) = \arg\min_h \sum_{j=1}^m (h(\mathbf{x}^j) - y^j)^2

.. math:: \tilde{w} = \arg\min_{w \in \mathbb{R}^d} (\langle \mathbf{x}^j, \mathbf{w}\rangle - y^j)^2

Provided that the columns of :math:`X` are linearly independent, the
problem has a unique, analytical solution obtained by setting the
derivative of the objective function to zero,

.. math::

   \begin{array}{rl} 
   \nabla_w \left[\lVert X w - y \rVert\right] & = 0 \\
   \nabla_w \left[(Xw)^\top Xw-(Xw)^\top y - y^\top(Xw)+y^\top y\right] & = 0 \\
   \nabla_w \left[w^\top X^\top X w-2(Xw)^\top y + y^\top y\right] & = 0 \\
   2 X^\top X w - 2 X^\top y & = 0 \\
   X^\top X w & = X^\top y 
   \end{array}

which results in the following solution,

.. math:: \tilde{w} = (X^\top X)^{-1} X^\top \mathbf{y}.

.. code:: ipython3

    # standard matrix inversion, not numerically stable
    G = np.linalg.inv(X.transpose().dot(X))
    
    # more stable way to calculate matrix inversion
    G = np.linalg.solve(X.transpose().dot(X), np.eye(2))
    
    # finish calculating
    estimated_w = G.dot(X.transpose()).dot(y)
    
    # the empirical risk is not zero because of the noise in the dataset label
    print(f"Empirical risk between target and estimated functions: {np.linalg.norm(np.apply_along_axis(lambda x: estimated_w.dot(x), 1, X) - y)}")
    
    # nonetheless, we get decent generalization error
    print(f"True risk between target and estimated functions: {np.linalg.norm(estimated_w - unknown_w)}")


.. parsed-literal::

    Empirical error between target and estimated functions: 0.35801910246331653
    True error between target and estimated functions: 0.06560906515727047


.. warning ::

    When using linear regression, we are making certain assumptions, such as assuming that the target function is linear, the data has been independently and identically distributed (i.i.d.) from some distribution, and the noise's variance is constant. If the target function does *not* adhere to these assumptions, it will be challenging to solve the learning task with a linear classifier, and we should consider choosing a different, more suitable machine learning model.

Using the capabilities of the *scikit-learn* framework instead of
writing everything from scratch with NumPy is much easier and less
error-prone. The same example demonstrated earlier can be rephrased as
follows:

.. code:: ipython3

    from sklearn.linear_model import LinearRegression
    
    # create model
    lin_reg = LinearRegression(fit_intercept=False)
    
    # training
    lin_reg.fit(X, y)
    
    # retrieve the weight parameters
    another_estimated_w = lin_reg.coef_
    print("Weights with the LinearRegressor class:", another_estimated_w)


.. parsed-literal::

    Weights with the LinearRegressor class: [0.19503256 0.14576726]


Ridge regression
----------------

In solving the learning problem above, numerical errors can arise when
computing the inverse of :math:`X^\top X`, especially when it has nearly
singular values. To address this issue, we can introduce positive
elements on the principal diagonal. This adjustment reduces the
condition number and eases matrix inversion. In this case, we aim to
solve the following problem:

.. math:: \tilde{w} = \arg\min_{w \in \mathbf{R}^d} \sum_{j=1}^m (\langle \mathbf{x}^j, \mathbf{w}\rangle - y^j)^2 + \lambda \lVert \mathbf{w} \rVert_2

where :math:`\lambda \ge 0` is the regularization term.

A regressor whose associated learning problem is the one described above
is called a *ridge regressor*. The solution can still be found
analytically:

.. math:: \tilde{w} = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}.

.. code:: ipython3

    from sklearn.linear_model import Ridge
    
    rigde_reg = Ridge(alpha=10.0, fit_intercept=False)
    rigde_reg.fit(X, y)
    ridge_estimated_w = rigde_reg.coef_
    print("Weights with the Ridge class:", ridge_estimated_w)


.. parsed-literal::

    Weights with the Ridge class: [0.15521267 0.13250445]


The Lagrangian multiplier method
--------------------------------

The solution to the *ridge regularization* problem can be found via the
Lagrangian multipliers methods. It is applied with problems in the
following form:

.. math:: \begin{array}{rl} \min_w & f(w) \\ \text{constrained to} & h_j(w) = 0, \text{with }j = 1, ..., k\end{array}

We assume that both :math:`f` and any :math:`h_j` is convex. The
Lagrangian function is defined as

.. math:: \mathcal{L}(\mathbf{w}, \mathbf{\alpha}) = f(\mathbf{w}) + \sum_{j=1}^k \mathbf{\alpha}_j h_j(\mathbf{w})

and the parameters :math:`\mathbf{\alpha}_j` are called Lagrangian
multipliers. The solution of this optimization problem can be found by
setting the partial derivatives to zero,
:math:`\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 0, \frac{\partial \mathcal{L}}{\partial \mathbf{\alpha}} = 0`
and solve for :math:`\mathbf{w}` and :math:`\mathbf{\alpha}`.

In this case, we can solve two possible optimization problems,

.. math:: \min_w \max_\alpha \mathcal{L}(w, \alpha)

which is the *primal* form of the optimization problem, or

.. math:: \max_\alpha \min_w \mathcal{L}(w, \alpha)

which is the *dual* form. In general the solution of the two opimization
problems are different, but, under some assumptions (the KKT conditions)
they coincide. These assutptions hold for the optimization problem
underlying the Rigde regression.

Solving the Ridge regression with Lagrangian multipliers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When considering the following objective function,

.. math:: \mathcal{L}(\mathbf{w}) = \frac{1}{2}\sum_{j=1}^m (\mathbf{w}^\top \mathbf{x}^j - y^j)^2 + \frac{\lambda}{2} \langle \mathbf{w}, \mathbf{w}\rangle,

the optimization problem :math:`\arg\min_w \mathcal{L}(w)` is
unconstrained, and it is solved analytically without the Lagrangian
multiplier method. The :math:`1/2` factor simplifies the constant when
taking the derivative and does not influence the result of the
optimization.

We can, although, convert this problem into a constraint optimization.
It seems that we are making things more difficult, but truly this way
will reveal important insights. We introduce the variables
:math:`\mathbf{z}_j = \mathbf{w}^\top \mathbf{x}^j - y^j`, and this
variable can be associated with the constraint
:math:`h_j(\mathbf{z}, \mathbf{w}) \equiv \mathbf{z}_j - \mathbf{w}^\top \mathbf{x}^j + y^j = 0`.
The optimization problem becomes

.. math::

   \begin{array}{rl} 
   \arg\min_w & \frac{1}{2} \langle z, z \rangle + \frac{\lambda}{2} \langle w, w \rangle \\ 
   \text{constrained to} & h_j(w) = 0, \text{with }j = 1, ..., m
   \end{array}

Now, a trick to make the result look nicer is to multiply the objective
function by :math:`1/\lambda`, while the constraints remain untouched.
Again, this does not change the overall result.

.. math::

   \begin{array}{rl} 
   \arg\min_w & \frac{1}{2\lambda} \langle z, z \rangle + \frac{1}{2} \langle w, w \rangle \\ 
   \text{constrained to} & h_j(w) = 0, \text{with }j = 1, ..., m
   \end{array}

For each constraint, we define a Lagrangian multiplier :math:`\alpha_j`.
The associated Lagrangian is:

.. math::

   \mathcal{L}(\mathbf{w}, \mathbf{z}, \mathbf{\alpha}) 
   = \frac{1}{2\lambda} \langle \mathbf{z}, \mathbf{z} \rangle 
   + \frac{1}{2} \langle \mathbf{w}, \mathbf{w}\rangle
   + \mathbf{\alpha}^\top (\mathbf{z} - X \mathbf{w} + \mathbf{y}).

Solving this problem in its primal form leads to the solution that we
have already found. However, if we solve the dual form,

.. math:: \max_\alpha \min_{w, z} \mathcal{L}(\mathbf{w}, \mathbf{z}, \mathbf{\alpha}),

we obtain that

.. math::

   \begin{array}{rl} 
   \frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \mathbf{w} - \mathbf{\alpha}^\top X = 0 & \implies \mathbf{w}^\top = X^\top \mathbf{\alpha}. \\
   \frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \frac{1}{\lambda} \mathbf{z} + \mathbf{\alpha} = 0 & \implies \mathbf{z} =  - \lambda \mathbf{\alpha}.
   \end{array}

Once substituted :math:`w` and :math:`z`, the remaining objective takes
the form:

.. math::

   \begin{array}{rl}
   \mathcal{L}(\mathbf{\alpha}) 
   & = \frac{1}{2\lambda} \langle -\lambda\mathbf{\alpha}, -\lambda\mathbf{\alpha} \rangle 
   + \frac{1}{2} \langle X^\top \mathbf{\alpha}, X^\top \mathbf{\alpha}\rangle
   + \mathbf{\alpha}^\top (-\lambda\mathbf{\alpha} - X X^\top \mathbf{\alpha} + \mathbf{y}) \\
   & = \frac{\lambda}{2} \mathbf{\alpha}^\top \mathbf{\alpha}
   + \frac{1}{2} \mathbf{\alpha}^\top X X^\top \mathbf{\alpha}
   - \lambda \mathbf{\alpha}^\top \mathbf{\alpha}
   - \mathbf{\alpha}^\top X X^\top \mathbf{\alpha}
   + \mathbf{\alpha}^\top \mathbf{y} \\
   & = - \frac{\lambda}{2} \mathbf{\alpha}^\top \mathbf{\alpha}
   - \frac{1}{2} \mathbf{\alpha}^\top X X^\top \mathbf{\alpha}
   + \mathbf{\alpha}^\top \mathbf{y} \\
   & = - \frac{1}{2} \mathbf{\alpha}^\top (\lambda I) \mathbf{\alpha}
   - \frac{1}{2} \mathbf{\alpha}^\top X X^\top \mathbf{\alpha}
   + \mathbf{\alpha}^\top \mathbf{y} \\
   & = - \frac{1}{2} \mathbf{\alpha}^\top (X X^\top + \lambda I) \mathbf{\alpha}
   + \mathbf{\alpha}^\top \mathbf{y} \\
   & = - \frac{1}{2} \mathbf{\alpha}^\top (G + \lambda I) \mathbf{\alpha}
   + \mathbf{\alpha}^\top \mathbf{y} 
   \end{array}

where :math:`G = X X^\top` is the Gram matrix of the inner products
between :math:`x^1, ..., x^m`. The solution is

.. math:: \nabla_\alpha \mathcal{L} = - (G + \lambda I) \alpha + \mathbf{y} = 0 \implies \tilde\alpha = (G + \lambda I)^{-1} \mathbf{y}.

Now, we have two ways of expressing the same Ridge regressor: \* In the
primal form, we have
:math:`\tilde{f}(\mathbf{x}) = \sum_{j=1}^d \mathbf{w}_j \mathbf{x}^j`.
\* In the dual form, we have
:math:`\tilde{f}(\mathbf{x}) = \sum_{j=1}^m \alpha_j \langle \mathbf{x}, \mathbf{x}^j \rangle`.

The primal form has the advantage that, in case we are dealing with a
large amount of data (:math:`m \gg d`), the regressor is much more
efficient. The dual form has the advantage of expressing the regressor
in terms of its similarity to the elements of the training set, instead
of on :math:`\mathbf{x}` itself. This leads to new possibilities:
changing the notion of ‘similarity’ allows the creation of much more
powerful models.

References
----------

[foot1] as it can be incorporated as the weight vector
:math:`\mathbf{w}` via the equation
:math:`\mathbf{w}^\top \mathbf{x} + b = [\mathbf{w}, b]^\top [\mathbf{x}, 1] = (\mathbf{w}')^\top \mathbf{x}'`.
