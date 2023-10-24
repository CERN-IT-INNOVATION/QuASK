Intro to classical kernels
==========================

To understand how quantum kernel works, it is useful for the reader to have a basic grasp
of how kernel methods works in the classical machine learning setting. For this reason, 
we have created a series of introductory 
tutorials that focus on classical kernels. Subsequently, we will follow this with another 
series of tutorials focused on quantum kernels.

In this first collection, we do not use *quask* to address any classical 
kernel-related tasks. Instead, we rely on the *scikit learn* package to illustrate 
fundamental concepts in the field. These tutorials will cover various topics, 
including how to transform a regression model into a kernel regressor, the concept 
of a feature map, what constitutes a kernel function, how to construct a kernel 
function from a feature map, the fundamentals of Support Vector Machines, and 
their applications in various machine learning tasks.

.. note::

    This series of tutorial are **not** meant to cover all the basic notions regarding kernel methods. 
    There are many books covering these topics. A useful reference is:

        Steinwart Ingo and Christmann Andreas. (2008). Support Vector Machines. Springer Science & Business Media.


Contents
--------

.. toctree::
    :maxdepth: 2

    classical_1_linear_to_kernel
    classical_2_kernel_functions
    classical_3_svm