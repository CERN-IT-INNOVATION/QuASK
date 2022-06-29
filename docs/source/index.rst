.. QuASK documentation master file, created by
   sphinx-quickstart on Wed Jun 29 08:24:47 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to QuASK's documentation!
=================================
Quantum Advantage Seeker with Kernel

.. image:: ../images/high_level_arch.png
  :width: 80%
  :align: center
  :alt: Alternative text

QuASK is a quantum machine learning software written in Python that supports researchers in designing, experimenting, and assessing different quantum and classic kernels performance. This software is package agnostic and can be integrated with all major quantum software packages (e.g. IBM Qiskit, Xanadu’s Pennylane, Amazon Braket).

QuASK guides the user through a simple preprocessing of input data, definition and calculation of quantum and classic kernels, either custom or pre-defined ones. From this evaluation the package provide an assessment about potential quantum advantage and prediction bounds on generalization error.

Beyond theoretical framing, it allows for the generation of parametric quantum kernels that can be trained using gradient-descent-based optimization, grid search, or genetic algorithms. Projected quantum kernels, an effective solution to mitigate the curse of dimensionality induced by the exponential scaling dimension of large Hilbert spaces, is also calculated. QuASK can also generate the observable values of a quantum model and use them to study the prediction capabilities of the quantum and classical kernels.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   how_to_use



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
