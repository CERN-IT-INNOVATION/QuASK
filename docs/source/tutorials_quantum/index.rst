Intro to quantum kernels
=========================

As part of the educational mission of *quask*, we have developed a series of introductory 
tutorials focusing on quantum kernels. These tutorials cover fundamental theoretical 
concepts, including what a quantum kernel is, the defining properties that characterize 
it, and when their usage is supported by theoretical evidence. They also delve into 
implementation details and practical applications.

These tutorials serve as a platform to introduce the core components of *quask*: 
the classes 
`Operation <../modules.html#quask.core.operation.Operation>`__, 
`Ansatz <../modules.html#quask.core.ansatz.Ansatz>`__, 
and 
`Kernel <../modules.html#quask.core.kernel.Kernel>`__, 
which model quantum kernels, the 
`KernelFactory <../modules.html#quask.core.kernel_factory.KernelEvaluator>`__ 
class used to select the backend for executing quantum kernels, and the 
`KernelEvaluator <../modules.html#quask.evaluator.kernel_evaluator.KernelEvaluator>`__ 
classes that assign scores to quantum kernels based on 
specific criteria. The latter is an especially noteworthy family of classes, 
as their source code serves as a reference implementation for numerous 
theoretical papers.

.. note::

    This series of tutorial are **not** meant to cover the basic notions of quantum computing. 
    If you don't know what a quantum circuit is, you can refer to the many resources available 
    online. Alternatively, there are few books covering the needed topics. A recent and 
    complete reference is:

        Manenti Riccardo and Motta Mario. (2023). Quantum Information Science. Oxford University Press.

    For an introduction to quantum machine learning you can refer to:

        Schuld Maria and Petruccione Francesco. (2021). Machine learning with quantum computers. Springer.


Contents
--------

.. toctree::
    :maxdepth: 1

    quantum_0_intro
    quantum_1_expressibility
    quantum_2_projected
    quantum_3_spectralbias
    quantum_4_beyondnisq
