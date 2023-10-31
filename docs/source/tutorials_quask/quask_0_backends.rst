Backends in *quask*
===================

In this tutorial, we show how to use *quask* with different backends.
These APIs have been designed to allow us to work with quantum kernels
at a high level, hiding all the hardware (or simulator) details. The
current APIs are categorized as follows:

-  The ``core`` package contains the high-level APIs, with the
   ``Operation`` class representing a gate, the ``Ansatz`` class that
   represents a parametric quantum circuit, and the ``Kernel`` abstract
   class representing the quantum kernel.
-  The ``core_implementation`` package contains the low-level
   implementation of the ``Kernel`` class on some quantum programming
   framework.
-  The ``core.KernelFactory`` class is used to instantiate ``Kernel``
   objects on the chosen, pre-configured backend.

We have seen in the `first tutorial on quantum
kernels <quantum_0_intro.html>`__ how to work with the Pennylane Backend
in a noiseless simulation setting. Here we explore further ptions.

.. warning::

    During the installation of _quask_, not all the dependencies are installed: there is no need, for most users, to have all the possible backend available on the machine. For this reason, no backend library is installed and the user have to do it manually. 

Working with the Qiskit backend
-------------------------------

We support Qiskit SDK. It can be installed via *pip* using the following
command:

.. code:: ipython3

    # !pip install qiskit
    # !pip install qiskit_ibm_runtime

Once configured, the class ``core_implementation.QiskitKernel`` can be
used. The objects of this class need a few configurations: \* platform,
“BasicAer” or “QiskitRuntimeService” for simulation and access to IBM
cloud \* backend, the default ones for “BasicAer” platform, or one of
the available in your account for “QiskitRuntimeService” platform \* the
number of samples \* optimization_level, the `optimization
configuration <https://qiskit.org/ecosystem/ibm-runtime/how_to/error-suppression.html>`__
\* resilience_level, the `error mitigation
configuration <https://qiskit.org/ecosystem/ibm-runtime/how_to/error-mitigation.html>`__
\* the tokenn, if platform “QiskitRuntimeService” is used and the token
has not been configured yet on the device, None otherwise

.. code:: ipython3

    from quask.core import Ansatz, Kernel, KernelFactory, KernelType
    from quask.core_implementation import QiskitKernel
    
    def create_qiskit_noiseless(ansatz: Ansatz, measurement: str, type: KernelType):
        return QiskitKernel(ansatz, measurement, type, device_name="default.qubit", n_shots=None)
    
    KernelFactory.add_implementation('qiskit_noiseless', create_qiskit_noiseless)

Working with the Pennylane backend
----------------------------------

We support Pennylane SDK. It can be installed via *pip* using the
following command:

.. code:: ipython3

    # !pip install pennylane

Once configured, the class ``core_implementation.PennylaneKernel`` can
be used. The objects of this class need a few configurations: \* the
name of the device to be used (‘default.qubit’ being the noiseless
simulator); \* the number of shots (‘None’ being infinite shots).

.. code:: ipython3

    from quask.core import Ansatz, Kernel, KernelFactory, KernelType
    from quask.core_implementation import PennylaneKernel
    
    def create_pennylane_noiseless(ansatz: Ansatz, measurement: str, type: KernelType):
        return PennylaneKernel(ansatz, measurement, type, device_name="default.qubit", n_shots=None)
    
    KernelFactory.add_implementation('pennylane_noiseless', create_pennylane_noiseless)

.. note::

    The `PennylaneKernel` class supports the basic functionalities of Pennylane, but not the most advanced ones. In particular, if you want to work with JAX you should copy this class and modify the creation of the device to use JAX instead of the standard libraries. 

Working with the Amazon Braket backend
--------------------------------------

We support Amazon Braket via the PennyLane plugins. It can be installed
via *pip* using the following command:

.. code:: ipython3

    # !pip install amazon-braket-sdk
    # !pip install amazon-braket-pennylane-plugin

The Amazon SDK has to be configured via the Amazon CLI, whose procedure
is detailed in the
`documentation <https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html>`__.
Basically, you have to run ``aws configure`` command and follow the
instruction to insert the access key, secret access key, and region of
your account.

Once configured, the class ``core_implementation.BraketKernel`` can be
used. The objects of this class need a few configurations: \* the name
of the device to be used; \* the Amazon S3 bucket and prefix to save the
results; \* the number of shots.

For more detailed explaination on the setup of Amazon Braket objects,
you can follow the `Amazon Braket
documentation <https://docs.aws.amazon.com/braket/latest/developerguide/hybrid.html>`__.
Here’s an example on how to configure the backend in *quask*:

.. code:: ipython3

    import numpy as np
    from quask.core import Ansatz, Kernel, KernelFactory, KernelType
    from quask.core_implementation import BraketKernel
            
    def create_braket(ansatz: Ansatz, measurement: str, type: KernelType):
        return BraketKernel(ansatz, measurement, type,
                            device_name="arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3", 
                            s3_bucket="my_s3_bucket", 
                            s3_prefix="my_s3_prefix", 
                            n_shots=1024)
    
    KernelFactory.add_implementation('braket_example', create_braket)
    KernelFactory.set_current_implementation('braket_example')

.. note::

    If you need to extend _quask_ to support a certain hardware, the easiest way to implement it is by checking if there exists a plugin to make it work with PennyLane. In that case, you can copy the `BraketKernel` and change the setup of Braket with the setup of the hardware you want to support.

Working with the Qibo backend
-----------------------------

Add your own backend
--------------------

Do you want to extend the list of *quask* backends? Do you need a
particular feature that is not supported built-in by our classes? If so,
you should consider writing your own backend.

We have designed *quask* in a modular fashion so that users can modify
it to suit their every need with minimal effort. To start the creation
of a brand new backend, you need to create a class that extends
``Kernel`` and implements the abstract methods. These are:

1. ``kappa``: calculates the inner product between a pair of data
   points.
2. ``phi``: calculates the feature map corresponding to a single data
   point if the ``kernel_type`` field is *observable*, throws an error
   otherwise.

Furthermore, the initialization method should set up the backend’s
environment properly. For example, if the backend is meant to work on
some quantum hardware accessed via the cloud, it should set up all the
necessary configurations.

Once you have set this up, you can make it available via the
``KernelFactory``. Here follows an example of a mock backend:

.. code:: ipython3

    import numpy as np
    from quask.core import Ansatz, Kernel, KernelFactory, KernelType
    
    class MockKernel(Kernel):
    
        def __init__(self, ansatz: Ansatz, measurement: str, type: KernelType):
            super().__init__(ansatz, measurement, type)
    
        def kappa(self, x1, x2) -> float:
            if self.type == KernelType.OBSERVABLE:
                return 1.0 if np.isclose(x1, x2) else 0.0
            elif self.type == KernelType.FIDELITY:
                return 1.0 if np.isclose(x1, x2) else 0.0
            elif self.type == KernelType.SWAP_TEST:
                return 1.0 if np.isclose(x1, x2) else 0.0
    
        def phi(self, x) -> float:
            if self.type == KernelType.OBSERVABLE:
                return np.array([1.0])
            elif self.type in [KernelType.FIDELITY, KernelType.SWAP_TEST]:
                raise ValueError("phi not available for fidelity kernels")
            else:
                raise ValueError("Unknown type, possible erroneous loading from a numpy array")
            
    def create_mock(ansatz: Ansatz, measurement: str, type: KernelType):
        return MockKernel(ansatz, measurement, type)
    
    KernelFactory.add_implementation('mock', create_mock)
    KernelFactory.set_current_implementation('mock')

.. note::

    If you have added a particular functionality to _quask_, consider reaching out to us if you want if included in a future version of our software. 

