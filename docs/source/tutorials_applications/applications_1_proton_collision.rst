Anomaly detection in proton collision
=====================================

.. code:: ipython3

    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

.. code:: ipython3

    # !pip install quask==2.0.0-alpha1

.. code:: ipython3

    import quask
    from quask.core import Ansatz, Kernel, KernelFactory, KernelType
    from quask.core_implementation import QiskitKernel

Introduction
------------

We illustrate an application of *quask* in High Energy Physics (HEP):
specifically, the detection of anomalies in proton collision events.
Successfully addressing this task is crucial for advancing our
comprehension of novel phenomena, with the potential to shed light on
enduring questions within the Standard Model of particle physics—an
integral focus of the LHC (Large Hadron Collider) physics program
objectives.

For a more comprehensive introduction, see [woz23], [sch23], [inc23].

Loading the dataset
-------------------

How the dataset has been constructed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dataset consists of Monte Carlo simulated samples of standard model
physics with the addition of BSM processes related to the actual
sensitivity of LHC experiments. To allow a realistic loading and
analysis of the adopted dataset on current quantum devices the data are
mapped to a latent representation of reduced dimensionality using an
autoencoder model.

The datasets used in this work are publicly available at [pie23], it
represents a dijet collision event. Each jet contains 100 particles and
each particle is described by three spatial coordinates for a total of
300 features. The BSM processes (aka anomalies) to benchmark the model
performance are: Randall-Sundrum gravitons decaying into W-bosons
(narrow G), broad Randall-Sundrum gravitons decaying into W-bosons
(broad G), and a scalar boson A decaying into Higgs and Z bosons
(:math:`A \to HZ`). Events are produced with PYTHIA, Jets are clustered
from reconstructed particles using the anti-kT clustering algorithm,
with typical kinematic cut emulating the effect of a typical LHC online
event selection. Subsequently, data are processed with DELPHES to
emulate detector effects, specifically with the adoption of CMS
configuration card. Then, this huge dataset is manipulated using a
classical autoencoder that is trained to compress particle jet objects
individually without access to truth labels. This approach is preferred
over the more standard PCA method due to its ability to capture
nonlinear relationships within the data. The output dimension of the
autoencoder is the *latent dimension*. We have used the pre-processed
dataset, after the simulation and the autoencoding. These datasets takes
the form
:math:`\{ ({x}^{(i)}, y^{(i)}) \in \mathbb{R}^{2\ell} \times \{\mathrm{sm}, \mathrm{bsm}\}\}_{i=1}^m`.
The factor of :math:`2\ell` arises from the fact that we are studying
dijet events. As a result, we have :math:`\ell` features for each jet.

For this tutorial, we have subsampled the smallest useful subset of this
huge dataset. It results in a background signal and one single BSM
signal.

Download
~~~~~~~~

We can load a HEP dataset. This dataset is proton collision and compares
the predictions of the standard model (``background_subsampled.npy``)
with the prediction of A->HZ decay
(``signal_AtoH_to_ZZZ_subsampled.npy``). These files are a small subset
of the true one, built for educational purposes only.

.. code:: ipython3

    !curl "https://zenodo.org/records/10570949/files/background_subsampled.npy?download=1" --output 'background_subsampled.npy'

.. code:: ipython3

    !curl "https://zenodo.org/records/10570949/files/signal_AtoH_to_ZZZ_subsampled.npy?download=1" --output 'signal_AtoH_to_ZZZ_subsampled.npy'

Once downloaded the files, they can be easily loaded via ``numpy``.

.. code:: ipython3

    qX1 = np.load('background_subsampled.npy')
    qX2 = np.load('signal_AtoH_to_ZZZ_subsampled.npy')

Analyzing all this data might take a while. To cut the computational
time, here, we analyze only the first 10 samples of each class.

.. code:: ipython3

    qX1 = qX1[:10,:]
    qX2 = qX2[:10,:]

The dataset is finally constructed as it follows. The background samples
are labelled with target :math:`-1` while the BSM samples are labelled
with :math:`+1`.

.. code:: ipython3

    qX = np.row_stack([qX1, qX2])
    qy = np.array([-1] * len(qX1) + [1] * len(qX2))
    
    print(f"{qX.shape=}")
    print(f"{qy.shape=}")

Split the dataset in training and testing set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    qX_train, qX_test, qy_train, qy_test = train_test_split(qX, qy, test_size=0.2, random_state=42)
    
    print("Shape training set:", qX_train.shape, qy_train.shape)
    print("Shape testing set:", qX_test.shape, qy_test.shape)
    print("Example of feature:", qX_train[0])
    print("Example of label:", qy_train[0])

Anomaly detection using fixed kernels
-------------------------------------

We can solve our tasks with *quask*. The first step is always setting
the backend. We can rely on Qiskit and noiseless simulation.

.. code:: ipython3

    def create_qiskit_noiseless(ansatz: Ansatz, measurement: str, type: KernelType):
        return QiskitKernel(ansatz, measurement, type, n_shots=None)
    
    # KernelFactory.add_implementation('qiskit_noiseless', create_qiskit_noiseless)
    KernelFactory.set_current_implementation('qiskit_noiseless')

Then, we can create the ansats. The number of features has to be 8, as
they are in the dataset. The number of qubits and operation is
arbitrary; in particular, we are not forced to use one qubit per
feature.

.. code:: ipython3

    # Create a quantum kernel
    ansatz = Ansatz(n_features=8, n_qubits=2, n_operations=1)
    ansatz.initialize_to_identity()
    ansatz.change_operation(0, new_feature=0, new_wires=[0, 1], new_generator="XX", new_bandwidth=0.1)
    kernel = KernelFactory.create_kernel(ansatz, "ZZ", KernelType.FIDELITY)
    
    # ansatz.change_operation(1, new_feature=1, new_wires=[1, 2], new_generator="XY", new_bandwidth=0.1)
    # ansatz.change_operation(2, new_feature=2, new_wires=[2, 3], new_generator="XZ", new_bandwidth=0.1)
    # ansatz.change_operation(3, new_feature=3, new_wires=[3, 0], new_generator="YX", new_bandwidth=0.1)
    # ansatz.change_operation(4, new_feature=4, new_wires=[0, 1], new_generator="YY", new_bandwidth=0.1)
    # ansatz.change_operation(5, new_feature=5, new_wires=[1, 2], new_generator="YZ", new_bandwidth=0.1)
    # ansatz.change_operation(6, new_feature=6, new_wires=[2, 3], new_generator="ZX", new_bandwidth=0.1)
    # ansatz.change_operation(7, new_feature=7, new_wires=[3, 0], new_generator="ZY", new_bandwidth=0.1)
    # kernel = KernelFactory.create_kernel(ansatz, "ZZZZ", KernelType.FIDELITY)

Once defined the kernel, the only information we need to solve the task
is the kernel Gram matrices, for both the training and testing set. The
rest of the process is done on a classical machine learning pipeline.

.. code:: ipython3

    # Create the kernel Gram matrices
    K_train = kernel.build_kernel(qX_train, qX_train)
    K_test = kernel.build_kernel(qX_test, qX_train)

We use a simple support vector classifier.

.. code:: ipython3

    # Fit the model to the training data
    model = SVC(kernel='precomputed')
    model.fit(K_train, qy_train)

Finally, we get the accuracy of the model.

.. code:: ipython3

    # Test the model and calculate the score
    y_pred = model.predict(K_test)
    accuracy = np.sum(qy_test == y_pred) / len(qy_test)
    print("Accuracy:", accuracy)

Anomaly detection using optimized kernels
-----------------------------------------

*quask* support also the creation of optimized kernels, tailored for
each task. This approach has been used in [inc23]. To test an
implementation of this, have a look at the optimizers/evaluators API.

References
----------

[pie23] Pierini M. and Wozniak K. A. Dataset for Quantum anomaly
detection in the latent space of proton collision events at the LHC.
Zenodo. https://doi.org/10.5281/zenodo.7673769 (2023).

[woz23] Woźniak K. A., Belis V., Puljak E., Barkoutsos P., Dissertori
G., Grossi M., … & Vallecorsa S. Quantum anomaly detection in the latent
space of proton collision events at the LHC. arXiv preprint
arXiv:2301.10780 (2023).

[sch23] Schuhmacher J., Boggia L., Belis V., Puljak E., Grossi M.,
Pierini M., … & Tavernelli I. Unravelling physics beyond the standard
model with classical and quantum anomaly detection. arXiv preprint
arXiv:2301.10787 (2023).

[inc23] Incudini M., Lizzio Bosco D., Martini F., Grossi M., Serra G.,
Di Pierro A. Automatic and effective discovery of quantum kernels. arXiv
preprint arXiv:2209.11144 (2023).

