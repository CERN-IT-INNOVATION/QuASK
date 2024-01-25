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
(:math:`A\textrightarrow{}HZ`). Events are produced with PYTHIA, Jets
are clustered from reconstructed particles using the anti-kT clustering
algorithm, with typical kinematic cut emulating the effect of a typical
LHC online event selection. Subsequently, data are processed with
DELPHES to emulate detector effects, specifically with the adoption of
CMS configuration card. Then, this huge dataset is manipulated using a
classical autoencoder that is trained to compress particle jet objects
individually without access to truth labels. This approach is preferred
over the more standard PCA method due to its ability to capture
nonlinear relationships within the data. The output dimension of the
autoencoder is the :raw-latex:`\emph{latent dimension}`. We have used
the pre-processed dataset, after the simulation and the autoencoding.
These datasets takes the form
:math:`\{ ({x}^{(i)}, y^{(i)}) \in \mathbb{R}^{2\ell} \times \{\mathrm{sm}, \mathrm{bsm}\}\}_{i=1}^m`.
The factor of :math:`2\ell` arises from the fact that we are studying
dijet events, where two jets collide with each other. As a result, we
have :math:`\ell` features for each jet.

For this tutorial, we have subsampled the smallest useful subset of this
huge dataset. It results in a background signal and one single BSM
signal.

Download
~~~~~~~~

We can load a HEP dataset. This dataset is proton collision and compares
the predictions of the standard model (background.npy) with the
prediction of A->HZ decay (signal_AtoH_to_ZZZ_subsampled.npy). These
files are a small subset of the true one, built for educational purposes
only.

.. code:: ipython3

    !curl "https://zenodo.org/records/10570949/files/background_subsampled.npy?download=1" --output 'background_subsampled.npy'


.. parsed-literal::

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  3328  100  3328    0     0   7068      0 --:--:-- --:--:-- --:--:--  7080


.. code:: ipython3

    !curl "https://zenodo.org/records/10570949/files/signal_AtoH_to_ZZZ_subsampled.npy?download=1" --output 'signal_AtoH_to_ZZZ_subsampled.npy'


.. parsed-literal::

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  3328  100  3328    0     0   6277      0 --:--:-- --:--:-- --:--:--  6279


.. code:: ipython3

    qX1 = np.load('background_subsampled.npy')
    qX2 = np.load('signal_AtoH_to_ZZZ_subsampled.npy')
    qX1 = qX1[:10,:]
    qX2 = qX2[:10,:]
    
    qX = np.row_stack([qX1, qX2])
    qy = np.array([-1] * len(qX1) + [1] * len(qX2))
    
    print(f"{qX.shape=}")
    print(f"{qy.shape=}")


.. parsed-literal::

    qX.shape=(20, 8)
    qy.shape=(20,)


Split the dataset in training and testing set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    qX_train, qX_test, qy_train, qy_test = train_test_split(qX, qy, test_size=0.2, random_state=42)
    
    print("Shape training set:", qX_train.shape, qy_train.shape)
    print("Shape testing set:", qX_test.shape, qy_test.shape)
    print("Example of feature:", qX_train[0])
    print("Example of label:", qy_train[0])


.. parsed-literal::

    Shape training set: (16, 8) (16,)
    Shape testing set: (4, 8) (4,)
    Example of feature: [-0.23470192 -0.49848634 -0.13048592  0.6318868  -0.5250736  -0.6175051
      0.08415551  0.43143862]
    Example of label: -1


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




.. raw:: html

    <style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SVC(kernel=&#x27;precomputed&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">SVC</label><div class="sk-toggleable__content"><pre>SVC(kernel=&#x27;precomputed&#x27;)</pre></div></div></div></div></div>



Finally, we get the accuracy of the model.

.. code:: ipython3

    # Test the model and calculate the score
    y_pred = model.predict(K_test)
    accuracy = np.sum(qy_test == y_pred) / len(qy_test)
    print("Accuracy:", accuracy)


.. parsed-literal::

    Accuracy: 1.0


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

