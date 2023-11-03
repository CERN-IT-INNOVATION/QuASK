===================
Installation
===================

In this section, we will show you how to install *quask* on your local platform. The software is meant to run
on Python 3 with a version equal to or higher than 3.8. The software has been tested on Ubuntu 22.04; however,
any platform supporting Python should be able to run *quask* successfully.

We recommend installing *quask* in a virtual environment. You can follow this guide or simply run:

.. code:: sh

   python3 -m venv my_quask_env
   source my_quask_env/bin/activate

Afterward, you can install *quask* using the *pip* package manager. You can check if you have the latest version of *pip* with:

.. code:: sh

   python3 -m ensurepip --upgrade

Then, install *quask* by running the command:

.. code:: sh

   python3 -m pip -U quask==2.0.0-alpha1

Finally, you need to install one or more quantum SDKs as a backend for *quask*. If you plan to work with Qiskit, run:

.. code:: sh

   python3 -m pip install qiskit qiskit-aer qiskit_ibm_runtime

If you plan to work with Pennylane, run:

.. code:: sh

   python3 -m pip install pennylane

If you plan to work with Amazon Braket, run:

.. code:: sh

   python3 -m pip install pennylane amazon-braket-sdk amazon-braket-pennylane-plugin

If you plan to work with Qibo, run:

.. code:: sh

   python3 -m pip install qibo

Dependencies
============

There are a few software dependencies used by a small subset of features that are not installed by default
with *quask*, mainly due to their significant space requirements. You can install them separately.

To support the creation of reinforcement learning agents for quantum kernel optimization, run:

.. code:: sh

   python3 -m pip install mushroom-rl
