==============
How to use it
==============

Installation
==============

The software has been tested on Python 3.9.10. We recommend using this version or a newer one.

You can install the software directly from the repository using the command:

``python3 -m pip install https://github.com/CERN-IT-INNOVATION/QuASK/releases/download/1.0.0-beta/quask-1.0.0b0-py3-none-any.whl``

If the software shows dependencies related problems, download the repository and from the main directory run the command:

``python3 -m pip install -r requirements.txt``

Usage
==============
Use quask as a library of software components
--------------------------------------------------------

QuASK can be used as a library to extend your own software. Check if everything's working with:

| ``python``
| ``import numpy as np``
| ``import quask.metrics``
| ``A = np.array([[1,2], [3,4]])``
| ``B = np.array([[5,6], [7,8]])`` 
| ``print(quask.metrics.calculate_frobenius_inner_product(A, B))``

Use quask as a command-line interface tool
--------------------------------------------------------

QuASK can be used as a command-line interface to analyze the dataset with the
kernel methods. These are the commands implemented so far.

To retrieve the datasets available:

     ``python3.9 -m quask get-dataset``

To preprocess a dataset:

    ``python3.9 -m quask preprocess-dataset``

To analyze a dataset using quantum and classical kernels:

    ``python3.9 -m apply-kernel``

To create some plot of the property related to the generated Gram matrices:

    ``python3.9 -m quask plot-metric --metric accuracy --train-gram training_linear_kernel.npy --train-y Y_train.npy --test-gram testing_linear_kernel.npy --test-y Y_test.npy --label linear``
