==============
How to use it
==============

Installation
==============

The software has been tested on Python 3.9.10. We recommend using this version or a newer one. 

The repository contains a ``requirements.txt`` file which can be installed 
by running from the main directory of the project using the command:

``python3 -m pip install -r requirements.txt``

You can install the software directly from the repository using the command:

```python3 -m pip install https://github.com/QML-HEP/quask/archive/master.zip```

or through PiP packet manager using the command:

``python3 -m pip install quask``

Usage
==============
Use quask as a library of software components
----------------------------

QuASK can be used as a library to extend your own software. Check if everything's working with:

```python
| import numpy as np 
| import quask.metrics 
| A = np.array([[1,2], [3,4]]) 
| B = np.array([[5,6], [7,8]]) 
| print(quask.metrics.calculate_frobenius_inner_product(A, B))  
```

Use quask as a command-line interface tool
----------------------------

QuASK can be used as a command-line interface to analyze the dataset with the
kernel methods. These are the commands implemented so far.

To retrieve the datasets available:

    $ python3.9 -m quask get-dataset

To preprocess a dataset:

    $ python3.9 -m quask preprocess-dataset

To analyze a dataset using quantum and classical kernels:

    $ python3.9 -m apply-kernel

To create some plot of the property related to the generated Gram matrices:

    $ python3.9 -m quask plot-metric --metric accuracy --train-gram training_linear_kernel.npy --train-y Y_train.npy --test-gram testing_linear_kernel.npy --test-y Y_test.npy --label linear
