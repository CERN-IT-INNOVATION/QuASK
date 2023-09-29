# QuASK  [![Made at CERN!](https://img.shields.io/badge/CERN-CERN%20openlab-brightgreen)](https://openlab.cern/) [![Made at CERN!](https://img.shields.io/badge/CERN-Open%20Source-%232980b9.svg)](https://home.cern) [![Made at CERN!](https://img.shields.io/badge/CERN-QTI-blue)](https://quantum.cern/our-governance)

## Quantum Advantage Seeker with Kernel

QuASK is a quantum machine learning software written in Python that 
supports researchers in designing, experimenting, and assessing 
different quantum and classic kernels performance. This software 
is package agnostic and can be integrated with all major quantum 
software packages (e.g. IBM Qiskit, Xanadu’s Pennylane, Amazon Braket).

QuASK guides the user through a simple preprocessing of input data, 
definition and calculation of quantum and classic kernels, 
either custom or pre-defined ones. From this evaluation the package 
provide an assessment about potential quantum advantage and prediction 
bounds on generalization error.

Beyond theoretical framing, it allows for the generation of parametric
quantum kernels that can be trained using gradient-descent-based 
optimization, grid search, or genetic algorithms. Projected quantum 
kernels, an effective solution to mitigate the curse of dimensionality 
induced by the exponential scaling dimension of large Hilbert spaces,
is also calculated. QuASK can also generate the observable values of
a quantum model and use them to study the prediction capabilities of
the quantum and classical kernels.

The initial release is accompanied by the journal article ["QuASK - Quantum
Advantage Seeker with Kernels" available on arxiv.org](https://arxiv.org/abs/2206.15284).

## Documentation

The documentation for QuASK can be accessed on the website [Read The Docs](https://quask.readthedocs.io/en/latest/index.html).

## Installation

The software has been tested on Python 3.9.10. We recommend using this version or a newer one. 

The library is available on the Python Package Index (PyPI) with ```pip install quask```.

## Usage

### Use quask as a library of software components

QuASK can be used as a library to extend your own software. Check if everything's working with:

```python
import numpy as np
import quask.metrics
A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])
print(quask.metrics.calculate_frobenius_inner_product(A, B))  # 70
```

### Use quask as a command-line interface tool

QuASK can be used as a command-line interface to analyze the dataset with the
kernel methods. These are the commands implemented so far.

To retrieve the datasets available:

    $ python3.9 -m quask get-dataset

To preprocess a dataset:

    $ python3.9 -m quask preprocess-dataset

To analyze a dataset using quantum and classical kernels:

    $ python3.9 -m quask apply-kernel

To create some plot of the property related to the generated Gram matrices:

    $ python3.9 -m quask plot-metric --metric accuracy --train-gram training_linear_kernel.npy --train-y Y_train.npy --test-gram testing_linear_kernel.npy --test-y Y_test.npy --label linear


## Credits

Please cite the work using the following Bibtex entry:

```text
@article{dimarcantonio2023quantum,
  title={Quantum Advantage Seeker with Kernels (QuASK): a software framework to speed up the research in quantum machine learning},
  author={Di Marcantonio, Francesco and Incudini, Massimiliano and Tezza, Davide and Grossi, Michele},
  journal={Quantum Machine Intelligence},
  volume={5},
  number={1},
  pages={20},
  year={2023},
  publisher={Springer}
}
```
