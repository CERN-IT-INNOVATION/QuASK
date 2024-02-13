# Quantum Advantage Seeker with Kernels (QuASK)

QuASK is an actively maintained library for constructing, studying, and benchmarking quantum kernel methods.

It is designed to simplify the process of choosing a quantum kernel, automate the machine learning pipeline at all its stages, and provide pedagogical guidance for early-stage researchers to utilize these tools to their full potential.

QuASK promotes the use of reusable code and is available as a library that can be seamlessly integrated into existing code bases. It is written in Python 3, can be easily installed using pip, and is accessible on PyPI.


*Homepage*: [quask.web.cern.ch](https://quask.web.cern.ch/)

*Documentation*: [quask.readthedocs.io](https://quask.readthedocs.io/en/latest/)

## Installation 

The easiest way to use *quask* is by installing it in your Python3
environment (version >= 3.10) via the *pip* packet manager,

    python3 -m pip install -U quask==2.0.0-alpha1

You also need any quantum SDK installed on your system. For example, we can install Qiskit (but we can also work with Pennylane, Braket, Qibo, and the modular nature of the software allows the creation of your own custom backends).

    python3 -m pip install qiskit qiskit_ibm_runtime
    python3 -m pip install qiskit_ibm_runtime --upgrade
    python3 -m pip install qiskit-aer

See the [Installation section](https://quask.readthedocs.io/en/latest/installation.html) 
of our documentation page for more information.

## Examples

The fastest way to start developing using _quask_ is via our [Getting started](https://quask.readthedocs.io/en/latest/getting_started.html) guide.

If you are not familiar with the concept of kernel methods in classical machine learning, we have developed a [series of introductory tutorials](https://quask.readthedocs.io/en/latest/tutorials_classical/index.html) on the topic. 

If you are not familiar with the concept of quantum kernels, we have developed a [series of introductory tutorials](https://quask.readthedocs.io/en/latest/tutorials_quantum/index.html) on the topic, which is also used to showcase the basic functionalities of _quask_. 

Then [advanced features of _quask_](https://quask.readthedocs.io/en/latest/tutorials_quask/index.html) are shown, including the use of different backends, the criteria to evaluate a quantum kernel, and the automatic optimization approach.

Finally, [look here for some applications](https://quask.readthedocs.io/en/latest/tutorials_applications/index.html). 


## Source 


### Deployment to PyPI

The software is uploaded to [PyPI](https://pypi.org/project/quask/).

### Test

The suite of test for _quask_ is currently under development.To run the available tests, type 

    pytest


You can also specify specific test scripts.

    pytest tests/test_example.py

 _quask_ has been developed and tested with the following versions of the quantum frameworks: 

* PennyLane==0.32.0
* PennyLane-Lightning==0.32.0
* qiskit==0.44.1
* qiskit-aer==0.12.2
* qiskit-ibm-runtime==0.14.0


## Documentation 

The documentation is available at our [Read the Docs](https://quask.readthedocs.io/en/latest/) domain. 

### Generate the documentation

The documentation has been generated with Sphinx (v7.2.6) and uses the Furo theme. To install it, run

    python3 -m pip install -U sphinx
    python3 -m pip install furo

To generate the documentation, run

    cd docs
    make clean && make html

The Sphinx configuration file (`conf.py`) has the following, non-standard options:

    html_theme = 'furo'
    html_theme_options = {
        "sidebar_hide_name": True
    }
    autodoc_mock_imports = ["skopt", "skopt.space", "django", "mushroom_rl", "opytimizer", "pennylane", "qiskit", "qiskit_ibm_runtime", "qiskit_aer"]

### Generate the UML diagrams

Currently, the pages generated from the Python notebooks has to be compiled to RST format manually. We could use in the future the [nbsphinx extension](https://docs.readthedocs.io/en/stable/guides/jupyter.html) to automatize this process. This has the advantage that the documentation is always up to date, the disadvantage is that the process is much slower. 

### Generate the UML diagrams

The UML diagrams in the [Platform overview](https://quask.readthedocs.io/en/latest/platform_overview.html) page of the documentation are generated using pyreverse and Graphviz. They can be installed via:

    sudo apt-get install graphviz
    python3 -m pip install pylint

The UML diagrams are created via: 

    cd src/quask
    pyreverse -o png -p QUASK .


## Acknowledgements

The platform has been developed with the contribution of [Massimiliano Incudini](https://incud.github.io), Francesco Di Marcantonio, Davide Tezza, Roman Wixinger, Sofia Vallecorsa, and [Michele Grossi](https://scholar.google.com/citations?user=cnfcO7cAAAAJ&hl=en). 

If you have used _quask_ for your project, please consider citing us.

    @article{dimarcantonio2023quask,
        title={Quantum Advantage Seeker with Kernels (QuASK): a software framework to accelerate research in quantum machine learning},
        author={Di Marcantonio, Francesco and Incudini, Massimiliano and Tezza, Davide and Grossi, Michele},
        journal={Quantum Machine Intelligence},
        volume={5},
        number={1},
        pages={20},
        year={2023},
        publisher={Springer}
    }
