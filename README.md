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

## Installation

The software requires Python3 version 3.9.10 or newer. The easiest way
to install the software is through PiP packet manager, using the command:

```python3 -m pip install quask```

or by downloading the software directly from this repository:

```python3 -m pip install https://github.com/QML-HEP/quask/archive/master.zip```

## Usage

## Credits

Please cite the work using the following Bibtex entry:

```text
@article{CitekeyArticle,
  author   = "P. J. Cohen",
  title    = "The independence of the continuum hypothesis",
  journal  = "Proceedings of the National Academy of Sciences",
  year     = 1963,
  volume   = "50",
  number   = "6",
  pages    = "1143--1148",
}
```


## Project name, project webpage and project GitHub repository

- [ ] use `main` branch for production-ready state only
- [ ] create `develop` branch for the latest delivered development changes for the next release
- [ ] create your development branch where each contributor works on a daily basis

##  Missing Requirements for the README
 Full description of the project
- [ ] Description of the project 
- [ ] How to install 
  - [ ] definition of virtual environment (anaconda/venv) used
  - [ ] instruction to install the package (requirements.txt or setup.cfg etc)
  - [ ] instruction how to run the code
- [ ] Quick start: minimal working example / tutorials / demos

##  Requirements for the CODE
- [ ] `requirements.txt` or `environment.yaml`(for conda) or `setup.cfg + pyproject.toml` or `setup.py`([setuptools](https://setuptools.pypa.io/en/latest/))
- [ ] `src/packagename` folder with source files
- [ ] formatting: production code must be formatted with [Black](https://github.com/psf/black)
- [ ] function annotations: augment all functions and modules with [dosctrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html)

##  Requirements before promoting the project from private to public
- [ ] `bibliography.md`: Zenodo link to external papers and datasets used
- [ ] Semantic versioning: comply with [semver.org](https://github.com/semver/semver/blob/master/semver.md) and [apache.org](https://apr.apache.org/versioning.html)
- [ ] documentation: using [readthedocs](https://docs.readthedocs.io/en/stable/tutorial/) and [simple formatting rules](https://hplgit.github.io/teamods/sphinx_api/html/sphinx_api.html). Please, use one of the following two standards: [Google's docstring](https://google.github.io/styleguide/pyguide.html) or [Numpy's docstring](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).
<!--
- [ ] [Sphynx](https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html) with Napoleon theme and Autodoc, include it in `docs` folder
-->
- [X] citation policy: how to use and cite the code (e.g. BibTex reference)
