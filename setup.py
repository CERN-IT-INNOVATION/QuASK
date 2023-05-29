import setuptools
from setuptools import setup

setup(
    name='quask',
    version='1.0.0-beta',
    description='Quantum Advantage Seeker withKernels',
    url='https://github.com/CERN-IT-INNOVATION/QuASK',
    author='Massimiliano Incudini <massimiliano.incudini@univr.it>, Francesco Di Marcantonio <francesco.di.marcantonio@cern.ch>, Michele Grossi <michele.grossi@cern.ch>',
    license='Apache License Version 2.0',
    packages=setuptools.find_packages(),
    python_requires='>=3.7.0',
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'click',
        'scikit-learn',
        'imbalanced-learn',
        'prince',
        'openml',
        'jax',
        'jaxlib',
        'PennyLane',
        'PennyLane-qiskit',
        'simanneal',
        'optax',
    ],
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Unix',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3.9',
    ],
)
