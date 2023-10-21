===================
Installation
===================

.. note::
    Ensure you have at least Python 3.10 installed.

Installation via `pip`
======================

_Quask_ has been designed to work on a Python3 environment and requires version 3.10 or higher. 
The easiest way to use _quask_ is by installing it in your Python3 environment via the _pip_ packet manager. Usually, this latter tool is already installed, however, we can check it is correctly installed via:

-- code-block:: sh

    python3 -m ensurepip --upgrade

You can install "QuASK" using `pip` by running the following command:

.. code-block:: sh

    pip install quask

Alternatively, you can download and install a specific release from GitHub:

.. code-block:: sh

    pip install https://github.com/CERN-IT-INNOVATION/QuASK/releases/download/1.0.0-beta/quask-1.0.0b0-py3-none-any.whl

Dependencies
============

QuASK has the following dependencies, which can be installed using `pip`:

- `dependency1`
- `dependency2`
- `dependency3`

To install these dependencies, run:

.. code-block:: sh

    pip install dependency1 dependency2 dependency3

Installation from Source
========================

To install "QuASK" from the source code, follow these steps:

1. Download the source code repository

   .. code-block:: sh

      git clone https://github.com/CERN-IT-INNOVATION/QuASK.git

2. Change to the QuASK source code directory:

   .. code-block:: sh

      cd quask

3. Install the required dependencies listed in `requirements.txt` using Python 3's `pip`:

   .. code-block:: sh

      python3 -m pip install -r requirements.txt

4. Install QuASK itself:

   .. code-block:: sh

      pip install .

Running with Docker
===================

To run "QuASK" using Docker, ensure Docker is installed on your system. Follow these steps:

1. Build a Docker image from the QuASK source code:

   .. code-block:: sh

      docker build -t quask .

2. Run a Docker container based on the image:

   .. code-block:: sh

      docker run quask

