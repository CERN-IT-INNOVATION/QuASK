import pytest
import sys

sys.path.append("./src/")


from quask.core import Ansatz


def test_trivial():
    """ Trivial test. """
    print("Passed")


def test_ansatz_init():
    ansatz = Ansatz(n_features=2, n_qubits=2, n_operations=2, allow_midcircuit_measurement=False)
    assert ansatz is not None, "Could not create an Ansatz object."
