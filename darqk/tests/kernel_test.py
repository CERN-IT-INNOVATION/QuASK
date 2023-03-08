from darqk import Operation, Ansatz, KernelType, PennylaneKernel
import numpy as np


def test_static_single_qubit(KernelClass):

    # circuit:  |0> - RX(pi) -
    #           |0> - ID     -
    ansats = Ansatz(n_features=1, n_qubits=2, n_operations=1)
    ansats.initialize_to_identity()
    ansats.change_generators(0, "XI")
    ansats.change_feature(0, -1)
    ansats.change_wires(0, [0, 1])
    ansats.change_bandwidth(0, 1)

    # measurement operation = <1|Z|1>
    # probabilities: [0.0, 1.0]
    # observable: 0.0 * (+1) + 1.0 * (-1) = -1.0
    kernel = KernelClass(ansats, "ZI", KernelType.OBSERVABLE)
    x = kernel.phi(np.array([np.inf]))
    assert np.allclose(kernel.get_last_probabilities(), np.array([0, 1])), f"Incorrect measurement: {kernel.get_last_probabilities()}"
    assert np.isclose(x, -1), "Incorrect observable"

    # measurement operation = <1|X|1> = <1H|Z|H1> = <+|Z|+>
    # probabilities: [0.5, 0.5]
    # observable: 0.5 * (+1) + 0.5 * (-1) = 0.0
    kernel = KernelClass(ansats, "XI", KernelType.OBSERVABLE)
    x = kernel.phi(np.array([np.inf]))
    assert np.allclose(kernel.get_last_probabilities(), np.array([0.5, 0.5])), f"Incorrect measurement: {kernel.get_last_probabilities()}"
    assert np.isclose(x, 0), "Incorrect observable"

    # measurement operation = <1|Y|1> = <1HSdag|Z|SdagH1> = <[1/sqrt(2), -i/sqrt(2)]|Z|[1/sqrt(2), -i/sqrt(2)]>
    # probabilities: [0.5, 0.5]
    # observable: 0.5 * (+1) + 0.5 * (-1) = 0.0
    kernel = KernelClass(ansats, "YI", KernelType.OBSERVABLE)
    x = kernel.phi(np.array([np.inf]))
    assert np.allclose(kernel.get_last_probabilities(), np.array([0.5, 0.5])), f"Incorrect measurement: {kernel.get_last_probabilities()}"
    assert np.isclose(x, 0), "Incorrect observable"


def test_static_two_qubit(KernelClass):

    # circuit:  |0> - XY(#0) -
    #           |0> - XY(#0) -
    ansats = Ansatz(n_features=1, n_qubits=2, n_operations=1)
    ansats.initialize_to_identity()
    ansats.change_generators(0, "XY")
    ansats.change_feature(0, 0)
    ansats.change_wires(0, [0, 1])
    ansats.change_bandwidth(0, 1)

    kernel = KernelClass(ansats, "ZZ", KernelType.OBSERVABLE)
    x = kernel.phi(np.array([np.pi / 2]))
    assert np.allclose(kernel.get_last_probabilities(), np.array([0.5, 0.0, 0.0, 0.5])), "Incorrect measurement"
    assert np.isclose(x, 0), "Incorrect observable"


test_static_single_qubit(PennylaneKernel)
test_static_two_qubit(PennylaneKernel)
