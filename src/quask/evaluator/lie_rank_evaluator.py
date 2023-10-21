import copy
import numpy as np
from typing import Set
from ..core import Kernel
from . import KernelEvaluator


class LieRankKernelEvaluator(KernelEvaluator):
    """
    Expressibility and 'Efficient classical simulability' measure based on the rank of the Lie algebra obtained by spanning
    the generators of the circuits.
    See: Larocca, Martin, et al. "Diagnosing barren plateaus with tools from quantum optimal control." Quantum 6 (2022): 824.
    """

    def __init__(self, T):
        """
        Initializer
        :param T: threshold T > 0 telling how is the minimum dimension of a 'hard-to-simulate' Lie algebra
        """
        super().__init__()
        self.T = T

    def evaluate(self, kernel: Kernel, K: np.ndarray, X: np.ndarray, y: np.ndarray):
        """
        Evaluate the current kernel and return the corresponding cost. Lower cost values corresponds to better solutions
        :param kernel: kernel object
        :param K: optional kernel matrix \kappa(X, X)
        :param X: datapoints
        :param y: labels
        :return: cost of the kernel, the lower the better
        """
        self.last_result = self.braket_generators(kernel, self.T)
        return -len(self.last_result)

    def braket_pair(self, a: str, b: str):
        """
        Calculate the commutator between two pauli matrices
        :param a: first Pauli (one of the strings 'I', 'X', 'Y', 'Z')
        :param b: second Pauli (one of the strings 'I', 'X', 'Y', 'Z')
        :return: [a, b]
        """
        assert a in ['I', 'X', 'Y', 'Z'] and b in ['I', 'X', 'Y', 'Z']
        if a == b: return 'I'
        if a == 'I': return b
        if b == 'I': return a
        return list(set(['X', 'Y', 'Z']).difference([a, b]))[0]

    def braket_strings(self, s1: str, s2: str):
        """
        Calculate the communtator between two pauli strings
        :param s1: first Pauli string
        :param s2: second Pauli string
        :return: [s1, s2]
        """
        assert len(s1) == len(s2), "Tha Pauli strings have different lengths"
        return [self.braket_pair(a, b) for (a, b) in zip(s1, s2)]

    def __braket_generators(self, initial_generators: Set[str], new_generators: Set[str]):
        """
        Return the set of generators obtained by commutating pairwise the elements in the given set
        :param initial_generators: first set of generators
        :param new_generators: second set of generators
        :return: generators obtained with the pairwise commutation of the given elements (only new ones)
        """
        out_generators = []
        for gen_new in new_generators:
            for gen_old in initial_generators:
                braket = "".join(self.braket_strings(gen_new, gen_old))
                if braket not in initial_generators and braket not in new_generators:
                    out_generators.append(braket)
        return set(out_generators)

    def get_initial_generators(self, kernel):
        """
        Create the initial generators of a kernel, i.e. for each operation apply the generator to the correct wires
        and identity everywhere else
        :param kernel: kernel object
        :return set of initial generators corresponding to the operations of the kernel
        """
        # get the generators of each operation
        generators = [kernel.ansatz.operation_list[i].generator for i in range(kernel.ansatz.n_operations)]
        # get the wires on which each operation acts
        wires = [kernel.ansatz.operation_list[i].wires for i in range(kernel.ansatz.n_operations)]
        initial_generators = []
        for i in range(kernel.ansatz.n_operations):
            # initialize each generator with identity everyone, as list of char and not as string (the latter is immutable)
            initial_generator = ['I'] * kernel.ansatz.n_qubits
            # assign the generator to each qubit
            q0, q1 = wires[i][0], wires[i][1]
            g0, g1 = generators[i][0], generators[i][1]
            initial_generator[q0] = g0
            initial_generator[q1] = g1
            # convert the list of char to string, now
            initial_generator = "".join(initial_generator)
            # print(f"{i=} {q0=} {q1=} {g0=} {g1=} {initial_generator=}")
            initial_generators.append(initial_generator)
            # print(f"{initial_generators}")
        return set(initial_generators)

    def braket_generators(self, kernel, T):
        """
        Return the basis of the lie algebra of the circuit defined by the kernel. The number of elements is truncated at T
        :param kernel: kernel object
        :param T: threshold
        :return: basis of the lie algebra of the generators in kernel
        """
        initial_generators = self.get_initial_generators(kernel)
        new_generators = copy.deepcopy(initial_generators)
        all_generators = copy.deepcopy(initial_generators)
        while len(all_generators) < T and len(new_generators) > 0:
            new_generators = self.__braket_generators(all_generators, new_generators)
            all_generators = all_generators.union(new_generators)
        return all_generators

    def __str__(self):
        return str(self.last_result)
