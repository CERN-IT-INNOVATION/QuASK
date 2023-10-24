import functools
import itertools

import numpy as np
from scipy.linalg import expm, logm


class Pauli:
    """
    Class representing a Pauli string with its operations
    """
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    sigma_id = np.eye(2)
    sigma_map = {'I': sigma_id, 'X': sigma_x, 'Y': sigma_y, 'Z': sigma_z}

    def __init__(self, phase, string):
        """
        Create a Pauli object
        :param phase: phase
        :param string: pauli string
        """
        assert set
        self.phase = phase
        self.string = string

    @classmethod
    def from_string_iterable(cls, string_iterable):
        return set(Pauli(1, string) for string in string_iterable)

    @staticmethod
    def commute_pair(a: str, b: str):
        """
        Commute a pair of Paulis
        :param a: 'I', 'X', 'Y', or 'Z'
        :param b: 'I', 'X', 'Y', or 'Z'
        :return: (1, 'I') if they commute, (phase, P) otherwise
        """
        # identity commute with everything
        if a == 'I': return (1, b)
        if b == 'I': return (1, a)
        # any item commutes with itself
        elif a == b: return (1, 'I')
        # cyclic rule
        elif (a, b) == ('X', 'Y'): return (1j, 'Z')
        elif (a, b) == ('Y', 'Z'): return (1j, 'X')
        elif (a, b) == ('Z', 'X'): return (1j, 'Y')
        # flip commutator and negate results
        elif (b, a) == ('X', 'Y'): return (-1j, 'Z')
        elif (b, a) == ('Y', 'Z'): return (-1j, 'X')
        elif (b, a) == ('Z', 'X'): return (-1j, 'Y')
        assert False, "End of the function reached"

    def __mul__(self, other):
        """
        Multiplication of two pauli strings
        :param other:
        :return:
        """
        assert len(self.string) == len(other.string), "Pauli strings must have the same length"
        p1, s1 = self.phase, self.string
        p2, s2 = other.phase, other.string
        s3 = [Pauli.commute_pair(a, b) for (a, b) in zip(s1, s2)]
        phase = functools.reduce(lambda acc, item: acc * item[0], s3, p1 * p2)
        string = functools.reduce(lambda acc, item: acc + item[1], s3, '')
        return Pauli(phase, string)

    def __floordiv__(self, other):
        """
        Commutator of two pauli strings
        :param other:
        :return:
        """
        t1 = self * other
        t2 = other * self
        assert t1.string == t2.string, "The Pauli strings must be equal"
        return Pauli(np.round((t1.phase - t2.phase)/2), t1.string)

    def __repr__(self):
        return f"{self.phase} {self.string}"

    def to_matrix(self, keep_phase=False):
        phase = self.phase if keep_phase else 1
        matrix = functools.reduce(lambda acc, pauli: np.kron(acc, Pauli.sigma_map[pauli]), self.string, np.array([1]))
        return phase * matrix


class Unitary:

    @staticmethod
    def decompose_hamiltonian(hamiltonian):
        n = int(np.rint(np.log2(hamiltonian.shape[0])))
        hamiltonian_decomposition = {}
        for pauli_string in itertools.product(['I', 'X', 'Y', 'Z'], repeat=n):
            pauli_string = "".join(pauli_string)
            coeff = np.round(np.trace(hamiltonian @ Pauli(1, pauli_string).to_matrix()), decimals=4)
            hamiltonian_decomposition[pauli_string] = coeff
        return hamiltonian_decomposition

    @staticmethod
    def decompose_unitary(unitary):
        hamiltonian = 1j * logm(unitary)
        return Unitary.decompose_hamiltonian(hamiltonian)

    @staticmethod
    def encode_unitary(phase, pauli):
        return expm(-1j * phase * pauli.to_matrix())

    @staticmethod
    def encode_unitary_from_ham_sum(phases, paulis):
        n = len(phases)
        return expm(-1j * np.sum(phases[i] * paulis[i].to_matrix() for i in range(n)))


def span_new_strings(steady_terms, new_terms):
    generated_terms = []
    for term1 in steady_terms:
        for term2 in new_terms:
            term3 = commute_strings(term1, term2)
            if not np.isclose(term3[0], 0) and term3 not in steady_terms and term3 not in new_terms:
                generated_terms.append(term3)
    return generated_terms


def span_strings(initial_terms, keep_phase=True):
    all_terms = set(initial_terms)
    last_terms = set(initial_terms)
    last_size = 0
    while len(all_terms) > last_size:
        last_size = len(all_terms)
        print(f"{last_terms=}")
        last_terms = span_new_strings(all_terms, last_terms)
        all_terms = all_terms.union(last_terms)
    return all_terms


def string_to_matrix():
    pauli_matrices = [sigma_map[p] for p in self.pauli_string]
    result = np.array([1])
    for p in pauli_matrices:
        result = np.kron(result, p)
    return self.coeff * result