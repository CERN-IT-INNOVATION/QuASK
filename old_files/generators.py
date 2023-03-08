import numpy as np

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_id = np.eye(2)
sigma_map = {'I': sigma_id, 'X': sigma_x, 'Y': sigma_y, 'Z': sigma_z}


class ProjectedPauli:

    paulis_list = set(['X', 'Y', 'Z'])

    @staticmethod
    def c_single(p1, p2):
        assert p1 in ProjectedPauli.paulis_list and p2 in ProjectedPauli.paulis_list
        if p1 == 'I':
            return p2
        elif p2 == 'I':
            return p1
        elif p1 == p2:
            return 'I'
        else:
            return list(ProjectedPauli.paulis_list.difference([p1, p2]))[0]

    @staticmethod
    def c_string(ps1, ps2):
        return "".join(ProjectedPauli.c_single(p1, p2) for p1, p2 in zip(ps1, ps2))

prod_map = {
    ('I', 'I'): (1, 'I'),
    ('I', 'X'): (1, 'X'),
    ('I', 'Y'): (1, 'Y'),
    ('I', 'Z'): (1, 'Z'),
    ('X', 'I'): (1, 'X'),
    ('X', 'X'): (1, 'I'),
    ('X', 'Y'): (1j, 'Z'),
    ('X', 'Z'): (-1j, 'Y'),
    ('Y', 'I'): (1, 'Y'),
    ('Y', 'X'): (-1j, 'Z'),
    ('Y', 'Y'): (1, 'I'),
    ('Y', 'Z'): (1j, 'X'),
    ('Z', 'I'): (1, 'Z'),
    ('Z', 'X'): (1j, 'Y'),
    ('Z', 'Y'): (-1j, 'X'),
    ('Z', 'Z'): (1, 'I'),
}


class PauliString:

    def __init__(self, coeff, pauli_string):
        assert set(pauli_string).difference(set("IXYZ")) == set(), "Some unallowed character present"
        self.coeff = coeff
        self.pauli_string = pauli_string

    def commutes_with(self, rhs):
        """
        Return true if the two Pauli strings commute
        :param rhs: second pauli string
        :return: True if [A, B] = 0
        """
        count = 0
        for a, b in zip(self.pauli_string, rhs.pauli_string):
            if a != b and a != 'I' and b != 'I':
                count += 1
        return count % 2 == 0

    def commute(self, rhs):
        """

        :param rhs:
        :return:
        """
        result = PauliString(self.coeff * rhs.coeff, "")
        for p1, p2 in zip(self.pauli_string, rhs.pauli_string):
            (c, p1p2) = prod_map[p1, p2]
            result.coeff *= c
            result.pauli_string += p1p2
        return result

    def commute_matrices(self, rhs):
        A = self.to_matrix()
        B = rhs.to_matrix()
        return A @ B - B @ A

    def to_matrix(self):
        pauli_matrices = [sigma_map[p] for p in self.pauli_string]
        result = np.array([1])
        for p in pauli_matrices:
            result = np.kron(result, p)
        return self.coeff * result

    def __str__(self):
        return self.pauli_string

    def __repr__(self):
        return self.pauli_string

    def __hash__(self):
        return self.coeff.__hash__() * self.pauli_string.__hash__()


def check(pauli_string_1, pauli_string_2):
    A1 = pauli_string_1.commute(pauli_string_2).to_matrix()
    A2 = pauli_string_1.commute_matrices(pauli_string_2)
    return np.allclose(A1 / np.max(A1), A2 / np.max(A2))


def check_str(str_1, str_2):
    pauli_string_1 = PauliString(1, str_1)
    pauli_string_2 = PauliString(1, str_2)
    A1 = pauli_string_1.commute(pauli_string_2).to_matrix()
    A2 = pauli_string_1.commute_matrices(pauli_string_2)
    return np.allclose(A1 / np.max(A1), A2 / np.max(A2))


xzx = PauliString(1, "XZX")
xzz = PauliString(1, "XZZ")
xzz = PauliString(1, "XZZ")
x = PauliString(1, "X")
y = PauliString(1, "Y")
z = PauliString(1, "Z")


def span_operators(list_pauli_strings):
    old_list = list_pauli_strings
    current_list = list_pauli_strings
    new_list = []
    for P1 in current_list:
        for P2 in old_list:
            if not P1.commutes_with(P2):
                P3 = P1.commute(P2)
                if P3 not in old_list and P3 not in current_list:
                    print(f"[{P1},{P2}] = {P3}")
                    new_list.add(P3)
        current_list = new_list