"""
template_qiskit
================
This module collects functions to create the three encodings used in the 
'Power of Data in QML' paper. In particular Section 12.A of the Supplementary

"""

# =============================================================================
# Import modules
# =============================================================================
# Import general purpose module(s)
import functools as ft
from scipy.stats import rv_continuous
import numpy as np

# Import specific purpose module(s)
from qiskit import Aer
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import PauliFeatureMap
from qiskit.circuit.library import TwoLocal
from qiskit_machine_learning.kernels import QuantumKernel

from typing import Optional

"""
In the first part we define 4 encodings:
- single rotation on x (no entanglement - 1 reps)
- zz feature map normalized
- Heisenberg Trotter inspired to many-body physics problem
- dense feature map useful in NISQ era

Then we have a method that calls these encodings to form a quantum circuit

In the last part we define quantum kernels and projected kernels
"""

# =============================================================================
# Custom data map function
# =============================================================================
def custom_data_map_func(x):
    """
    custom data map function

    This function is used in the Encoding two (E2) to respect the normalization 
    convention used with the datasets. It replaces the default map function 
    used in ZZFeatureMap according to the 'Supervised learning with 
    quantum-enhanced feature space' paper.

    x: datapoint used to parametrize the feature map

    """
    coeff = x[0] if len(x) == 1 else ft.reduce(lambda m , n : m * n, x)
    return coeff


def single_rot_x_feature_map(qubits):

    feature_map = TwoLocal(num_qubits=qubits, reps=1, rotation_blocks=['rx'], 
                entanglement_blocks=None, entanglement='full',  
                skip_unentangled_qubits=False, 
                skip_final_rotation_layer=True, 
                parameter_prefix='x', insert_barriers=False, 
                initial_state=None)
    return feature_map

def zz_norm_feature_map(qubits):

    feature_map = PauliFeatureMap(feature_dimension=qubits, reps=2, paulis=['Z','ZZ'], 
            entanglement='linear', parameter_prefix='x', 
            data_map_func=custom_data_map_func, insert_barriers=False)
    return feature_map

# setting the Haar states
# =============================================================================
# Sine probability distribution
# =============================================================================
class sin_prob_dist(rv_continuous):
    """
    sin prob dist

    This class defines new varibles by subclassing the rv_continuous class
    and re-dfining the _pdf method.

    """
    def _pdf(self, theta):
        # The 0.5 is so that the distribution is normalized
        return 0.5 * np.sin(theta)


# =============================================================================
# Heisenberg Trotter model
# =============================================================================
class HTmodel(TwoLocal):
    """
    Heisenberg Trotter model

    This function produces the Encoding three (E3). It is an evolution
    Hamiltonian ansatz with Trotter steps. It has an initial state given by
    the haar random unitary.

    qubits: number of qubits of the state. It corresponds to the number of 
            features in the dataset + 1
    T: Trotter steps 

    """
    def __init__(
        self,
        num_qubits: Optional[int] = None,
        T: Optional[int] = 20,
        name: Optional[str] = "HT",
    ) -> None:

        super().__init__(name=name,
                        num_qubits = None,
                        )

        self.num_qubits = num_qubits 
        self._data = None
        self.T = T

    # =============================================================================
    # Haar random unitary
    # =============================================================================
    def haar_random_unitary(self, qubit, sampler, ancilla_circuit):
        """
        haar random unitary

        This function applies a unitary gate to each qubit in order to initialize
        the state into the so-called "Haar random states". This is done by
        passing into the unitary random values for the angles that will rotate
        the ground state.

        qubit: qubit where the unitary u(theta,phi,lambda) is applied
        sampler: variable defined thanks to the sin_prob_dist class to have 
                unifrom sampling of the variable theta
        ancilla_circuit: ancilla quantum circuit used to apply u(theta,phi,lambda)

        """
        phi, lam = 2 * np.pi * np.random.uniform(size=2) # Sample phi and omega as normal
        theta = sampler # Sample theta from our new distribution
        ancilla_circuit.u(theta,phi,lam,qubit)
        return ancilla_circuit

    def heisenberg_trotter_feature_map(self):
        # setting the parametrized gates
        qent = QuantumCircuit(self.num_qubits+1) # quantum circuit used for entanglement
        qcirc = QuantumCircuit(self.num_qubits+1) # quantum circuit used for trotterization
        params_ent = ParameterVector('x', self.num_qubits)
        t = self.num_qubits/3
        # entanglement block
        for j in range(self.num_qubits):
            qent.rzz(t/self.T*params_ent[j],j,j+1)
            qent.ryy(t/self.T*params_ent[j],j,j+1)
            qent.rxx(t/self.T*params_ent[j],j,j+1)

        # trotterization block: we repeat the above circuit T times
        for i in range(self.T):
            qcirc.compose(qent,inplace=True)
        
        
        # Samples of theta should be drawn between 0 and pi
        sin_sampler = sin_prob_dist(a=0, b=np.pi)
        aqci = QuantumCircuit(self.num_qubits+1) # ancilla quantum circuit to set the Haar states

        for i in range(self.num_qubits+1):
            haar = self.haar_random_unitary(i, sin_sampler.rvs(size=1), aqci)
        
        feature_map = haar.compose(qcirc)
        
        return feature_map

def dense_feature_map(nqubits, nfeatures) -> QuantumCircuit:
    """
    Constructs the u2Reuploading feature map.
    @nqubits   :: Int number of qubits used.
    @nfeatures :: Number of variables in the dataset to be processed.
    returns :: The quantum circuit object form qiskit.
    """
    x = ParameterVector("x", nfeatures)
    qc = QuantumCircuit(nqubits)
    for feature, qubit in zip(range(0, 2 * nqubits, 2), range(nqubits)):
        qc.u(np.pi / 2, x[feature], x[feature + 1], qubit)  # u2(φ,λ) = u(π/2,φ,λ)
    for i in range(nqubits):
        if i == nqubits - 1:
            break
        qc.cx(i, i + 1)
    for feature, qubit in zip(range(2 * nqubits, nfeatures, 2), range(nqubits)):
        qc.u(np.pi / 2, x[feature], x[feature + 1], qubit)

    for feature, qubit in zip(range(0, 2 * nqubits, 2), range(nqubits)):
        qc.u(x[feature], x[feature + 1], 0, qubit)

    return qc


# =============================================================================
# Encoding
# =============================================================================
def encoding(enc, qubits, observable=False):
    """
    encoding

    This function gives in return a quantum circuit with the selected encoding.
    The circuit is not parametrized yet with numerical values.

    enc: string indicating which encoding to implement
    qubits: n° of qubits, thus of features_DIM, to use in the Quantum Circuit.
    pennylane: Defines the encodings in PennyLane instead on qiskit.
    observable: By default False gives the simple encoded circuit. If True
                the circuit will be created with an additional qubit in order
                to have matching dimensions with the QNN circuit used in 
                observable mesurements.

    """
    
    # fix the qubit = 1 case since can be problematic for some encodings
    if qubits == 1 and enc == 'E2':
        return print('Feature map E2 not suited for 1 qubit')
    else:
        
        # Single qubit wall of rotations along x axis (of the Bloch sphere) 
        if enc == 'E1':
            feature_map = single_rot_x_feature_map(qubits)
            
        # ZZFeatureMap adapred for our normalization convention
        elif enc == 'E2':
            feature_map = zz_norm_feature_map(qubits)

        # Hamiltonian evolution ansatz initialized with random Haar states
        elif enc == 'E3':
            T = 20 # Trotter steps

            feature_map = HTmodel(num_qubits=qubits, T=T)
            feature_map = feature_map.heisenberg_trotter_feature_map()
            # print(feature_map)

        elif enc == 'dense':
            feature_map = dense_feature_map(nqubits=round(qubits/2), nfeatures=qubits)

        # the True condition imposes the circuit to aquire an extra qubit
        if observable:
            qc_comp = QuantumCircuit(qubits+1)
            return  qc_comp.compose(feature_map)
        else:
            return feature_map

def qiskit_quantum_kernel(feature_map, X_1, X_2=None):
    """
    Create a Quantum Kernel given the template written in Qiskit framework
    :param feature_map: Qiskit template for the quantum feature map
    :param X_1: First dataset
    :param X_2: Second dataset
    :return: Gram matrix
    """
    if X_2 == None: X_2 = X_1  # Training Gram matrix
    assert X_1.shape[1] == X_2.shape[1], "The training and testing data must have the same dimensionality"

    bcknd = Aer.get_backend("statevector_simulator")

    qu_kernel = QuantumKernel(feature_map, quantum_instance=bcknd)

    gram = qu_kernel.evaluate(x_vec=X_1, y_vec=X_2)  # if y_vec=None the function calculates self inner product with x_tr
    # scaling of the kernel
    if gram.trace() != X_1.shape[0]:
        gram = gram * X_1.shape[0] / gram.trace()
    

def qiskit_projected_feature_map(feature_map, X):
    """
    Create a Quantum Kernel given the template written in Qiskit framework
    :param feature_map: Qiskit template for the quantum feature map
    :param X: First dataset
    :return: projected quantum feature map X
    """
    pass