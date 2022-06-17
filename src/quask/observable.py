"""
observable
================
This module collects functions to create quantum lables/target function
according to the Section 12.D.2 of the 'Power of data in QML' Supplementary. 

"""

# =============================================================================
# Import modules
# =============================================================================
# Import general purpose module(s)
import time
import numpy as np

# Import specific purpose module(s)
from qiskit.circuit import QuantumCircuit,  ParameterVector
import qiskit.quantum_info as qi
from qiskit.quantum_info.operators import Operator, Pauli
import pennylane as qml
from .template_qiskit import encoding
from .utils import display_time, gather_file, save_file, pennylane_node, obs_pennylane


# =============================================================================
# Quantum Neural Network
# =============================================================================
def QNN(T1, qubits, parametrized=False):
    """
    Quantum Neural Network

    This function creates and appends a quantum neural network to the selected
    encoding. It follows formula S(116) in the Supplementary.

    T1: n° of Trotter steps
    qubits: n° of qubits used in the current encoding
    parametrized: if True gives random parameters from a gaussian distribution. 
                  By default False
    
    """
    # ancilla quantum circuit
    qc = QuantumCircuit(qubits+1)

    qent_qnn = QuantumCircuit(qubits+1) # quantum circuit for entanglement
    params_ent_qnn = ParameterVector('x', qubits)
    # entanglement block
    for j in range(qubits):
        qent_qnn.rzz(params_ent_qnn[j],j,j+1)
        qent_qnn.ryy(params_ent_qnn[j],j,j+1)
        qent_qnn.rxx(params_ent_qnn[j],j,j+1)
    # trotterization block
    for _ in range(T1):
        qc.compose(qent_qnn,inplace=True) # inplace keeps the qent_qnn appended to qc  

    if parametrized:
        param_vec = [np.random.normal()]*qc.num_parameters
        # assing parameters to the qnn circuit
        qc = qc.assign_parameters(param_vec) 

    return qc

# =============================================================================
# Observables
# =============================================================================
def observables(datatype, N_TRAIN, features_DIM, enc, pennylane=True):
    """
    observables

    This function gives the expectation value of the QNN evolution of a choosen observable 
    (Z over the first qubit), with respect to the quantum state defined by the enconding 
    strategy selected. It resembles forumla S(117) of the Supplementary.

    datatype: string specifing if we use a training or test dataset
    N_TRAIN: n° of datapoints in the training dataset
    features_DIM: n° of features for each datapoint. It is equal to the number of
                  parametrized qubits
    enc: encoding strategy selected

    """
    
    # gather dataset by fixing the encoding to 'C'
    data = gather_file(f'x_{datatype}_mnist_{N_TRAIN}_{features_DIM}', '../data')
    
    # embed the classical dataset in a quantum circuit through a specific encoding
    # set observable to True in this case
    feature_map = encoding(enc, features_DIM, observable=True)
    if feature_map is None:
        return
    else:
        # when features_DIM = 1 each element of the dataset is of the type numpy.float32
        # but we need it to be a list/numpy.array to pass in the assign_parameters() function
        if features_DIM == 1:
            data = [[k] for k in data]
        
        # recall the quantum neural network
        qnn_map_parametrized = QNN(10, features_DIM, parametrized=True)   
        exp_val = []
        
        # take track of the time required to evaluate the expectation value for a given embedding strategy
        print('=================================')
        print('Computing observables for quantum labels:')
        print(f'\n{enc} encoding, {datatype}, n of qubits {features_DIM}\n')
        time_start = time.perf_counter()
        
        # loop over the dataset 
        if datatype == 'train':
            for i in range(N_TRAIN):
                # assign parameters to the embedded circuit
                enc_map_parametrized = feature_map.assign_parameters(data[i])
                # compose the encoded and qnn parametrized circuits
                circuit = (enc_map_parametrized).compose(qnn_map_parametrized)

                if pennylane:
                    pl_circ = qml.from_qiskit(circuit.decompose())
                    pl_node = pennylane_node(obs_pennylane, wires=circuit.num_qubits, jax=True)
                    exp_val.append(pl_node(pl_circ))
            
                else:
                    # get the density matrix
                    rho = qi.DensityMatrix.from_instruction(circuit)
                    # get the expectation value of the observable on the quantum state defined by rho
                    exp_val.append(round(rho.expectation_value(Operator(Pauli('I'*(features_DIM)+'Z'))).real, 4))

        elif datatype == 'test':
            for i in range(round(N_TRAIN*0.2)):
                # assign parameters to the embedded circuit
                enc_map_parametrized = feature_map.assign_parameters(data[i])
                # compose the encoded and qnn parametrized circuits
                circuit = (enc_map_parametrized).compose(qnn_map_parametrized)
                if pennylane:
                    pl_circ = qml.from_qiskit(circuit.decompose())
                    pl_node = pennylane_node(obs_pennylane, wires=circuit.num_qubits, jax=True)
                    exp_val.append(pl_node(pl_circ))

                else:
                    # get the density matrix
                    rho = qi.DensityMatrix.from_instruction(circuit)
                    # get the expectation value of the observable on the quantum state defined by rho
                    exp_val.append(round(rho.expectation_value(Operator(Pauli('I'*(features_DIM)+'Z'))).real, 4))

        display_time(time_start)
        print('=================================\n')
        # save the observable in the folder /data    
        save_file(exp_val, f'y_quantum_{datatype}_{N_TRAIN}_{features_DIM}_{enc}', '../data')
        return exp_val
