"""
utils
================
This module collects functions that can be used in multiple situations

"""

# =============================================================================
# Import modules
# =============================================================================
# Import general purpose module(s)
import time
import numpy as np
import matplotlib.pylab as plt

# Import qiskit module(s)
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi

# Import pennylane module(s)
import pennylane as qml

# =============================================================================
# Pennylane node
# =============================================================================
def pennylane_node(q_func, wires, jax=True):
    """
    Pennylane node

    This function create a quantum node in pennylane for a quantum circuit.
    It uses the jax interface.

    q_func: quantum function that defines the quantum circuit in pennylane
    wires: number of qubits of the pennylane circuit
    jax: by default it is True. It defines the device interface to jax

    """
    if jax:
        # define the device with jax
        dev = qml.device("default.qubit.jax", wires)
        # define and return the quantum node with jax interface for the given function
        q_circuit = qml.QNode(q_func, dev, interface="jax")
    else:
        # define the device
        dev = qml.device("default.qubit", wires)
        # define and return the quantum node for the given function
        q_circuit = qml.QNode(q_func, dev)
    return q_circuit


# =============================================================================
# Quantum Circuit
# =============================================================================
def q_circuit(qc, k):
    """
    Quantum Circuit

    This function creates a quantum circuit in pennylane using a predefined
    quantum circuit qc. It is helpful if we want to import circuits directly from qiskit.
    Use it in conjunction with qml.from_qiskit(qc.decompose()).
    Then it computes the reduced density matrix of the circuit on the qubit k.

    k: qubit where to compute the local density matrix
    qc: quantum circuit. Can be imported from qiskit

    """
    qc()
    return qml.density_matrix([k])

# =============================================================================
# Observable Pennylane
# =============================================================================
def obs_pennylane(qc, k=0):
    """
    Observable pennylane

    This function uses a quantum circuit qc to compute an observable on it.
    In particular we perform a PauliZ local observable on the k-th qubit.

    k: qubit where to perform the local measurement. By default we measure 
       on the first qubit.
    qc: quantum circuit. Can be imported from qiskit

    """
    qc()
    return qml.expval(qml.PauliZ([k]))


# =============================================================================
# display time
# =============================================================================
def display_time(time_start):
    """
    display time

    This function is a tool to display easily the computation time.

    time_start: when the computation started

    """
    time_elapsed = time.perf_counter() - time_start
    if time_elapsed < 60:
        print("%5.1f secs" % time_elapsed)
        return time_elapsed
    elif time_elapsed > 60 and time_elapsed < 3600:
        print("%5.1f mins" % round(time_elapsed / 60, 2))
        return time_elapsed
    elif time_elapsed > 3600:
        print("%5.1f hours" % round(time_elapsed / 3600, 2))
        return time_elapsed

# =============================================================================
# Save file
# =============================================================================
def save_file(datum, filename, folder, fmt="%.18e"):
    """
    save file

    This function saves a generic numpy data with a given filename.

    datum: numpy file to be saved
    filename: name given to the datum to save
    folder: where to save the file

    """
    np.savetxt(f"{folder}/{filename}", datum, fmt)


# =============================================================================
# Gather file
# =============================================================================
def gather_file(filename, folder, compl=False):
    """
    gather file

    This function gathers files with the name filename from a specified folder.

    filename: name of the file to gather
    folder: name of the folder where to gather the file

    """
    if folder is None:
        return np.loadtxt(filename)
    else:
        if compl:
            return np.loadtxt(f"{folder}/{filename}", dtype=np.complex_)
        else:
            return np.loadtxt(f"{folder}/{filename}")


# =============================================================================
# Reconstruct array
# =============================================================================
def rec_array(array):
    """
    reconstruct array

    This function reconstruct the array that was flattened during the saving.
    It has saved, in its last slots, the how many dimensions and their values
    for the original array.

    array: array to reconstruct after the flattening required to save it

    """
    dims = []
    # in the last value we have the n° of dimensions
    len_dims = list(range(round(array[-1].real)))
    len_dims.reverse()

    # save the dimensions before deleting the last values from the array
    for i in len_dims:
        dims.append(round(array[-i - 2].real))

    # delete the dimensions to properly reshape the array
    for i in range(len_dims[0] + 2):
        array = np.delete(array, -1)

    # reshape
    array = np.reshape(array, (dims))
    return array

# =============================================================================
# reduced density matrix k-th qubit
# =============================================================================
def rdm_k(rho, k, qubits, subsys=False):
    """
    Reduced density matrix k-th qubit

    This function computes the reduced density matrix for the qubit k
    by tracing out the other qubits in the quantum circuit represented
    by the density matrix rho.

    rho: global density matrix of the circuit
    k: qubit to take the local density matrix
    qubits: qubits in the quantum circuit
    subsys: It returns the list of the subsystem used to trace out

    """
    # create an array with all the qubits
    ray = np.arange(qubits)
    # delet the k-th qubit to the array to trace out
    a = np.delete(ray, k)
    # the function partial_trace needs a list object
    b = list(a)
    # tracing out the all the states different to k from our rho
    red_rho = qi.partial_trace(rho, b)
    if subsys:
        return red_rho._data, b
    else:
        return red_rho._data


# =============================================================================
# Reduce density matrix
# =============================================================================
def rdm(
    x,
    i,
    pl_node=qml.QNode,
    enc_map=QuantumCircuit,
    pennylane=False,
    multpr=False,
    subsys=False,
):
    """
    reduced density matrix

    This function computes the reduced density matrix for the datum
    x_i for all qubits of the circuit, given a specific enconding enc.

    x: training or test datapoint. Its lenght represents the number of features
    i: position of the datapoint in the original dataset, useful to sort results
       in multiprocessing
    datatype: type of dataset used: 'train' or 'test'
    enc: encoding that embbeds the datapoint
    dataset_type: which dataset is used: 'higgs', 'mnist', 'SM'...

    """
    # to check the time of a computation
    # time_start = time.perf_counter()
    
    qubits = enc_map.num_qubits
    
    rdm_tot = []
    # for batch processing use the commented part below. x is the batch
    if multpr:
        print(f'start {i} in rdm')
        for data in x:

            if pennylane:

                circ_param = enc_map.bind_parameters(data).decompose()
                pl_enc_map = qml.from_qiskit(circ_param)
                red_rho_k = []
                for k in range(qubits):
                    red_rho = pl_node(pl_enc_map, k)
                    red_rho_k.append(red_rho)
                rdm_tot.append(red_rho_k)
            else:

                circ_param = enc_map.bind_parameters(data)
                rho = qi.DensityMatrix(circ_param)
                red_rho_k = []

                for k in range(qubits):

                    if subsys:
                        # to define what to do with b
                        red_rho, b = rdm_k(rho, k, qubits)
                    else:
                        red_rho = rdm_k(rho, k, qubits)
                    red_rho_k.append(red_rho)
                rdm_tot.append(red_rho_k)
                # print(f'\n {np.asarray(rdm_tot).shape} \n')
        print(f'end {i} in rdm')
        
        return rdm_tot, i

    else:

        if pennylane:

            circ_param = enc_map.bind_parameters(x).decompose()
            pl_enc_map = qml.from_qiskit(circ_param)

            red_rho_k = []
            for k in range(qubits):
                red_rho = pl_node(pl_enc_map, k)
                rdm_tot.append(red_rho)

        else:

            circ_param = enc_map.bind_parameters(x)
            rho = qi.DensityMatrix(circ_param)

            for k in range(qubits):
                if subsys:
                    # to define what to do with b
                    red_rho, b = rdm_k(rho, k, qubits)
                else:
                    red_rho = rdm_k(rho, k, qubits)

                rdm_tot.append(red_rho)
        

        return rdm_tot

# =============================================================================
# Product and trace
# =============================================================================
def prod_and_trace(x, y, i=None, multpr=False):
    """
    Product and trace

    This function computes the product between two matrices x and y and than
    gets the trace of the final matrix saving only the real part.
    It can be used to compute the expectation value between two quantum states
    represented by global or local density matrices x,y.

    x: matrix
    y: matrix

    """
    # make the product and than trace of the final matrix given by x and y
    pat = (x @ y).trace().real
    if multpr:
        return pat, i
    else:
        return pat


# =============================================================================
# Sum traces
# =============================================================================
def sum_traces(x, y, qubits):
    """
    Sum traces

    This function sum traces given by the product of two rdms x,y. The x file
    is an array containing the reduced density matrices for each qubit extracted
    from the quantum circuit encoding the whole datapoint x.

    x,y: arrays of rdms. we have (qubits * (2x2)) matrices
    qubits: total number of qubits of the global density matrix

    """
    # with concurrent.futures.ThreadPoolExecutor() as executor:

    # define an array where to save the traces
    trace_sum = []
    # loop over the qubits to get the traces
    for k in range(qubits):
        # t_pat = executor.submit(prod_and_trace, x[k], y[k])
        # result = t_pat.result()

        # multiply and trace the k-th rdm for x and y
        result = prod_and_trace(x[k], y[k])
        # append the traces
        trace_sum.append(result)

    # sum the traces
    trace_sum = sum(trace_sum).real
    return trace_sum


# ==================================================================================================
# Plot approximate dimension, geometric difference, model complexity or prediction error upper bound
# ==================================================================================================
def plot_d_g_s_p(
    d_g_s_p_tot_best, type, encs, feat_array, dataset_type, pr_kern, label_type=None
):
    """
    plot approximate dimension, geometric difference, model complexity, and peub

    This function creates plot of the d,g,s,p between kernels
    for different quantum encodings and number of qubits/features.

    d_g_s_p_tot_best: array of values of the best geometric difference for two kernels.
                It is a tensor 2,i,j where i is the encoding and j the feature,
                in the first position are stored values for the projected kernel
    type: str labelling if we plot geometric difference 'g', approximate dimension 'd',
          model complexity 's', prediction error upper bound 'p'
    encs: str of encoding strategies used to evaluate the geometric difference
    feat_array: array of features used to evaluate the geometric difference
    dataset_type: indicates if it is a 'mnist', 'higgs' or another specific dataset

    """
    colors = [
        ["blue", "darkblue"],
        ["goldenrod", "darkgoldenrod"],
        ["red", "darkred"],
        ["orange", "darkorange"],
        ["yellowgreen", "olive"],
        ["deepskyblue", "darkcyan"],
    ]

    plt.figure(figsize=(19, 13), facecolor="whitesmoke")

    if type == "d" or type == "g":
        for i in range(len(encs)):
            plt.plot(
                feat_array,
                d_g_s_p_tot_best[0, i, :],
                linewidth=1.2,
                linestyle="--",
                color=colors[i][0],
            )
            plt.plot(
                feat_array,
                d_g_s_p_tot_best[0, i, :],
                "2",
                color=colors[i][0],
                markersize=22,
                label=f"PQ ({encs[i]})",
            )
            """y1 = accuracy[i,:,0,0] + accuracy[i,:,0,1]
            y2 = accuracy[i,:,0,0] - accuracy[i,:,0,1]
            if i == 2:
                plt.fill_between(vec, y1, y2, alpha=0.1, color=colors[2][0], linewidth=0, label='σ standard deviation')
            else:
                plt.fill_between(vec, y1, y2, alpha=0.1, color=colors[i][0], linewidth=0)"""
            plt.plot(
                feat_array,
                d_g_s_p_tot_best[1, i, :],
                linewidth=1.2,
                linestyle="-",
                color=colors[i][1],
            )
            plt.plot(
                feat_array,
                d_g_s_p_tot_best[1, i, :],
                "x",
                color=colors[i][1],
                markersize=22,
                label=f"Q ({encs[i]})",
            )

        if type == "d":
            plt.title(
                f"Approximate Dimension - {dataset_type} N = {600} fidelity-{pr_kern}",
                fontsize=20,
            )
            plt.ylabel("d (dimension)", fontsize=18)
        elif type == "g":
            plt.title(
                f"Geometric Difference - {dataset_type} N = {600} gaussian-fidelity-{pr_kern}",
                fontsize=20,
            )
            # plt.ylim(13,25)
            plt.ylabel("g (geometric difference)", fontsize=18)

    if label_type == "quantum":
        for i in range(len(encs)):
            plt.plot(
                feat_array,
                d_g_s_p_tot_best[0, 0, i, :],
                linewidth=1.2,
                linestyle="--",
                color=colors[i][0],
            )
            plt.plot(
                feat_array,
                d_g_s_p_tot_best[0, 0, i, :],
                "2",
                color=colors[i][0],
                markersize=22,
                label=f"PQ ({encs[i]})",
            )
            """y1 = accuracy[i,:,0,0] + accuracy[i,:,0,1]
            y2 = accuracy[i,:,0,0] - accuracy[i,:,0,1]
            if i == 2:
                plt.fill_between(vec, y1, y2, alpha=0.1, color=colors[2][0], linewidth=0, label='σ standard deviation')
            else:
                plt.fill_between(vec, y1, y2, alpha=0.1, color=colors[i][0], linewidth=0)"""
            plt.plot(
                feat_array,
                d_g_s_p_tot_best[1, 0, i, :],
                linewidth=1.2,
                linestyle="-",
                color=colors[i][1],
            )
            plt.plot(
                feat_array,
                d_g_s_p_tot_best[1, 0, i, :],
                "x",
                color=colors[i][1],
                markersize=22,
                label=f"Q ({encs[i]})",
            )
            """y1 = accuracy[i,:,1,0] + accuracy[i,:,1,1]
            y2 = accuracy[i,:,1,0] - accuracy[i,:,1,1]
            plt.fill_between(vec, y1, y2, alpha=0.1, color=colors[i][1], linewidth=0)"""

            if type == "s" or type == "p":
                plt.plot(
                    feat_array,
                    d_g_s_p_tot_best[2, 0, i, :],
                    linewidth=1.2,
                    linestyle="-.",
                    color=colors[i + 3][0],
                )
                plt.plot(
                    feat_array,
                    d_g_s_p_tot_best[2, 0, i, :],
                    "o",
                    color=colors[i + 3][0],
                    markersize=10,
                    label=f"C ({encs[i]})",
                )

            if type == "s":
                plt.title(
                    f"Model complexity - {dataset_type} for N = {600} and {pr_kern}",
                    fontsize=20,
                )
                plt.ylabel("$s_K$(N) (model complexity)", fontsize=18)

            if type == "p":
                plt.title(
                    f"Prediction Error Upper Bound - {dataset_type} for N = {600} and {pr_kern}",
                    fontsize=20,
                )
                plt.ylabel("PEUB (prediction error upper bound)", fontsize=18)

    elif label_type == "classical":
        for i in range(len(encs)):
            plt.plot(
                feat_array,
                d_g_s_p_tot_best[0, 1, i, :],
                linewidth=1.2,
                linestyle="--",
                color=colors[i][0],
            )
            plt.plot(
                feat_array,
                d_g_s_p_tot_best[0, 1, i, :],
                "2",
                color=colors[i][0],
                markersize=22,
                label=f"PQ ({encs[i]})",
            )
            """y1 = accuracy[i,:,0,0] + accuracy[i,:,0,1]
            y2 = accuracy[i,:,0,0] - accuracy[i,:,0,1]
            if i == 2:
                plt.fill_between(vec, y1, y2, alpha=0.1, color=colors[2][0], linewidth=0, label='σ standard deviation')
            else:
                plt.fill_between(vec, y1, y2, alpha=0.1, color=colors[i][0], linewidth=0)"""
            plt.plot(
                feat_array,
                d_g_s_p_tot_best[1, 1, i, :],
                linewidth=1.2,
                linestyle="-",
                color=colors[i][1],
            )
            plt.plot(
                feat_array,
                d_g_s_p_tot_best[1, 1, i, :],
                "x",
                color=colors[i][1],
                markersize=22,
                label=f"Q ({encs[i]})",
            )
            """y1 = accuracy[i,:,1,0] + accuracy[i,:,1,1]
            y2 = accuracy[i,:,1,0] - accuracy[i,:,1,1]
            plt.fill_between(vec, y1, y2, alpha=0.1, color=colors[i][1], linewidth=0)"""

        if type == "s" or type == "p":
            plt.plot(
                feat_array,
                d_g_s_p_tot_best[2, 1, 0, :],
                linewidth=1.2,
                linestyle="-.",
                color=colors[4][0],
            )
            plt.plot(
                feat_array,
                d_g_s_p_tot_best[2, 1, 0, :],
                "o",
                color=colors[4][0],
                markersize=10,
                label=f"C",
            )

        if type == "s":
            plt.title(
                f"Model complexity - {dataset_type} for N = {600} and {pr_kern}",
                fontsize=20,
            )
            plt.ylabel("$s_K$(N) (model complexity)", fontsize=18)

        if type == "p":
            plt.title(
                f"Prediction Error Upper Bound - {dataset_type} for N = {600} and {pr_kern}",
                fontsize=20,
            )
            plt.ylabel("PEUB (prediction error upper bound)", fontsize=18)

    # plt.xticks(ticks=vec, labels=labels)
    plt.xlabel("system size (n)", fontsize=18)
    # plt.ylim(20,26)
    plt.tick_params(axis="both", labelsize=16)
    plt.legend(loc="best", fontsize=17)
    # plt.legend(bbox_to_anchor=(1,1), fontsize=17)

    if dataset_type == "mnist":
        folder = "../data/plots"
    elif dataset_type == "higgs":
        folder = "../Higgs_data/plots"

    if type == "d":
        plt.savefig(
            f"{folder}/approximate_dimension_{dataset_type}_{600}_gaussian-fidelity-{pr_kern}.png"
        )
    if type == "g":
        plt.savefig(
            f"{folder}/geometric_difference_{dataset_type}_{600}_gaussian-fidelity-{pr_kern}.png"
        )
    if type == "s":
        plt.savefig(
            f"{folder}/model_complexity_{dataset_type}_{label_type}_{600}_gaussian-fidelity-{pr_kern}.png"
        )
    if type == "p":
        plt.savefig(
            f"{folder}/peub_{dataset_type}_{label_type}_{600}_gaussian-fidelity-{pr_kern}.png"
        )
