"""
Module dedicated to the applying classical and quantum kernel on the given datasets.

Args:
    the_kernel_register: singleton global instance of KernelRegister
"""


from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel
from .template_pennylane import (
    zz_fullentanglement_embedding,
    pennylane_quantum_kernel,
    pennylane_projected_quantum_kernel,
    random_qnn_encoding,
)


class KernelRegister:
    """
    List of datasets available in this module. The object is iterable.
    """

    def __init__(self):
        """
        Init method.

        Returns:
            None
        """
        self.kernel_functions = []
        self.kernel_names = []
        self.parameters = []
        self.current = 0

    def register(self, fn, name, params):
        """
        Register a new kernel.

        Args:
            fn: function pointer to a kernel function having exactly three parameters: X_1, X_2 numpy training and
                testing Gram matrices, and a possibly empty list of parameters
            name: name of the kernel (str)
            params: list of parameters (List[str])

        Returns:
            None
        """
        self.kernel_functions.append(fn)
        self.kernel_names.append(name)
        self.parameters.append(params)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= len(self.kernel_functions):
            raise StopIteration
        ret = (
            self.kernel_functions[self.current],
            self.kernel_names[self.current],
            self.parameters[self.current],
        )
        self.current += 1
        return ret


# create the global register
the_kernel_register = KernelRegister()

# register linear kernel
linear_kernel_wrapper = lambda X1, X2, params: linear_kernel(X1, X2)
the_kernel_register.register(linear_kernel_wrapper, "linear_kernel", [])

# register gaussian kernel
rbf_kernel_wrapper = lambda X1, X2, params: rbf_kernel(X1, X2, gamma=float(params[0]))
the_kernel_register.register(rbf_kernel_wrapper, "rbf_kernel", ["gamma"])

# register polynomial kernel
poly_kernel_wrapper = lambda X1, X2, params: polynomial_kernel(
    X1, X2, degree=int(params[0])
)
the_kernel_register.register(poly_kernel_wrapper, "poly_kernel", ["degree"])

# register custom quantum kernels
zz_quantum_kernel = lambda X_1, X_2, params: pennylane_quantum_kernel(
    zz_fullentanglement_embedding, X_1, X_2
)
the_kernel_register.register(zz_quantum_kernel, "zz_quantum_kernel", [])

projected_zz_quantum_kernel = (
    lambda X_1, X_2, params: pennylane_projected_quantum_kernel(
        zz_fullentanglement_embedding, X_1, X_2, params
    )
)
the_kernel_register.register(
    projected_zz_quantum_kernel, "projected_zz_quantum_kernel", ["gamma"]
)

random_quantum_kernel = lambda X_1, X_2, params: pennylane_quantum_kernel(
    random_qnn_encoding, X_1, X_2, params
)
the_kernel_register.register(random_quantum_kernel, "random_quantum_kernel", ["gamma"])

projected_random_quantum_kernel = (
    lambda X_1, X_2, params: pennylane_projected_quantum_kernel(
        random_qnn_encoding, X_1, X_2, params
    )
)
the_kernel_register.register(
    projected_random_quantum_kernel, "projected_random_quantum_kernel", ["gamma"]
)
