from . import Ansatz, KernelType


class KernelFactory:
    """
    Instantiate the concrete object from classes that inherit from (abstract class) Kernel.
    Implement the self-registering factory pattern
    """

    __implementations = {}
    """Dictionary containing pairs (name, function to create the kernel)."""

    __current_implementation: str = ""
    """Name of the implementation to use right now to create the kernels"""

    @staticmethod
    def add_implementation(name, fn):
        """
        Add the current closure function as one of the possible implementations available

        :param name: name of the implementation
        :param fn: function that creates the quantum kernel
        """
        if name in KernelFactory.__implementations:
            raise ValueError("This name is already present in the register of available implementations")
        if fn.__code__.co_argcount != 3:
            raise ValueError("The function must have these three arguments, 'ansatz', 'measurement', and 'type': the number of argument does not match")
        if fn.__code__.co_varnames != ('ansatz', 'measurement', 'type'):
            raise ValueError("The function must have these three arguments, 'ansatz', 'measurement', and 'type': the name of some argument does not match")
        KernelFactory.__implementations[name] = fn


    @staticmethod
    def set_current_implementation(name):
        if name not in KernelFactory.__implementations:
            raise ValueError("This name is not present in the register of available implementations")
        KernelFactory.__current_implementation = name

    @staticmethod
    def create_kernel(ansatz: Ansatz, measurement: str, type: KernelType):
        """
        Create a kernel object using the default class chosen.

        :param ansatz: Ansatz object representing the unitary transformation
        :param measurement: Pauli string representing the measurement
        :param type: type of kernel, fidelity, swap test or observable
        :return: kernel object of the default concrete class
        """
        fn = KernelFactory.__implementations[KernelFactory.__current_implementation]
        return fn(ansatz, measurement, type)