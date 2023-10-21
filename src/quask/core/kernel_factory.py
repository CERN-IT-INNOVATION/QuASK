from . import Ansatz, KernelType


class KernelFactory:
    """
    Instantiate the concrete object from classes that inherit from (abstract class) Kernel.
    Implement the self-registering factory pattern
    """

    @staticmethod
    def create_kernel(ansatz: Ansatz, measurement: str, type: KernelType):
        """
        Create a kernel object using the default class chosen
        :param ansatz: Ansatz object representing the unitary transformation
        :param measurement: Pauli string representing the measurement
        :param type: type of kernel, fidelity or observable
        :return: kernel object of the default concrete class
        """
        from ..core_implementation import PennylaneKernel
        return PennylaneKernel(ansatz, measurement, type)
