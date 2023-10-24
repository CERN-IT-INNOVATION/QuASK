from enum import Enum


class KernelType(Enum):
    """
    Possible types of kernel
    """
    FIDELITY = 0
    OBSERVABLE = 1
    SWAP_TEST = 2

    @staticmethod
    def convert(item):
        if isinstance(item, KernelType):
            return item
        elif item < 0.5: 
            return KernelType.FIDELITY
        elif 0.5 <= item < 1.5: 
            return KernelType.OBSERVABLE
        else: 
            return KernelType.SWAP_TEST
