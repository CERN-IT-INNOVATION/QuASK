from enum import Enum


class KernelType(Enum):
    """
    Possible types of kernel
    """
    FIDELITY = 0
    OBSERVABLE = 1

    @staticmethod
    def convert(item):
        if isinstance(item, KernelType):
            return item
        else:
            if item < 0.5:
                return KernelType.FIDELITY
            else:
                return KernelType.OBSERVABLE