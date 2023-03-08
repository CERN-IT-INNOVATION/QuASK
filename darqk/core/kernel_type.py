from enum import Enum


class KernelType(Enum):
    """
    Possible types of kernel
    """
    FIDELITY = 0
    OBSERVABLE = 1