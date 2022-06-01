
class TrainableKernelRegister:
    def __init__(self):
        self.kernel_functions = []
        self.kernel_names = []
        self.kernel_hyperparams = []
        self.current = 0

    def register(self, fn, name, params):
        self.kernel_functions.append(fn)
        self.kernel_names.append(name)
        self.kernel_hyperparams.append(params)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= len(self.kernel_functions):
            raise StopIteration
        ret = (self.kernel_functions[self.current], self.kernel_names[self.current], self.kernel_hyperparams[self.current])
        self.current += 1
        return ret


the_trainable_kernel_register = TrainableKernelRegister()







