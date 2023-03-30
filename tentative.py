import numpy as np

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

def sphere(x):
  return np.sum(x ** 2)

n_agents = 20
n_variables = 2
lower_bound = [-10, -10]
upper_bound = [10, 10]

space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()
function = Function(sphere)





opt = Opytimizer(space, optimizer, function)
opt.start(n_iterations=2, callbacks=[CustomCallback()])
