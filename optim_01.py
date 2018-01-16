# Unconstrained minimization of multivariate scalar functions (minimize)
# the problem of minimizing the Rosenbrock function of N variables:

import numpy as np
from scipy.optimize import minimize

def Rosenbrock(x, a):
    return sum(100.0*(x[1:]-x[:-1]**2)**2 + (a-x[:-1])**2)

# method = 'nelder-mead'
# method = 'BFGS'
# method = 'Newton-CG' # required Jacobian
# method = 'trust-ncg' # required Jacobian

x0 = np.ones(5)
for a in np.linspace(1, 0.1, 11):
    fun = lambda x: Rosenbrock(x,a)
    res = minimize(fun, x0, method=method,        options={'xtol': 1e-8, 'disp': True})
    print(a, res.x)
    x0 = res.x
