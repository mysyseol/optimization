# Custom minimizers
# Sometimes, it may be useful to use a custom method as a (multivariate or univariate) minimizer, for example when using some library wrappers of minimize (e.g. basinhopping).
#
# We can achieve that by, instead of passing a method name, we pass a callable (either a function or an object implementing a __call__ method) as the method parameter.
#
# Let us consider an (admittedly rather virtual) need to use a trivial custom multivariate minimization method that will just search the neighborhood in each dimension independently with a fixed step size:


import numpy as np
from scipy.optimize import OptimizeResult, minimize
# import Rosenbrock

def custmin(fun, x0, args=(), maxfev=None, stepsize=0.1, maxiter=100, callback=None, **options):

    bestx = x0
    besty = fun(x0)
    funcalls = 1
    niter = 0
    improved = True
    stop = False

    while improved and not stop and niter < maxiter:
        improved = False
        niter += 1
        for dim in range(np.size(x0)):
            for s in [bestx[dim] - stepsize, bestx[dim] + stepsize]:
                testx = np.copy(bestx)
                testx[dim] = s
                testy = fun(testx, *args)
                funcalls += 1
                if testy < besty:
                    besty = testy
                    bestx = testx
                    improved = True

            if callback is not None:
                callback(bestx)
            if maxfev is not None and funcalls >= maxfev:
                stop = True
                break

        return OptimizeResult(fun=besty, x=bestx, nit=niter, nfev=funcalls, success=(niter > 1))

def Rosenbrock(x):
    return sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)


x0 = [1.3, 0.9, 0.8, 1.1, 1.1]
res = minimize(Rosenbrock, x0, method=custmin, options=dict(stepsize=0.05))
print(res.x)
