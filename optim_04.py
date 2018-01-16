# Bounded minimization (method='bounded')
# Very often, there are constraints that can be placed on the solution space before minimization occurs. The bounded method in minimize_scalar is an example of a constrained minimization procedure that provides a rudimentary interval constraint for scalar functions. The interval constraint allows the minimization to occur only between two fixed endpoints, specified using the mandatory bounds parameter.
#
# For example, to find the minimum of J1(x) near x=5 , minimize_scalar can be called using the interval [4,7] as a constraint. The result is xmin=5.3314 :

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import j1

res = minimize_scalar(j1, bounds=(4, 7), method='bounded')
print(res.x)

import matplotlib.pyplot as plt
x = np.linspace(4,7,100)
y = j1(x)
plt.plot(x,y)
plt.show()
