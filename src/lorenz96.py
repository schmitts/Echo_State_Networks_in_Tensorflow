from numba import jit
import numpy as np
from scipy.integrate import odeint


@jit(nopython=True)
def Lorenz96(x, t, N, F):

    # compute state derivatives
    d = np.zeros(N)
    # first the 3 edge cases: i=1,2,N
    d[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
    d[1] = (x[2] - x[N-1]) * x[0] - x[1]
    d[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
    # then the general case
    for i in range(2, N-1):
        d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
    # add the forcing term
    d = d + F

    # return the state derivatives
    return d


def calcLorenz(N, F, tmax=50, dt=0.001, disP=[3], disStr=[0.01], offSetT=0):
    x0 = F*np.ones(N)  # initial state (equilibrium)
    for dP, dS in zip(disP, disStr):
        x0[dP] += dS  # add small perturbation to 20th variable
    t = np.arange(0.0, tmax+(offSetT*dt), dt)

    x = odeint(Lorenz96, x0, t, args=(N, F))
    x = x.transpose()

    return x
