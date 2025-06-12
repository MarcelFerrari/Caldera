# Operators for heat problem
import numpy as np
import numba as nb

@nb.njit(cache=True, parallel=True)
def compute_qx(nx1, ny1, dx, k, T, qx):
    # qx = - k * dT/dx
    for i in nb.prange(ny1 - 1):
        for j in range(1, nx1 - 1):
            kij = 0.5 * (k[i, j] + k[i, j - 1]) # Average conductivity
            qx[i, j] = - kij * (T[i, j] - T[i, j - 1])/dx

    return qx

@nb.njit(cache=True, parallel=True)
def compute_qy(nx1, ny1, dy, k, T, qy):
    # qy = - k * dT/dy
    for i in nb.prange(1, ny1 - 1):
        for j in range(nx1 - 1):
            kij = 0.5 * (k[i, j] + k[i-1, j]) # Average conductivity
            qy[i, j] = - kij * (T[i, j] - T[i-1, j])/dy

    return qy

    