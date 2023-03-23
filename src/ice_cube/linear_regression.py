import numpy as np
from numba import njit


@njit(cache=True)
def compute_angle_numba(x, y, z):
    covx = np.cov(z, x)
    mx = covx[0, 1] / covx[0, 0]
    covy = np.cov(z, y)
    my = covy[0, 1] / covy[0, 0]
    zr = 1
    xr = mx * zr
    yr = my * zr
    r = np.sqrt(zr**2 + xr**2 + yr**2)
    azimuth = np.arctan2(yr, xr)
    if azimuth < 0:
        azimuth = 2 * np.pi + azimuth
    zenith = np.arccos(zr / r)
    return azimuth, zenith
