"""
Module to compute and calibrate sm2rain.
"""

import numpy as np
from scipy.optimize import minimize


def sm2rain(sm, a, b, z, thr=None):
    """
    Retrieve rainfall from soil moisture.

    Parameters
    ----------
    sm : numpy.ndarray
        Soil moisture time series.
    a : float
        a parameter, units mm.
    b : float
        b parameter, units -.
    z : float
        Z parameter, units mm.

    Returns
    -------
    p_sim : numpy.ndarray
        Precipitation time series.
    """
    p_sim = z * (sm[1:] - sm[:-1]) + ((a * sm[1:]**b + a * sm[:-1]**b) / 2.)
    p_sim = np.clip(p_sim, 0, thr)

    return p_sim


def calib_sm2rain(sm, p_obs, x0=None, bounds=None,
                  options=None, method='TNC'):
    """
    Calibrate parameters z, a, b.

    Parameters
    ----------
    sm : numpy.ndarray
        Soil moisture time series.
    p_obs : numpy.ndarray
        Precipitation time series.

    Returns
    -------
    a : float
        a parameter, units mm.
    b : float
        b parameter, units -.
    z : float
        Z parameter, units mm.
    """
    if x0 is None:
        x0 = np.array([20., 5., 80.])

    if bounds is None:
        bounds = ((0, 200), (0.01, 50), (1, 800))

    if options is None:
        options = {'ftol': 1e-8, 'maxiter': 3000, 'disp': False}

    result = minimize(cost_func, x0, args=(sm, p_obs),
                      method=method, bounds=bounds, options=options)

    a, b, z = result.x

    return a, b, z


def cost_func(x0, sm, p_obs):
    """
    Cost function.

    Parameters
    ----------
    x0 : tuple
        Start parameters (a, b, z).
    sm : numpy.ndarray
        Soil moisture time series.
    p_obs : numpy.ndarray
        Observed precipitation time series.

    Returns
    -------
    rmsd : float
        Root mean square difference between p_sim and p_obs.
    """
    p_sim = sm2rain(sm, x0[0], x0[1], x0[2])
    rmsd = np.nanmean((p_obs - p_sim)**2)**0.5

    return rmsd


def soil_water_index(sm, jd, t=2.):
    """
    Soil water index computation.

    Parameters
    ----------
    sm : numpy.ndarray
        Soil moisture time series.
    jd : numpy.ndarray
        Julian date time series.
    t : float, optional
        t parameter (default: 2).

    Returns
    -------
    swi : numpy.ndarray
        Soil water index time series.
    k : numpy.ndarray
        Gain parameter time series.
    """
    n = sm.size

    swi = np.zeros(n)
    swi[np.isnan(sm)] = np.nan

    k = np.ones(n)
    k[np.isnan(sm)] = np.nan

    idx = np.arange(n)[~np.isnan(sm)]
    n_not_nan = idx.size

    for i in np.arange(1, n_not_nan):
        dt = jd[idx[i]] - jd[idx[i-1]]
        k[idx[i]] = k[idx[i-1]] / (k[idx[i-1]] + np.exp(-dt/t))
        swi[idx[i]] = swi[idx[i-1]] + k[idx[i]] * (sm[idx[i]] - swi[idx[i-1]])

    return swi, k
