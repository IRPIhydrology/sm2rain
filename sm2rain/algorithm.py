"""Module to compute and calibrate sm2rain."""

import numpy as np
from scipy.optimize import minimize
np.seterr(invalid='ignore')

# handle the case when numba is not available
try:
    from numba import jit
    _numba_available = True
except ImportError:
    _numba_available = False


def ts_sm2rain(sm, a, b, z, jdates=None, T=None, c=None, thr=None):
    """
    Retrieve rainfall from soil moisture.

    Parameters
    ----------
    sm : numpy.ndarray
        Single or multiple soil moisture time series.
    a : float, numpy.ndarray
        a parameter, units mm.
    b : float, numpy.ndarray
        b parameter, units -.
    z : float, numpy.ndarray
        Z parameter, units mm.
    jdates: numpy.ndarray
        Julian date time series.
    T : float, numpy.ndarray
        T parameter, units days.
    c : float, numpy.ndarray
        Tpot parameter, units days.
    thr : float, optional
        Upper threshold of p_sim (default: None).

    Returns
    -------
    p_sim : numpy.ndarray
        Single or multiple simulated precipitation time series.
    """
    if T is not None and c is None:
        swi = swicomp_nan(sm, jdates, T)
        swi = (swi-np.nanmin(swi))/(np.nanmax(swi)-np.nanmin(swi))
    elif c is not None:
        swi = swi_pot_nan(sm, jdates, T, c)
        swi = (swi-np.nanmin(swi))/(np.nanmax(swi)-np.nanmin(swi))
    else:
        swi = sm

    if jdates is None:
        jdates = np.arange(0, len(sm))

    p_sim = z * (swi[1:] - swi[:-1]) + \
        ((a * swi[1:]**b + a * swi[:-1]**b)*(jdates[1:]-jdates[:-1]) / 2.)

    p_sim[abs(np.diff(swi)) <= 0.0001] = 0
    p_sim[p_sim > 999999] = np.nan

    return np.clip(p_sim, 0, thr)


def calib_sm2rain(sm, p_obs, x0=None, bounds=None,
                  options=None, method='TNC'):
    """
    Calibrate sm2rain parameters a, b, z.

    Parameters
    ----------
    sm : numpy.ndarray
        Soil moisture time series.
    p_obs : numpy.ndarray
        Precipitation time series.
    x0 : tuple, optional
        Initial guess of a, b, z (default: (20, 5, 80)).
    bounds : tuple of tuples, optional
        Boundary values for a, b, z
        (default: ((0, 200), (0.01, 50), (1, 800)).
    options : dict, optional
        A dictionary of solver options
        (default: {'ftol': 1e-8, 'maxiter': 3000, 'disp': False}).
        For more explanation/options see scipy.minimize.
    method : str, optional
        Type of solver (default: 'TNC', i.e. Truncated Newton).
        For more explanation/options see scipy.minimize.

    Returns
    -------
    a : float
        a parameter, units mm.
    b : float
        b parameter, units -.
    z : float
        z parameter, units mm.
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
        Initial guess of parameters a, b, z.
    sm : numpy.ndarray
        Single soil moisture time series.
    p_obs : numpy.ndarray
        Observed precipitation time series.

    Returns
    -------
    rmsd : float
        Root mean square difference between p_sim and p_obs.
    """
    p_sim = ts_sm2rain(sm, x0[0], x0[1], x0[2])
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
        t parameter, the unit is fraction of days (default: 2).

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


def calib_sm2rain_T(jdates, sm, p_obs, NN, x0=None, bounds=None,
                    options=None, method='TNC'):
    """
    Calibrate sm2rain parameters a, b, z, T.

    Parameters
    ----------
    jdates: numpy.ndarray
        Julian date time series.
    sm : numpy.ndarray
        Soil moisture time series.
    p_obs : numpy.ndarray
        Precipitation time series.
    NN : integer
        Data aggregation coefficient
    x0 : tuple, optional
        Initial guess of a, b, z, T (default: (8, 5.9, 49, 1.5)).
    bounds : tuple of tuples, optional
        Boundary values for a, b, z, T
        (default: ((0, 160), (1, 50), (10, 400), (0, 15))).
    options : dict, optional
        A dictionary of solver options
        (default: {'ftol': 1e-8, 'maxiter': 3000, 'disp': False}).
        For more explanation/options see scipy.minimize.
    method : str, optional
        Type of solver (default: 'TNC', i.e. Truncated Newton).
        For more explanation/options see scipy.minimize.

    Returns
    -------
    a : float
        a parameter, units mm.
    b : float
        b parameter, units -.
    z : float
        z parameter, units mm.
    T : float
        T parameter, units days.
    """
    if bounds is None:
        bounds = ((0, 200), (0.01, 50), (1, 800), (0.0001, 8))

    if x0 is None:
        p = [0.1, 0.1, 0.1, 0.]
        x0 = np.array([(bounds[i][1] - bounds[i][0])*p[i] + bounds[i][0]
                       for i in range(len(bounds))])

    if options is None:
        options = {'ftol': 1e-8, 'maxiter': 3000, 'disp': False}

    result = minimize(cost_func_T, x0, args=(jdates, sm, p_obs, NN),
                      method=method, bounds=bounds, options=options)

    a, b, z, T = result.x

    return a, b, z, T


def cost_func_T(x0, jdates, sm, p_obs, NN):
    """
    Cost function.

    Parameters
    ----------
    x0 : tuple
        Initial guess of parameters a, b, z, T.
    jdates: numpy.ndarray
        Julian date time series.
    sm : numpy.ndarray
        Soil moisture time series.
    p_obs : numpy.ndarray
        Observed precipitation time series.
    NN : integer
        Data aggregation coefficient

    Returns
    -------
    rmsd : float
        Root mean square difference between p_sim and p_obs.
    """
    p_sim = ts_sm2rain(sm, x0[0], x0[1], x0[2], jdates, x0[3])

    p_sim1 = np.add.reduceat(p_sim, np.arange(0, len(p_sim), NN))
    p_obs1 = np.add.reduceat(p_obs, np.arange(0, len(p_obs), NN))
    rmsd = np.nanmean((p_obs1 - p_sim1)**2)**0.5

    return rmsd


def swicomp_nan(in_data, in_jd, ctime):
    """
    Calculates exponentially smoothed time series using an
    iterative algorithm

    Parameters
    ----------
    in_data : double numpy.array
        input data
    in_jd : double numpy.array
        julian dates of input data
    ctime : int
        characteristic time used for calculating
        the weight
    """

    filtered = np.empty(len(in_data))
    gain = 1
    filtered.fill(np.nan)

    ID = np.where(~np.isnan(in_data))
    D = in_jd[ID]
    SWI = in_data[ID]
    tdiff = np.diff(D)

    # find the first non nan value in the time series

    for i in range(2, SWI.size):
        gain = gain / (gain + np.exp(- tdiff[i - 1] / ctime))
        SWI[i] = SWI[i - 1] + gain * (SWI[i] - SWI[i-1])

    filtered[ID] = SWI

    return filtered


def calib_sm2rain_Tpot(jdates, sm, p_obs, NN, x0=None, bounds=None,
                       options=None, method='TNC'):
    """
    Calibrate sm2rain parameters a, b, z, T, c.

    Parameters
    ----------
    jdates: numpy.ndarray
        Julian date time series.
    sm : numpy.ndarray
        Soil moisture time series.
    p_obs : numpy.ndarray
        Precipitation time series.
    NN : integer
        Data aggregation coefficient
    x0 : tuple, optional
        Initial guess of a, b, z, T, c
        (default: (10%, 5%, 10%, 10%, 10%) of bounds limits).
    bounds : tuple of tuples, optional
        Boundary values for a, b, z, T, c
        (default: ((0, 160), (1, 50), (10, 400), (0.05 3.00), (0.05 0.75))).
    options : dict, optional
        A dictionary of solver options
        (default: {'ftol': 1e-8, 'maxiter': 3000, 'disp': False}).
        For more explanation/options see scipy.minimize.
    method : str, optional
        Type of solver (default: 'TNC', i.e. Truncated Newton).
        For more explanation/options see scipy.minimize.

    Returns
    -------
    a : float
        a parameter, units mm.
    b : float
        b parameter, units -.
    z : float
        z parameter, units mm.
    T : float
        Tbase parameter, units days.
    c : float
        Tpot parameter, units -.
    """
    if bounds is None:
        bounds = ((0, 200), (0.01, 50), (1, 800), (0.05, 3.), (0.05, 0.75))

    if x0 is None:
        p = [0.1, 0.05, 0.1, 0.1, 0.1]
        x0 = np.array([(bounds[i][1] - bounds[i][0]) * p[i] + bounds[i][0]
                       for i in range(len(bounds))])

    if options is None:
        options = {'ftol': 1e-8, 'maxiter': 3000, 'disp': False}

    result = minimize(cost_func_Tpot, x0, args=(jdates, sm, p_obs, NN),
                      method=method, bounds=bounds, options=options)

    a, b, z, T, c = result.x

    return a, b, z, T, c


def cost_func_Tpot(x0, jdates, sm, p_obs, NN):
    """
    Cost function.

    Parameters
    ----------
    x0 : tuple
        Initial guess of parameters a, b, z, T, Tpot.
    jdates: numpy.ndarray
        Julian date time series.
    sm : numpy.ndarray
        Soil moisture time series.
    p_obs : numpy.ndarray
        Observed precipitation time series.
    NN : integer
        Data aggregation coefficient

    Returns
    -------
    rmsd : float
        Root mean square difference between p_sim and p_obs.
    """
    p_sim = ts_sm2rain(sm, x0[0], x0[1], x0[2], jdates, x0[3], x0[4])

    p_sim1 = np.add.reduceat(p_sim, np.arange(0, len(p_sim), NN))
    p_obs1 = np.add.reduceat(p_obs, np.arange(0, len(p_obs), NN))
    rmsd = np.nanmean((p_obs1 - p_sim1)**2)**0.5

    return rmsd


def swi_pot_nan(sm, jd, t, POT):
    """
    Soil water index computation.

    Parameters
    ----------
    sm : numpy.ndarray
        Soil moisture time series.
    jd : numpy.ndarray
        Julian date time series.
    t : float, optional
        t parameter, the unit is fraction of days (default: 2).

    Returns
    -------
    swi : numpy.ndarray
        Soil water index time series.
    k : numpy.ndarray
        Gain parameter time series.
    """

    idx = np.where(~np.isnan(sm))[0]

    swi = np.empty(len(sm))
    swi[:] = np.nan
    swi[idx[0]] = sm[idx[0]]

    Tupd = t * sm[idx[0]] ** (- POT)
    gain = 1

    for i in range(1, idx.size):

        dt = jd[idx[i]] - jd[idx[i-1]]

        gain0 = gain / (gain + np.exp(- dt / Tupd))
        swi[idx[i]] = swi[idx[i - 1]] + gain0 * (sm[idx[i]] - swi[idx[i - 1]])
        Tupd = t * swi[idx[i]] ** (- POT)

        gain0 = gain / (gain + np.exp(- dt / Tupd))
        swi[idx[i]] = swi[idx[i - 1]] + gain0 * (sm[idx[i]] - swi[idx[i - 1]])
        Tupd = t * swi[idx[i]] ** (- POT)

        gain0 = gain / (gain + np.exp(- dt / Tupd))
        swi[idx[i]] = swi[idx[i - 1]] + gain0 * (sm[idx[i]] - swi[idx[i - 1]])
        Tupd = t * swi[idx[i]] ** (- POT)

        gain = gain0

    return swi


if _numba_available:
    # perform numba-jit compilation of swi_pot_nan to speed up
    # recursive calculations
    swi_pot_nan = jit(swi_pot_nan, nopython=True, nogil=True)
    swicomp_nan = jit(swicomp_nan, nopython=True, nogil=True)
