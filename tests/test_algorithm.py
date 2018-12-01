import numpy as np

from sm2rain.algorithm import sm2rain
from sm2rain.algorithm import calib_sm2rain
from sm2rain.algorithm import cost_func


def test_sm2rain():
    """
    Test sm2rain algorithm.
    """
    a = 1
    b = 2
    z = 3

    sm = np.arange(100)
    p_obs = z * (sm[1:] - sm[:-1]) + ((a * sm[1:]**b + a * sm[:-1]**b) / 2.)
    p_sim = sm2rain(sm, a, b, z)

    np.testing.assert_array_equal(p_sim, p_obs)


def test_calib_sm2rain():
    """
    Test calibration of sm2rain.

    """
    a = 1
    b = 2
    z = 3

    sm = np.arange(100)
    p_obs = z * (sm[1:] - sm[:-1]) + ((a * sm[1:]**b + a * sm[:-1]**b) / 2.)

    result = calib_sm2rain(sm, p_obs)
    np.testing.assert_almost_equal(np.array([a, b, z]), result, decimal=1)


def test_cost_func():
    """
    Test cost function.
    """
    a = 1
    b = 2
    z = 3

    x0 = (a, b, z)
    sm = np.arange(100)
    p_obs = z * (sm[1:] - sm[:-1]) + ((a * sm[1:]**b + a * sm[:-1]**b) / 2.)

    rmsd = cost_func(x0, sm, p_obs)

    np.testing.assert_almost_equal(rmsd, 0, decimal=3)
