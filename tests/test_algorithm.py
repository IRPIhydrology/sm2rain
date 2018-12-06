import numpy as np

from sm2rain.algorithm import ts_sm2rain
from sm2rain.algorithm import calib_sm2rain
from sm2rain.algorithm import cost_func


def test_single_ts_sm2rain():
    """
    Test single soil moisture time series input into ts_sm2rain algorithm.
    """
    a = 1
    b = 2
    z = 3

    sm = np.arange(100)
    p_obs = z * (sm[1:] - sm[:-1]) + (
        (a * sm[1:]**b + a * sm[:-1]**b) / 2.)

    p_sim = ts_sm2rain(sm, a, b, z)

    np.testing.assert_array_equal(p_sim, p_obs)


def test_multi_ts_sm2rain():
    """
    Test multiple soil moisture time series input into ts_sm2rain algorithm.
    """
    n = 10
    a = 1
    b = 2
    z = 3

    sm = np.arange(100)[:, np.newaxis].repeat(n, axis=1)

    p_obs = z * (sm[1:] - sm[:-1]) + (
        (a * sm[1:]**b + a * sm[:-1]**b) / 2.)

    p_sim = ts_sm2rain(sm, a, b, z)

    np.testing.assert_array_equal(p_sim, p_obs)


def test_multi_ts_multi_param_ts_sm2rain():
    """
    Test multiple soil moisture time series input and multiple parameter
    input into ts_sm2rain algorithm.
    """
    n = 10

    a = np.ones(n)
    b = a + 2
    z = a + 4

    sm = np.arange(100)[:, np.newaxis].repeat(n, axis=1)

    p_obs = z * (sm[1:] - sm[:-1]) + (
        (a * sm[1:]**b + a * sm[:-1]**b) / 2.)

    p_sim = ts_sm2rain(sm, a, b, z)

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


if __name__ == '__main__':
    test_single_ts_sm2rain()
    test_multi_ts_sm2rain()
    test_multi_ts_multi_param_ts_sm2rain()
    test_calib_sm2rain()
    test_cost_func()
