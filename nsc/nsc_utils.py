"""Shared NSC-MP utilities: psi_mp with caching."""
import numpy as np
from scipy import integrate

_PSI_CACHE = {}


def mp_density(x, gamma):
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2
    if x < lm or x > lp:
        return 0.0
    return np.sqrt((lp - x) * (x - lm)) / (2 * np.pi * gamma * x)


def xavier_sigma(m, n):
    """Xavier/Glorot initialization std for an m x n matrix: sqrt(2 / (m + n))."""
    return np.sqrt(2.0 / (m + n))


he_sigma = xavier_sigma


def psi_mp(m, n, sigma=None):
    """Marchenko-Pastur spectral capacity: psi = n * E[log(1 + sigma^2 * m * x)].
    Default sigma: Xavier/Glorot init sqrt(2/(m+n)).
    """
    if sigma is None:
        sigma = xavier_sigma(m, n)
    key = (m, n, round(sigma, 10))
    if key in _PSI_CACHE:
        return _PSI_CACHE[key]
    if m < n:
        m, n = n, m
    if n == 0 or m == 0:
        _PSI_CACHE[key] = 0.0
        return 0.0
    gamma = n / m
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2
    s2m = sigma ** 2 * m

    def integrand(x):
        d = mp_density(x, gamma)
        return np.log(1 + s2m * x) * d if d > 0 else 0.0

    val = n * integrate.quad(integrand, lm, lp, limit=200)[0]
    _PSI_CACHE[key] = val
    return val
