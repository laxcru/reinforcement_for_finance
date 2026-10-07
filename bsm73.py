#
# Valuation of European call options
# in Black-Scholes-Merton (1973) model
#
# (c) Dr. Yves J. Hilpisch
# Reinforcement Learning for Finance
#

from math import log, sqrt, exp
from scipy import stats


def bsm_call_value(St, K, T, t, r, sigma):
    ''' Valuation of European call option in BSM model.
    Analytical formula.

    Parameters
    ==========
    St: float
        stock/index level at date/time t
    K: float
        fixed strike price
    T: float
        maturity date/time (in year fractions)
    t: float
        current data/time
    r: float
        constant risk-free short rate
    sigma: float
        volatility factor in diffusion term

    Returns
    =======
    value: float
        present value of the European call option
    '''
    St = float(St)
    d1 = (log(St / K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * sqrt(T - t))
    d2 = (log(St / K) + (r - 0.5 * sigma ** 2) * (T - t)) / (sigma * sqrt(T - t))
    # stats.norm.cdf --> cumulative distribution function
    #                    for normal distribution
    value = (St * stats.norm.cdf(d1, 0, 1) -
             K * exp(-r * (T - t)) * stats.norm.cdf(d2, 0, 1))
    return value

